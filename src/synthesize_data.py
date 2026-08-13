"""
Generates synthetic training pairs to plug texture-coverage gaps in the real
3,200-pair training set. Root cause of hard cases like `002929` (water-ripple)
was diagnosed by direct measurement, not guesswork: searching all training GT
images for a similar frequency-spectrum signature found only ~1 independent
real example of that texture family. No loss/architecture change can fix a
coverage gap -- this targets it directly by resampling/recomposing texture
from the few real examples that DO have it into new spatial arrangements,
then re-degrading with the real dataset's own fitted noise model so the
synthetic pairs are statistically consistent with real training data.

Pipeline:
  1. fit_noise_model()      -- multiplicative speckle noise, std(brightness),
                                fit from real (NoisyLR, GT_full) pairs.
  2. find_rare_textures()   -- radially-averaged PSD per GT image, flag images
                                with few close neighbors as coverage gaps.
  3. synthesize_variant()   -- patch-quilt a rare source into a new 256x256
                                canvas (new spatial arrangement of its local
                                texture, not a crop/flip of the same layout),
                                then degrade with the fitted noise model.

Everything here operates only on the officially provided training set --
no external imagery.
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import resize as sk_resize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
NOISY_DIR = os.path.join(BASE, "data", "train", "NoisyLR", "NoisyLR")
GT_DIR = os.path.join(BASE, "data", "train", "GT_full", "GT")
NOISY_SYNTH_DIR = os.path.join(BASE, "data", "train", "NoisyLR_synth")
GT_SYNTH_DIR = os.path.join(BASE, "data", "train", "GT_synth")
PREVIEW_DIR = os.path.join(BASE, "outputs", "synth_preview")


# ── 1. Noise model ──────────────────────────────────────────────────────────

def fit_noise_model(n_samples=400, seed=0):
    """Multiplicative noise: noisy ~= gt_downsampled * (mean(b) + std(b)*z),
    z ~ N(0,1). Returns (bin_centers, mean_ratio, std_ratio) for interpolation."""
    rng = random.Random(seed)
    files = sorted(os.listdir(NOISY_DIR))
    sample = rng.sample(files, min(n_samples, len(files)))

    edges = np.linspace(0, 1, 11)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_vals = [[] for _ in range(10)]

    for f in sample:
        n = np.load(os.path.join(NOISY_DIR, f)).astype(np.float32)
        g = np.load(os.path.join(GT_DIR, f)).astype(np.float32)
        g_ds = sk_resize(g, n.shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
        ratio = n / (g_ds + 1e-3)
        b = np.clip(np.digitize(g_ds.ravel(), edges) - 1, 0, 9)
        r = ratio.ravel()
        for i in range(10):
            bin_vals[i].extend(r[b == i].tolist())

    mean_ratio = np.array([np.mean(v) if v else 1.0 for v in bin_vals])
    std_ratio = np.array([np.std(v) if v else 0.15 for v in bin_vals])
    return centers, mean_ratio, std_ratio


def apply_noise_model(clean_128, noise_curve, rng):
    """clean_128: (128,128) float32 in [0,1]. Returns synthetic noisy input,
    same multiplicative-noise statistics as the real dataset, not clipped to
    [0,1] (real NoisyLR files aren't either -- see model.py's evidence map)."""
    centers, mean_ratio, std_ratio = noise_curve
    mean_b = np.interp(clean_128, centers, mean_ratio)
    std_b = np.interp(clean_128, centers, std_ratio)
    z = rng.standard_normal(clean_128.shape).astype(np.float32)
    ratio = mean_b + std_b * z
    return (clean_128 * ratio).astype(np.float32)


# ── 2. Rare-texture detection via PSD similarity ────────────────────────────

def radial_psd(img, n_bins=32):
    """Radially-averaged power spectral density, log-scaled, L2-normalized --
    a compact rotation-invariant texture-frequency fingerprint."""
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = min(cy, cx)
    edges = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if mask.any():
            profile[i] = power[mask].mean()
    profile = np.log1p(profile)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def load_holdout_filenames():
    """Files reserved as the held-out validation split (train_big.py's fixed-
    seed 90/10 split, written to outputs/val_filenames.txt). Synthesis must
    never touch these -- using a held-out image as texture source material
    would leak validation content into training, even indirectly."""
    path = os.path.join(BASE, "outputs", "val_filenames.txt")
    if not os.path.exists(path):
        print("[warn] outputs/val_filenames.txt not found -- can't exclude the held-out "
              "split from synthesis sources. Run train_big.py at least once first "
              "(it writes this file before any training happens), or synthesis will "
              "risk leaking held-out content into training data.")
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def compute_fingerprints(exclude=None):
    exclude = exclude or set()
    files = [f for f in sorted(os.listdir(GT_DIR)) if f not in exclude]
    fingerprints = np.zeros((len(files), 32), dtype=np.float64)
    for i, f in enumerate(files):
        g = np.load(os.path.join(GT_DIR, f)).astype(np.float32)
        fingerprints[i] = radial_psd(g)
    return files, fingerprints


def find_rare_textures(near_k=2, far_k=6, isolation_ratio_thresh=1.8,
                        cluster_dist_factor=1.5, top_n=20, exclude=None, verbose=True):
    """Flags GT images that sit in a small, tightly-similar cluster (e.g. crops
    of the same source photo) with a sharp gap to the rest of the dataset --
    the exact signature confirmed for the diagnosed `002929` coverage-gap case
    (near neighbors ~0.002-0.014 away, then a jump to ~0.023+ for everything
    else). A plain "few neighbors within a fixed distance" threshold doesn't
    work here: most images in this dataset have few close matches by that
    measure, since the corpus is broadly diverse -- what's actually diagnostic
    of a *coverage gap* specifically is the isolation gap (near_k-th neighbor
    distance vs far_k-th), not raw neighbor scarcity.

    Returns a list of (filenames_in_cluster, isolation_ratio), sorted by
    isolation_ratio descending, for the top_n most isolated clusters."""
    files, fingerprints = compute_fingerprints(exclude=exclude)

    sq_norms = (fingerprints ** 2).sum(axis=1)
    dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * fingerprints @ fingerprints.T
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.clip(dists_sq, 0, None))

    sorted_dists = np.sort(dists, axis=1)
    d_near = sorted_dists[:, near_k - 1]
    d_far = sorted_dists[:, far_k - 1]
    isolation_ratio = d_far / np.clip(d_near, 1e-6, None)

    order = np.argsort(-isolation_ratio)
    seen = set()
    clusters = []
    for i in order:
        if isolation_ratio[i] < isolation_ratio_thresh or files[i] in seen:
            continue
        cluster_thresh = d_near[i] * cluster_dist_factor
        cluster_idx = np.where(dists[i] < cluster_thresh)[0]
        cluster_files = [files[i]] + [files[j] for j in cluster_idx]
        cluster_files = [f for f in dict.fromkeys(cluster_files) if f not in seen]
        if not cluster_files:
            continue
        seen.update(cluster_files)
        clusters.append((cluster_files, float(isolation_ratio[i])))
        if len(clusters) >= top_n:
            break

    if verbose:
        print(f"Scanned {len(files)} GT images. Top isolated clusters (ratio = "
              f"{far_k}th-neighbor-dist / {near_k}th-neighbor-dist):")
        for cluster_files, ratio in clusters:
            print(f"  {cluster_files} ratio={ratio:.2f}")
    return clusters


# ── 3. Patch-quilt synthesis ─────────────────────────────────────────────────

def _elastic_warp(img, alpha=2.0, sigma=25.0, rng=None):
    """Gentle, large-scale wobble only -- alpha/sigma are tuned to nudge
    texture without introducing fine streaking (a high alpha with a small
    sigma produces sharp directional smearing that doesn't look like any
    real texture, badly enough to actively mislead training)."""
    rng = rng or np.random.default_rng()
    h, w = img.shape
    dx = gaussian_filter((rng.random((h, w)) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random((h, w)) * 2 - 1), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = (np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1))
    return map_coordinates(img, coords, order=1, mode="reflect").astype(np.float32)


def quilt_canvas(sources, out_size=256, patch_size=128, overlap=48, rng=None):
    """Blend a few large, randomly-positioned crops from `sources` (one array,
    or a list of same-cluster arrays to pool from) onto a new out_size x
    out_size canvas with wide feathered blending at the seams -- a new
    composition of the cluster's real content (different crop positions,
    different source mix at the seams), not a crop or flip of any single one
    of their real layouts.

    Only horizontal/vertical flips are applied to patches, never 90-degree
    rotation: this texture family is directional (e.g. ripple lines all run
    one way), and rotating a patch turns its lines perpendicular to its
    neighbors' at the seam -- a highly visible, unrealistic crosshatch
    artifact that would actively teach the model the wrong thing."""
    rng = rng or np.random.default_rng()
    if isinstance(sources, np.ndarray):
        sources = [sources]
    canvas = np.zeros((out_size, out_size), dtype=np.float32)
    weight = np.zeros((out_size, out_size), dtype=np.float32)

    # cosine feather: smoother falloff at the seam than a linear ramp
    t = np.linspace(0, np.pi, overlap)
    ramp = (1 - np.cos(t)) / 2
    feather_1d = np.concatenate([ramp, np.ones(patch_size - 2 * overlap), ramp[::-1]])
    feather = np.outer(feather_1d, feather_1d).astype(np.float32)

    stride = patch_size - overlap
    for ty in range(-overlap, out_size, stride):
        for tx in range(-overlap, out_size, stride):
            source = sources[rng.integers(0, len(sources))]
            sh, sw = source.shape
            py = rng.integers(0, max(1, sh - patch_size))
            px = rng.integers(0, max(1, sw - patch_size))
            patch = source[py:py + patch_size, px:px + patch_size]
            if patch.shape != (patch_size, patch_size):
                patch = sk_resize(patch, (patch_size, patch_size), preserve_range=True).astype(np.float32)

            if rng.random() < 0.5:
                patch = np.fliplr(patch)
            if rng.random() < 0.5:
                patch = np.flipud(patch)

            y0, x0 = max(ty, 0), max(tx, 0)
            y1, x1 = min(ty + patch_size, out_size), min(tx + patch_size, out_size)
            if y1 <= y0 or x1 <= x0:
                continue
            fy0, fx0 = y0 - ty, x0 - tx
            fy1, fx1 = fy0 + (y1 - y0), fx0 + (x1 - x0)

            canvas[y0:y1, x0:x1] += patch[fy0:fy1, fx0:fx1] * feather[fy0:fy1, fx0:fx1]
            weight[y0:y1, x0:x1] += feather[fy0:fy1, fx0:fx1]

    weight = np.clip(weight, 1e-6, None)
    canvas = canvas / weight

    # mild global warp + jitter so repeated quilts of the same source aren't
    # geometrically identical, and brightness/contrast varies a bit -- kept
    # gentle (large sigma, small alpha) to avoid the streaking artifacts a
    # sharper warp produces (see _elastic_warp docstring)
    canvas = _elastic_warp(canvas, alpha=rng.uniform(1, 3), sigma=rng.uniform(20, 35), rng=rng)
    gain = rng.uniform(0.92, 1.08)
    bias = rng.uniform(-0.03, 0.03)
    canvas = np.clip(canvas * gain + bias, 0.0, 1.0).astype(np.float32)
    return canvas


# ── driver ───────────────────────────────────────────────────────────────────

def _ensure_target_cluster(clusters, files, fingerprints, dists, target_file,
                            near_k=1, cluster_dist_factor=1.5):
    """Guarantee a specific known hard case (e.g. 002929's training-set
    siblings) gets synthetic coverage even if its isolation ratio didn't rank
    in the auto-discovered top_n. Uses near_k=1 (nearest-neighbor distance,
    not the near_k=2 used elsewhere for the general 3-member-cluster case) --
    excluding a held-out member can shrink a target's real cluster down to a
    single remaining sibling, and sorted_dists[1] would already land past
    that boundary into unrelated images."""
    if any(target_file in cf for cf, _ in clusters):
        return clusters
    if target_file not in files:
        print(f"[warn] {target_file} not found in the (held-out-excluded) training "
              f"corpus -- can't force-include its cluster.")
        return clusters
    i = files.index(target_file)
    sorted_d = np.sort(dists[i])
    d_near = sorted_d[near_k - 1]
    cluster_idx = np.where(dists[i] < d_near * cluster_dist_factor)[0]
    cluster_files = list(dict.fromkeys([target_file] + [files[j] for j in cluster_idx]))
    print(f"Force-including {target_file}'s cluster (not in auto-discovered top_n): {cluster_files}")
    return clusters + [(cluster_files, None)]


def generate(top_n=50, n_per_source=10, near_k=2, far_k=6, isolation_ratio_thresh=1.8,
             cluster_dist_factor=1.5, force_include=("002930.npy",), seed=0):
    os.makedirs(NOISY_SYNTH_DIR, exist_ok=True)
    os.makedirs(GT_SYNTH_DIR, exist_ok=True)

    holdout = load_holdout_filenames()
    print(f"Excluding {len(holdout)} held-out validation files from synthesis sources.")

    print("Fitting real noise model...")
    noise_curve = fit_noise_model()

    print("Scanning for isolated/underrepresented texture clusters...")
    files, fingerprints = compute_fingerprints(exclude=holdout)
    sq = (fingerprints ** 2).sum(axis=1)
    dsq = sq[:, None] + sq[None, :] - 2 * fingerprints @ fingerprints.T
    np.fill_diagonal(dsq, np.inf)
    dists = np.sqrt(np.clip(dsq, 0, None))

    clusters = find_rare_textures(near_k=near_k, far_k=far_k, isolation_ratio_thresh=isolation_ratio_thresh,
                                   cluster_dist_factor=cluster_dist_factor, top_n=top_n, exclude=holdout)
    for target in force_include:
        clusters = _ensure_target_cluster(clusters, files, fingerprints, dists, target,
                                           near_k=1, cluster_dist_factor=cluster_dist_factor)

    rng = np.random.default_rng(seed)
    count = 0
    for cluster_files, ratio in clusters:
        sources = [np.load(os.path.join(GT_DIR, f)).astype(np.float32) for f in cluster_files]
        tag = os.path.splitext(cluster_files[0])[0]
        for i in range(n_per_source):
            gt_synth = quilt_canvas(sources, out_size=256, rng=rng)
            noisy_128 = sk_resize(gt_synth, (128, 128), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
            noisy_synth = apply_noise_model(noisy_128, noise_curve, rng)

            out_name = f"synth_{tag}_{i:03d}.npy"
            np.save(os.path.join(GT_SYNTH_DIR, out_name), gt_synth)
            np.save(os.path.join(NOISY_SYNTH_DIR, out_name), noisy_synth)
            count += 1

    print(f"Generated {count} synthetic pairs from {len(clusters)} isolated-texture clusters "
          f"-> {NOISY_SYNTH_DIR} / {GT_SYNTH_DIR}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=50, help="Number of most-isolated texture clusters to synthesize for")
    parser.add_argument("--n_per_source", type=int, default=10, help="Synthetic variants generated per cluster")
    parser.add_argument("--isolation_ratio_thresh", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(top_n=args.top_n, n_per_source=args.n_per_source,
              isolation_ratio_thresh=args.isolation_ratio_thresh, seed=args.seed)
