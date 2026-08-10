"""
Metrics script: compares restored predictions against ground truth.

Computes the three metrics KLA's Slide 6 asks for:
  - SSIM  (structural similarity - higher is better, max 1.0)
  - PSNR  (peak signal-to-noise ratio in dB - higher is better)
  - LPIPS (learned perceptual similarity - LOWER is better, unlike the other two)

Usage (as a library):
    from metrics import evaluate_directory
    results = evaluate_directory(pred_dir, gt_dir)

Usage (from command line):
    python metrics.py --pred_dir path/to/predictions --gt_dir path/to/ground_truth
"""

import os
import glob
import argparse

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


_lpips_model = None
_lpips_unavailable = False


def get_lpips_model(device="cpu"):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


def compute_lpips(pred, gt, device="cpu"):
    """pred, gt: numpy arrays (H,W) in [0,1]. LPIPS expects 3-channel, range [-1,1].
    Returns None if the pretrained LPIPS weights can't be downloaded (e.g. no
    internet access) - this does not affect SSIM/PSNR, only skips LPIPS."""
    global _lpips_unavailable
    if _lpips_unavailable:
        return None

    try:
        model = get_lpips_model(device)
    except Exception as e:
        print(f"[warn] LPIPS unavailable (could not load pretrained weights: {e}). "
              f"Skipping LPIPS - SSIM/PSNR are unaffected. This requires internet "
              f"access to download AlexNet weights on first run.")
        _lpips_unavailable = True
        return None

    def to_lpips_tensor(arr):
        t = torch.from_numpy(arr).float()
        t = t.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
        t = t.repeat(1, 3, 1, 1)                 # (1,3,H,W) - LPIPS expects RGB-like input
        t = t * 2 - 1                            # [0,1] -> [-1,1]
        return t.to(device)

    with torch.no_grad():
        d = model(to_lpips_tensor(pred), to_lpips_tensor(gt))
    return d.item()


def evaluate_directory(pred_dir, gt_dir, device="cpu", verbose=True):
    pred_files = {os.path.basename(f) for f in glob.glob(os.path.join(pred_dir, "*.npy"))}
    gt_files = {os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.npy"))}
    matched = sorted(pred_files & gt_files)

    if not matched:
        raise RuntimeError(f"No matching files between {pred_dir} and {gt_dir}")

    ssim_scores, psnr_scores, lpips_scores = [], [], []

    for fname in matched:
        pred = np.load(os.path.join(pred_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(gt_dir, fname)).astype(np.float32)
        pred = np.clip(pred, 0, 1)
        gt = np.clip(gt, 0, 1)

        ssim_scores.append(sk_ssim(gt, pred, data_range=1.0))
        # avoid divide-by-zero warning when pred == gt exactly (PSNR is
        # mathematically infinite for identical images - cap it for display)
        mse = np.mean((gt - pred) ** 2)
        if mse == 0:
            psnr_scores.append(100.0)  # treat as effectively perfect
        else:
            psnr_scores.append(sk_psnr(gt, pred, data_range=1.0))

        lp = compute_lpips(pred, gt, device)
        if lp is not None:
            lpips_scores.append(lp)

    results = {
        "n_images": len(matched),
        "SSIM_mean": float(np.mean(ssim_scores)),
        "PSNR_mean": float(np.mean(psnr_scores)),
        "LPIPS_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
    }

    if verbose:
        print(f"Evaluated {results['n_images']} image pairs")
        print(f"  SSIM  (higher better): {results['SSIM_mean']:.4f}")
        print(f"  PSNR  (higher better): {results['PSNR_mean']:.2f} dB")
        if results["LPIPS_mean"] is not None:
            print(f"  LPIPS (lower better):  {results['LPIPS_mean']:.4f}")
        else:
            print(f"  LPIPS: skipped (no internet access for pretrained weights)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSIM/PSNR/LPIPS between predictions and ground truth")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    args = parser.parse_args()

    evaluate_directory(args.pred_dir, args.gt_dir)
