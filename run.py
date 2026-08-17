"""
KLA image restoration challenge -- graded entry point.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it, and writes a same-named
.npy to <output-dir> (created if it doesn't exist). No manual edits,
flags, internet access, or API keys required -- weights are loaded
from ./models relative to this file, resolved by this script's own
location so it works regardless of the caller's working directory.
"""

import os
import sys
import glob
import time

import numpy as np
import torch

# Make sibling model.py importable regardless of caller's cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import NAFNetRestorer

CHECKPOINT = os.path.join(SCRIPT_DIR, "models", "model_nafnet_synth_v1.pt")
CHECKPOINT2 = os.path.join(SCRIPT_DIR, "models", "model_nafnet_pre_lpips.pt")
WIDTH = 48


def load_model(checkpoint_path, device):
    model = NAFNetRestorer(width=WIDTH, enc_blks=(2, 2, 4), middle_blks=12, dec_blks=(4, 2, 2))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_views(model, arr, device):
    """Restore one (H,W) array with 4-view flip/rotate TTA, returned un-averaged."""
    variants = [arr, np.fliplr(arr), np.flipud(arr), np.rot90(arr, 2)]
    preds = []
    with torch.no_grad():
        for i, v in enumerate(variants):
            x = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
            p = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()
            if i == 1:
                p = np.fliplr(p)
            elif i == 2:
                p = np.flipud(p)
            elif i == 3:
                p = np.rot90(p, -2)
            preds.append(p)
    return preds


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    models = [load_model(CHECKPOINT, device)]
    if os.path.exists(CHECKPOINT2):
        models.append(load_model(CHECKPOINT2, device))
        print(f"Averaging predictions across {len(models)} checkpoints (TTA x{len(models)*4} views/image)")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(input_files) == 0:
        raise RuntimeError(f"No .npy files found in {input_dir}")
    print(f"Found {len(input_files)} input files.")

    times = []
    for filepath in input_files:
        fname = os.path.basename(filepath)
        arr = np.load(filepath).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]

        t0 = time.time()
        all_views = [v for m in models for v in predict_views(m, arr, device)]
        pred = np.stack(all_views, axis=0).mean(axis=0)
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        times.append(time.time() - t0)

        np.save(os.path.join(output_dir, fname), pred)

    avg_time = sum(times) / len(times)
    print(f"Processed {len(input_files)} images.")
    print(f"Average inference time per image: {avg_time*1000:.2f} ms")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
