"""
Sanity check on REAL in-distribution data: pick held-out validation
filenames, run them through the trained model, and compare against
ground truth with SSIM/PSNR. Contrast against test_degradations.py,
which deliberately probes out-of-distribution synthetic noise/blur.

Runs on CPU so it doesn't compete with an in-progress GPU training run.

Usage: python test_real_sample.py [n_samples]
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

from model import NAFNetRestorer

CHECKPOINT_PATH = "../outputs/model_nafnet_my_run.pt"
VAL_LIST = "../outputs/val_filenames.txt"
NOISY_DIR = "../data/train/NoisyLR/NoisyLR"
GT_DIR = "../data/train/GT_full/GT"
OUT_DIR = "../outputs/degradation_test"
DEVICE = "cpu"

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3

with open(VAL_LIST) as f:
    val_filenames = [line.strip() for line in f if line.strip()][:n_samples]

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
width = state_dict["head.weight"].shape[0]
model = NAFNetRestorer(width=width, enc_blks=(2, 2, 4), middle_blks=12, dec_blks=(4, 2, 2)).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

os.makedirs(OUT_DIR, exist_ok=True)

for fname in val_filenames:
    noisy = np.load(os.path.join(NOISY_DIR, fname)).astype(np.float32)
    gt = np.load(os.path.join(GT_DIR, fname)).astype(np.float32)

    x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()

    ssim_val = sk_ssim(pred, gt, data_range=1.0)
    psnr_val = sk_psnr(gt, pred, data_range=1.0)
    print(f"{fname}: SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f} dB")

    stem = os.path.splitext(fname)[0]
    Image.fromarray((np.clip(noisy, 0, 1) * 255).astype(np.uint8)).save(f"{OUT_DIR}/real_{stem}_input.png")
    Image.fromarray((pred * 255).astype(np.uint8)).save(f"{OUT_DIR}/real_{stem}_restored.png")
    Image.fromarray((gt * 255).astype(np.uint8)).save(f"{OUT_DIR}/real_{stem}_gt.png")

print(f"\nSaved input/restored/gt triplets to {OUT_DIR}/")
