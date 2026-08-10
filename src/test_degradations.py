"""
Quick sanity test: synthesize a noisy and a blurry version of an image,
run both through the trained NAFNet model, and save all outputs for comparison.

Runs on CPU by default so it doesn't compete for GPU memory with an
in-progress training run.

Usage: python test_degradations.py [path/to/clean_image.png]
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch

from model import NAFNetRestorer

CHECKPOINT_PATH = "../outputs/model_nafnet_my_run.pt"
OUT_DIR = "../outputs/degradation_test"
DEVICE = "cpu"

image_path = sys.argv[1] if len(sys.argv) > 1 else "../outputs/lenna.png_input_128.png"
os.makedirs(OUT_DIR, exist_ok=True)

# ── build a clean 128x128 base + two degraded versions ──────────────────────
clean_img = Image.open(image_path).convert("L").resize((128, 128))
clean = np.array(clean_img).astype(np.float32) / 255.0

rng = np.random.default_rng(0)
noisy = np.clip(clean + rng.normal(0, 0.15, clean.shape), 0, 1).astype(np.float32)
blurry = gaussian_filter(clean, sigma=2.5).astype(np.float32)

# ── load model ────────────────────────────────────────────────────────────
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
width = state_dict["head.weight"].shape[0]
model = NAFNetRestorer(width=width, enc_blks=(2, 2, 4), middle_blks=12, dec_blks=(4, 2, 2)).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()


def restore(arr):
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()
    return pred


noisy_restored = restore(noisy)
blurry_restored = restore(blurry)


def save(arr, name):
    Image.fromarray((arr * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, name))


save(clean, "0_clean_128.png")
save(noisy, "1_noisy_input_128.png")
save(noisy_restored, "2_noisy_restored_256.png")
save(blurry, "3_blurry_input_128.png")
save(blurry_restored, "4_blurry_restored_256.png")

print(f"Saved 5 images to {OUT_DIR}/")
print("  0_clean_128.png            - clean base image")
print("  1_noisy_input_128.png      - clean + synthetic Gaussian noise")
print("  2_noisy_restored_256.png   - model output on the noisy input")
print("  3_blurry_input_128.png     - clean + Gaussian blur")
print("  4_blurry_restored_256.png  - model output on the blurry input")
