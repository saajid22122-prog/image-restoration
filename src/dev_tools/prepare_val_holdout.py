import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
val_list_path = os.path.join(BASE, "outputs", "val_filenames.txt")
noisy_src = os.path.join(BASE, "data", "train", "NoisyLR", "NoisyLR")
gt_src = os.path.join(BASE, "data", "train", "GT_full", "GT")
noisy_holdout = os.path.join(BASE, "data", "train", "NoisyLR_holdout")
gt_holdout = os.path.join(BASE, "data", "train", "GT_holdout")

os.makedirs(noisy_holdout, exist_ok=True)
os.makedirs(gt_holdout, exist_ok=True)

with open(val_list_path) as f:
    val_files = [line.strip() for line in f if line.strip()]

copied = 0
for fname in val_files:
    src_n = os.path.join(noisy_src, fname)
    src_g = os.path.join(gt_src, fname)
    if os.path.exists(src_n) and os.path.exists(src_g):
        shutil.copy(src_n, os.path.join(noisy_holdout, fname))
        shutil.copy(src_g, os.path.join(gt_holdout, fname))
        copied += 1

print(f"Copied {copied} held-out pairs (never seen during training)")