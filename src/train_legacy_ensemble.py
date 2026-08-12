"""
Training script for the RETIRED four-model ensemble members (baseline /
noise_aware / unet variants) -- kept for reference only. This is NOT the
training script for the submitted model. The submission is NAFNet, trained
by train_big.py; that is the script to run to reproduce the submission
from scratch.

Loss: Charbonnier (L1-smooth) + SSIM + FFT frequency-domain
  - Charbonnier avoids L1's gradient discontinuity at zero
  - SSIM preserves perceptual structure
  - FFT loss penalises frequency-domain mismatch -> sharp, noise-free output

Optimizer: AdamW with cosine LR + linear warmup + gradient clipping.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import ssim

from dataset import RestorationDataset
from model import RestorationNet, NoiseAwareRestorationNet, UNetRestorationNet, NAFNetRestorer


# ── Loss functions ─────────────────────────────────────────────────────────────

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target).pow(2) + eps * eps))


def frequency_loss(pred, target):
    """L1 on FFT magnitude spectrum — enforces high-frequency sharpness."""
    pred_f   = torch.fft.rfft2(pred,   norm='ortho')
    target_f = torch.fft.rfft2(target, norm='ortho')
    return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))


def combined_loss(pred, target, char_w=0.5, ssim_w=0.3, freq_w=0.2):
    pred_c   = torch.clamp(pred, 0, 1)
    char     = charbonnier_loss(pred_c, target)
    ssim_l   = 1.0 - ssim(pred_c, target, data_range=1.0, size_average=True)
    freq     = frequency_loss(pred_c, target)
    return char_w * char + ssim_w * ssim_l + freq_w * freq


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    noisy_dir,
    gt_dir,
    epochs=30,
    batch_size=16,
    lr=1e-3,
    val_split=0.1,
    checkpoint_path="outputs/best_model.pt",
    device=None,
    model_variant="nafnet",
    num_res_blocks=8,
    width=32,
    warmup_epochs=5,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = RestorationDataset(noisy_dir, gt_dir, augment=False)
    val_size  = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    split_gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=split_gen)

    print(f"Train pairs: {train_size} | Val pairs: {val_size}")

    val_filenames = [full_dataset.filenames[i] for i in val_ds.indices]
    val_list_path = os.path.join(os.path.dirname(checkpoint_path), "val_filenames.txt")
    os.makedirs(os.path.dirname(val_list_path), exist_ok=True)
    with open(val_list_path, "w") as f:
        f.write("\n".join(val_filenames))
    print(f"Saved {len(val_filenames)} held-out filenames to {val_list_path}")

    # augmented view of the training split only — val indices are never touched
    from torch.utils.data import Subset
    train_aug_ds = RestorationDataset(noisy_dir, gt_dir, augment=True)
    train_aug_subset = Subset(train_aug_ds, train_ds.indices)

    train_loader = DataLoader(
        train_aug_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    if model_variant == "nafnet":
        model = NAFNetRestorer(width=width).to(device)
    elif model_variant == "unet":
        model = UNetRestorationNet().to(device)
    elif model_variant == "noise_aware":
        model = NoiseAwareRestorationNet(num_res_blocks=num_res_blocks).to(device)
    else:
        model = RestorationNet(num_res_blocks=num_res_blocks).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_variant} | Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = combined_loss(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)
        train_loss /= train_size
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                val_loss += combined_loss(pred, gt).item() * noisy.size(0)
        val_loss /= val_size

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | lr: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> best saved (val={val_loss:.4f})")


if __name__ == "__main__":
    import sys

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
    OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")

    epochs  = int(sys.argv[1])   if len(sys.argv) > 1 else 30
    variant = sys.argv[2]        if len(sys.argv) > 2 else "nafnet"
    run_name = sys.argv[3]       if len(sys.argv) > 3 else "run"
    width   = int(sys.argv[4])   if len(sys.argv) > 4 else 32

    checkpoint_name = f"model_{variant}_{run_name}.pt"
    print(f"Checkpoint: {checkpoint_name}")

    train(
        noisy_dir=os.path.join(DATA_DIR, "train", "NoisyLR", "NoisyLR"),
        gt_dir=os.path.join(DATA_DIR, "train", "GT_full", "GT"),
        epochs=epochs,
        batch_size=16,
        model_variant=variant,
        width=width,
        checkpoint_path=os.path.join(OUTPUTS_DIR, checkpoint_name),
    )
