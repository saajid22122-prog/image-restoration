"""
Training script — full-quality run.

Changes vs train.py:
  - Bigger NAFNetRestorer config (width=48, more blocks)
  - More epochs, larger batch size
  - Same loss: Charbonnier + SSIM + FFT frequency-domain
  - AdamW + linear warmup + cosine LR + gradient clipping
  - Data augmentation always on
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pytorch_msssim import ssim

from dataset import RestorationDataset
from model import (
    RestorationNet, NoiseAwareRestorationNet, UNetRestorationNet,
    NAFNetRestorer, NAFNetRestorerV2,
)


# ── Loss functions ─────────────────────────────────────────────────────────────

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target).pow(2) + eps * eps))


def frequency_loss(pred, target):
    """L1 on FFT magnitude spectrum — enforces high-frequency sharpness."""
    pred_f   = torch.fft.rfft2(pred,   norm='ortho')
    target_f = torch.fft.rfft2(target, norm='ortho')
    return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))


def combined_loss(pred, target, char_w=0.4, ssim_w=0.25, freq_w=0.35):
    pred_c = torch.clamp(pred, 0, 1)
    char   = charbonnier_loss(pred_c, target)
    ssim_l = 1.0 - ssim(pred_c, target, data_range=1.0, size_average=True)
    freq   = frequency_loss(pred_c, target)
    return char_w * char + ssim_w * ssim_l + freq_w * freq


def deep_supervision_loss(aux_outputs, target, weight=0.1):
    """
    Charbonnier loss between each decoder-stage aux head and the ground
    truth downsampled to that stage's resolution. Gives early decoder
    layers a direct gradient instead of only what backprops through the
    final tail, at a small weight so it nudges training without
    dominating the main full-resolution loss.
    """
    if not aux_outputs:
        return 0.0
    loss = 0.0
    for aux in aux_outputs:
        aux_c = torch.clamp(aux, 0, 1)
        target_ds = F.interpolate(target, size=aux_c.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + charbonnier_loss(aux_c, target_ds)
    return weight * loss / len(aux_outputs)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    noisy_dir,
    gt_dir,
    epochs=100,
    batch_size=16,
    lr=1e-3,
    val_split=0.1,
    checkpoint_path="outputs/best_model.pt",
    device=None,
    model_variant="nafnet",
    num_res_blocks=12,
    width=48,
    warmup_epochs=10,
    use_augmentation=True,
    resume=False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_state_path = os.path.splitext(checkpoint_path)[0] + ".trainstate.pt"
    log_path = os.path.splitext(checkpoint_path)[0] + ".log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")

    # use a no-augment dataset to compute the split reproducibly
    split_ref = RestorationDataset(noisy_dir, gt_dir, augment=False)
    val_size  = max(1, int(len(split_ref) * val_split))
    train_size = len(split_ref) - val_size

    split_gen = torch.Generator().manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(split_ref)), [train_size, val_size], generator=split_gen
    )

    log(f"Train pairs: {train_size} | Val pairs: {val_size}")

    val_filenames = [split_ref.filenames[i] for i in val_indices.indices]
    val_list_path = os.path.join(os.path.dirname(checkpoint_path), "val_filenames.txt")
    os.makedirs(os.path.dirname(val_list_path), exist_ok=True)
    if not (resume and os.path.exists(val_list_path)):
        with open(val_list_path, "w") as f:
            f.write("\n".join(val_filenames))
        log(f"Saved {len(val_filenames)} held-out filenames to {val_list_path}")
    else:
        log(f"Resume: kept existing held-out split at {val_list_path}")

    train_dataset = RestorationDataset(noisy_dir, gt_dir, augment=use_augmentation)
    train_ds = Subset(train_dataset, train_indices.indices)
    val_ds   = Subset(split_ref,    val_indices.indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    if model_variant == "nafnet":
        model = NAFNetRestorer(
            width=width,
            enc_blks=(2, 2, 4),
            middle_blks=12,
            dec_blks=(4, 2, 2),
        ).to(device)
    elif model_variant == "nafnet_v2":
        model = NAFNetRestorerV2(
            width=width,
            enc_blks=(2, 2, 4),
            middle_blks=12,
            dec_blks=(4, 2, 2),
        ).to(device)
    elif model_variant == "nafnet_large":
        model = NAFNetRestorer(
            width=64,
            enc_blks=(2, 2, 4, 8),
            middle_blks=16,
            dec_blks=(8, 4, 2, 2),
        ).to(device)
    elif model_variant == "unet":
        model = UNetRestorationNet(base_channels=width).to(device)
    elif model_variant == "noise_aware":
        model = NoiseAwareRestorationNet(num_res_blocks=num_res_blocks, base_channels=width).to(device)
    else:
        model = RestorationNet(num_res_blocks=num_res_blocks, base_channels=width).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {model_variant} | width={width} | Parameters: {num_params:,}")
    log(f"Augmentation: {use_augmentation}")

    use_deep_supervision = model_variant == "nafnet_v2"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    best_val_loss = float("inf")

    if resume:
        if os.path.exists(train_state_path):
            state = torch.load(train_state_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state["best_val_loss"]
            log(f"Resumed full training state from {train_state_path} "
                f"(epoch {start_epoch}, best_val={best_val_loss:.4f})")
        elif os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            warm_val_loss = 0.0
            with torch.no_grad():
                for noisy, gt in val_loader:
                    noisy, gt = noisy.to(device), gt.to(device)
                    pred = model(noisy)
                    warm_val_loss += combined_loss(pred, gt).item() * noisy.size(0)
            best_val_loss = warm_val_loss / val_size
            log(f"Resume requested but no train-state file found; warm-started weights "
                f"from {checkpoint_path} (optimizer/scheduler restart at epoch 1, "
                f"measured val={best_val_loss:.4f} — only an improvement on this will overwrite the checkpoint)")
        else:
            log("Resume requested but no checkpoint or train-state found; starting fresh")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (noisy, gt) in enumerate(train_loader):
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            if use_deep_supervision:
                pred, aux_outputs = model(noisy, return_aux=True)
                loss = combined_loss(pred, gt) + deep_supervision_loss(aux_outputs, gt)
            else:
                pred = model(noisy)
                loss = combined_loss(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                print(f"  epoch {epoch} batch {batch_idx+1}/{num_batches}", flush=True)

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
        log(f"Epoch {epoch:3d}/{epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | lr: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            log(f"  -> best saved (val={val_loss:.4f})")

        # full training state, saved every epoch so a crash never loses more than one epoch
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }, train_state_path)

    log_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, nargs="?", default=100)
    parser.add_argument("variant", nargs="?", default="nafnet")
    parser.add_argument("run_name", nargs="?", default="run")
    parser.add_argument("width", type=int, nargs="?", default=48)
    parser.add_argument("--resume", action="store_true",
                         help="Resume training for this run_name from its saved train-state "
                              "(or warm-start from its best-weights checkpoint if no train-state exists yet)")
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Peak LR after warmup. Use a much lower value (e.g. 1e-4) when resuming/warm-"
                              "starting from an already-converged checkpoint -- the from-scratch default "
                              "(1e-3) can kick good weights out of their minimum once warmup peaks.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                         help="Linear LR warmup length. Shorten this (e.g. 2) when resuming, since warm-"
                              "started weights don't need a long ramp-up.")
    args = parser.parse_args()

    checkpoint_name = f"model_{args.variant}_{args.run_name}.pt"
    print(f"Checkpoint: {checkpoint_name}")

    train(
        noisy_dir="../data/train/NoisyLR/NoisyLR",
        gt_dir="../data/train/GT_full/GT",
        epochs=args.epochs,
        batch_size=16,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        model_variant=args.variant,
        width=args.width,
        checkpoint_path=f"../outputs/{checkpoint_name}",
        resume=args.resume,
    )
