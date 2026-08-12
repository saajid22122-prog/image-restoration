"""
Standalone inference script for the KLA image restoration challenge.
"""

import os
import argparse
import glob
import time

import numpy as np
import torch

from model import (
    RestorationNet, NoiseAwareRestorationNet, UNetRestorationNet,
    NAFNetRestorer, NAFNetRestorerV2,
)

# Resolved relative to this file, not the caller's working directory --
# `python infer.py ...` and `python src/infer.py ...` from anywhere both work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "outputs", "model_nafnet_my_run.pt")
# Second checkpoint averaged in by default -- same NAFNet architecture, an
# earlier snapshot from before the LPIPS fine-tune. Averaging the two
# recovers most of the fine-tune's small SSIM/PSNR cost while keeping most
# of its LPIPS gain (see README). Only meaningful for model_variant="nafnet".
DEFAULT_CHECKPOINT2 = os.path.join(SCRIPT_DIR, "..", "outputs", "model_nafnet_pre_lpips.pt")


def load_model(checkpoint_path, device, model_variant="nafnet", num_res_blocks=8, base_channels=64, width=32):
    if model_variant == "nafnet":
        model = NAFNetRestorer(width=width, enc_blks=(2, 2, 4), middle_blks=12, dec_blks=(4, 2, 2))
    elif model_variant == "nafnet_v2":
        model = NAFNetRestorerV2(width=width)
    elif model_variant == "nafnet_large":
        model = NAFNetRestorer(width=64, enc_blks=(2, 2, 4, 8), middle_blks=16, dec_blks=(8, 4, 2, 2))
    elif model_variant == "unet":
        model = UNetRestorationNet(base_channels=base_channels)
    elif model_variant == "noise_aware":
        model = NoiseAwareRestorationNet(num_res_blocks=num_res_blocks, base_channels=base_channels)
    else:
        model = RestorationNet(num_res_blocks=num_res_blocks, base_channels=base_channels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_one(model, arr, device, use_tta):
    """Restore a single (H,W) array. With TTA, averages 4 flip/rotate views."""
    if not use_tta:
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.clamp(model(x), 0, 1)
        return pred.squeeze(0).squeeze(0).cpu().numpy()

    variants = [arr, np.fliplr(arr), np.flipud(arr), np.rot90(arr, 2)]
    preds = []
    with torch.no_grad():
        for i, v in enumerate(variants):
            x = torch.from_numpy(v.copy()).unsqueeze(0).unsqueeze(0).to(device)
            p = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()
            if i == 1:
                p = np.fliplr(p)
            elif i == 2:
                p = np.flipud(p)
            elif i == 3:
                p = np.rot90(p, -2)
            preds.append(p)
    return np.mean(preds, axis=0)


def run_inference(input_dir, output_dir, checkpoint_path, checkpoint_path2=None, device=None,
                   model_variant="nafnet", use_tta=True, num_res_blocks=8, base_channels=64, width=32):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if use_tta:
        print("Test-time augmentation (TTA) enabled - averaging 4 views per image")

    os.makedirs(output_dir, exist_ok=True)

    models = [load_model(checkpoint_path, device, model_variant, num_res_blocks=num_res_blocks, base_channels=base_channels, width=width)]
    if checkpoint_path2 and os.path.exists(checkpoint_path2):
        models.append(load_model(checkpoint_path2, device, model_variant, num_res_blocks=num_res_blocks, base_channels=base_channels, width=width))
        print(f"Averaging predictions across 2 checkpoints ({os.path.basename(checkpoint_path)} + {os.path.basename(checkpoint_path2)})")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(input_files) == 0:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    print(f"Found {len(input_files)} input files.")

    times = []
    for filepath in input_files:
        fname = os.path.basename(filepath)
        arr = np.load(filepath).astype(np.float32)

        t0 = time.time()

        preds = [_predict_one(m, arr, device, use_tta) for m in models]
        pred_np = np.mean(preds, axis=0) if len(preds) > 1 else preds[0]

        elapsed = time.time() - t0
        times.append(elapsed)

        out_path = os.path.join(output_dir, fname)
        np.save(out_path, pred_np)

    avg_time = sum(times) / len(times)
    print(f"Processed {len(input_files)} images.")
    print(f"Average inference time per image: {avg_time*1000:.2f} ms")
    print(f"Total time: {sum(times):.2f} s")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run restoration inference on a directory of test images.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Defaults to the final submitted NAFNet checkpoint")
    parser.add_argument("--checkpoint2", type=str, default=DEFAULT_CHECKPOINT2,
                        help="Second checkpoint averaged in with --checkpoint (same architecture). "
                             "Defaults to the pre-LPIPS-fine-tune NAFNet snapshot -- averaging the "
                             "two recovers most of the fine-tune's SSIM/PSNR cost while keeping most "
                             "of its LPIPS gain. Only used when model_variant=nafnet.")
    parser.add_argument("--single_checkpoint", action="store_true",
                        help="Use only --checkpoint, skip averaging in --checkpoint2.")
    parser.add_argument("--model_variant", type=str, default="nafnet",
                        choices=["nafnet", "nafnet_v2", "nafnet_large", "baseline", "noise_aware", "unet"])
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation (4-view flip/rotate averaging). "
                             "TTA is ON by default -- it measurably improves SSIM/PSNR on hard "
                             "cases for ~3x the inference time (still well under 100ms/image).")
    parser.add_argument("--width", type=int, default=48, help="NAFNet width (must match training)")
    parser.add_argument("--num_res_blocks", type=int, default=8, help="Must match how the checkpoint was trained")
    parser.add_argument("--base_channels", type=int, default=64, help="Must match how the checkpoint was trained")
    args = parser.parse_args()

    checkpoint2 = None if (args.single_checkpoint or args.model_variant != "nafnet") else args.checkpoint2

    run_inference(
        args.input_dir, args.output_dir, args.checkpoint, checkpoint_path2=checkpoint2,
        model_variant=args.model_variant, use_tta=not args.no_tta,
        width=args.width, num_res_blocks=args.num_res_blocks, base_channels=args.base_channels,
    )