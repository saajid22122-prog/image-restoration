"""
Standalone inference script for the KLA image restoration challenge.
"""

import os
import argparse
import glob
import time

import numpy as np
import torch

from model import RestorationNet, NoiseAwareRestorationNet, UNetRestorationNet, NAFNetRestorer


def load_model(checkpoint_path, device, model_variant="nafnet", num_res_blocks=8, base_channels=64, width=32):
    if model_variant == "nafnet":
        model = NAFNetRestorer(width=width)
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


def run_inference(input_dir, output_dir, checkpoint_path, device=None, model_variant="nafnet", use_tta=False, num_res_blocks=8, base_channels=64, width=32):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if use_tta:
        print("Test-time augmentation (TTA) enabled - averaging 4 views per image")

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(checkpoint_path, device, model_variant, num_res_blocks=num_res_blocks, base_channels=base_channels, width=width)

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if len(input_files) == 0:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    print(f"Found {len(input_files)} input files.")

    times = []
    with torch.no_grad():
        for filepath in input_files:
            fname = os.path.basename(filepath)
            arr = np.load(filepath).astype(np.float32)

            t0 = time.time()

            if use_tta:
                variants = [arr, np.fliplr(arr), np.flipud(arr), np.rot90(arr, 2)]
                preds = []
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
                pred_np = np.mean(preds, axis=0)
            else:
                x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(x)
                pred = torch.clamp(pred, 0, 1)
                pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default="nafnet",
                        choices=["nafnet", "nafnet_large", "baseline", "noise_aware", "unet"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--width", type=int, default=32, help="NAFNet width (must match training)")
    parser.add_argument("--num_res_blocks", type=int, default=8, help="Must match how the checkpoint was trained")
    parser.add_argument("--base_channels", type=int, default=64, help="Must match how the checkpoint was trained")
    args = parser.parse_args()

    run_inference(
        args.input_dir, args.output_dir, args.checkpoint,
        model_variant=args.model_variant, use_tta=args.tta,
        width=args.width, num_res_blocks=args.num_res_blocks, base_channels=args.base_channels,
    )