"""
Ensemble inference: combines predictions from SEVERAL already-trained
models by a WEIGHTED average - better-performing models get more say,
weaker-but-different models (like TV_v2) still contribute a little
diversity without dominating the result.

No retraining needed - this only uses checkpoints you already have.

Usage:
    python ensemble_infer.py --input_dir path --output_dir path [--tta]
"""

import os
import argparse
import glob
import time

import numpy as np
import torch

from model import RestorationNet, NoiseAwareRestorationNet, UNetRestorationNet

# Resolved relative to this file, not the caller's working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")


def load_one_model(checkpoint_path, device, model_variant, num_res_blocks, base_channels):
    if model_variant == "unet":
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


# Checkpoint filenames match the trained weights committed under outputs/ — no
# edits needed to run this script. "weight" controls how much influence each
# model has in the final average.
ENSEMBLE_MEMBERS = [
    {"checkpoint": os.path.join(OUTPUTS_DIR, "model_noise_aware_LOCAL_v1.pt"), "variant": "noise_aware", "num_res_blocks": 8, "base_channels": 64, "weight": 1.2},
    {"checkpoint": os.path.join(OUTPUTS_DIR, "model_unet_UNET_v1.pt"), "variant": "unet", "num_res_blocks": 8, "base_channels": 48, "weight": 1.0},
    {"checkpoint": os.path.join(OUTPUTS_DIR, "model_noise_aware_BIG_v4.pt"), "variant": "noise_aware", "num_res_blocks": 12, "base_channels": 80, "weight": 1.2},
    {"checkpoint": os.path.join(OUTPUTS_DIR, "model_noise_aware_TV_v2.pt"), "variant": "noise_aware", "num_res_blocks": 8, "base_channels": 64, "weight": 0.6},
]


def run_ensemble_inference(input_dir, output_dir, device=None, use_tta=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Ensemble of {len(ENSEMBLE_MEMBERS)} models" + (" + TTA (4 views each)" if use_tta else ""))

    models = []
    weights = []
    for m in ENSEMBLE_MEMBERS:
        print(f"  - {m['checkpoint']} ({m['variant']}, blocks={m['num_res_blocks']}, channels={m['base_channels']}, weight={m['weight']})")
        models.append(load_one_model(m["checkpoint"], device, m["variant"], m["num_res_blocks"], m["base_channels"]))
        weights.append(m["weight"])
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    os.makedirs(output_dir, exist_ok=True)
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
                weighted_sum = np.zeros((256, 256), dtype=np.float32)
                for i, v in enumerate(variants):
                    x = torch.from_numpy(v.copy()).unsqueeze(0).unsqueeze(0).to(device)
                    for model, w in zip(models, weights):
                        p = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()
                        if i == 1:
                            p = np.fliplr(p)
                        elif i == 2:
                            p = np.flipud(p)
                        elif i == 3:
                            p = np.rot90(p, -2)
                        weighted_sum += p * w
                pred_np = weighted_sum / 4
            else:
                x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                weighted_sum = np.zeros((256, 256), dtype=np.float32)
                for model, w in zip(models, weights):
                    p = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()
                    weighted_sum += p * w
                pred_np = weighted_sum

            elapsed = time.time() - t0
            times.append(elapsed)

            np.save(os.path.join(output_dir, fname), pred_np)

    avg_time = sum(times) / len(times)
    print(f"Processed {len(input_files)} images.")
    print(f"Average inference time per image: {avg_time*1000:.2f} ms")
    print(f"Total time: {sum(times):.2f} s")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tta", action="store_true", help="Also apply test-time augmentation")
    args = parser.parse_args()
    run_ensemble_inference(args.input_dir, args.output_dir, use_tta=args.tta)