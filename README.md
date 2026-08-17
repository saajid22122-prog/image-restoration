# HAL 9001 — AI-Based Restoration of Degraded Images (KLA Problem Statement)

Restores degraded semiconductor inspection images: removes noise and upsamples 128x128 -> 256x256 in a single pass.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12 (verified via CI on Windows and Linux). No internet access, API keys, or manual configuration needed at run time — all weights are bundled under `models/` and loaded from paths resolved relative to this script's own location.

## Run

```bash
python run.py <input-dir> <output-dir>
```

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` if it doesn't already exist.
- Writes one restored `.npy` per input, under the same filename.
- Outputs are grayscale `(H, W)` float32 arrays, values clamped to `[0, 1]`, guaranteed free of NaN/Inf.
- Output resolution is 256x256 regardless of input resolution.
- Runs on GPU automatically if an NVIDIA GPU is available (`torch.cuda.is_available()`), otherwise falls back to CPU.

No flags, environment variables, or manual edits are required — the two lines above are the complete setup and run procedure.

## What it does

The model is **NAFNet** (Chen et al., ECCV 2022 architecture), run as an average of two training checkpoints of the same architecture (`models/model_nafnet_synth_v1.pt` + `models/model_nafnet_pre_lpips.pt`), each evaluated with 4-view flip/rotate test-time augmentation (8 forward passes per image total, averaged into the final output). This configuration was the best of every checkpoint pairing measured on a held-out validation set — SSIM 0.7894, PSNR 28.78 dB, LPIPS 0.1770, ~97 ms/image.

## Folder contents

```text
HAL_9001/
├── run.py              Entry point: python run.py <input-dir> <output-dir>
├── model.py             NAFNet model definition (imported by run.py)
├── requirements.txt      Pinned dependencies
├── models/
│   ├── model_nafnet_synth_v1.pt      Primary checkpoint
│   └── model_nafnet_pre_lpips.pt     Second checkpoint, averaged in by default
└── README.md
```
