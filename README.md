# AI-Based Restoration of Degraded Images — KLA Hackathon (PS1)

Restores degraded semiconductor inspection images: removes speckle noise and upsamples 128×128 → 256×256 in a single pass.

There are two model tracks in this repo:

- **The submission**: an ensemble of three trained networks + test-time augmentation (TTA). This is what `ensemble_infer.py` runs, and it's what the results below and the official grading are based on.
- **NAFNet**: a newer, larger single model (`model_nafnet_my_run.pt`). It's finished training and now beats the ensemble on every metric while running about 4x faster — see the numbers below — but it isn't wired into `ensemble_infer.py` yet, so it's not part of the graded submission.

## Results

Ensemble (the graded submission), full held-out validation set:

| Metric | Value |
|---|---|
| SSIM | 0.7678 |
| PSNR | 28.18 dB |
| LPIPS | 0.3105 |
| Avg. inference time | ~64 ms/image |

NAFNet (single model, width=48, ~15.9M params), same held-out set:

| Metric | Value |
|---|---|
| SSIM | 0.7872 |
| PSNR | 28.39 dB |
| LPIPS | 0.2394 |
| Avg. inference time | ~16 ms/image |

More on how NAFNet got here, and why it isn't the submission yet, further down.

## Repository structure

```text
ps1_restoration/
├── data/                        Not committed to git — see "About the data" below
│   ├── train/
│   │   ├── NoisyLR/         Degraded training inputs (128×128 .npy files)
│   │   ├── GT_full/         Matching ground-truth targets (256×256 .npy files)
│   │   ├── NoisyLR_holdout/ Held-out validation inputs (never used for training)
│   │   └── GT_holdout/      Ground truth for the held-out validation set
│   └── test/
│       └── NoisyLR/         Official test inputs (128×128 .npy files, no ground truth)
├── src/
│   ├── dataset.py              Loads matched NoisyLR/GT pairs
│   ├── model.py                All model architectures, including NAFNetRestorer
│   ├── train.py                Training script for the ensemble members
│   ├── train_big.py            Training script for NAFNet, with resume support
│   ├── infer.py                Single-model inference
│   ├── ensemble_infer.py       Final inference script — this is what gets graded
│   ├── metrics.py              SSIM / PSNR / LPIPS
│   ├── prepare_val_holdout.py  Rebuilds the held-out validation split
│   ├── run_custom.py           Runs a model on any regular image file (jpg/png)
│   ├── test_real_sample.py     Sanity-checks a checkpoint against real held-out data
│   └── test_degradations.py    Probes a checkpoint against synthetic noise/blur
├── outputs/
│   ├── model_noise_aware_LOCAL_v1.pt   \
│   ├── model_noise_aware_BIG_v4.pt      | the three ensemble models
│   ├── model_unet_UNET_v1.pt           /
│   ├── model_nafnet_my_run.pt          NAFNet, finished checkpoint
│   ├── model_nafnet_my_run.log         Full epoch-by-epoch training history
│   ├── final_test_predictions/         Ensemble output on the official test set (400 files)
│   ├── val_filenames.txt               Held-out split filenames, shared by both tracks
│   └── degradation_test/               Sample outputs from the two test scripts above
├── requirements.txt
└── README.md
```

`model_nafnet_my_run.trainstate.pt` (optimizer/scheduler state, for resuming that specific run) isn't committed — it's about 190MB, over GitHub's limit, and only useful on the machine that produced it.

## About the data

`data/` isn't in git. Two reasons: the training set is roughly 1.3GB, which is more than a repo should carry, and it isn't needed anyway — the checkpoints under `outputs/` are already trained, so cloning this repo is enough to run inference (`infer.py`, `ensemble_infer.py`, `run_custom.py`). You only need the dataset if you want to retrain from scratch, in which case set it up as described below.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is strongly recommended for training — CPU training on this dataset is very slow — but inference runs fine on either.

## Training

Put the dataset here:

```text
data/train/NoisyLR/
data/train/GT_full/
```

Every degraded image needs a matching ground-truth file with the same name, e.g. `000000.npy`.

For the ensemble models:

```bash
cd src
python train.py <num_epochs> <model_variant> <run_name>
```

Model variants: `baseline`, `noise_aware` (this is the one used for the submission).

For NAFNet, use `train_big.py`, which also supports resuming:

```bash
cd src
python train_big.py <num_epochs> nafnet <run_name> <width> [--resume] [--lr 1e-3] [--warmup_epochs 10]
```

- `--resume` continues training under the same `run_name`. If a `.trainstate.pt` exists it picks up the optimizer/scheduler exactly where they left off; otherwise it warm-starts from the best checkpoint and measures a fresh baseline val loss, so a bad run can't silently overwrite a good checkpoint.
- Lower `--lr` and shorter `--warmup_epochs` (e.g. `1e-4` / `2`) when resuming an already-converged checkpoint — using the from-scratch defaults on converged weights caused a real divergence during development (loss jumped from ~0.085 to ~0.51 around epoch 11). Worth knowing if you resume this run yourself.

A few other things during training: GPU is used automatically when available, the train/val split is a fixed-seed 90/10 so it's reproducible, the best checkpoint by validation loss gets saved to `outputs/model_<variant>_<run_name>.pt`, and `train_big.py` also writes a `.log` (survives a crash) and a `.trainstate.pt` (full resumable state, every epoch). Loss is a combination of Charbonnier, SSIM, and an FFT frequency term.

## Inference

This is the script that gets graded:

```bash
cd src
python ensemble_infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output --tta
```

It reads every `.npy` in the input directory, runs all three trained models, optionally averages in 4-view test-time augmentation (`--tta`), clamps to `[0,1]`, and writes 256×256 `.npy` outputs under the original filenames. No manual edits needed — just point it at an input and output directory.

Single-model inference, if you want something faster and don't need the full ensemble:

```bash
python infer.py --checkpoint path/to/model.pt --model_variant noise_aware
```

(use `--model_variant nafnet --width 48` for the NAFNet checkpoint)

To try a checkpoint on a regular image file instead of the `.npy` format:

```bash
python run_custom.py path/to/image.png
```

## Evaluation

```bash
cd src
python metrics.py --pred_dir /path/to/your/predictions --gt_dir /path/to/matching/GT
```

Reports mean SSIM, PSNR, and LPIPS. Only works where ground truth exists — the official test set doesn't have any, so this only runs against the held-out split.

LPIPS pulls down pretrained AlexNet weights the first time it runs. If that download fails, the script warns and just falls back to SSIM/PSNR instead of crashing.

## NAFNet: where it stands

`model_nafnet_my_run.pt` (width=48, ~15.9M params) trains with the same Charbonnier + SSIM + FFT loss as the ensemble, but weighted differently — 0.4 / 0.25 / 0.35 — with the FFT term pushed up specifically to fight the blur that pixel-wise losses tend to produce on fine, repeating texture. It finished its full 20-epoch schedule and the numbers in the Results table above are from the complete 320-image held-out set, not a spot check.

It isn't in `ensemble_infer.py` yet even though it now outperforms the ensemble standalone. Folding it in — as a fourth member, or replacing the ensemble outright — is the obvious next step, just not done yet.

## Known limitations

The model favors structural accuracy over sharp texture. On heavily-noised, fine-detail regions (dense speckle, repeating patterns) output tends to look a bit smoother than the ground truth — that's the loss function's bias, not a bug, and it's the main thing NAFNet's FFT-weighted loss is trying to claw back.

It's also trained specifically on this dataset's degradation: multiplicative speckle noise (std roughly 0.06–0.16 depending on brightness) on structured wafer imagery. Thrown at a different noise type — flat additive Gaussian on an unrelated photo, via `test_degradations.py` — it still produces something structurally reasonable, just noticeably softer. That's expected; the training distribution never included that noise type or image domain. Extending it would mean training on a broader mix of degradations, which wasn't the goal here.

## What we tried (original ensemble)

A few things that didn't make the cut, in case it's useful:

| Technique | Result |
|-----------|--------|
| Gradient / edge-sharpness loss | Reduced overall performance |
| U-Net architecture (skip connections) | No meaningful improvement |
| Larger model (4× parameters) with augmentation | No meaningful improvement |
| Total Variation loss | Worse results, repeatedly |
| Test-time augmentation | Helped — kept it |
| Three-model ensemble | Helped — kept it |
| Weighted four-model ensemble | Slightly worse than the plain three-model average |
| Unsharp masking post-processing | Explored as a training-free sharpening step |

The architecture and loss choices weren't arbitrary — they came out of actually looking at the training data's noise characteristics before writing any model code.
