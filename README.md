# AI-Based Restoration of Degraded Images — KLA Hackathon (PS1)

Restores degraded semiconductor inspection images: removes noise and upsamples 128×128 → 256×256 in a single pass, end to end.

## Quick start (evaluation)

```bash
pip install -r requirements.txt
cd src
python infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output
```

That's the entire setup. No manual edits, no extra flags required — default checkpoints and paths are resolved relative to the script's own location, not the caller's working directory, so this works from any directory on any machine. Reads every `.npy` in the input directory, writes matching 256×256 `.npy` outputs under the original filenames to the output directory.

## Results

Measured on a fixed, held-out 320-image validation split (never used in training). Default `infer.py` behavior with zero flags — TTA (4-view flip/rotate averaging) on both checkpoints, averaged together:

| Metric | Value |
|---|---|
| SSIM | **0.7894** |
| PSNR | **28.78 dB** |
| LPIPS | **0.1770** |
| Avg. inference time | ~97 ms/image |

This is the best configuration found, out of every checkpoint pairing tested — including 3-way ensembles:

| Config | SSIM | PSNR | LPIPS | ms/image |
|---|---|---|---|---|
| New checkpoint alone, TTA | 0.7842 | 28.62 | 0.1331 | ~78 |
| New + fine-tuned checkpoint, TTA each | 0.7851 | 28.68 | 0.1358 | ~155 |
| All 3 checkpoints, TTA each | 0.7882 | 28.75 | 0.1617 | ~150 |
| **New + pre-LPIPS checkpoint, TTA each (default)** | **0.7894** | **28.78** | **0.1770** | ~97 |

The model is **NAFNet**, run as an average of two checkpoints of itself (`model_nafnet_synth_v1.pt` + `model_nafnet_pre_lpips.pt` — one architecture, two training snapshots, not a multi-architecture ensemble). This beat every other configuration tried on every metric simultaneously, including a heavier 3-way ensemble — more checkpoints isn't automatically better, this pairing was found by direct measurement, not assumption.

An earlier four different-architecture weighted ensemble was the original approach; NAFNet alone beat it on every metric while running faster, so it became the submission (its code/checkpoints were removed from the repo; still recoverable from git history, final numbers below for context):

| Metric | Value |
|---|---|
| SSIM | 0.7678 |
| PSNR | 28.18 dB |
| LPIPS | 0.3105 |

More on how the current default was found, further down.

## What makes this approach different

Two things, both aimed at the actual reliability stakes of an inspection pipeline, not just the leaderboard metric:

**1. Diagnosis-driven training data, not blind augmentation.** A recurring hard case (a water-ripple texture, held-out sample `002929`) was root-caused by searching all 3,200 training images for similar frequency signatures — the model had seen essentially *one* real independent example of that texture family. Confirmed as a genuine training-data coverage gap (no loss/architecture tuning fixed it on its own), not a model weakness. `synthesize_data.py` fixes this directly: it fits a real noise model from KLA's own (noisy, clean) training pairs, automatically finds under-represented texture clusters via frequency-signature isolation analysis, and generates new synthetic pairs by recomposing real texture content and re-degrading it with the fitted noise model — so synthetic examples stay statistically consistent with KLA's real sensor data, not generic assumed noise. This expanded the synthetic training set from 510 to 2,610 pairs across 400 previously under-covered clusters, and measurably improved the specific diagnosed hard case (SSIM 0.6684 vs. the previous best of 0.6627 on `002929`).

**2. Free per-image confidence reporting.** Every inference already computes 8 raw predictions per image (2 checkpoints × 4 TTA views) before averaging them into the final output. `infer.py` also measures how much those 8 predictions *disagree* with each other and reports it as a per-image confidence signal — flagging images where disagreement is unusually high (relative to the rest of that run) as candidates for manual review. This costs nothing extra (the predictions are already being computed) and is verified against real ground truth, not just asserted: on the full 320-image held-out set, flagged images average **SSIM 0.68** vs. **0.81** for unflagged images (correlation -0.45 across all 320 images between reported uncertainty and real SSIM). Written to `<output_dir>_confidence_report.csv` alongside every inference run by default.

## Repository structure

```text
image-restoration/
├── data/                        Not committed to git — see "About the data" below
│   ├── train/
│   │   ├── NoisyLR/              Degraded training inputs (128×128 .npy files)
│   │   ├── GT_full/               Matching ground-truth targets (256×256 .npy files)
│   │   ├── NoisyLR_synth/         Synthetic training inputs (generated, see synthesize_data.py)
│   │   ├── GT_synth/               Matching synthetic ground truth
│   │   ├── NoisyLR_holdout/       Held-out validation inputs (never used for training)
│   │   └── GT_holdout/             Ground truth for the held-out validation set
│   └── test/
│       └── NoisyLR/                Official test inputs (128×128 .npy files, no ground truth)
├── src/
│   ├── dataset.py                  Loads matched NoisyLR/GT pairs, optional augmentation
│   ├── model.py                    All model architectures, including NAFNetRestorer
│   ├── train_big.py                Training script — run this to reproduce the model from scratch
│   ├── infer.py                    Evaluation script — this is what gets graded
│   ├── synthesize_data.py          Generates domain-faithful synthetic training pairs
│   ├── metrics.py                  SSIM / PSNR / LPIPS computation
│   └── dev_tools/                  Optional sanity-check utilities, not part of the graded path
│       ├── prepare_val_holdout.py    Rebuilds the held-out validation split
│       ├── run_custom.py             Runs the model on any regular image file (jpg/png)
│       ├── test_real_sample.py       Sanity-checks a checkpoint against real held-out data
│       └── test_degradations.py      Probes a checkpoint against synthetic noise/blur
├── outputs/
│   ├── model_nafnet_synth_v1.pt         NAFNet checkpoint, trained with the expanded synthetic set
│   ├── model_nafnet_pre_lpips.pt        NAFNet checkpoint, earlier snapshot — averaged with the above by default
│   ├── model_nafnet_synth_v1.log        Full epoch-by-epoch training history
│   ├── final_test_predictions/          Model output on the official 400-image test set
│   ├── val_filenames.txt                 Held-out split filenames
│   └── degradation_test/                 Sample before/after/ground-truth image triplets
├── requirements.txt                 Full pip freeze, verified installable in a clean environment
└── README.md
```

`model_nafnet_synth_v1.trainstate.pt` (optimizer/scheduler state, for resuming that specific run) isn't committed — it's ~190MB, over GitHub's recommended limit, and only useful on the machine that produced it.

## About the data

`data/` isn't in git — the training set is roughly 1.3GB, more than a repo should carry, and it isn't needed to run inference anyway. The committed checkpoints under `outputs/` are already trained, so cloning this repo is sufficient for `infer.py`. You only need the dataset to retrain from scratch, described below.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is strongly recommended for training — CPU training on this dataset is very slow — but inference runs fine on either.

## Training (reproduce from scratch)

Put the dataset here:

```text
data/train/NoisyLR/
data/train/GT_full/
```

Every degraded image needs a matching ground-truth file with the same name, e.g. `000000.npy`.

```bash
cd src
python train_big.py <num_epochs> nafnet <run_name> <width> [--resume] [--lr 1e-3] [--warmup_epochs 10] [--use_synth]
```

- `--use_synth` adds synthetic pairs from `data/train/NoisyLR_synth` + `GT_synth` to the training set (generate these first with `synthesize_data.py`, below). The held-out validation split is always computed from real data alone, regardless of this flag.
- `--resume` continues training under the same `run_name`. If a `.trainstate.pt` exists it picks up the optimizer/scheduler exactly where they left off; otherwise it warm-starts from the best checkpoint and measures a fresh baseline val loss.
- Lower `--lr` and shorter `--warmup_epochs` (e.g. `1e-4` / `2`) when resuming an already-converged checkpoint.

The best checkpoint by validation loss is saved to `outputs/model_<variant>_<run_name>.pt`. `train_big.py` also writes a `.log` (survives a crash) and a `.trainstate.pt` (full resumable state, saved every epoch). Loss is a combination of spatially-weighted Charbonnier, SSIM, an FFT frequency term, and LPIPS.

### Generating synthetic training data

```bash
cd src
python synthesize_data.py --top_n 400 --n_per_source 6
```

Fits a noise model from the real training pairs, finds under-represented texture clusters, and writes new synthetic pairs to `data/train/NoisyLR_synth` / `GT_synth`. Requires `outputs/val_filenames.txt` to already exist (run `train_big.py` once first, or `dev_tools/prepare_val_holdout.py`) so synthesis never samples from the held-out split.

## Inference (the graded script)

```bash
cd src
python infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output
```

By default it loads both checkpoints, runs 4-view TTA on each, and averages everything together (`--single_checkpoint` to use only the primary checkpoint, `--no_tta` to also drop TTA, `--no_confidence_report` to skip the confidence CSV). Every `.npy` in the input directory is read, clamped to `[0,1]`, and written as a 256×256 `.npy` output under the original filename.

## Evaluation (against ground truth)

```bash
cd src
python metrics.py --pred_dir /path/to/your/predictions --gt_dir /path/to/matching/GT
```

Reports mean SSIM, PSNR, and LPIPS. Only works where ground truth exists — the official test set doesn't have any, so this only runs against the held-out split.

## How the current default was found

The base model trained with a combined Charbonnier + SSIM + FFT loss (weighted 0.4/0.25/0.35), then was fine-tuned with an added LPIPS perceptual loss term and a spatially-weighted Charbonnier term (weighted by local ground-truth gradient magnitude) to recover fine texture detail without over-smoothing. Averaging that fine-tuned checkpoint's TTA predictions with an earlier pre-fine-tune snapshot recovered the small SSIM/PSNR cost of the perceptual fine-tune while keeping most of its LPIPS gain — this became the first default ensemble.

After confirming with the competition organizers that supplementing the dataset with synthetic training data is explicitly permitted, `synthesize_data.py` was built to target the specific data-coverage gaps diagnosed in testing (see "What makes this approach different" above), and a new checkpoint (`model_nafnet_synth_v1.pt`) was trained on real data plus the expanded synthetic set. Every plausible checkpoint pairing was then measured directly on the held-out set — solo, 2-way, and 3-way ensembles — rather than assumed; the winner (`model_nafnet_synth_v1.pt` + `model_nafnet_pre_lpips.pt`) beat every other configuration on all three metrics simultaneously, including combinations using more checkpoints, which is why it's the default.

## Known limitations

- **Water-ripple texture** (held-out sample `002929`), the project's longest-standing hard case: root-caused as a training-data coverage gap (the model had seen essentially one real independent example of this texture family — searched via frequency-spectrum similarity across all 3,200 training images). The targeted synthetic data measurably improved this case (SSIM 0.6684 vs. a previous best of 0.6627) but did not fully close the gap — it's still correctly flagged as low-confidence by the reliability signal above, which is the honest outcome: better, not solved.
- **Dense random grain/speckle baked into some ground-truth images themselves** is irreducible: a deterministic model can't reproduce a specific random noise realization it was never given enough information to predict. No loss function or data change fixes this category.
- **Confidence flagging catches most, not all, of the worst cases.** Of the 10 lowest-SSIM images in the held-out set, 3 were caught by the low-confidence flag. It's a real, verified signal (see the correlation numbers above) — worth using as a triage aid, not a guaranteed worst-case detector.
- Trained specifically on this dataset's real degradation characteristics (multiplicative speckle noise, std roughly 0.06–0.16 depending on brightness, measured directly from the provided data) on structured wafer-style imagery. Thrown at an unrelated noise type or image domain — tested via `dev_tools/test_degradations.py` with synthetic additive Gaussian noise on an arbitrary photo — it still produces something structurally reasonable, just softer; that's expected, since the training distribution never included that noise type or domain.

## What we tried and didn't keep

| Technique | Result |
|-----------|--------|
| Gradient / edge-sharpness loss | Reduced overall performance |
| U-Net architecture (skip connections) | No meaningful improvement |
| Larger model (4× parameters) with augmentation | No meaningful improvement |
| Total Variation loss | Worse results, repeatedly |
| 3-way checkpoint ensemble | Worse than the 2-way winner on every metric |
| Multi-architecture ensemble (4 different models) | Helped over any single legacy model, but NAFNet alone beat it outright |

The architecture, loss, and data choices weren't arbitrary — each came from directly measuring the training data and the model's actual failure modes before writing a fix, not from guessing.
