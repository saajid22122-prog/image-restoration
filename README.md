AI-Based Restoration of Degraded Images — KLA Hackathon (PS1)

This project restores degraded semiconductor inspection images by removing speckle noise and upsampling images from 128×128 to 256×256 in a single pass.

Two model tracks live in this repo:

- **The original submission**: an ensemble of three trained networks + test-time augmentation (TTA), used for grading.
- **NAFNet upgrade (in progress)**: a newer, larger single model (`model_nafnet_my_run.pt`) using NAFNet blocks, currently mid-training. Not yet folded into the ensemble.

Final results on the held-out validation set (original three-model ensemble):

SSIM: 0.7678  
PSNR: 28.18 dB  
LPIPS: 0.3105  
Average inference time: ~64 ms per image

Repository Structure

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
│   ├── model.py                Contains all model architectures (incl. NAFNetRestorer)
│   ├── train.py                Main training script using the proven configuration
│   ├── train_big.py            Training script for the larger NAFNet model, with resume support
│   ├── infer.py                Single-model inference
│   ├── ensemble_infer.py       Final standalone inference script used for grading
│   ├── metrics.py              Calculates SSIM, PSNR, and LPIPS
│   ├── prepare_val_holdout.py  Rebuilds the held-out validation split
│   ├── run_custom.py           Runs a model on any regular image file (jpg/png)
│   ├── test_real_sample.py     Sanity-checks a checkpoint against real held-out data (SSIM/PSNR)
│   └── test_degradations.py    Probes a checkpoint against synthetic noise/blur (out-of-distribution)
├── outputs/
│   ├── model_noise_aware_LOCAL_v1.pt   \
│   ├── model_noise_aware_BIG_v4.pt      | original three-model ensemble
│   ├── model_unet_UNET_v1.pt           /
│   ├── model_nafnet_my_run.pt          NAFNet upgrade, best checkpoint so far
│   ├── model_nafnet_my_run.log         Full epoch-by-epoch training history
│   ├── val_filenames.txt               Held-out split filenames (shared by both tracks)
│   └── degradation_test/               Sample outputs from the two test scripts above
├── requirements.txt
└── README.md
```

Note: `model_nafnet_my_run.trainstate.pt` (optimizer/scheduler state for resuming that specific run) is intentionally **not** committed — it's ~190MB, over GitHub's file size limit, and only useful for resuming training on the machine that made it, not for using the model.

About the data

`data/` is deliberately excluded from git (see `.gitignore`) for two reasons:

- **Size**: the training set alone is roughly 1.3GB — far more than a git repo should carry, and GitHub isn't built for dataset hosting.
- **It's not needed to use the trained models**: the checkpoints in `outputs/` are already trained. Anyone cloning this repo can run inference (`infer.py`, `ensemble_infer.py`, `run_custom.py`) immediately without the dataset. It's only required if you want to retrain from scratch — in which case, place it locally as described below.

Setup

Install the required packages using:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. A CUDA-enabled GPU is highly recommended for training because CPU training is extremely slow on this dataset. Inference, however, runs comfortably on either GPU or CPU.

1. Training

Place the complete training dataset inside:

```text
data/train/NoisyLR/
data/train/GT_full/
```

Each degraded image should have a matching ground-truth file with the same filename (for example, `000000.npy`).

Run training with:

```bash
cd src
python train.py <num_epochs> <model_variant> <run_name>
```

Available model variants:

- baseline
- noise_aware (recommended)

The larger NAFNet model can be trained using `train_big.py`, which also supports resuming:

```bash
cd src
python train_big.py <num_epochs> nafnet <run_name> <width> [--resume] [--lr 1e-3] [--warmup_epochs 10]
```

- `--resume`: continues training for the same `run_name`. If a `.trainstate.pt` exists (saved every epoch), it resumes the optimizer/scheduler exactly where they left off. Otherwise it warm-starts from the best-weights checkpoint with a freshly measured baseline val loss, so the checkpoint is never overwritten by a regression.
- `--lr` / `--warmup_epochs`: use a lower peak LR (e.g. `1e-4`) and a shorter warmup (e.g. `2`) when resuming an already-converged checkpoint — see "What Could Be Better" below for why this matters.

During training:

- GPU is used automatically if available.
- A reproducible 90/10 train-validation split is created using a fixed random seed.
- The best checkpoint (lowest validation loss) is saved as `outputs/model_<variant>_<run_name>.pt`.
- `train_big.py` additionally writes `outputs/model_<variant>_<run_name>.log` (full history, survives a crash) and `outputs/model_<variant>_<run_name>.trainstate.pt` (full resumable state, every epoch).
- Each run creates a unique checkpoint name, so previous models are never overwritten.
- Training uses a combined Charbonnier + SSIM + FFT frequency-domain loss.

2. Inference (Final Submission Script)

Run the complete ensemble using:

```bash
cd src
python ensemble_infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output --tta
```

This script:

- Reads every `.npy` file from the input directory.
- Runs all three trained models.
- Optionally applies test-time augmentation (`--tta`) using four flipped and rotated views for each model.
- Averages all predictions together.
- Saves restored 256×256 `.npy` files using the original filenames.
- Clamps output values to the `[0,1]` range.
- Prints total inference time and average inference time per image.

No manual code changes are required. Simply provide the input and output directories.

For faster but slightly lower accuracy, single-model inference is available:

```bash
python infer.py --checkpoint path/to/model.pt --model_variant noise_aware
```

To test a checkpoint on an arbitrary image file instead of the `.npy` dataset format:

```bash
python run_custom.py path/to/image.png
```

3. Evaluation

Evaluate predictions against ground truth using:

```bash
cd src
python metrics.py --pred_dir /path/to/your/predictions --gt_dir /path/to/matching/GT
```

The script reports:

- Mean SSIM
- Mean PSNR
- Mean LPIPS

Use this only on datasets where ground truth is available, such as the held-out validation split. Since the official test set does not contain ground truth, evaluation cannot be performed there.

LPIPS downloads pretrained AlexNet weights the first time it runs. If the download is unavailable, the script prints a warning and continues with SSIM and PSNR instead of stopping.

NAFNet Upgrade: Current Accuracy

`model_nafnet_my_run.pt` (NAFNet, width=48, 15.9M params) is a single model, not yet part of the ensemble above, trained with a combined Charbonnier + SSIM + FFT loss (`train_big.py`). It is still training as of this writing — numbers below will keep improving.

Validation loss (combined loss, lower is better, same held-out split as the ensemble):

| Stage | Val loss |
|---|---|
| First training pass (interrupted by a system shutdown, recovered from checkpoint) | 0.0872 |
| Resumed at original LR (diverged at epoch 11 — see below) | 0.0853 before divergence |
| Resumed again at a lower LR, in progress | 0.0847 and improving |

Spot-check against 3 real held-out samples (`test_real_sample.py`), for a sense of per-image quality — not a full-set average:

| Sample | SSIM | PSNR |
|---|---|---|
| 002929 (heavy fine texture — hardest case) | 0.647 | 22.6 dB |
| 000325 | 0.928 | 32.2 dB |
| 000832 | 0.915 | 27.5 dB |

For a trustworthy overall number, run `metrics.py` against the full held-out set once training finishes — 3 samples is only a sanity check, not a benchmark.

What Could Be Better

| Area | Issue | Fix / next step |
|---|---|---|
| Training stability | Resuming from a converged checkpoint with the from-scratch LR schedule (peak 1e-3, 10-epoch warmup) caused the loss to spike from ~0.085 to ~0.51 at epoch 11 — warmup pushed already-good weights out of their minimum | `train_big.py` now accepts `--lr` and `--warmup_epochs`; use a much lower peak LR (e.g. 1e-4) and short warmup (e.g. 2 epochs) when continuing a converged run, not the from-scratch defaults |
| Evaluation coverage | Current NAFNet accuracy numbers are from 3 hand-picked samples, not the full 320-file held-out set | Run `metrics.py` on the full held-out split once training completes, for a statistically meaningful SSIM/PSNR/LPIPS |
| Convergence | The current low-LR resume run hasn't finished its 20-epoch schedule yet | Let it complete the full cosine decay before judging final quality |
| Ensemble integration | The NAFNet model isn't included in `ensemble_infer.py` yet, so the graded submission still only reflects the original three models | Once NAFNet training stabilizes and beats the individual ensemble members, add it as a fourth ensemble member and re-tune TTA/weighting |
| Checkpoint size | `model_nafnet_my_run.pt` is ~61MB, over GitHub's 50MB soft limit (still pushes fine, just triggers a warning) | Consider Git LFS if checkpoint sizes keep growing |
| Out-of-distribution generalization | See "Known Limitations" below — this is architectural/data-driven, not something the current training run fixes | Would need training data covering more noise types, if that becomes a goal |

What We Tried (original ensemble)

| Technique | Result |
|-----------|--------|
| Gradient / edge-sharpness loss | Reduced overall performance |
| U-Net architecture (skip connections) | No meaningful improvement |
| Larger model (4× parameters) with augmentation | No meaningful improvement |
| Total Variation loss | Produced worse results in repeated testing |
| Test-time augmentation | Consistent improvement and included in the final solution |
| Three-model ensemble | Consistent improvement and included in the final solution |
| Weighted four-model ensemble | Slightly worse than the simple three-model ensemble |
| Unsharp masking post-processing | Explored as a training-free sharpening technique |

Known Limitations

The model focuses on preserving structural accuracy rather than producing extremely sharp textures. As a result, very fine details may appear slightly smoother than the original ground truth, which is a deliberate tradeoff considering the heavy noise present in the input images.

The model was also tested on images with a completely different type of noise from outside the training distribution. It continued to produce structurally reasonable outputs, although the results were noticeably softer than those obtained on the semiconductor dataset. This reflects a genuine generalization limit rather than a failure of the model.

Concretely: the training data's degradation is **speckle noise** (multiplicative, std scaling from ~0.06 in dark regions to ~0.16 in bright regions), applied to structured semiconductor wafer imagery. Testing the NAFNet checkpoint against flat additive Gaussian noise on an unrelated photograph (`test_degradations.py`) produced a much weaker result — a mismatch in both noise statistics and image domain, not evidence the model is broken. Blur (which removes information rather than adding noise) is out-of-distribution in a different way and is also handled only softly, as expected.

Every major design decision—including the choice of architecture, loss function, and the experiments listed above—was based on careful analysis of the training data before model development. The final noise-aware architecture was designed after studying the characteristics and range of noise present in the dataset.
