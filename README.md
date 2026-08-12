# AI-Based Restoration of Degraded Images — KLA Hackathon (PS1)

Restores degraded semiconductor inspection images: removes speckle noise and upsamples 128×128 → 256×256 in a single pass.

The submission is a single architecture, **NAFNet**, used as an average of two checkpoints of itself (`model_nafnet_my_run.pt` + `model_nafnet_pre_lpips.pt` — one architecture, two training snapshots, not a multi-architecture ensemble). An earlier four-model weighted ensemble of genuinely different architectures is also still in this repo — it was the original approach, and it's kept for reference since it's part of the project's history — but NAFNet beats it on every metric while running faster, so NAFNet is what `infer.py` runs by default and what the results below are based on.

## Results

NAFNet (submission, width=48, ~15.9M params), full held-out validation set. Default `infer.py` behavior with zero flags — TTA (4-view flip/rotate averaging) on, both checkpoints averaged:

| Metric | Value |
|---|---|
| SSIM | 0.7879 |
| PSNR | 28.70 dB |
| LPIPS | 0.1796 |
| Avg. inference time | ~101 ms/image |

This beats the pre-fine-tune checkpoint on every metric (SSIM 0.7872, PSNR 28.39dB, LPIPS 0.2394) while still keeping a large LPIPS improvement. Other configurations, same held-out set, for reference:

| Config | SSIM | PSNR | LPIPS | ms/image |
|---|---|---|---|---|
| Pre-fine-tune checkpoint alone, no TTA | 0.7872 | 28.39 | 0.2394 | ~15 |
| Fine-tuned checkpoint alone, no TTA | 0.7768 | 28.30 | 0.1264 | ~15 |
| Fine-tuned checkpoint alone, TTA | 0.7826 | 28.58 | 0.1397 | ~52 |
| **Both checkpoints averaged, TTA each (default)** | **0.7879** | **28.70** | **0.1796** | ~101 |

For comparison, the earlier four different-architecture ensemble + TTA, same held-out set:

| Metric | Value |
|---|---|
| SSIM | 0.7678 |
| PSNR | 28.18 dB |
| LPIPS | 0.3105 |
| Avg. inference time | ~64–110 ms/image |

More on how NAFNet got here, further down.

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
│   ├── train_big.py            Training script for NAFNet (the submission) — run this to reproduce it from scratch
│   ├── train_legacy_ensemble.py  Training script for the retired ensemble's members — reference only, NOT the submission
│   ├── infer.py                Inference script — this is what gets graded. Defaults to NAFNet.
│   ├── ensemble_infer.py       The earlier four-model ensemble, kept for reference
│   ├── metrics.py              SSIM / PSNR / LPIPS
│   ├── prepare_val_holdout.py  Rebuilds the held-out validation split
│   ├── run_custom.py           Runs a model on any regular image file (jpg/png)
│   ├── test_real_sample.py     Sanity-checks a checkpoint against real held-out data
│   └── test_degradations.py    Probes a checkpoint against synthetic noise/blur
├── outputs/
│   ├── model_nafnet_my_run.pt          NAFNet, fine-tuned checkpoint (the primary submitted weights)
│   ├── model_nafnet_pre_lpips.pt       NAFNet, pre-fine-tune checkpoint -- averaged with the above by default
│   ├── model_nafnet_my_run.log         Full epoch-by-epoch training history
│   ├── model_noise_aware_LOCAL_v1.pt   \
│   ├── model_noise_aware_BIG_v4.pt      | earlier four-model ensemble, reference only
│   ├── model_noise_aware_TV_v2.pt      |
│   ├── model_unet_UNET_v1.pt           /
│   ├── final_test_predictions/         NAFNet output on the official test set (400 files)
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

For NAFNet, use `train_big.py`, which also supports resuming:

```bash
cd src
python train_big.py <num_epochs> nafnet <run_name> <width> [--resume] [--lr 1e-3] [--warmup_epochs 10]
```

- `--resume` continues training under the same `run_name`. If a `.trainstate.pt` exists it picks up the optimizer/scheduler exactly where they left off; otherwise it warm-starts from the best checkpoint and measures a fresh baseline val loss, so a bad run can't silently overwrite a good checkpoint.
- Lower `--lr` and shorter `--warmup_epochs` (e.g. `1e-4` / `2`) when resuming an already-converged checkpoint — using the from-scratch defaults on converged weights caused a real divergence during development (loss jumped from ~0.085 to ~0.51 around epoch 11). Worth knowing if you resume this run yourself.

A few other things during training: GPU is used automatically when available, the train/val split is a fixed-seed 90/10 so it's reproducible, the best checkpoint by validation loss gets saved to `outputs/model_<variant>_<run_name>.pt`, and `train_big.py` also writes a `.log` (survives a crash) and a `.trainstate.pt` (full resumable state, every epoch). Loss is a combination of Charbonnier, SSIM, and an FFT frequency term.

For the retired ensemble's members (not the submission — reference only): `python train_legacy_ensemble.py <num_epochs> <model_variant> <run_name>`, with `model_variant` one of `baseline` / `noise_aware` / `unet`.

## Inference

This is the script that gets graded:

```bash
cd src
python infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output
```

No manual edits needed — `--input_dir`/`--output_dir` is all you need to supply. By default it loads both `model_nafnet_my_run.pt` and `model_nafnet_pre_lpips.pt` (same NAFNet architecture, two checkpoints), runs 4-view TTA on each, and averages everything together (pass `--single_checkpoint` to use only `model_nafnet_my_run.pt`, or `--no_tta` to also drop TTA, e.g. for a raw speed benchmark). It reads every `.npy` in the input directory, clamps to `[0,1]`, and writes 256×256 `.npy` outputs under the original filenames.

The earlier four-model ensemble is still runnable the same way, if you want to compare:

```bash
python ensemble_infer.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output --tta
```

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

## NAFNet: how it got here

`model_nafnet_my_run.pt` first trained with a combined Charbonnier + SSIM + FFT loss — weighted 0.4 / 0.25 / 0.35 — with the FFT term pushed up specifically to fight the blur that pixel-wise losses tend to produce on fine, repeating texture. That version finished its full 20-epoch schedule and beat the four-model ensemble on every metric while running several times faster, which is what made it the actual submission over the ensemble track.

It was then fine-tuned for another 15 epochs with two additions aimed at the remaining texture-blur gap: an **LPIPS perceptual loss term** (rewards "looks like the same texture" in a learned feature space, not just per-pixel closeness) and a **spatially-weighted Charbonnier term** (weights the pixel loss by local ground-truth gradient magnitude, so texture-dense regions get more gradient signal instead of being averaged into flat background). Combined loss is now Charbonnier(spatially-weighted) / SSIM / FFT / LPIPS weighted 0.30 / 0.20 / 0.25 / 0.25.

This is a real, measured trade at the model level: without TTA, LPIPS improved 47% relative (0.2394 → 0.1264) and fine texture is visibly closer to ground truth on cases with real structured detail, at the cost of SSIM dropping ~1.3% (0.7872 → 0.7768) and PSNR ~0.09 dB. Verified on the full 320-image held-out set before and after, not a spot check.

Turning TTA on recovers most of that SSIM/PSNR cost on its own: SSIM 0.7826, PSNR 28.58 dB (both close to or above the pre-fine-tune numbers), LPIPS still 42% better (0.1397).

Going further, **averaging the two checkpoints' TTA predictions together** (the pre-fine-tune and fine-tuned NAFNet, same architecture, both run with TTA) turned out to beat every single-checkpoint configuration on SSIM and PSNR — including the original pre-fine-tune checkpoint alone — while still keeping a real 25% LPIPS improvement over it. This is why it's the default: it isn't a trade at all on the aggregate metrics, it's close to a clean win on all three, for roughly double the inference time of single-checkpoint TTA (~101ms/image vs ~52ms, still fast). This was found by testing, not assumed going in — see the full comparison table in Results above.

## Known limitations

The model still favors structural accuracy over sharp texture to some degree — the LPIPS + spatially-weighted fine-tune narrows this gap but doesn't eliminate it, and it doesn't help uniformly. Distinct patterns showed up during evaluation, worth being upfront about:

- **Structured fine detail** (wire, fabric weave, repeating patterns): the fine-tune measurably helps here — this is the case the added loss terms specifically target, and the clearest visible win (e.g. held-out sample `000325`).
- **Dense random grain/speckle baked into the ground truth itself** (the model's per-pixel randomness, not recoverable structure): no loss function change can fix this. A deterministic model can't reproduce a specific random noise realization it was never given enough information to predict — the "smooth" prediction is actually the theoretically correct hedge against unpredictable per-pixel noise. The held-out set's worst-SSIM grain cases improved slightly anyway under the fine-tune, though that's not guaranteed to generalize.
- **Water-ripple texture, diagnosed as a training-data coverage gap, not a loss/architecture problem** (held-out sample `002929`): this is the project's longest-standing hard case, and nothing tried today moved it much — SSIM and PSNR regress slightly under the fine-tune alone (0.6550 → 0.6298 SSIM, 22.55 → 22.34 dB PSNR, no TTA), and the default two-checkpoint TTA ensemble only partially recovers it (SSIM 0.6577, PSNR 22.76 dB vs the pre-fine-tune checkpoint's own TTA score of 0.6627 / 22.79 dB).

  Root cause, found by searching all 3,200 training images for a similar frequency-spectrum signature (radially-averaged power spectral density) to `002929`: only 2 other training images (`002930`, `002931`) share this content, and they're clearly crops/frames of the *same source photo* (same wall, same water body), not independent examples. Every other spectrally-similar match turned out to be visually unrelated content (gravel, animal fur, calm water) that just happens to share a coincidental frequency profile. So the network saw essentially one real example of this texture during training — nowhere near enough to learn a general "plausible ripple" prior, only enough to partially memorize that one scene. No loss function change or architecture swap fixes a data coverage gap; the correct fix would be adding more training examples of this texture type (real or synthetic), which is a bigger scope change than a loss/inference tweak and wasn't pursued here. Documented rather than chased further, since diagnosing the actual cause is more useful than another blind attempt at a fix.

It's also trained specifically on this dataset's degradation: multiplicative speckle noise (std roughly 0.06–0.16 depending on brightness) on structured wafer imagery. Thrown at a different noise type — flat additive Gaussian on an unrelated photo, via `test_degradations.py` — it still produces something structurally reasonable, just noticeably softer. That's expected; the training distribution never included that noise type or image domain. Extending it would mean training on a broader mix of degradations, which wasn't the goal here.

## What we tried (earlier ensemble track)

A few things that didn't make the cut, in case it's useful:

| Technique | Result |
|-----------|--------|
| Gradient / edge-sharpness loss | Reduced overall performance |
| U-Net architecture (skip connections) | No meaningful improvement |
| Larger model (4× parameters) with augmentation | No meaningful improvement |
| Total Variation loss | Worse results, repeatedly |
| Test-time augmentation | Helped — kept it |
| Multi-model ensemble | Helped — kept it |
| Unsharp masking post-processing | Explored as a training-free sharpening step |

The architecture and loss choices weren't arbitrary — they came out of actually looking at the training data's noise characteristics before writing any model code.
