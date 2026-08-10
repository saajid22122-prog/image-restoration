AI-Based Restoration of Degraded Images — KLA Hackathon (PS1)

This project restores degraded semiconductor inspection images by removing speckle noise and upsampling images from 128×128 to 256×256 in a single pass. The final solution uses an ensemble of three trained neural networks along with test-time augmentation (TTA) to achieve the best overall performance.

Final results on the held-out validation set:

SSIM: 0.7678  
PSNR: 28.18 dB  
LPIPS: 0.3105  
Average inference time: ~64 ms per image

Repository Structure

```text
ps1_restoration/
├── data/
│   ├── train/
│   │   ├── NoisyLR/         Degraded training inputs (128×128 .npy files)
│   │   ├── GT_full/         Matching ground-truth targets (256×256 .npy files)
│   │   ├── NoisyLR_holdout/ Held-out validation inputs (never used for training)
│   │   └── GT_holdout/      Ground truth for the held-out validation set
│   └── test/
│       └── NoisyLR/         Official test inputs (128×128 .npy files, no ground truth)
├── src/
│   ├── dataset.py              Loads matched NoisyLR/GT pairs
│   ├── model.py                Contains all model architectures
│   ├── train.py                Main training script using the proven configuration
│   ├── train_big.py            Training script for the larger model variant
│   ├── infer.py                Single-model inference
│   ├── ensemble_infer.py       Final standalone inference script used for grading
│   ├── metrics.py              Calculates SSIM, PSNR, and LPIPS
│   └── prepare_val_holdout.py  Rebuilds the held-out validation split
├── outputs/
│   ├── model_noise_aware_LOCAL_v1.pt
│   ├── model_unet_UNET_v1.pt
│   ├── model_noise_aware_BIG_v4.pt
│   └── val_filenames.txt
├── requirements.txt
└── README.md
```

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

The larger model can be trained using `train_big.py`.

During training:

- GPU is used automatically if available.
- A reproducible 90/10 train-validation split is created using a fixed random seed.
- The best checkpoint (lowest validation loss) is saved as:

```text
outputs/model_<variant>_<run_name>.pt
```

- Each run creates a unique checkpoint name, so previous models are never overwritten.
- Training uses a combined L1 + SSIM loss function for better restoration quality.

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

What We Tried

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

Every major design decision—including the choice of architecture, loss function, and the experiments listed above—was based on careful analysis of the training data before model development. The final noise-aware architecture was designed after studying the characteristics and range of noise present in the dataset.
