"""
Dataset loader for the KLA image restoration task.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, augment=False):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.augment = augment

        noisy_files = set(os.path.basename(f) for f in glob.glob(os.path.join(noisy_dir, "*.npy")))
        gt_files = set(os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.npy")))
        self.filenames = sorted(noisy_files & gt_files)

        if len(self.filenames) == 0:
            raise RuntimeError(f"No matching pairs found between:\n  {noisy_dir}\n  {gt_dir}")

        missing_gt = noisy_files - gt_files
        missing_noisy = gt_files - noisy_files
        if missing_gt:
            print(f"[warn] {len(missing_gt)} NoisyLR files have no matching GT (skipped)")
        if missing_noisy:
            print(f"[warn] {len(missing_noisy)} GT files have no matching NoisyLR (skipped)")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        noisy = np.load(os.path.join(self.noisy_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                noisy = np.fliplr(noisy).copy()
                gt = np.fliplr(gt).copy()
            if random.random() < 0.5:
                noisy = np.flipud(noisy).copy()
                gt = np.flipud(gt).copy()
            k = random.randint(0, 3)
            if k > 0:
                noisy = np.rot90(noisy, k).copy()
                gt = np.rot90(gt, k).copy()
            # diagonal transpose (both images are square, so this is valid)
            if random.random() < 0.5:
                noisy = noisy.T.copy()
                gt = gt.T.copy()

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        return noisy, gt