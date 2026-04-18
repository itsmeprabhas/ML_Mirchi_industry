"""
ML Mirchi — Dataset Class
Handles loading chilli images with Albumentations transforms.
"""

import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config import CLASSES, CLASS_TO_IDX


class ChilliDataset(Dataset):
    """
    Dataset for chilli quality grading.
    Expects directory structure: root/split/Class/image.jpg
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.class_counts = {cls: 0 for cls in CLASSES}

        for cls in CLASSES:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                for img_path in cls_dir.glob(ext):
                    self.samples.append((str(img_path), CLASS_TO_IDX[cls]))
                    self.class_counts[cls] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    def get_class_weights(self):
        """Compute inverse frequency weights for class balancing."""
        total = len(self.samples)
        weights = {}
        for cls, count in self.class_counts.items():
            weights[cls] = total / (len(CLASSES) * count) if count > 0 else 0
        return weights

    def __repr__(self):
        lines = [f"ChilliDataset(split={self.root_dir.name}, {len(self)} images)"]
        for cls, count in self.class_counts.items():
            lines.append(f"  {cls}: {count}")
        return "\n".join(lines)
