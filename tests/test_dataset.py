"""Tests for dataset loading."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

from src.dataset import ChilliDataset
from src.augmentations import get_train_transforms, get_val_transforms
from config import CLASSES


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a minimal dataset for testing."""
    for split in ["train", "val"]:
        for cls in CLASSES:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
                img.save(cls_dir / f"{cls}_{i}.jpg")
    return tmp_path


class TestDataset:

    def test_len(self, temp_dataset):
        ds = ChilliDataset(temp_dataset, "train")
        assert len(ds) == len(CLASSES) * 3

    def test_getitem_shape(self, temp_dataset):
        ds = ChilliDataset(temp_dataset, "train", get_val_transforms(224))
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert 0 <= label < len(CLASSES)

    def test_class_counts(self, temp_dataset):
        ds = ChilliDataset(temp_dataset, "train")
        for cls in CLASSES:
            assert ds.class_counts[cls] == 3

    def test_repr(self, temp_dataset):
        ds = ChilliDataset(temp_dataset, "train")
        text = repr(ds)
        assert "ChilliDataset" in text
        assert "train" in text

    def test_missing_split_returns_empty(self, tmp_path):
        ds = ChilliDataset(tmp_path, "nonexistent")
        assert len(ds) == 0

    def test_train_transforms_change_image(self, temp_dataset):
        ds_no_aug = ChilliDataset(temp_dataset, "train", get_val_transforms(224))
        ds_aug = ChilliDataset(temp_dataset, "train", get_train_transforms(224))
        img1, _ = ds_no_aug[0]
        img2, _ = ds_aug[0]
        # With augmentation, images should differ
        assert not torch.equal(img1, img2)
