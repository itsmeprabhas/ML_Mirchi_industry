"""
ML Mirchi — Augmentation Pipelines
Albumentations-based transforms for train/val/test.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import NORMALIZATION


def get_train_transforms(image_size=224):
    """
    Heavy augmentation for training.
    Simulates: different camera angles, lighting, seasons, defects.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=25, p=0.7, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=224):
    """Minimal augmentation for validation/test."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size=224):
    """Transforms for production inference (same as val)."""
    return get_val_transforms(image_size)
