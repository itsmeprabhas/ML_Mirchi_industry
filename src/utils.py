"""
ML Mirchi — Utility Functions
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


def _rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
