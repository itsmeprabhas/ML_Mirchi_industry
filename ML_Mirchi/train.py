"""
ML Mirchi — Chilli Quality Grading Model Training
Uses Transfer Learning with EfficientNet-B0 for fast, accurate training.
Architecture chosen for low latency (< 30ms) as required by real-time grading.
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

CLASSES = ["Grade_A", "Grade_B", "Grade_C", "Reject"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Training hyperparameters
DEFAULT_CONFIG = {
    "model_name": "efficientnet_b0",
    "pretrained": True,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "label_smoothing": 0.1,
    "dropout_rate": 0.3,
    "num_workers": 4,
    "seed": 42,
    "early_stopping_patience": 7,
    "save_best_only": True,
}


# ============================================================
# CUSTOM DATASET WITH ALBUMENTATIONS
# ============================================================
class ChilliDataset(torch.utils.data.Dataset):
    """
    Custom dataset that applies Albumentations augmentations.
    Much more powerful than torchvision transforms for this task.
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []

        for cls in CLASSES:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), CLASS_TO_IDX[cls]))
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), CLASS_TO_IDX[cls]))
            for img_path in cls_dir.glob("*.jpeg"):
                self.samples.append((str(img_path), CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# ============================================================
# AUGMENTATION PIPELINES
# ============================================================
def get_train_augmentations(image_size=224):
    """
    Heavy augmentations for training.
    Simulates real-world variation: lighting, rotation, camera angle, noise.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        # Geometric transforms — simulates different camera angles
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=25, p=0.7, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=10,
            border_mode=0, p=0.5
        ),
        # Color/lighting — simulates different camera conditions & seasons
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.7
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
        ),
        # Quality — simulates different camera resolutions
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        # Defect simulation — helps model learn defect features
        A.CoarseDropout(
            max_holes=8, max_height=20, max_width=20,
            min_holes=1, fill_value=0, p=0.3
        ),
        # Normalization for pretrained models
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_augmentations(image_size=224):
    """Minimal augmentation for validation/test — just resize and normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
def create_model(config):
    """
    Create the grading model using transfer learning.
    EfficientNet-B0: 5.3M params, ~5ms inference on GPU, ~30ms on CPU.
    Perfect for real-time conveyor belt grading.
    """
    if config["model_name"] == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if config["pretrained"] else None
        model = models.efficientnet_b0(weights=weights)
        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config["dropout_rate"]),
            nn.Linear(in_features, NUM_CLASSES),
        )

    elif config["model_name"] == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if config["pretrained"] else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config["dropout_rate"]),
            nn.Linear(in_features, NUM_CLASSES),
        )

    elif config["model_name"] == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if config["pretrained"] else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=config["dropout_rate"]),
            nn.Linear(in_features, NUM_CLASSES),
        )

    else:
        raise ValueError(f"Unknown model: {config['model_name']}")

    return model


# ============================================================
# MIXUP & CUTMIX (Regularization techniques)
# ============================================================
def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, config, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Randomly apply MixUp or CutMix
        r = np.random.random()
        if r < 0.3 and config["mixup_alpha"] > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, config["mixup_alpha"])
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            # For accuracy, use original labels approximation
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() +
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        elif r < 0.5 and config["cutmix_alpha"] > 0:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, config["cutmix_alpha"])
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() +
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        total += labels.size(0)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc="Validating", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    return {
        "loss": total_loss / len(loader),
        "accuracy": 100.0 * correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


def train(config):
    """Main training function."""

    # Setup
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} — {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Datasets
    print("[DATA] Loading datasets...")
    train_dataset = ChilliDataset(DATA_DIR, "train", get_train_augmentations(config["image_size"]))
    val_dataset = ChilliDataset(DATA_DIR, "val", get_val_augmentations(config["image_size"]))

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # Class-balanced sampling (handles class imbalance)
    targets = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Model
    print(f"[MODEL] Creating {config['model_name']}...")
    model = create_model(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    no_improve_count = 0

    print(f"\n[TRAIN] Starting training for {config['epochs']} epochs...")
    print("=" * 70)

    total_start = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )

        val_result = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_result["loss"])
        history["val_acc"].append(val_result["accuracy"])

        is_best = val_result["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_result["accuracy"]
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save checkpoint
        if is_best or not config["save_best_only"]:
            ckpt_path = CHECKPOINT_DIR / f"best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_result["accuracy"],
                "config": config,
                "classes": CLASSES,
            }, ckpt_path)
            if is_best:
                print(f"  [SAVED] New best: {best_val_acc:.2f}% at epoch {epoch + 1}")

        print(
            f"Epoch {epoch + 1:3d}/{config['epochs']} | "
            f"Train: {train_acc:5.1f}% (loss {train_loss:.4f}) | "
            f"Val: {val_result['accuracy']:5.1f}% (loss {val_result['loss']:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
            + (" *" if is_best else "")
        )

        # Early stopping
        if no_improve_count >= config["early_stopping_patience"]:
            print(f"\n[STOP] No improvement for {config['early_stopping_patience']} epochs. Stopping.")
            break

    total_time = time.time() - total_start
    print("=" * 70)
    print(f"[DONE] Training completed in {total_time / 60:.1f} minutes")
    print(f"[BEST] Validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}")

    # Save training history
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(CHECKPOINT_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model, history


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ML Mirchi — Train Chilli Grading Model")
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["image_size"] = args.image_size
    config["num_workers"] = args.workers
    config["pretrained"] = not args.no_pretrained

    print("=" * 60)
    print("ML MIRCHI — Chilli Quality Grading Model Training")
    print("=" * 60)
    print(f"Model:         {config['model_name']}")
    print(f"Epochs:        {config['epochs']}")
    print(f"Batch Size:    {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Image Size:    {config['image_size']}x{config['image_size']}")
    print(f"Pretrained:    {config['pretrained']}")
    print(f"Classes:       {CLASSES}")
    print("=" * 60 + "\n")

    train(config)


if __name__ == "__main__":
    main()