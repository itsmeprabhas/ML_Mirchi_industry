
"""
ML Mirchi — Centralized Configuration
Single source of truth for all hyperparameters and paths.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
EXPORT_DIR = BASE_DIR / "exports"
RESULTS_DIR = BASE_DIR / "results"

# ============================================================
# CLASSES
# ============================================================
CLASSES = ["Grade_A", "Grade_B", "Grade_C", "Reject"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

CLASS_DESCRIPTIONS = {
    "Grade_A": "Premium: Deep red, uniform color, no defects, proper shape",
    "Grade_B": "Standard: Minor color variation, slight imperfections",
    "Grade_C": "Sub-standard: Visible discoloration, wrinkles, green spots",
    "Reject": "Rejected: Severe rot, mold, deep cracks, pest damage",
}

# ============================================================
# MODEL
# ============================================================
MODEL_CONFIG = {
    "name": "efficientnet_b0",
    "pretrained": True,
    "dropout_rate": 0.3,
    "num_classes": NUM_CLASSES,
}

# ============================================================
# TRAINING
# ============================================================
TRAIN_CONFIG = {
    "image_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "label_smoothing": 0.1,
    "early_stopping_patience": 7,
    "num_workers": 4,
    "seed": 42,
}

# ============================================================
# DATA SPLITS
# ============================================================
DATA_SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# ============================================================
# NORMALIZATION (ImageNet stats for pretrained models)
# ============================================================
NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# ============================================================
# DEPLOYMENT
# ============================================================
DEPLOYMENT = {
    "confidence_threshold": 0.70,
    "max_latency_ms": 30.0,
    "input_format": "NCHW",  # [batch, channels, height, width]
    "input_dtype": "float32",
}

# ============================================================
# BUSINESS TARGETS (from CRISP-DM spec)
# ============================================================
BUSINESS_TARGETS = {
    "post_harvest_loss_reduction_pct": 15.0,
    "customer_satisfaction_increase_pct": 10.0,
    "cogs_reduction_pct": 8.0,
    "deployment_timeline_months": 6,
    "min_accuracy_pct": 90.0,
    "max_latency_ms": 30.0,
    "min_precision": 0.90,
    "min_recall": 0.90,
}
