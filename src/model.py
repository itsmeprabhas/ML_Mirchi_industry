"""
ML Mirchi — Model Architecture
Clean separation of model definition from training logic.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def create_efficientnet_b0(pretrained=True, dropout=0.3, num_classes=NUM_CLASSES):
    """EfficientNet-B0: 5.3M params, ~5ms GPU / ~15ms CPU inference."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_mobilenet_v3_small(pretrained=True, dropout=0.3, num_classes=NUM_CLASSES):
    """MobileNet-V3-Small: 2.5M params, ultra-lightweight for edge devices."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_resnet18(pretrained=True, dropout=0.3, num_classes=NUM_CLASSES):
    """ResNet-18: 11.7M params, solid baseline."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


MODEL_REGISTRY = {
    "efficientnet_b0": create_efficientnet_b0,
    "mobilenet_v3_small": create_mobilenet_v3_small,
    "resnet18": create_resnet18,
}


def create_model(name="efficientnet_b0", **kwargs):
    """Factory function to create models by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
