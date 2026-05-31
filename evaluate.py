"""
Evaluate trained model and generate metrics/plots.
"""

import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from train import (
    ChilliDataset, get_val_augmentations, create_model,
    CLASSES, NUM_CLASSES, CLASS_TO_IDX, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR,
)

# Style
plt.style.use("dark_background")
sns.set_palette("husl")

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[LOADED] Model from epoch {checkpoint['epoch'] + 1}, "
          f"val accuracy: {checkpoint['val_accuracy']:.2f}%")
    return model, config

def measure_latency(model, device, num_runs=200, image_size=224):
    """
    Measure inference latency — critical metric for real-time grading.
    Target: < 30ms per image for conveyor belt throughput.
    """
    print(f"\n[LATENCY] Measuring over {num_runs} inferences...")

    # Warmup
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    for _ in range(20):
        _ = model(dummy)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  Min:    {latencies.min():.2f} ms")
    print(f"  Max:    {latencies.max():.2f} ms")

    # Throughput calculation
    throughput = 1000.0 / latencies.mean()  # images per second
    print(f"  Throughput: {throughput:.0f} images/sec ({throughput * 3600:.0f}/hr)")

    return latencies

def evaluate_model(model, loader, device):
    """Run full evaluation on test set."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_latencies = []

    print(f"Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_latencies.extend([elapsed / images.size(0)] * images.size(0))

    return {
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
        "latencies": np.array(all_latencies),
    }

def print_metrics(results):
    """Print all metrics from the project spec."""
    y_true = results["labels"]
    y_pred = results["predictions"]
    y_prob = results["probabilities"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — ML Mirchi Output Metrics (per project spec)")
    

    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  (i) Accuracy:      {acc * 100:.2f}%")
    print(f"      Target:        > 90% for production")

    # 2. Latency
    lats = results["latencies"]
    print(f"\n  (ii) Latency:")
    print(f"       Mean:         {lats.mean():.2f} ms")
    print(f"       P95:          {np.percentile(lats, 95):.2f} ms")
    print(f"       Target:       < 30ms for real-time conveyor belt")

    # 3. Precision and Recall (per class and macro)
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    print(f"\n  (iii) Precision & Recall:")
    print(f"       {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"       {'-' * 52}")
    for i, cls in enumerate(CLASSES):
        support = np.sum(y_true == i)
        print(f"       {cls:<12} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f} {support:>10}")

    macro_prec = precision_score(y_true, y_pred, average="macro")
    macro_rec = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"       {'-' * 52}")
    print(f"       {'Macro Avg':<12} {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f1:>10.4f}")

    # Additional metrics
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        print(f"\n  ROC-AUC (OvR):   {auc:.4f}")
    except ValueError:
        print("\n  ROC-AUC:         Cannot compute (need all classes in test set)")

    print(f"\n  Avg Inference:  {lats.mean():.2f} ms/image")
    print(f"  Throughput:     {1000 / lats.mean():.0f} images/sec")
    

    # Full classification report
    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    return {
        "accuracy": acc,
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1.tolist(),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "mean_latency_ms": float(lats.mean()),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        "throughput_per_sec": float(1000 / lats.mean()),
    }

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax1,
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap="Reds", ax=ax2,
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax2.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion matrix -> {save_path}")

def plot_training_history(history_path, save_path):
    """Plot training curves from saved history."""
    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", color="#e63926")
    ax1.plot(history["val_loss"], label="Val Loss", color="#f4a623")
    ax1.set_title("Loss Over Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.plot(history["train_acc"], label="Train Accuracy", color="#e63926")
    ax2.plot(history["val_acc"], label="Val Accuracy", color="#f4a623")
    ax2.set_title("Accuracy Over Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Training curves -> {save_path}")

def plot_latency_distribution(latencies, save_path):
    """Plot latency distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(latencies, bins=40, color="#e63926", alpha=0.7, edgecolor="#211a14")
    ax.axvline(latencies.mean(), color="#f4a623", linestyle="--", linewidth=2,
               label=f"Mean: {latencies.mean():.2f}ms")
    ax.axvline(np.percentile(latencies, 95), color="#22c55e", linestyle="--", linewidth=2,
               label=f"P95: {np.percentile(latencies, 95):.2f}ms")
    ax.axvline(30, color="#ef4444", linestyle=":", linewidth=2,
               label="Target: 30ms")
    ax.set_title("Inference Latency Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Latency distribution -> {save_path}")

def main():
    parser = argparse.ArgumentParser(description="ML Mirchi — Evaluate Model")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best_model.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Measure standalone latency
    latencies = measure_latency(model, device, image_size=config["image_size"])
    plot_latency_distribution(latencies, RESULTS_DIR / "latency_distribution.png")

    # Test set evaluation
    test_dataset = ChilliDataset(DATA_DIR, "test", get_val_augmentations(config["image_size"]))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_dataset)} images")

    results = evaluate_model(model, test_loader, device)

    # Print all metrics
    metrics = print_metrics(results)

    # Generate plots
    plot_confusion_matrix(results["labels"], results["predictions"],
                          RESULTS_DIR / "confusion_matrix.png")

    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        plot_training_history(history_path, RESULTS_DIR / "training_curves.png")

    # Save metrics to JSON
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved Metrics -> {RESULTS_DIR / 'evaluation_metrics.json'}")

    print("\n[DONE] All evaluation results saved to results/")

if __name__ == "__main__":
    main()