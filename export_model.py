"""
ML Mirchi — Model Export for Production
Exports the trained PyTorch model to ONNX format for deployment.
ONNX Runtime provides consistent sub-30ms inference across platforms.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from train import create_model, CLASSES, CHECKPOINT_DIR

EXPORT_DIR = Path(__file__).parent / "exports"


def export_to_onnx(checkpoint_path, image_size=224):
    """Export model to ONNX format."""
    EXPORT_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")  # Always export on CPU for portability
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export
    onnx_path = EXPORT_DIR / "mirchi_grader.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"[ONNX] Exported to {onnx_path}")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Model verification passed")

    # Test with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    outputs = session.run(None, inputs)
    print(f"[ONNX] Runtime test passed — output shape: {outputs[0].shape}")

    # Measure ONNX Runtime latency
    import time
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        _ = session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies = np.array(latencies)
    print(f"[ONNX] CPU Latency: mean={latencies.mean():.2f}ms, P95={np.percentile(latencies, 95):.2f}ms")

    # Save metadata
    import json
    metadata = {
        "model_name": config["model_name"],
        "classes": CLASSES,
        "image_size": image_size,
        "onnx_latency_mean_ms": float(latencies.mean()),
        "onnx_latency_p95_ms": float(np.percentile(latencies, 95)),
        "input_name": "image",
        "output_name": "logits",
        "format": "NCHW — [batch, 3, height, width], float32",
        "normalization": "mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]",
    }
    with open(EXPORT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[ONNX] Metadata saved to {EXPORT_DIR / 'model_metadata.json'}")


def export_to_torchscript(checkpoint_path, image_size=224):
    """Export to TorchScript for PyTorch-based deployment."""
    EXPORT_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    scripted = torch.jit.trace(model, dummy_input)

    ts_path = EXPORT_DIR / "mirchi_grader.pt"
    scripted.save(str(ts_path))
    print(f"[TSC] TorchScript saved to {ts_path}")

    # Test
    loaded = torch.jit.load(str(ts_path))
    out = loaded(dummy_input)
    print(f"[TSC] Test passed — output shape: {out.shape}")


def main():
    parser = argparse.ArgumentParser(description="ML Mirchi — Export Model")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "both"])
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    print("=" * 60)
    print("ML MIRCHI — Model Export")
    print("=" * 60)

    if args.format in ["onnx", "both"]:
        export_to_onnx(args.checkpoint, args.image_size)
    if args.format in ["torchscript", "both"]:
        export_to_torchscript(args.checkpoint, args.image_size)

    print("\n[DONE] Export complete. Files in exports/")


if __name__ == "__main__":
    main()