ML Mirchi — System Architecture

Overview

 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ Camera │────>│ Preprocessing│────>│ Model │────>│ Decision │ │ (Factory) │ │ Pipeline │ │ (EfficientNet│ │ Engine │ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │ Resize 224x224 ┌────────┼────────┐ Normalize │ │ │ NCHW format ↓ ↓ ↓ Confidence Auto Flag Auto > 0.70 Grade Review Reject
 
## Model Selection Rationale

| Model | Params | GPU Latency | CPU Latency | Accuracy |
|-------|--------|-------------|-------------|----------|
| EfficientNet-B0 | 5.3M | ~5ms | ~15ms | High |
| MobileNet-V3-Small | 2.5M | ~3ms | ~10ms | Medium |
| ResNet-18 | 11.7M | ~8ms | ~25ms | High |

**Selected: EfficientNet-B0** — best accuracy-to-latency ratio.

## Human-in-the-Loop Design

Per the CRISP-DM spec:
- 95% of images auto-graded (confidence > threshold)
- 5% flagged for human review (confidence < threshold)
- Flagged samples collected for retraining
