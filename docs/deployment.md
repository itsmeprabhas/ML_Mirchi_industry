**ML Mirchi — Deployment Guide**

**ONNX Runtime Deployment**

**Requirements**

Python 3.9+
onnxruntime >= 1.15.0

**Quick Start**

from src.inference import MirchiGradergrader = MirchiGrader("exports/mirchi_grader.onnx", confidence_threshold=0.70)result = grader.grade("chilli_photo.jpg")print(result)# {'grade': 'Grade_A', 'confidence': 0.9412, 'latency_ms': 12.3, 'flagged': False}
 
** Monitoring**

**Concept Drift Detection**

 Track grade distribution over time
 Alert if distribution shifts > 10% from baseline
 Trigger monthly retraining
 
 **Latency Monitoring**

 P95 latency must stay < 30ms
 Alert if P95 > 25ms (early warning)
 Alert if P95 > 30ms (critical)
