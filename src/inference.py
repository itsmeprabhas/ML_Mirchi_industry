"""
ML Mirchi — Production Inference Wrapper
Clean API for loading ONNX model and running predictions.
"""

import time
import numpy as np
from PIL import Image
from pathlib import Path

import onnxruntime as ort

from config import CLASSES, NORMALIZATION, DEPLOYMENT


class MirchiGrader:
    """
    Production-ready inference class.
    Usage:
        grader = MirchiGrader("exports/mirchi_grader.onnx")
        result = grader.grade("chilli_photo.jpg")
        print(result)  # {'grade': 'Grade_A', 'confidence': 0.94, 'latency_ms': 12.3}
    """

    def __init__(self, model_path, confidence_threshold=None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.classes = CLASSES
        self.mean = np.array(NORMALIZATION["mean"], dtype=np.float32)
        self.std = np.array(NORMALIZATION["std"], dtype=np.float32)
        self.confidence_threshold = confidence_threshold or DEPLOYMENT["confidence_threshold"]

        # Warmup
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        _ = self.session.run(None, {self.input_name: dummy})

    def preprocess(self, image, image_size=224):
        """Convert PIL Image or numpy array to model input tensor."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Resize
        from PIL import Image as PILImage
        image = PILImage.fromarray(image.astype(np.uint8)).resize((image_size, image_size))
        image = np.array(image, dtype=np.float32)

        # Normalize: [0,255] -> [0,1] -> ImageNet normalization
        image = image / 255.0
        image = (image - self.mean) / self.std

        # HWC -> NCHW
        image = image.transpose(2, 0, 1)[np.newaxis]
        return image.astype(np.float32)

    def grade(self, image, return_probabilities=False):
        """
        Grade a single chilli image.

        Args:
            image: File path (str), PIL Image, or numpy array
            return_probabilities: If True, include all class probabilities

        Returns:
            dict with grade, confidence, latency_ms, flagged, probabilities
        """
        input_tensor = self.preprocess(image)

        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        latency = (time.perf_counter() - start) * 1000

        probs = self.softmax(logits[0])
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        result = {
            "grade": self.classes[top_idx],
            "confidence": round(confidence, 4),
            "latency_ms": round(latency, 2),
            "flagged": confidence < self.confidence_threshold,
            "threshold": self.confidence_threshold,
        }

        if return_probabilities:
            result["probabilities"] = {
                cls: round(float(probs[i]), 4) for i, cls in enumerate(self.classes)
            }

        return result

    def grade_batch(self, images, image_size=224):
        """Grade multiple images in a single batch."""
        batch = np.concatenate([self.preprocess(img, image_size) for img in images], axis=0)

        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: batch})[0]
        latency = (time.perf_counter() - start) * 1000

        results = []
        for i in range(len(images)):
            probs = self.softmax(logits[i])
            top_idx = int(np.argmax(probs))
            results.append({
                "grade": self.classes[top_idx],
                "confidence": round(float(probs[top_idx]), 4),
                "latency_ms": round(latency / len(images), 2),
                "flagged": float(probs[top_idx]) < self.confidence_threshold,
            })
        return results

    @staticmethod
    def softmax(x):
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
