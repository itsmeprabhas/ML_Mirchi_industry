"""Tests for inference pipeline."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

from src.inference import MirchiGrader
from config import CLASSES


@pytest.fixture
def temp_onnx_model(tmp_path):
    """Create a minimal ONNX model for testing."""
    import torch
    from src.model import create_model

    model = create_model("mobilenet_v3_small", pretrained=False)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = tmp_path / "test_model.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=13,
                      input_names=["image"], output_names=["logits"])
    return str(onnx_path)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return Image.fromarray(np.random.randint(100, 255, (300, 300, 3), dtype=np.uint8))


class TestInference:

    def test_load_model(self, temp_onnx_model):
        grader = MirchiGrader(temp_onnx_model)
        assert grader is not None

    def test_grade_returns_dict(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model)
        result = grader.grade(sample_image)
        assert isinstance(result, dict)
        assert "grade" in result
        assert "confidence" in result
        assert "latency_ms" in result
        assert "flagged" in result

    def test_grade_valid_class(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model)
        result = grader.grade(sample_image)
        assert result["grade"] in CLASSES

    def test_confidence_range(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model)
        result = grader.grade(sample_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_latency_positive(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model)
        result = grader.grade(sample_image)
        assert result["latency_ms"] > 0

    def test_flagged_low_confidence(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model, confidence_threshold=0.999)
        result = grader.grade(sample_image)
        assert result["flagged"] is True

    def test_not_flagged_high_confidence(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model, confidence_threshold=0.0)
        result = grader.grade(sample_image)
        assert result["flagged"] is False

    def test_return_probabilities(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model)
        result = grader.grade(sample_image, return_probabilities=True)
        assert "probabilities" in result
        assert len(result["probabilities"]) == len(CLASSES)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=0.01)

    def test_grade_from_file_path(self, temp_onnx_model, sample_image, tmp_path):
        img_path = tmp_path / "test.jpg"
        sample_image.save(str(img_path))
        grader = MirchiGrader(temp_onnx_model)
        result = grader.grade(str(img_path))
        assert "grade" in result

    def test_batch_grading(self, temp_onnx_model, sample_image):
        grader = MirchiGrader(temp_onnx_model)
        results = grader.grade_batch([sample_image, sample_image])
        assert len(results) == 2
        for r in results:
            assert "grade" in r

    def test_model_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            MirchiGrader("/nonexistent/model.onnx")
