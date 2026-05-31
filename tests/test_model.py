"""Tests for model creation and forward pass."""

import pytest
import torch
from src.model import create_model, count_parameters, MODEL_REGISTRY


class TestModelCreation:

    def test_efficientnet_b0_output_shape(self):
        model = create_model("efficientnet_b0")
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    def test_mobilenet_v3_output_shape(self):
        model = create_model("mobilenet_v3_small")
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4)

    def test_resnet18_output_shape(self):
        model = create_model("resnet18")
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4)

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_parameter_count_efficientnet(self):
        model = create_model("efficientnet_b0")
        params = count_parameters(model)
        assert params["total"] > 4_000_000
        assert params["trainable"] == params["total"]

    def test_parameter_count_mobilenet(self):
        model = create_model("mobilenet_v3_small")
        params = count_parameters(model)
        assert params["total"] < 3_000_000

    def test_all_models_in_registry(self):
        for name in MODEL_REGISTRY:
            model = create_model(name)
            assert model is not None
