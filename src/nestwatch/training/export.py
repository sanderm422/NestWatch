"""Utility helpers to export trained PyTorch models for deployment."""

from __future__ import annotations

from pathlib import Path

import torch

from nestwatch.config import TrainingConfig
from nestwatch.models.mobilenet import build_mobilenet


def export_torchscript(
    state_dict_path: Path,
    output_path: Path,
    config: TrainingConfig,
) -> Path:
    """Export a trained MobileNetV2 model to TorchScript."""

    model = build_mobilenet(num_classes=config.num_classes, pretrained=False)
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummy_input)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(output_path)
    print(f"[INFO] TorchScript model saved to {output_path}")
    return output_path


__all__ = ["export_torchscript"]
