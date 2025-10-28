"""Model factory functions used for bird classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mobilenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Return a MobileNetV2 classifier tailored to the requested number of classes."""

    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


__all__ = ["build_mobilenet"]
