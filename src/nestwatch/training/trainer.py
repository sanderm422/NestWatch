"""Training entry points for the NestWatch classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from nestwatch.config import TrainingConfig
from nestwatch.data.dataloaders import DataLoaders, build_dataloaders
from nestwatch.models.mobilenet import build_mobilenet


def _train_one_epoch(
    model: nn.Module,
    dataloaders: DataLoaders,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(
        dataloaders.train,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloaders.train), 1)


def _evaluate(model: nn.Module, dataloaders: DataLoaders, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloaders.val:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def train_model(config: TrainingConfig) -> Tuple[Path, Dict[str, float]]:
    """Train the MobileNetV2 classifier and persist the resulting weights."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = build_dataloaders(config)
    model = build_mobilenet(num_classes=config.num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    history: Dict[str, float] = {}

    for epoch in range(1, config.num_epochs + 1):
        train_loss = _train_one_epoch(
            model,
            dataloaders,
            optimizer,
            criterion,
            device,
            epoch,
            config.num_epochs,
        )
        accuracy = _evaluate(model, dataloaders, device)

        history[f"epoch_{epoch}_loss"] = train_loss
        history[f"epoch_{epoch}_accuracy"] = accuracy

        print(f"[TRAIN] Epoch {epoch}/{config.num_epochs} - loss: {train_loss:.4f}")
        print(f"[VAL]   Epoch {epoch}/{config.num_epochs} - accuracy: {accuracy:.2f}%")

    output_path = Path(config.output_dir) / "bird_classifier_state_dict.pth"
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Training complete. Weights saved to {output_path}")

    return output_path, history


__all__ = ["train_model"]
