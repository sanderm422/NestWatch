"""Utilities for constructing dataloaders used during training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from nestwatch.config import TrainingConfig


@dataclass
class DataLoaders:
    """Container for train/validation dataloaders and class metadata."""

    train: DataLoader
    val: DataLoader
    classes: Iterable[str]


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return the train and validation transforms used for MobileNetV2."""

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transforms, val_transforms


def build_dataloaders(config: TrainingConfig) -> DataLoaders:
    """Create dataloaders according to the provided configuration."""

    train_transforms, val_transforms = build_transforms()

    dataset: Dataset = datasets.ImageFolder(config.dataset_path, transform=train_transforms)
    val_size = int(config.validation_split * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataLoaders(train=train_loader, val=val_loader, classes=dataset.classes)


__all__ = ["DataLoaders", "build_dataloaders", "build_transforms"]
