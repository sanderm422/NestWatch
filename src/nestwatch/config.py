"""Configuration dataclasses used across the NestWatch project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from nestwatch.utils import ensure_directory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "models" / "bird_classifier.pt"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    dataset_path: Path = DEFAULT_DATA_DIR / "birds"
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    validation_split: float = 0.2
    num_workers: int = 2
    num_classes: int = 5
    output_dir: Path = DEFAULT_ARTIFACT_DIR / "models"

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        ensure_directory(self.output_dir)


@dataclass
class DeviceConfig:
    """Configuration for inference on the Raspberry Pi device."""

    model_path: Path = DEFAULT_MODEL_PATH
    classes: List[str] = field(
        default_factory=lambda: [
            "blåmes",
            "gråsparv",
            "koltrast",
            "none",
            "talgoxe",
        ]
    )
    confidence_threshold: float = 0.4
    detection_duration: float = 2.0
    snapshot_interval: float = 1.0
    num_snapshots: int = 5
    snapshot_dir: Path = DEFAULT_ARTIFACT_DIR / "snapshots"
    capture_resolution: Tuple[int, int] = (640, 480)
    rotation_degrees: int = 180

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.snapshot_dir = Path(self.snapshot_dir)
        ensure_directory(self.snapshot_dir)


@dataclass
class StreamConfig:
    """Configuration for consuming the MJPEG stream published by the Pi."""

    stream_url: str = "http://raspberrypi.local:5000/video_feed"
    model_path: Path = DEFAULT_MODEL_PATH
    classes: List[str] = field(
        default_factory=lambda: [
            "blåmes",
            "gråsparv",
            "koltrast",
            "none",
            "talgoxe",
        ]
    )

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)


@dataclass
class VideoTestConfig:
    """Configuration for running inference on a stored video file."""

    video_path: Path = PROJECT_ROOT / "data" / "test_videos" / "garden_birds.mp4"
    model_path: Path = DEFAULT_MODEL_PATH
    classes: List[str] = field(
        default_factory=lambda: [
            "blåmes",
            "gråsparv",
            "koltrast",
            "none",
            "talgoxe",
        ]
    )

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.model_path = Path(self.model_path)


__all__ = [
    "TrainingConfig",
    "DeviceConfig",
    "StreamConfig",
    "VideoTestConfig",
    "PROJECT_ROOT",
    "DEFAULT_DATA_DIR",
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_MODEL_PATH",
]
