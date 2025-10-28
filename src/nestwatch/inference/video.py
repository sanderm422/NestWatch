"""Offline video inference utilities."""

from __future__ import annotations

import sys

import cv2
import torch
from torchvision import transforms

from nestwatch.config import VideoTestConfig


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def run_video_test(config: VideoTestConfig) -> None:
    """Run inference on a recorded video clip."""

    transform = _build_transform()
    model = torch.jit.load(config.model_path)
    model.eval()
    print(f"[INFO] Model loaded from {config.model_path}.")

    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        print(f"[ERROR] Could not open video at {config.video_path}.")
        sys.exit(1)

    print(f"[INFO] Processing video {config.video_path}…")

    while True:
        success, frame = capture.read()
        if not success:
            break

        input_tensor = transform(frame).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, dim=0)
            label = f"{config.classes[int(predicted_idx)]} ({confidence:.2f})"

        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("NestWatch Video Test", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


__all__ = ["run_video_test"]
