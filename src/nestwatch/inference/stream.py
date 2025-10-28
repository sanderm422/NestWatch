"""Utilities for consuming the remote NestWatch video stream."""

from __future__ import annotations

import cv2
import numpy as np
import requests
import torch
from torchvision import transforms

from nestwatch.config import StreamConfig


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def consume_stream(config: StreamConfig) -> None:
    """Connect to the MJPEG stream and display predictions locally."""

    transform = _build_transform()
    model = torch.jit.load(config.model_path)
    model.eval()

    print(f"[INFO] Model loaded from {config.model_path}. Connecting to stream…")

    stream = requests.get(config.stream_url, stream=True)
    bytes_data = b""

    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        start = bytes_data.find(b"\xff\xd8")
        end = bytes_data.find(b"\xff\xd9")

        if start != -1 and end != -1:
            jpeg_bytes = bytes_data[start : end + 2]
            bytes_data = bytes_data[end + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            input_tensor = transform(frame).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probs, dim=0)
                prediction = config.classes[int(predicted_idx)]

            label = f"{prediction} ({confidence:.2f})"
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

            cv2.imshow("NestWatch Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


__all__ = ["consume_stream"]
