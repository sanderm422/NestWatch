"""Edge inference loop executed on the Raspberry Pi."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from picamera2 import Picamera2

from nestwatch.config import DeviceConfig


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


def run_device_inference(config: DeviceConfig) -> None:
    """Run the real-time recognition loop on the Raspberry Pi camera."""

    from picamera2 import Picamera2  # type: ignore import-not-found

    transform = _build_transform()
    model = torch.jit.load(config.model_path, map_location="cpu")
    model.eval()

    print("[INFO] Model loaded. Starting headless inference…")

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = config.capture_resolution
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(2)

    last_detected_time: float | None = None
    detection_confirmed = False

    try:
        while True:
            frame = picam2.capture_array()
            if config.rotation_degrees == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif config.rotation_degrees == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif config.rotation_degrees == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            input_tensor = transform(frame).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probs, dim=0)
                predicted_class = config.classes[int(predicted_idx)]

            if predicted_class != "none" and confidence.item() >= config.confidence_threshold:
                if last_detected_time is None:
                    last_detected_time = time.time()
                elif time.time() - last_detected_time >= config.detection_duration:
                    detection_confirmed = True
            else:
                last_detected_time = None
                detection_confirmed = False

            if detection_confirmed:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(
                    f"[INFO] Confirmed bird detected: {predicted_class} ({confidence:.2f})"
                )
                for index in range(config.num_snapshots):
                    frame = picam2.capture_array()
                    if config.rotation_degrees == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif config.rotation_degrees == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif config.rotation_degrees == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    filename = config.snapshot_dir / f"{timestamp}_{predicted_class}_{index}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"[INFO] Saved snapshot: {filename}")
                    time.sleep(config.snapshot_interval)

                last_detected_time = None
                detection_confirmed = False

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    finally:
        picam2.stop()


__all__ = ["run_device_inference"]
