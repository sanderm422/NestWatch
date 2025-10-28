import time
import torch
import os
import cv2
import numpy as np
from picamera2 import Picamera2
from torchvision import transforms
from datetime import datetime

# ====== CONFIG ======
MODEL_PATH = "bird_model.pt"
CLASSES = ['blåmes', 'gråsparv', 'koltrast', 'none', 'talgoxe']
CONFIDENCE_THRESHOLD = 0.4
DETECTION_DURATION = 2  # seconds
SNAPSHOT_INTERVAL = 1.0  # seconds between saved frames
NUM_SNAPSHOTS = 5
SAVE_DIR = "snapshots"

os.makedirs(SAVE_DIR, exist_ok=True)

# ====== Preprocessing ======
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Load Model ======
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
print("[INFO] Model loaded. Starting headless inference...")

# ====== Start Camera ======
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

last_detected_time = None
detection_confirmed = False

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Flip 180
        input_tensor = transform(frame).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, dim=0)
            predicted_class = CLASSES[predicted_idx]

        if predicted_class != "none" and confidence.item() >= CONFIDENCE_THRESHOLD:
            if last_detected_time is None:
                last_detected_time = time.time()
            elif time.time() - last_detected_time >= DETECTION_DURATION:
                detection_confirmed = True
        else:
            last_detected_time = None
            detection_confirmed = False

        if detection_confirmed:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[INFO] Confirmed bird detected: {predicted_class} ({confidence:.2f})")
            for i in range(NUM_SNAPSHOTS):
                frame = picam2.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_180)  # Flip 180
                filename = f"{SAVE_DIR}/{timestamp}_{predicted_class}_{i}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Saved snapshot: {filename}")
                time.sleep(SNAPSHOT_INTERVAL)
            # Reset after saving
            last_detected_time = None
            detection_confirmed = False

        time.sleep(1)

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")

finally:
    picam2.stop()
