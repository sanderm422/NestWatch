import cv2
import torch
import numpy as np
import requests
from torchvision import transforms

# URL to the Pi video stream
VIDEO_STREAM_URL = 'http://192.168.0.234:5000/video_feed'  # update if needed

# TorchScript model path
MODEL_PATH = 'bird_model.pt'

# Class names (must match your training labels order)
CLASSES = ['blåmes', 'gråsparv', 'koltrast', 'none', 'talgoxe']

# Preprocessing to match training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load TorchScript model
model = torch.jit.load(MODEL_PATH)
model.eval()

print("[INFO] Model loaded. Starting stream...")

# Read MJPEG stream from Pi
stream = requests.get(VIDEO_STREAM_URL, stream=True)
bytes_data = b''

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Preprocess frame for model
        input_tensor = transform(frame).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, dim=0)
            prediction = CLASSES[predicted_idx]

        # Display result on frame
        label = f"{prediction} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Bird Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
