import cv2
import torch
import numpy as np
from torchvision import transforms

# ====== CONFIG ======
VIDEO_PATH = "data/test_videos/garden_birds.mp4"  # Your local video file
MODEL_PATH = "bird_model.pt"                         # TorchScript model
CLASSES = ['blåmes', 'gråsparv', 'koltrast', 'none', 'talgoxe']  # In training order

# ====== Preprocessing ======
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Load model ======
model = torch.jit.load(MODEL_PATH)
model.eval()
print("[INFO] Model loaded.")

# ====== Load video ======
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("[ERROR] Could not open video.")
    exit()

print("[INFO] Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, dim=0)
        label = f"{CLASSES[predicted_idx]} ({confidence:.2f})"

    # ====== Display frame with overlay ======
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Model test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
