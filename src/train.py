import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model import get_model
from dataset import get_dataloaders
from config import NUM_EPOCHS, LR

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load data
train_loader, val_loader, class_names = get_dataloaders()
print(f"[INFO] Classes: {class_names}")

# Load model
model = get_model()
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"[TRAIN] Epoch {epoch+1} - Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"[VAL] Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "bird_model.pth")
print("[INFO] Training complete. Model saved to bird_model.pth")
