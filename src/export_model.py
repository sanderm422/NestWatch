import torch
from model import get_model
from config import NUM_CLASSES

# Load model structure
model = get_model(pretrained=False)
model.load_state_dict(torch.load("bird_model.pth", map_location="cpu"))
model.eval()

# Example dummy input (batch size 1, 3 channels, 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# Convert to TorchScript
traced_model = torch.jit.trace(model, dummy_input)

# Save as .pt file
traced_model.save("bird_model.pt")
print("[INFO] TorchScript model saved as bird_model.pt")
