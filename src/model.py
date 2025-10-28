import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_model(pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model
