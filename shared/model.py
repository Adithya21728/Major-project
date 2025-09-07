import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification (Pneumonia vs Normal)
    return model


