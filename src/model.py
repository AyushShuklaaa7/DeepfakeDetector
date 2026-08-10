import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2):
    """Create the EfficientNet-B0 architecture used during training."""

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        num_classes
    )

    return model


def load_model(model_path, device):
    """Load the trained deepfake detection checkpoint."""

    model = create_model(num_classes=2)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint
