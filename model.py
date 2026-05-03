# model.py
# ResNet-18 based classifier for four vehicle classes.

import torch
import torch.nn as nn
from torchvision import models

import config


class VehicleClassifier(nn.Module):
    """Vehicle classifier using a ResNet-18 backbone."""

    def __init__(self, num_classes=config.num_classes, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


if __name__ == "__main__":
    model = VehicleClassifier(pretrained=False)
    dummy = torch.randn(2, 3, config.resize_y, config.resize_x)
    output = model(dummy)
    print("Input shape: - model.py:40", dummy.shape)
    print("Output shape: - model.py:41", output.shape)
