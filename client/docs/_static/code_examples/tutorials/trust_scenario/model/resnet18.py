import torch
from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    """Simple CNN model: ResNet18 + Linear + Linear."""

    def __init__(self, n_classes: int) -> None:
        """Model initialization."""
        super().__init__()

        self.conv_block = nn.Sequential(models.resnet18(weights=models.ResNet18_Weights.DEFAULT), nn.ReLU())

        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, n_classes),
        )

    def forward(self, x) -> torch.Tensor:
        """Feed-forward method."""
        x = self.conv_block(x)
        x = x.view((x.shape[0], -1))
        x = self.fc_layers(x)

        return x
