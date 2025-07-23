import torch
from torch import nn
from torch.nn import Conv2d
from torchvision import models
from torchvision.models import ResNet18_Weights


class BWResNet18(nn.Module):
    """Simple 1-channel CNN model: ResNet18 + Linear + Linear."""

    def __init__(self, n_classes: int, pretrained: bool, hid_lay_size: int, dropout: float) -> None:
        """Model initialization."""
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv_block = nn.Sequential(backbone, torch.nn.ReLU())
        self.fc_layers = nn.Sequential(
            torch.nn.Linear(1000, hid_lay_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hid_lay_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward method."""
        x = self.conv_block(x)
        x = x.view((x.shape[0], -1))
        x = self.fc_layers(x)

        return x
