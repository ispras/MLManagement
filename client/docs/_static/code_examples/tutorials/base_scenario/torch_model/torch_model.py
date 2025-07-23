from pathlib import Path
from typing import Dict, List

import torch
from ML_management import mlmanagement
from ML_management.model.patterns.trainable_model import TrainableModel
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .resnet18 import BWResNet18


class BWResNet18Wrapper(TrainableModel):
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        hid_lay_size: int = 100,
        dropout: float = 0.1,
        weights_path: str = "best_model.pth",
    ) -> None:
        # model definition
        self.model = BWResNet18(num_classes, pretrained, hid_lay_size, dropout)
        weights_path = Path(self.artifacts) / weights_path
        if weights_path.exists():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.load_state_dict(torch.load(weights_path, map_location=device))

    def train_function(
        self,
        lr: float = 0.001,
        momentum: float = 0.9,
        num_epochs: int = 5,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> Dict[str, str]:
        """Implement train_function interface."""
        # classic torch train loop
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.train()

        # we will use prepared torchvision.datasets.MNIST divided into 2 parts
        train_loader = DataLoader(self.dataset["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        test_loader = DataLoader(self.dataset["validation"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        scheduler = CosineAnnealingLR(optimizer, T_max=int(len(self.dataset) / batch_size + 1) * num_epochs)

        for epoch_num in range(num_epochs):

            running_loss = 0.0
            correct = 0.0
            total = 0.0

            for inputs, labels in train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total += inputs.data.size(0)
                predicted = torch.argmax(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                lr = scheduler.optimizer.param_groups[0]["lr"]
                scheduler.step()

            mlmanagement.log_metric("lr", lr, step=epoch_num)
            mlmanagement.log_metric("Train Accuracy", float(correct / total * 100.0), step=epoch_num)
            mlmanagement.log_metric("Train Loss", float(running_loss / total), step=epoch_num)
            self.validation(test_loader, device)

        weights_path = Path(self.artifacts) / "best_model.pth"
        torch.save(self.model.state_dict(), weights_path)

    def validation(self, validation_loader: DataLoader, device: str) -> None:
        """Evaluate the quality of the model on the validation part of the data."""
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            mlmanagement.log_metric("Validation Accuracy", float(correct / total * 100.0))

    def predict_function(self, input_batch: List[List[List[float]]]) -> torch.Tensor:
        """Implement required predict_function interface."""
        self.to_device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_batch.to(self.device))
            predicted = torch.argmax(outputs.data, 1)
        return predicted

    def to_device(self, device: str) -> None:
        """Implement to_device interface."""
        self.device = device
        self.model.to(device)
