from pathlib import Path
from typing import Callable, Annotated

import numpy as np
import torch
from ML_management import mlmanagement
from ML_management.jsonschema_inference import SkipJsonSchema
from ML_management.model.patterns.evaluatable_model import EvaluatableModel
from ML_management.model.patterns.gradient_model import GradientModel

from .resnet18 import ResNet18


class Resnet18Wrapper(EvaluatableModel, GradientModel):
    """Wrapper of ResNet18 model."""

    def __init__(self, num_classes: int = 10, weights_path: str = "resnet18_cifar10.pth") -> None:
        # model definition
        self.model = ResNet18(num_classes)

        weights_path = Path(self.artifacts) / weights_path

        if weights_path.exists():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.load_state_dict(torch.load(weights_path, map_location=device))

    def evaluate_function(
        self,
        shuffle: bool,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """Implement evaluate_function interface."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        mlmanagement.log_metric("Accuracy", float(correct / total * 100.0))

    def predict_function(self, input_batch: Annotated[torch.Tensor, SkipJsonSchema]) -> torch.Tensor:
        """Implement required predict_function interface."""
        self.to_device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            input_batch = torch.as_tensor(input_batch, dtype=torch.float32)
            outputs = self.model(input_batch.to(self.device))
            predicted = torch.argmax(outputs.data, 1)
        return predicted

    def get_grad(self, loss_fn: Annotated[Callable, SkipJsonSchema], input_batch: Annotated[np.ndarray, SkipJsonSchema]) -> np.ndarray:
        """Implement get_grad interface."""
        self.model.eval()
        input_batch = torch.as_tensor(input_batch, dtype=torch.float32)
        input_batch.requires_grad_()
        outputs = self.model(input_batch.to(self.device)).cpu()
        loss = loss_fn(outputs)
        grad = torch.autograd.grad(loss, input_batch)[0]
        return np.asarray(grad)

    def to_device(self, device: str) -> None:
        """Implement to_device interface."""
        self.device = device
        self.model.to(device)
