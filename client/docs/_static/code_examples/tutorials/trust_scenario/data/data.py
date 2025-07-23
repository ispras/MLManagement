from typing import List

import torchvision
import torchvision.transforms as transforms
from ML_management.dataset_loader.dataset_loader_pattern import DatasetLoaderPattern


class DataWrapper(DatasetLoaderPattern):
    """Cifar10 DatasetLoader class."""

    def get_dataset(self, train_part: bool = True, mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5]):
        """Return mix of validation part of Cifar10 and train part of MNIST."""
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data = torchvision.datasets.CIFAR10(root=self.data_path, train=train_part, transform=transformation)
        return data
