from typing import Dict

import torchvision
import torchvision.transforms as transforms
from ML_management.dataset_loader.dataset_loader_pattern import DatasetLoaderPattern
from torch.utils.data import Dataset


class MNISTWrapper(DatasetLoaderPattern):
    """MNIST DatasetLoader class."""

    def get_dataset(self, mean: float = 0.7, std: float = 0.7) -> Dict[str, Dataset]:
        """Return MNIST torchvision.dataset."""
        data = {
            "train": torchvision.datasets.MNIST(
                root=self.data_path, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            ),
            "validation": torchvision.datasets.MNIST(
                root=self.data_path, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            ),
        }
        return data
