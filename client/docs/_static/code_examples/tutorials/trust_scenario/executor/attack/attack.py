"""Define PGD attack on PyTorch model executor class."""
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from ML_management.executor.base_executor import BaseExecutor
from ML_management.executor.patterns import OneModelPattern
from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.s3 import S3Manager
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .pgd import ProjectedGradientDescent


class PGDExecutor(BaseExecutor):
    """PGD attack on PyTorch model executor from pattern with defined settings parameters."""

    def __init__(self):
        super().__init__(
            executor_models_pattern=OneModelPattern(
                desired_model_methods=[ModelMethodName.get_grad, ModelMethodName.predict_function, ModelMethodName.evaluate_function],
            )
        )

    def execute(
        self,
        bucket_name: str,
        batch_size: int = 64,
        eps: float = 4 / 255,
        eps_step: float = 1 / 255,
        max_iter: int = 100,
        shuffle: bool = False,
        num_workers: int = 0,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        num_images_to_save: int = 5,
        example_bucket_name: Optional[str] = None,
    ):
        """Define execute function that calls evaluate_function of model with corresponding params."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to_device(device)
        attack = ProjectedGradientDescent(get_grad=self.model.get_grad, eps=eps, eps_step=eps_step, max_iter=max_iter, mean=mean, std=std)

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        Path(bucket_name).mkdir(parents=True, exist_ok=True)

        if example_bucket_name is not None:
            Path(example_bucket_name).mkdir(parents=True, exist_ok=True)

        images_attacked = []
        example_counter = 0
        counter = 0

        for data in tqdm(dataloader):
            images, targets = data
            batch_attacked = images.numpy()
            batch_attacked = attack.generate(images.numpy(), targets)
            if example_counter < num_images_to_save:
                pred = self.model.predict_function(batch_attacked)

                for orig_img, label, crushed_img, pr in zip(images, targets, batch_attacked, pred):
                    npimg = orig_img.numpy()
                    original_file_name = Path(example_bucket_name) / f"original_image{example_counter}_label{label}.png"
                    self.save_image(npimg, original_file_name, mean, std)
                    crushed_file_name = Path(example_bucket_name) / f"crushed_image{example_counter}_label{pr}.png"
                    self.save_image(crushed_img, crushed_file_name, mean, std)
                    example_counter += 1
                    if example_counter >= num_images_to_save:
                        break

            if bucket_name:
                for attack_img, label in zip(batch_attacked, targets):
                    attack_file_name = Path(bucket_name) / f"attack_image{counter}_label{label}.png"
                    self.save_image(attack_img, attack_file_name, mean, std)
                    counter += 1

            images_attacked.append(batch_attacked)

        if example_bucket_name:
            S3Manager().upload(example_bucket_name, example_bucket_name)

        S3Manager().upload(bucket_name, bucket_name, upload_as_tar=True)

        images_attacked = np.vstack(images_attacked)
        self.model.dataset = TensorDataset(torch.Tensor(images_attacked), torch.Tensor(self.dataset.targets))

        # do not forget to pass model's parameters to run all model's methods correctly!
        self.model.evaluate_function(**self.model_methods_parameters[ModelMethodName.evaluate_function])
        return None

    def save_image(self, image, name, mean, std):
        """Save attacked image."""
        image = image * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)
        image = np.clip(image, a_min=0, a_max=1)
        im = Image.fromarray((np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8))
        im.save(name)
