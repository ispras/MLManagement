from functools import partial
from typing import Callable, List

import numpy as np
from torch import nn


class ProjectedGradientDescent:
    """Projected Gradient Descent attack on images."""

    def __init__(
        self,
        get_grad: Callable,
        eps: float = 4.0 / 255,
        eps_step: float = 1.0 / 255,
        max_iter: int = 10,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
    ) -> None:
        self.get_grad = get_grad
        self.eps = eps
        self.alpha = eps_step
        self.max_iter = max_iter
        self.mean = np.expand_dims(np.asarray(mean), axis=[0, 2, 3])
        self.std = np.expand_dims(np.asarray(std), axis=[0, 2, 3])

        self.noise = None
        self.loss_computer = nn.CrossEntropyLoss()

    def normalize(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize inputs."""
        return (inputs - self.mean) / self.std

    def unnormalize(self, inputs: np.ndarray) -> np.ndarray:
        """Unnormalize inputs."""
        return inputs * self.std + self.mean

    def generate(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Generate PGD attack on inputs."""
        inputs = self.unnormalize(inputs)
        noise = np.random.uniform(low=-self.eps, high=self.eps, size=inputs.shape)

        for _ in range(self.max_iter):
            noisy_inputs = inputs + noise
            noisy_inputs = np.clip(noisy_inputs, a_min=0.0, a_max=1.0)
            loss_fn = partial(self.loss_computer, target=targets)
            noisy_inputs = self.normalize(noisy_inputs)
            grad = self.get_grad(loss_fn, noisy_inputs)

            noise += self.alpha * np.sign(grad)
            noise = np.clip(noise, a_min=-self.eps, a_max=self.eps)

        attacked = inputs + noise
        attacked = np.clip(attacked, a_min=0.0, a_max=1.0)
        return self.normalize(attacked)
