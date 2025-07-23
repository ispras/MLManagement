from pathlib import Path
from typing import Dict, List

import numpy as np
from ML_management import mlmanagement
from ML_management.model.patterns.trainable_model import TrainableModel
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from skops.io import dump, get_untrusted_types, load


class SklearnModelWrapper(TrainableModel):
    """Wrapper for simple sklearn model."""

    def __init__(
        self,
        n_neighbors: int = 5,
        obj_path: str = "model.skops",
    ) -> None:
        # to load existing model weights
        model_path = Path(self.artifacts) / obj_path

        if model_path.exists():
            # initialization of the prepared model
            unknown_types = get_untrusted_types(file=model_path)
            self.model = load(model_path, trusted=unknown_types)
        else:
            # base initialization
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
            )

    def train_function(self) -> None:
        """Implement required train_function interface."""
        iris = load_iris()
        self.model.fit(iris.data, iris.target)
        accuracy = self.evaluate_accuracy(iris)
        # log metric using mlmanagement's client
        mlmanagement.log_metric("Train Accuracy", accuracy)
        # save model artifacts for later use(initialization)
        model_path = Path(self.artifacts) / "model.skops"
        dump(self.model, model_path)

    def evaluate_accuracy(self, data) -> float:
        """Implement additional quality evaluation interface."""
        predictions = self.model.predict(data.data)
        return accuracy_score(data.target, predictions)

    def predict_function(self, input_batch: List[List[float]]) -> np.ndarray:
        """Implement required predict_function interface."""
        return self.model.predict(input_batch)
