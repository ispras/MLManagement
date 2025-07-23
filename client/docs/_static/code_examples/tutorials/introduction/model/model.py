from pathlib import Path
from typing import Dict, List

from ML_management import mlmanagement
from ML_management.model.patterns.trainable_model import TrainableModel


class MyModelWrapper(TrainableModel):
   # to use the platform, you must specify parameter types for all required methods
   def __init__(self, param1: int, param2: str, ..., paramN: bool, weight_path: str) -> None:
        # here we define a model for two scenarios: primary initialization and initialization from existing weights/object
        
        # to load existing model weights
        model_path = Path(self.artifacts) / weights_path

        if model_path.exists():
            # initialization of the prepared model
            self.model = load_model(model_path, ...).loaded_object  # it may be skops.io.load/load_state_dict(torch.load(...))/etc.
        else:
            self.model = ...

   def train_function(self, num_epochs: int, ...) -> Dict[str, str]:
        ...
        for epoch in num_epochs:
            for input_data, output_data in self.dataset:
                ...
                ... = self.model(input_data)
                ...

            ...
            mlmanagement.log_metric("Accuracy", accuracy)
        
        weight_path = Path(self.artifacts) / "model_weight.*"
        
        save_model(weight_path)  # it may be skops.io.dump/torch.save(self.model.state_dict(), weighs_path)/etc.

   def predict_function(self, input_batch: List[List[List[float]]]):
       ...
