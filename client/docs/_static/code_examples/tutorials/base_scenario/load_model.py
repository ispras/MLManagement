from .names import model_name
from ML_management.mlmanagement import load_model

# model_name = <YOUR_MODEL_NAME>
# version = <YOUR_MODEL_VERSION>

loaded_object = load_model(
    name=model_name,
    version=version
)
