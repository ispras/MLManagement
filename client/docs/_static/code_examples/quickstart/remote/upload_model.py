from .names import model_name
from ML_management.mlmanagement import log_model_src


# model_name = <YOUR_MODEL_NAME>

log_model_src(
    model_path="model",
    registered_name=model_name,
    description="Sklearn KNN Iris classifier with default params",
)
