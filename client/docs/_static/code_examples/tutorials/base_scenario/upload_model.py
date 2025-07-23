from .names import model_name
from ML_management.mlmanagement import log_model_src

# model_name = <YOUR_MODEL_NAME>

metainfo = log_model_src(
    model_path="torch_model",
    registered_name=model_name,
    description="1-channel ResNet18 for image classification.",
)
