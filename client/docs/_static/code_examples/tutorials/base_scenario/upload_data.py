from .names import dataset_loader_name
from ML_management.mlmanagement import log_dataset_loader_src


# dataset_loader_name = <YOUR_DATASET_LOADER_NAME>

log_dataset_loader_src(
    model_path="data",
    registered_name=dataset_loader_name,
    description="MNIST PyTorch Dataset",
)
