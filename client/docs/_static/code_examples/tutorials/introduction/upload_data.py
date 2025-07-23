from ML_management.mlmanagement import set_mlm_credentials, set_server_url, log_dataset_loader_src

set_server_url("SERVER_URL")
set_mlm_credentials(login="login", password="password")

log_dataset_loader_src(
    model_path="data",
    registered_name="data_name",
    description="data description",
)
