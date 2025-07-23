import os

simple_job_name = os.environ.get("DEBUG_JOB_SIMPLE", "debug-scenario-job-docs-simple")
base_job_name = os.environ.get("DEBUG_JOB_BASE", "debug-scenario-job-docs-base")
model_path = os.path.abspath(os.environ.get("DEBUG_MODEL_PATH", "../base_scenario/torch_model"))
new_model_name = os.environ.get("DEBUG_NEW_MODEL_NAME", "new_model_name")
bucket_name = os.environ.get("DEBUG_BUCKET_NAME", "mnist")
dataset_loader_path = os.path.abspath(os.environ.get("DEBUG_DATASET_LOADER_PATH", "../base_scenario/data"))
executor_path = os.path.abspath(
    os.environ.get("DEBUG_EXECUTOR_PATH", "../base_scenario/data"))
local_path = os.environ.get("DEBUG_LOCAL_PATH", "./mnist")


model_path_adv = os.path.abspath(
    os.environ.get("DEBUG_MODEL_PATH_ADV", "../../../../../../../../tests/coin_model/models/coin_model_no_torch"))
dataset_loader_path_adv = os.path.abspath(os.environ.get("DEBUG_DATASET_LOADER_PATH_ADV",
                                                         "../../../../../../../../tests/coin_model/test/coin_model/datasets/FashionMNIST"))
