import os


job_name = os.environ.get("BASE_JOB_NAME", "base-scenario-job-docs-test")
job_id = int(os.environ.get("BASE_JOB_ID", 1))
model_name = os.environ.get("BASE_MODEL_NAME", "BlackWhiteResNet18")
new_model_name = os.environ.get("BASE_NEW_MODEL_NAME", "BlackWhiteResNet18")
bucket_name = os.environ.get("BASE_BUCKET_NAME", "mnist")
dataset_loader_name = os.environ.get("DATASET_LOADER_NAME", "MNIST")