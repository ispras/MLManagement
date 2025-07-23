from .names import job_name
from ML_management.mlmanagement import download_job_artifacts

artifacts_path = download_job_artifacts(job_name)
