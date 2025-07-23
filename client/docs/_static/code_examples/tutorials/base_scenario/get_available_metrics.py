from .names import job_name
from ML_management.sdk import available_metrics


av_metrics = available_metrics(job_name)
print(av_metrics)