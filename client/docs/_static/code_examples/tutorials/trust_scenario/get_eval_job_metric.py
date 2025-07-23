from .names import eval_job_name
from ML_management.sdk import job_metric_by_name


eval_metrics = job_metric_by_name(eval_job_name)
print(eval_metrics)