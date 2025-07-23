from .names import job_name
from ML_management.sdk import job_metric_by_name


metric = job_metric_by_name(job_name)
print(metric)