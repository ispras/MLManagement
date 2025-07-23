from .names import attack_job_name
from ML_management.sdk import job_metric_by_name


attack_metrics = job_metric_by_name(attack_job_name)
print(attack_metrics)