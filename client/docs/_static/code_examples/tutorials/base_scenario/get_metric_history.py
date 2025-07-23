from .names import job_name
from ML_management.sdk import metric_history


history = metric_history(job_name, "Train Accuracy", make_graph=True)
print(history)