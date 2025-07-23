from .names import job_name
from ML_management.mlmanagement import start_job, log_metric, log_artifact
import numpy as np

# YOUR MODEL INITIALIZATION

#############################
### ╰( ° ʖ ° )つ──────☆*:・ﾟ
#############################

# metric_value = compute_accuracy(model(data), target) # metric_value = 0.88
# preds = model(data) # preds = [1, 0, 0, 1, 0, 1]

# job_name = <YOUR_JOB_NAME>

np.savetxt('artifacts.txt', [1, 0, 0, 1, 0, 1], fmt='%d')

with start_job(job_name):
    log_metric("accuracy", 0.88)
    log_artifact("artifacts.txt")
