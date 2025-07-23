from .names import bucket_name
from ML_management.s3 import S3Manager


# bucket_name = <YOUR_BUCKET_NAME>

S3Manager().upload("./mnist", bucket_name)
