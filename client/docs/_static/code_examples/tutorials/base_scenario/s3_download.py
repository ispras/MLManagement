from .names import bucket_name
from ML_management.s3 import S3Manager


# bucket_name = <YOUR_BUCKET_NAME>

S3Manager().set_data(local_path="local_folder", bucket=bucket_name, sync=True)
