from .names import attacked_examples
from ML_management.s3 import S3Manager


S3Manager().set_data(local_path='examples', bucket=attacked_examples)