import os


job_name = os.environ.get("REMOTE_JOB_NAME", "docs_test")
model_name = os.environ.get("REMOTE_MODEL_NAME", "KNN_Iris_Classifier")
new_model_name = os.environ.get("REMOTE_NEW_MODEL_NAME", "KNN_Iris_Classifier")
