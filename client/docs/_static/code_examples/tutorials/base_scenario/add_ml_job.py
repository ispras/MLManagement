from .names import job_name, model_name, new_model_name, bucket_name, dataset_loader_name
from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.sdk import DatasetLoaderMethodParams, ModelMethodParams, ModelVersionChoice, DatasetLoaderForm, ModelForm, ResourcesForm, add_ml_job


# job_name = <YOUR_JOB_NAME>
# new_model_name = <YOUR_NEW_MODEL_NAME>

train_job = add_ml_job(
    job_executor_name="train",
    executor_params={},
    models_pattern=ModelForm(
        model_version_choice=ModelVersionChoice(name=model_name),
        params=[
            ModelMethodParams(method=ModelMethodName.train_function, params={}),
            ModelMethodParams(method=ModelMethodName.init, params={"num_classes": 10}),
        ],
        new_model_name=new_model_name,
        new_model_description="ResNet18 PyTorch trainable model for image classification trained on MNIST",
    ),
    data_pattern=DatasetLoaderForm(
        name=dataset_loader_name, collector_name="s3", collector_params={"bucket": bucket_name}, params=[DatasetLoaderMethodParams(params={})]
    ),
    job_name=job_name,
    resources=ResourcesForm(gpu_number=1)
)
