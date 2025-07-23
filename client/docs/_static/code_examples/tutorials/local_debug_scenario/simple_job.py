from .names import simple_job_name as job_name, new_model_name, model_path, bucket_name
from ML_management.sdk import DatasetLoaderMethodParams, ModelVersionChoice, ModelMethodParams, DatasetLoaderForm, ModelForm, add_ml_job_local
from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.mlmanagement import set_debug

# job_name = <YOUR_JOB_NAME>
# new_model_name = <YOUR_NEW_MODEL_NAME>
# model_path = 'path/to/model/dir'
# bucket_name = 'path/to/model/dir'

set_debug(True)

train_job = add_ml_job_local(
    job_executor_name="train",
    executor_params={},
    models_pattern=ModelForm(
        model_version_choice=ModelVersionChoice(name="BlackWhiteResNet18"),
        params=[
            ModelMethodParams(method=ModelMethodName.train_function, params={'num_epochs': 1}),
            ModelMethodParams(method=ModelMethodName.init, params={"num_classes": 10}),
        ],
        new_model_name=new_model_name,
        new_model_description='Description model'

    ),
    data_pattern=DatasetLoaderForm(
        name="MNIST",
        version=1,
        collector_name="s3",
        collector_params={"bucket": bucket_name},
        params=[DatasetLoaderMethodParams(params={})]
    ),
    model_paths=model_path,
    job_name=job_name
)
