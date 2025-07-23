from .names import eval_job_name, model_name, dataset_loader_name, bucket_name
from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.sdk import DatasetLoaderMethodParams, ModelMethodParams, ResourcesForm, ModelVersionChoice, DatasetLoaderForm, ModelForm, add_ml_job


eval_job = add_ml_job(
    job_executor_name="eval",
    executor_params={},
    models_pattern=ModelForm(
        model_version_choice=ModelVersionChoice(name=model_name),
        params=[ModelMethodParams(method=ModelMethodName.evaluate_function, params={"shuffle": False, "batch_size": 256})],
    ),
    data_pattern=DatasetLoaderForm(
        name=dataset_loader_name,
        collector_name="s3",
        params=[DatasetLoaderMethodParams(params={"train_part": False})],
        collector_params={"bucket": bucket_name},
    ),
    resources=ResourcesForm(gpu_number=1),
    job_name=eval_job_name
)
