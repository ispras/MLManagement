from .names import attack_job_name, model_name, dataset_loader_name, bucket_name, attacked_data, attacked_examples
from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.sdk import DatasetLoaderMethodParams, ModelMethodParams, ModelVersionChoice, DatasetLoaderForm, ModelForm, ResourcesForm, add_ml_job


# attacked_data = <BUCKET_NAME_FOR_ATTACKED_DATA>
# attacked_examples = <BUCKET_NAME_FOR_SOME_ATTACKED_EXAMPLES>

attack_job = add_ml_job(
    job_executor_name="ProjectedGradientDescent_Attack",
    executor_params={"bucket_name": attacked_data, "example_bucket_name": attacked_examples, "batch_size": 512, "max_iter": 50},
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
    job_name=attack_job_name,
    resources=ResourcesForm(gpu_number=1)
)
