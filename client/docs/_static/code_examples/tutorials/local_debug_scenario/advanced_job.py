from .names import base_job_name, model_path_adv, dataset_loader_path_adv, local_path
from ML_management.sdk import add_ml_job_local
from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.sdk import AnyDatasetLoaderForm, AnyModelForm, DatasetLoaderMethodParams, DatasetLoaderWithRole, ModelMethodParams, \
    ModelVersionChoice, ModelWithRole
from ML_management.mlmanagement import set_debug

# job_name = <YOUR_JOB_NAME>
# new_model_name = <YOUR_NEW_MODEL_NAME>
# local_path = 'local/path'
# dataset_loader_path_adv = 'local/path'
# model_path_adv = 'local/path'

set_debug(True)

job_result = add_ml_job_local(
    job_executor_name="multiple_two_model",
    executor_params={},
    data_pattern=AnyDatasetLoaderForm(
        dataset_loaders=[
            DatasetLoaderWithRole(
                role="single1",
                name="FashionMnist",
                version=1,  # Optional. Default: latest version
                params=[DatasetLoaderMethodParams(params={"train": True})],
                collector_name="s3",
                collector_params={"bucket": "fashion-mnist"},
            ),
            DatasetLoaderWithRole(
                role="new_custom_role",
                name="FashionMnist",
                params=[DatasetLoaderMethodParams(params={"train": False})],
                collector_name="local",
                collector_params={"local_path": local_path},
            ),
        ]
    ),
    models_pattern=AnyModelForm(
        models=[
            ModelWithRole(
                role="single1",
                model_version_choice=ModelVersionChoice(name="CoinModel"),
                params=[ModelMethodParams(method=ModelMethodName.finetune_function, params={"data_size": 100})],
            ),
            ModelWithRole(
                role="single2",
                model_version_choice=ModelVersionChoice(name="CoinModel", version=1),
                params=[ModelMethodParams(method=ModelMethodName.train_function, params={"data_size": 100})],
            ),
        ]
    ),
    job_name=base_job_name,
    model_paths={"single1": model_path_adv},
    dataset_loader_paths={"new_custom_role": dataset_loader_path_adv},
)
