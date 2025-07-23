from .names import job_name, model_name, new_model_name
from ML_management.mlmanagement.upload_model_mode import UploadModelMode
from ML_management.sdk import ModelVersionChoice, DatasetLoaderForm, ModelForm, add_ml_job, UploadOneNewModelForm


# job_name = <YOUR_JOB_NAME>
# new_model_name = <YOUR_NEW_MODEL_NAME>

train_job = add_ml_job(
    job_executor_name="train",
    executor_params={},
    data_pattern=DatasetLoaderForm(
        name="DummyData",
        # params=[DatasetLoaderMethodParams(params={})],  # if we use default params, we can skip this argument
        collector_name="dummy",
        collector_params={},
    ),
    models_pattern=ModelForm(
        model_version_choice=ModelVersionChoice(name=model_name),
        # params=[ModelMethodParams(method=ModelMethodName.train_function, params={})],  # if we use default params, we can skip this argument
    ),
    upload_models_params=UploadOneNewModelForm(
        upload_model_mode=UploadModelMode.new_model,
        new_model_name=new_model_name,
        new_model_description="Default sklearn KNN Classifier trained on Iris.",
    ),
    job_name=job_name
)
