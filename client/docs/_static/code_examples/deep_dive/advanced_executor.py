import gzip
import os

import numpy as np
from ML_management import mlmanagement
from ML_management.dataset_loader.dataset_loader_pattern import DatasetLoaderPattern
from ML_management.model.patterns.retrainable_model import RetrainableModel
from ML_management.model.patterns.trainable_model import TrainableModel
from ML_management.s3 import S3Manager


# https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
def load_mnist(path, kind="train"):
    """Load MNIST data from `path`."""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """Implemetation of call."""
        return sample


class FashionMnist:
    """Torch Dataset for fashion_mnist."""

    def __init__(self, path, train=True, image_transform=None, label_transform=None):
        """Set pathes and transforms."""
        if train:
            images, labels = load_mnist(f"{path}/raw")[:1000]
        else:
            images, labels = load_mnist(f"{path}/raw", kind="t10k")[:1000]

        self.images = images.reshape(-1, 28, 28).astype(np.float32)[:1000]

        self.labels = labels.reshape(-1, 1).astype(int)[:1000]

        self.path = path
        self.transform = ToTensor() if image_transform is None else image_transform
        self.label_transform = ToTensor() if label_transform is None else label_transform

    def __len__(
        self,
    ):
        """Length of current dataset."""
        length = len(self.images)
        return length

    def __getitem__(self, idx):
        """Return image  and label by index."""
        img, label = self.images[idx], self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label


class FashionMnistWrapper(DatasetLoaderPattern):
    """Fashion mnist dataset loader class."""

    def __init__(self):
        super().__init__()

    def get_dataset(self, train: bool, unused_params: str = ""):
        """Return FashionMnist dataset."""
        return FashionMnist(self.data_path, train)


class CoinModel:
    """Dummy coin model."""

    def __init__(self, p: float = 0.5) -> None:
        """
        Model initialization.

        Parameters
        ----------
        p : float
            Parameter of the binomial distribution, >= 0 and <=1.
        """
        if not (0.0 <= p <= 1):
            raise ValueError("p < 0, p > 1 or p is NaN")

        self.p = p

    def __call__(self, x):
        """Model call method."""
        return np.random.binomial(1, self.p)


class WrapperCoinModel(TrainableModel, RetrainableModel):
    """Wrapper for CoinModel."""

    def __init__(self, p: float = 0.5):
        """Model initialization."""
        super().__init__()

        self.model = CoinModel(p=p)

    def train_function(self, data_size: int, str_param: str = "param", bool_param: bool = True):
        """Train function implementation."""
        assert self.dataset is not None

        train_data = self.dataset

        train_acc = 0

        for item in train_data:
            prediction = self.model(item[0])
            train_acc += prediction == item[1]

        mlmanagement.log_metric("accuracy", float(train_acc / len(train_data)) * 0.8)
        # to test how metric history works, log one more time
        mlmanagement.log_metric("accuracy", float(train_acc / len(train_data)))
        artifacts = None

        return artifacts

    def finetune_function(self, data_size: int, str_param: str = "param", bool_param: bool = True):
        """Finetune function implementation."""
        assert self.dataset is not None

        train_data = self.dataset

        train_acc = 0

        for item in train_data:
            prediction = self.model(item[0])
            train_acc += prediction == item[1]

        mlmanagement.log_metric("accuracy", float(train_acc / len(train_data)) * 0.8)
        mlmanagement.log_metric("accuracy", float(train_acc / len(train_data)))

        artifacts = None

        return artifacts


S3Manager().upload("<PATH_TO_DATA>", "mnist")


from ML_management.executor import BaseExecutor
from ML_management.executor.patterns import (
    ArbitraryDatasetLoaderPattern,
    ArbitraryModelsPattern,
    OneDatasetLoaderPatternWithRole,
    OneModelPatternWithRole,
)
from ML_management.model.model_type_to_methods_map import ModelMethodName


class DemoTwoModelTwoDatasetLoaderExecutor(BaseExecutor):
    def __init__(self):
        """Executor initialization."""
        super().__init__(
            executor_models_pattern=ArbitraryModelsPattern(
                desired_models=[
                    OneModelPatternWithRole(
                        role="single1",
                        desired_model_methods=[ModelMethodName.finetune_function],
                    ),
                    OneModelPatternWithRole(
                        role="single2", desired_model_methods=[ModelMethodName.train_function]
                    ),
                ]
            ),
            executor_dataset_loaders_pattern=ArbitraryDatasetLoaderPattern(
                desired_dataset_loaders=[
                    OneDatasetLoaderPatternWithRole(
                        role="single1"
                    ),  # dataset loader may have same role as models, because model's roles and dataset loader's roles are independent.
                    OneDatasetLoaderPatternWithRole(role="new_custom_role"),
                ]
            ),
        )

    def execute(self):
        self.role_model_map["single1"].dataset = self.role_dataset_map["single1"]
        self.role_model_map["single2"].dataset = self.role_dataset_map["new_custom_role"]
        res1 = self.role_model_map["single1"].finetune_function(
            **self.model_method_parameters_dict["single1"][ModelMethodName.finetune_function]
        )
        res2 = self.role_model_map["single2"].train_function(**self.model_method_parameters_dict["single2"][ModelMethodName.train_function])

        with open("single1_artifacts.txt", "w") as stream:
            stream.write("some single1 artifacts")

        with open("single2_artifacts.txt", "w") as stream:
            stream.write("some single2 artifacts")

        return {"single1": "single1_artifacts.txt", "single2": "single2_artifacts.txt"}


from ML_management.model.model_type_to_methods_map import ModelMethodName
from ML_management.mlmanagement.upload_model_mode import UploadModelMode
from ML_management.sdk import (
    AnyDatasetLoaderForm,
    AnyModelForm,
    DatasetLoaderMethodParams,
    DatasetLoaderWithRole,
    ModelMethodParams,
    ModelVersionChoice,
    ModelWithRole,
    ResourcesForm,
    UploadAnyNewModelsForm,
    UploadOneNewModelWithRole,
    add_ml_job,
)

add_ml_job(
    job_executor_name="multiple_entities_executor",
    job_executor_version=1,
    experiment_name="MultipleEntities",
    executor_params={},
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
    data_pattern=AnyDatasetLoaderForm(
        dataset_loaders=[
            DatasetLoaderWithRole(
                role="single1",
                name="FashionMnist",
                version=1,  # Optional. Default: latest version
                params=[DatasetLoaderMethodParams(params={"train": True})],
                collector_name="s3",
                collector_params={"bucket": "mnist"},
            ),
            DatasetLoaderWithRole(
                role="new_custom_role",
                name="FashionMnist",
                version=1,  # Optional. Default: latest version
                params=[DatasetLoaderMethodParams(params={"train": False})],
                collector_name="s3",
                collector_params={"bucket": "mnist"},
            ),
        ]
    ),
    upload_models_params=UploadAnyNewModelsForm(
        upload_models_params=[
            UploadOneNewModelWithRole(
                role="single1",
                upload_model_mode=UploadModelMode.new_version,
                new_model_description="CoinModel new version description",
                prepare_new_model_inference=True,
            ),
            UploadOneNewModelWithRole(
                role="single2",
                upload_model_mode=UploadModelMode.new_model,
                new_model_description="CoinModelTrained description",
                new_model_name="CoinModelTrained",
            )
        ]
    ),
    resources=ResourcesForm(gpu_number=1)
)
