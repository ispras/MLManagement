Advanced Executor[WIP]
======================

!!!WORK IN PROGRESS!!!

This guide will tell you about more advanced features for writing executors.

Let's consider an executor that will work with 2 models and 2 dataset loaders.
This executor will train one model, finetune second and use FashionMnist for this.
Let's imagine that we have simple model and FashionMnist dataset loader.
In our example, to simplify the code, both the model and the dataset loader will be used in 2 different roles. Similarly, it would be possible to put a separate model and a separate dataset loader for each role.


.. _deep_drive:

First, let's define our dataset loader and model.
Since we want to train and finetune our :ref:`model <model>`, we need to use :class:`~ML_management.models.patterns.trainable_model.TrainableModel` and :class:`~ML_management.models.patterns.retrainable_model.RetrainableModel` Wrapper that will implement the necessary interface.

.. literalinclude:: ../_static/code_examples/deep_dive/advanced_executor.py
    :language: python
    :lines: 1-151

We also need to upload the data to s3

.. literalinclude:: ../_static/code_examples/deep_dive/advanced_executor.py
    :language: python
    :lines: 154


Now we will define our executor.
Important parameters to consider are ``executor_models_pattern`` and ``executor_dataset_loaders_pattern``.
They determine how many models or dataset loaders the executor is waiting for and what kind of interface each entity should provide.
To do this, classes are used that describe these requirements of the executor.
Since we have 2 models and dataset loaders, we will use the ArbitraryModelsPattern and ArbitraryDatasetLoaderPattern classes.
These classes wrap a list of objects describing each entity individually.
Since at the moment only 1 method is available for dataset loaders, 
it is enough for them to specify only the role - an arbitrary string describing the purpose of the dataset loader.
For models, you must specify the desired method as well as a parameter that determines the need to log a new model or version

.. literalinclude:: ../_static/code_examples/deep_dive/advanced_executor.py
    :language: python
    :lines: 157-208


After all the entities and data are uploaded to the system, you can use the sdk to create a job.
In general, the structure of the description of the parameters of models and data for the job is similar to the structure of the description of the executor.
We will use 2 classes for high-level parameters description and some nested classes.

.. literalinclude:: ../_static/code_examples/deep_dive/advanced_executor.py
    :language: python
    :lines: 210-267
