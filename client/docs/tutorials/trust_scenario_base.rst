.. _trust_base:

Trust scenario [Base]
=====================

This tutorial is dedicated to demonstrating the capabilities of trusted artificial intelligence in the platform.

Here we will:
    * prepare a pre-trained model for interacting with the platform
    * evaluate the quality of the original model
    * conduct an attack on the data and evaluate the resulting quality
    * demonstrate the attacked examples


Original model
--------------

This is the simplest image classification model - ResNet18 with two linear layers.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/model/resnet18.py
    :language: python
    :linenos:
    :caption: ``model/resnet18.py``

We trained this model on **Cifar10** and saved the weights to the ``"resnet18_cifar10.pth"``.

As already noted, we want to perform 2 operations with the model: evaluate the quality, conduct an attack. To do this, we will use 2 executors: **eval**, **ProjectedGradientDescent_Attack**. Let's find out which interfaces we need to implement for this using :func:`~ML_management.sdk.executor.get_required_classes_by_executor`.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/get_required_classes.py
    :language: python
    :linenos:

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/get_required_classes_output.txt
    :linenos:

So, we need to implement *evaluate_function* for :class:`~ML_management.model.patterns.evaluatable_model.EvaluatableModel`, *get_grad* for :class:`~ML_management.model.patterns.gradient_model.GradientModel` and also *predict_function* since all :class:`~ML_management.model.patterns.model_pattern.Model` must implement this interface.

Model Wrapper
-------------

Let's organize the correct structure of the model files again:

.. code::

    root
    ├── upload_model.py
    └── model 
        ├── __init__.py
        ├── model.py
        ├── resnet18.py
        ├── conda.yaml
        └── artifacts
            └── resnet18_cifar10.pth


.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/model/model.py
    :language: python
    :linenos:
    :caption: ``model/model.py``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/model/__init__.py
    :language: python
    :linenos:
    :caption: ``model/__init__.py``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/model/conda.yaml
    :linenos:
    :caption: ``model/conda.yaml``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/upload_model.py
    :linenos:
    :caption: ``upload_model.py``
    :lines: 2-


Data Wrapper
------------

Let's organize the correct structure of the data files:

.. code::

    root
    ├── upload_data.py
    └── data 
        ├── __init__.py
        ├── data.py
        └── conda.yaml
    

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/data/data.py
    :language: python
    :linenos:
    :caption: ``data/data.py``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/data/__init__.py
    :language: python
    :linenos:
    :caption: ``data/__init__.py``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/data/conda.yaml
    :linenos:
    :caption: ``data/conda.yaml``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/upload_data.py
    :linenos:
    :caption: ``upload_data.py``
    :lines: 2-

We also need to upload raw data (from local directory ``./cifar10``) to the remote storage, which will be downloaded to the execution container using the :ref:`Сollector <collector>`.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/upload_s3.py
    :linenos:
    :caption: ``upload_s3.py``
    :lines: 2-

Original quality
----------------

Firstly, we will evaluate quality of the original model. So let's create appropriate job:

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/add_eval_job.py
    :lines: 2-
    :language: python
    :linenos:

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/get_eval_job_metric.py
    :lines: 2-
    :language: python
    :linenos:

.. csv-table::
   :file: ../_static/code_examples/tutorials/trust_scenario/eval_job_metric.csv
   :widths: 30
   :header-rows: 1

.. _attack_job:

Attack
------

Now we will attack our model using **ProjectedGradientDescent_Attack** executor. Up to this point, we used basic executors that called the basic methods of the model (*train_function*, *evaluate_function*) and had no special parameters. 

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/get_eval_executor_schema.py
    :language: python
    :linenos:

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/eval_schema_output.txt
    :linenos:

So now we use the executor that not only calls some method of the model, but also has some logic, and therefore its own parameters.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/get_attack_executor_schema.py
    :language: python
    :linenos:

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/attack_schema_output.txt
    :linenos:

This way we can see list of all the executor parameters with default types and values, as well as a list of required parameters. In this case, the required parameter is - "bucket_name". It determines which bucket the attacked examples will be uploaded to. To view some of the attacked examples, we will also define the parameter: "example_bucket_name"

.. note::
    The sdk also has some user-friendly functions that will help you learn the methods and parameters of the executor, datasetloaders, as well as models for a specific executor. See :mod:`~ML_management.sdk.dataset_loader.print_dataset_loader_schema`, :mod:`~ML_management.sdk.executor.print_executor_schema`, :mod:`~ML_management.sdk.executor.print_executor_roles`, :mod:`~ML_management.sdk.executor.print_model_schema_for_executor`.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/add_attack_job.py
    :lines: 2-
    :language: python
    :linenos:

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/get_attack_job_metric.py
    :lines: 2-
    :language: python
    :linenos:

.. csv-table::
   :file: ../_static/code_examples/tutorials/trust_scenario/attack_job_metric.csv
   :widths: 30
   :header-rows: 1

The prediction quality of the model has declined significantly. Let's see how much the images have changed visually.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/s3_set_data.py
    :lines: 2-
    :language: python
    :linenos:

.. list-table::

    * - .. figure:: ../_static/images/trust_scenario/original_image0_label3.png
            :width: 100 %

            Fig 1. original_image0_label3

      - .. figure:: ../_static/images/trust_scenario/crushed_image0_label5.png
            :width: 100 %

            Fig 2. crushed_image0_label5

**Thus, we got a set of attacked images that visually practically do not differ from the original ones, but the model is almost always wrong on them.**