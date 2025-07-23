.. _remote_job:

Remote Job
==========

Our main goal is to train the simplest scikit-learn model, evaluate it's quality and use it to make a prediction. We will use the simplest Iris dataset. Let's implement :class:`~ML_management.model.patterns.trainable_model.TrainableModel` wrapper interfaces that will perform training as well as make predictions.

.. literalinclude:: ../_static/code_examples/quickstart/remote/model/model.py
    :caption: ``model.py``
    :language: python
    :linenos:

.. note::
    We assume that the model is the source code (model wrapper) and weights of the model. The model can be initialized for the first time or it can be initialized from existing weights. The model initialization code will be executed every time when model is loaded locally or within job, which means that the user always needs to handle two cases: the presence and absence of pre-trained model weights.

    Note that the training code of the model has only two differences from the traditional one: the name of the training method is *train_function* and *logging* of the performance metric is done through a call to the client library.

Now we need to upload the model wrapper to a remote server. To achieve it, you should make a directory (for example, "model") that contains all required source code, ``__init__.py`` file with *get_object()* function that returns a model instance, and all artifacts that must be in a subfolder called artifacts.

.. note::
    Artifacts are not executable code: they are files used for the functioning of the model.

The supposed directory structure:

.. code::

    root
    ├── upload_model.py
    └── model 
        ├── __init__.py
        ├── model.py
        └── conda.yaml

.. literalinclude:: ../_static/code_examples/quickstart/remote/model/__init__.py
    :caption: ``__init__.py``
    :language: python
    :linenos:

.. literalinclude:: ../_static/code_examples/quickstart/remote/model/conda.yaml
    :caption: ``conda.yaml``
    :linenos:

.. note::
    conda.yaml defines the model's environment.

As mentioned before, all user code is stored and executed on a remote server. To store user code on a remote server, connect to the server :func:`~ML_management.mlmanagement.backend_api.set_server_url`, specifying the credentials with :func:`~ML_management.mlmanagement.backend_api.set_mlm_credentials`. Then you can use :func:`~ML_management.mlmanagement.log_api.log_model_src` client function to upload a model to the server. 

.. literalinclude:: ../_static/code_examples/quickstart/remote/upload_model.py
    :caption: ``upload_model.py``
    :lines: 2-
    :language: python
    :linenos:


To use the uploaded model, create a remote task using :func:`~ML_management.sdk.job.add_ml_job`.

.. literalinclude:: ../_static/code_examples/quickstart/remote/add_ml_job.py
    :language: python
    :lines: 2-
    :linenos:

Now we can check :func:`~ML_management.sdk.job.job_metric_by_name`.

.. literalinclude:: ../_static/code_examples/quickstart/remote/get_job_metric.py
    :language: python
    :lines: 2-
    :linenos:

.. csv-table::
   :file: ../_static/code_examples/quickstart/remote/get_job_metric_output.csv
   :widths: 30
   :header-rows: 1


One Step Inference
------------------

First, load a model using its name and version. Than load some data and check predictions:

.. literalinclude:: ../_static/code_examples/quickstart/remote/local_model_predictions.py
    :language: python
    :linenos:
    :lines: 2-

This code is available as a notebook on Google Colab[TODO].
