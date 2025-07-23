.. _local_debug_mode:

Local Debug Mode
====================

This guide assumes that you've read the :ref:`Base scenario <base_scenario>` and this tutorial will demonstrate local **debugging** mode.

.. warning:: We strongly recommend against using the methodology described in this section for complete training. It was created solely to identify and correct errors, and not to conduct the main part of the training. When working with the Model, we RECOMMEND performing all operations, such as training, quality assessment, or any other manipulations, on SMALL TEST DATASETS consisting of a couple of examples. This will make sure that the implementation of the Model is working and does not contain critical errors before starting full-scale training or using the model.

The key differences from a remote task are that the task will be executed in the same runtime where you started it. There is no need to upload your entities to the server and you may also track and view your entities and jobs locally.


In general, the presence of an Executor in the registry who can train Models and the availability of suitable training data do not guarantee the successful completion of the task. The absence of a ready-made Model can lead to unforeseen errors, which can only be detected at late stages of development, which is undesirable.

In this regard, we will consider how to prepare the Model as efficiently as possible to minimize the risks of errors when performing time-consuming remote tasks and ensure their successful completion.



Simple job
----------

Let's say you want to run a task locally using a DatasetLoader and an Executor that have already been uploaded to the server and a Model located locally.

Immediately before launching, you must set the debug flag to True. This can be done using the function :mod:`~ML_management.mlmanagement.backend_api.set_debug`.
After that, you can use the :mod:`~ML_management.sdk.local.add_ml_job_local` function to run the task locally for debugging purposes.


* Model: :ref:`BlackWhiteResNet18 <torchmodel>` from local path (save structure of the Model described :ref:`here <file_structure>`)
* Executor: train from server
* DatasetLoader:  :ref:`MNIST <mnist_wrapper>` (pre-debugged and uploaded to a remote server as described :ref:`here <data_wrapper>`)
* Data: minst from s3

.. literalinclude:: ../_static/code_examples/tutorials/local_debug_scenario/simple_job.py
    :language: python
    :lines: 2-
    :linenos:

The example uses the standard train Executor and MNIST DatasetLoader from remote. It requires only one Model, the local path to which we have passed to `model_paths`.
This is the path to the directory where the Model is located, similar to `model_path` for :mod:`~ML_management.mlmanagement.log_api.log_model_src`.

Upon completion of the task, the variable `train_job` will contain an instance of the class :mod:`~ML_management.local_debug.debug_job_result.DebugJobResult`.


Job with local Model, DatasetLoader and raw data
------------------------------------------------

In this task, the DatasetLoader and the Model are used from the local path. However, all their parameters must be specified as if you are running a regular remote task.

* Model: :ref:`BlackWhiteResNet18 <torchmodel>` from local path
* Executor: train from server
* DatasetLoader: :ref:`MNIST <mnist_wrapper>` from local path
* Data: mnist from local path

.. literalinclude:: ../_static/code_examples/tutorials/local_debug_scenario/second_example.py
    :language: python
    :lines: 2-
    :linenos:

In order to use raw data from your local storage, use the local collector (see :ref:`"local" <collector>`) as in the example above. 

.. warning:: Don't forget to set :mod:`~ML_management.mlmanagement.backend_api.set_debug` to ``False`` after local debugging.

..
    Advanced local debug job
    ------------------------
    * Model: :ref:`CoinModel <deep_drive>` from local path; CoinModel from server
    * Executor: :ref:`multiple_two_model <deep_drive>` from server
    * DatasetLoader: :ref:`FashionMnist <deep_drive>` from server; :ref:`FashionMnist <deep_drive>`  from local path
    * Data: fashion-mnist from s3; fashion-mnist from local path




    In this example, we use an Executor that requires two models and two DatasetLoaders. One model will be loaded from the local storage, and the other from the registry, similar to the dataset loader.

    .. literalinclude:: ../_static/code_examples/tutorials/local_debug_scenario/advanced_job.py
        :language: python
        :lines: 2-
        :linenos: