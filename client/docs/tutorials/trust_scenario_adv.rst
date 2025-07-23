Trust scenario [Executor]
=========================

As we mentioned in the :ref:`introduction <introduction>`, in fact, the executor is a meta-algorithm that interacts with certain methods of model and data wrappers. The platform provides some set of basic :ref:`Executor <executor_base>`-s (**train**, **eval**, **fine-tune**) that perform basic operations. In particular, the training executor puts data into the model, and also calls the user-implemented *train_function* method.
At the same time, as we mentioned earlier, the platform is a block constructor and the executor is another extensible aspect of the platform.

In this demo, we will look at how to write and add a custom executor to the platform.

My First Executor
-----------------

Let's implement the executor that will perform a PGD attack. In fact, we used it in section :ref:`Trust scenario [Base] <trust_base>`, and now let's look at how it works from the inside.

The file structure stays the same.

.. code::

    root
    ├── upload_executor.py
    └── executor
        └── attack
            ├── __init__.py
            ├── attack.py
            ├── pgd.py
            └── conda.yaml

Here we will implement logic of PGD attack. You can view the content of the attack at the `link <https://arxiv.org/pdf/1706.06083>`_.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/executor/attack/pgd.py
    :language: python
    :linenos:
    :caption: ``executor/attack/pgd.py``


So we will prepare the attack wrapper, the actual body of the executor itself. 
    
    - ``__init__``: in order to interact with the model you need to declare a list of methods that model should implement; as well as what will happen to the model after the end of the execution job. Since PGD is an attack on data and the model remains unchanged, this executor will not generate a new version of the model, and therefore we define ``upload_model_mode=UploadModelMode.none``. 
    
    - In the ``execute`` method we initialize the attack instance, make predictions on the data and modify (attack) the images based on them; save the changed images to the new s3-bucket and also make it possible to save a number of typical examples for analysis. After the attack is over, we will call a method that evaluates the quality of the original model on the attacked data in order to understand how much the prediction quality has changed.

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/executor/attack/attack.py
    :language: python
    :linenos:
    :caption: ``executor/attack/attack.py``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/executor/attack/__init__.py
    :language: python
    :linenos:
    :caption: ``executor/attack/__init__.py``

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/executor/attack/conda.yaml
    :linenos:
    :caption: ``executor/attack/conda.yaml``

Now we can upload our executor to the server and run job with it (see section :ref:`Trust scenario [Base] <attack_job>`).

.. literalinclude:: ../_static/code_examples/tutorials/trust_scenario/upload_executor.py
    :linenos:
    :caption: ``upload_executor.py``

.. note:: Executor can be debugged in debug mode similar to the Model and DatasetLoader (see :ref:`local_debug_mode`).