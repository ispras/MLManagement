.. _executor_base:

Executor
========

.. toctree::
   :maxdepth: 1
   
   executor_params/executor_params.rst

.. autoclass:: ML_management.executor.base_executor.BaseExecutor
   :members: execute
   :special-members: __init__
   :undoc-members:

Examples
--------

.. autoclass:: ML_management.executor.templates.eval.eval_executor.EvalExecutor

.. autoclass:: ML_management.executor.templates.train.train_executor.TrainExecutor
   
.. autoclass:: ML_management.executor.templates.finetune.finetune_executor.FinetuneExecutor
