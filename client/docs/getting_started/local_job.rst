.. _local_job:

This guide assumes that the **ML-management** library is already installed. For installation instructions, see :ref:`installation`.

Local Job
=========

To connect to a remote server, use function :func:`~ML_management.mlmanagement.backend_api.set_server_url`, specifying the credentials with :func:`~ML_management.mlmanagement.backend_api.set_mlm_credentials`. Then you can use client function to communicate with remote server. 

.. literalinclude:: ../_static/code_examples/quickstart/set_server_url.py
    :language: python
    :linenos:

To create a local job on your machine and log metrics and artifacts to the registry use :func:`~ML_management.mlmanagement.active_job.start_job` with your ``job_name``:

.. literalinclude:: ../_static/code_examples/quickstart/local/local_job.py
    :language: python
    :lines: 2-
    :linenos:

Now we can check :func:`~ML_management.sdk.job.job_metric_by_name`.

.. literalinclude:: ../_static/code_examples/quickstart/local/get_job_metric.py
    :language: python
    :lines: 2-
    :linenos:

.. csv-table::
   :file: ../_static/code_examples/quickstart/local/get_job_metric_output.csv
   :widths: 30
   :header-rows: 1

Also we can download job artifacts by :func:`~ML_management.mlmanagement.load_api.download_job_artifacts`:

.. literalinclude:: ../_static/code_examples/quickstart/local/get_job_artifact.py
    :language: python
    :lines: 2-
    :linenos:
