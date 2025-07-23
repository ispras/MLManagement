.. _collector:

Collectors
==========

Collector puts the raw data from the remote storage into the execution container.

Supported data collectors.
--------------------------

.. list-table:: Supported collectors
   :widths: 15 30 45
   :header-rows: 1
   :align: center

   * - :mod:`~ML_management.sdk.parameters.SingleDatasetLoader.collector_name`
     - Description
     - Job Collector Params
   * - "dummy"
     - **"dummy"** collector implies that the job does not require downloading data from any remote storage.
     - No params
   * - "s3"
     - **"s3"** collector loads data from s3 storage according to the specified parameters.
     - * bucket: str,
       * untar_data: bool = False,
       * remote_paths: Optional[List[str]] = None,
       * verbose: bool = True,
       * sync: bool = True,
       * clear_local: bool = False

        For more information see :mod:`~ML_management.s3.manager.S3Manager.set_data`.
   * - "local"
     - **"local"** collector only for local debug job. Returns the specified path to the data in its original form.
     - * local_path: str - local path to the raw data.