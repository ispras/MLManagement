.. MLManagement documentation master file, created by
   sphinx-quickstart on Mon Apr 15 13:23:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLManagement: a cloud-native MLOps system. 
==========================================================================================

**MLM**\anagement is aimed at ensuring the full lifecycle of ML models. It is an extensible framework-agnostic tool that allows you to conduct a lot of experiments with the model: train, fine-tune, test and also serve your ML model.

Some of **MLM**'s features are:
   * Versioning models and data;
   * Conducting multiple experiments in a team;
   * Comparing results and analysis;
   * Efficiency of implementation and operation;
   * Support for non-standard scenarios: trust-ML, data drift, etc.;

.. _workflow:

Workflow
--------
The main workflow consists of the following stages:

   1. Prepare Wrappers for your ML model and data;
   2. Upload your Wrappers to the server;
   3. Run the experiment;
   4. Wait for the end of the experiment and analyze the results using a convenient UI;
   5. Serve your model;

.. _installation:

Installation
----------------

As **MLM** is a cloud-native platform, all user code is stored and executed on a remote server. The interaction is carried out using the client library **ML-management**.

Install the client library via pip:

    .. code-block:: bash
      
      pip install ML-management


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/local_job.rst
   getting_started/remote_job.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/introduction.rst
   tutorials/base_scenario.rst
   tutorials/local_debug_mode.rst
   tutorials/trust_scenario_base.rst
   tutorials/trust_scenario_adv.rst

.. toctree::
   :maxdepth: 1
   :caption: SDK usage

   sdk/sdk.rst
   sdk/parameters.rst
   sdk/utils.rst


.. toctree::
   :maxdepth: 1
   :caption: Abstractions

   models/model.rst
   executors/executor.rst
   dataset_loaders/datasetloader.rst
   collectors/collectors.rst

.. toctree::
   :maxdepth: 1
   :caption: Deep Dive

   tutorials/deep_dive.rst

.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

   changelog/changelog.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
