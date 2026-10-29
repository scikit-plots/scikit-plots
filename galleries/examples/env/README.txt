.. _env__examples:

Env
===

.. currentmodule:: scikitplot._env

Examples for :py:mod:`~scikitplot._env`, the environment builder behind
``scikitplot env``: describe a Python environment once, then build the same
environment on any machine with whichever tool it has (uv, ``venv`` + pip,
pipenv, micromamba, mamba or conda).

.. code-block:: text

    request      arguments, a scikitplot-env.toml, or both (arguments win)
        ↓
    plan         reads only: every command, its network use, its approvals
        ↓
    approvals    asked (or refused) before the first step runs
        ↓
    execute      the plan's steps, under a lock on the environment
        ↓
    receipt      what was built; the next run with the same request is a no-op

Every example runs from the command line and from Python with the same
fields; the command line form is shown next to each Python call:

.. prompt:: bash $

    scikitplot env plan -n analysis --package "pandas>=2" --verify-import pandas
    scikitplot env create -n analysis --package "pandas>=2" --verify-import pandas

The examples need no network: they plan with any backend, and build only
``venv`` environments from the standard library. Named environments go to
``SKPLT_ENV_HOME`` (default ``~/.scikitplot/env``); the examples point it at a
temporary folder and remove what they build.

Start with the plan, then build, run and remove, then configuration files,
policies and the errors an automation reads. See also :ref:`env-index`.
