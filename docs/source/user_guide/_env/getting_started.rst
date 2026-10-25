.. _env-getting-started:

======================================================================
Getting started
======================================================================

See the plan, then build
----------------------------------------------------------------------

``plan`` reads the request and the dependency files and prints every
command, without changing anything:

.. code-block:: console

   $ scikitplot env plan -n analysis --package "pandas>=2" --verify-import pandas
   plan 3c7c22ba3748  ensure ~/.scikitplot/env/envs/analysis
     target   host
     backend  uv (asked: auto) - PyPI only; uv found at /usr/local/bin/uv
     1. create a uv environment with Python 3.12.4  [network]
        $ uv venv ~/.scikitplot/env/envs/analysis --seed --python 3.12.4
     2. install PyPI packages with uv  [network]
        $ uv pip install --python ~/.scikitplot/env/envs/analysis/bin/python 'pandas>=2'
     3. import pandas
        $ ~/.scikitplot/env/envs/analysis/bin/python -I -c '...' pandas

``create`` runs exactly that plan. Run it again and nothing happens: the
environment's receipt matches the request.

.. code-block:: console

   $ scikitplot env create -n analysis --package "pandas>=2" --verify-import pandas
   $ scikitplot env create -n analysis --package "pandas>=2" --verify-import pandas
   ~/.scikitplot/env/envs/analysis: up to date (nothing to do)
   $ scikitplot env run -n analysis -- python -c "import pandas; print(pandas.__version__)"

Named environments live under ``~/.scikitplot/env/envs`` (``SKPLT_ENV_HOME``
moves the whole home); ``-p .venv`` builds in a directory of your choice.

From Python
----------------------------------------------------------------------

The functions take the same fields as keywords:

.. code-block:: python

   import tempfile

   from scikitplot import _env

   prefix = tempfile.mkdtemp() + "/demo"
   plan = _env.plan(prefix=prefix, backend="venv", verify_imports=["json"])
   print(plan.render())
   assert [step.id for step in plan.steps] == ["create", "verify"]

``_env.create(...)`` returns an ``ExecutionResult`` (``result.changed`` is
false when there was nothing to do); ``_env.remove(...)`` deletes the
environment; ``_env.run(["python", "-V"], ...)`` runs a command in it.

Conda packages and bootstrapping micromamba
----------------------------------------------------------------------

A Conda package (``--conda-package``) or an ``environment.yml`` selects
micromamba, mamba or conda. When none is installed, the builder can install
micromamba itself, the conda-forge build checked against a SHA-256 written in
the builder, into ``~/.scikitplot/env/tools/bin``. That needs both the
policy and your approval:

.. code-block:: console

   $ scikitplot env create -n geo --conda-package gdal --package rasterio \
         --bootstrap if-missing --approve bootstrap

``scikitplot env shell-init bash`` prints the lines that put the tools on
``PATH`` and hook ``micromamba activate`` into bash; ``--write --yes`` adds
them to ``~/.bashrc`` between marker lines (running it again changes
nothing).
