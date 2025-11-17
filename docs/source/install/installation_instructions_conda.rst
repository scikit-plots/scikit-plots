:orphan:
..
  # This file is included. So it needs to be marked as orphan to suppress warnings.
  .. include:: installation_instructions_conda.rst
      :start-after: :orphan:

.. _installation_instructions_conda:

..
  .. warning::
      Not Implemented...

..
    # Code syntax
    :: >>>
    python, r, c, cpp, sql, bash, shell, make, cmake, docker,
    html, markdown, javascript, json, yaml, toml, ini, diff, text, etc.
    .. 	General-purpose code (highlighting only)
    .. code-block:: python
    .. 'sphinx_prompt' Simulates interactive terminal prompts
    .. prompt:: bash $
    .. prompt:: python >>>
    .. prompt:: ipython In [1]:
    .. prompt:: sh $
    .. prompt:: powershell PS C:\>
    .. prompt:: docker root@container:~#

Install conda using the
`miniforge installers <https://github.com/conda-forge/miniforge#miniforge>`__ (no
administrator permission required). Then run:

.. prompt:: bash

  ## (conda, mamba or micromamba) Create New Env and install ``scikit-plots``
  ## Create a new environment and install Python 3.11 with IPython kernel support
  # conda create -y -n py311 python=3.11 ipykernel
  micromamba create -y -n py311 python=3.11 ipykernel

.. prompt:: bash

  ## (conda, mamba or micromamba) Activate the environment
  # conda activate py311
  micromamba activate py311

.. prompt:: bash

  ## (conda, mamba or micromamba) Install scikit-plots (Upcoming)
  # conda search conda-forge::scikit-plots
  # conda search --channel conda-forge scikit-plots
  micromamba search -c conda-forge scikit-plots

.. prompt:: bash

  ## (conda, mamba or micromamba) Install scikit-plots (Upcoming)
  # conda install -y conda-forge::scikit-plots
  # conda install --yes --channel conda-forge scikit-plots
  micromamba install -y -c conda-forge scikit-plots

  # Cause numpy>=2.0.0 but support old numpy
  # pip install numpy==1.26.4

In order to check your installation, you can use:

.. prompt:: bash

  # conda list               # show all installed packages in the environment
  conda list scikit-plots    # show scikit-plots version and location

  # Version Check
  python -c "import scikitplot; scikitplot.show_versions()"

  # (Optionally) CLI Version Check
  scikitplot
  scikitplot -h
  scikitplot -V
  scikitplot --version

(Optionally) Also available Nightly Version **can be cause conflict** required ``New Environment``:

.. prompt:: bash

  ## (Optionally) Install the lost packages "Runtime dependencies"
  ## wget https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  curl -O https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  pip install -r default.txt

  ## Try After Ensure all "Runtime dependencies" installed
  # pip install --extra-index-url https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
  pip install -i https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots

  # Cause numpy>=2.0.0 but support old numpy
  # pip install numpy==1.26.4
