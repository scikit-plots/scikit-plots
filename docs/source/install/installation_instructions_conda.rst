:orphan:
..
  # This file is included. So it needs to be marked as orphan to suppress warnings.
  .. https://www.sphinx-doc.org/en/master/usage/restructuredtext/field-lists.html#metadata
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

    .. .. jupyter-execute
    .. .. code-block:: python
    .. prompt:: python >>>

Install conda using the `miniforge installers
<https://github.com/conda-forge/miniforge#miniforge>`__ (no administrator permission
required). Then run:

.. prompt:: bash

  ## (conda, mamba or micromamba) Create New Env and install ``scikit-plots``
  ## Create a new environment and install Python 3.11 with IPython kernel support
  # conda create -y -n py311 python=3.11 ipykernel
  # mamba create --yes --name py311 python=3.11 ipykernel
  micromamba create -y -n py311 python=3.11 ipykernel

.. prompt:: bash

  ## (conda, mamba or micromamba) Activate the environment
  # conda activate py311
  # mamba activate py311
  micromamba activate py311

.. prompt:: bash

  ## (conda, mamba or micromamba) Deep Explore scikit-plots
  # conda repoquery search -c conda-forge "scikit-plots=0.4.0" --json
  # mamba repoquery search -c conda-forge "scikit-plots=0.4.0" --json
  # micromamba repoquery search -c conda-forge "scikit-plots=0.4.0" --json
  # micromamba repoquery search -c conda-forge "scikit-plots=0.4.0" --json --platform osx-64
  micromamba repoquery search -c conda-forge "scikit-plots=0.4.0" --json \
    | jq -r '.result.pkgs[] | "\(.subdir)  \(.build)"'

.. prompt:: bash

  ## (conda, mamba or micromamba) Install scikit-plots
  # conda install -y conda-forge::scikit-plots
  # mamba install --yes --channel conda-forge scikit-plots
  micromamba install -y -c conda-forge scikit-plots

  # Cause numpy>=2.0.0 but support old numpy
  # pip install numpy==1.26.4

.. prompt:: bash

  ## (conda, mamba or micromamba) Install newest compatible build scikit-plots
  # conda update -y conda-forge::scikit-plots
  # mamba update --yes --channel conda-forge scikit-plots
  micromamba update -y -c conda-forge scikit-plots

  # Cause numpy>=2.0.0 but support old numpy
  # pip install numpy==1.26.4

In order to check your installation, you can use:

.. prompt:: bash

  ## (conda, mamba or micromamba) Verify version and location scikit-plots
  # conda list | grep scikit-plots
  # mamba list | grep scikit-plots
  micromamba list | grep scikit-plots

  # Version Check
  python -c "import scikitplot; scikitplot.show_versions()"

.. prompt:: bash

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

.. prompt:: bash

  ## Try After Ensure all "Runtime dependencies" installed
  # pip install --extra-index-url https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
  pip install -i https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots

  # Cause numpy>=2.0.0 but support old numpy
  # pip install numpy==1.26.4
