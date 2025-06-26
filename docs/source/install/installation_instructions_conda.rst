:orphan:

.. # This file is included. So it needs to be marked as orphan to suppress warnings.

.. _installation_instructions_conda:

..
  .. warning::
      Not Implemented...

Install conda using the
`miniforge installers <https://github.com/conda-forge/miniforge#miniforge>`__ (no
administrator permission required). Then run:

.. prompt:: bash >>> auto

  # (conda or mamba) Create New Env and install ``scikit-plots``
  mamba create -n py311 python=3.11 ipykernel -y
  conda activate py311

.. prompt:: bash >>> auto

  ## (conda or mamba) Install scikit-plots (Upcoming)
  conda install --yes -c conda-forge scikit-plots

In order to check your installation, you can use:

.. prompt:: bash >>> auto

  # conda list               # show all installed packages in the environment
  conda list scikit-plots    # show scikit-plots version and location

  # Version Check
  python -c "import scikitplot; scikitplot.show_versions()"

  # (Optionally) CLI Version Check
  scikitplot --version
  scikitplot -V
  scikitplot -v

(Optionally) Also available Nightly Version **can be cause conflict** required ``New Environment``:

.. prompt:: bash >>> auto

  ## (Optionally) Install the lost packages "Runtime dependencies"
  ## wget https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  curl -O https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  pip install -r default.txt

  ## Try After Ensure all "Runtime dependencies" installed
  # pip install --extra-index-url https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
  pip install -i https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
