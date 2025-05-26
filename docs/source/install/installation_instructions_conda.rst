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

  # (conda or mamba) Create New Env and install scikit-plots
  conda create -n sklearn-env -c conda-forge scikit-plots
  conda activate sklearn-env

.. prompt:: bash >>> auto

  # (conda or mamba) Install scikit-plots
  conda install -c conda-forge scikit-plots

In order to check your installation, you can use:

.. prompt:: bash >>> auto

  conda list scikit-plots  # show scikit-plots version and location
  conda list               # show all installed packages in the environment
  python -c "import scikitplot; scikitplot.show_versions()"
  scikitplot --version
  scikitplot -v

(Not Recommended) Also available Nightly Version can cause conflict:

.. prompt:: bash >>> auto

  ## (Optionally) Install the lost packages
  ## wget https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  curl -O https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  pip install -r default.txt

  ## Try After Ensure all dependencies installed
  pip install -U -i https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
