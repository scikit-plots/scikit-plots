<!--
- https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/
- https://github.com/pypa/readme_renderer
-->

<h1 align=center>
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    Welcome to Scikit-plots 101
  </a>
</h1>

<div>
<!-- GitHub-flavored Markdown (GFM) does not support inline CSS or HTML layout tags -->
<!-- <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; max-width: 580px; margin: auto;">
  <a href="https://pypi.org/project/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="PyPI Status" src="https://img.shields.io/pypi/v/scikit-plots">
  </a>
  <a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="Anaconda Nightly Wheels Status" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/version.svg">
  </a>
  <a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/files" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="Anaconda Nightly Release Date" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/latest_release_date.svg">
  </a>
  <a href="https://github.com/scikit-plots/scikit-plots/actions/workflows/wheels.yml?query=event%3Aworkflow_dispatch" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="GitHub Actions CI Build Wheels Status" src="https://github.com/scikit-plots/scikit-plots/actions/workflows/ci_wheels_conda.yml/badge.svg?event=workflow_dispatch">
  </a>
  <a href="https://dl.circleci.com/status-badge/redirect/circleci/MzCciwxVsGS9w3PCUFjTaB/TPithCzV9DBEcZUACH7Zij/tree/main" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="CircleCI Status" src="https://dl.circleci.com/status-badge/img/circleci/MzCciwxVsGS9w3PCUFjTaB/TPithCzV9DBEcZUACH7Zij/tree/main.svg?style=shield">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/scikit-plots/scikit-plots/main" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="pre-commit.ci Status" src="https://results.pre-commit.ci/badge/github/scikit-plots/scikit-plots/main.svg">
  </a>
  <a href="https://github.com/pre-commit/pre-commit" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="pre-commit Status" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit">
  </a>
  <a href="https://github.com/astral-sh/ruff" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="Ruff Version" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
  </a>
  <a href="https://github.com/astral-sh/ruff" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="Ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg">
  </a>
  <a href="https://codecov.io/gh/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    # <img style="display:auto;width:auto;height:17px;" alt="Coverage Status" src="https://codecov.io/gh/scikit-plots/scikit-plots/graph/badge.svg?token=D9ET8W1I2P"/>
    <img style="height:17px;" alt="Coverage Status" src="https://codecov.io/gh/scikit-plots/scikit-plots/branch/main/graph/badge.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.13367000" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="Zenodo DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.13367000.svg">
  </a>
  <a href="https://pypi.org/project/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img style="height:17px;" alt="pyversions" src="https://img.shields.io/pypi/pyversions/scikit-plots.svg">
  </a>
</div> -->


<!-- GitHub- & PyPI-Compatible Badge Grid -->

<!-- [![build status](https://github.com/pre-commit/pre-commit/actions/workflows/main.yml/badge.svg)](https://github.com/pre-commit/pre-commit/actions/workflows/main.yml) -->
<!--
<a href="https://pypi.org/project/scikit-plots" target="_blank" rel="noopener noreferrer">
<img style="display:auto;width:auto;height:17px;" alt="PyPI Status" src="https://img.shields.io/pypi/v/scikit-plots">
</a>
<a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank" rel="noopener noreferrer">
<img style="display:auto;width:auto;height:17px;" alt="Anaconda Nightly Wheels Status" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/version.svg">
</a>
<a href="https://doi.org/10.5281/zenodo.13367000" target="_blank" rel="noopener noreferrer">
<img style="display:auto;width:auto;height:17px;" alt="Zenodo DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.13367000.svg">
</a> -->
<p align="center">
  <a href="https://pypi.org/project/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img alt="PyPI Status" height="17" src="https://img.shields.io/pypi/v/scikit-plots">
  </a>
  <a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img alt="Anaconda Nightly Wheels Status" height="17" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/version.svg">
  </a>
  <a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/files" target="_blank" rel="noopener noreferrer">
    <img alt="Anaconda Nightly Release Date" height="17" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/latest_release_date.svg">
  </a>
  <a href="https://github.com/scikit-plots/scikit-plots/actions/workflows/wheels.yml?query=event%3Aworkflow_dispatch" target="_blank" rel="noopener noreferrer">
    <img alt="GitHub Actions CI Build Wheels Status" height="17" src="https://github.com/scikit-plots/scikit-plots/actions/workflows/ci_wheels_conda.yml/badge.svg?event=workflow_dispatch">
  </a>
  <a href="https://dl.circleci.com/status-badge/redirect/gh/scikit-plots/scikit-plots/tree/main" target="_blank" rel="noopener noreferrer">
    <img alt="CircleCI Status" height="17" src="https://dl.circleci.com/status-badge/img/gh/scikit-plots/scikit-plots/tree/main.svg?style=shield">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/scikit-plots/scikit-plots/main" target="_blank" rel="noopener noreferrer">
    <img alt="pre-commit.ci Status" height="17" src="https://results.pre-commit.ci/badge/github/scikit-plots/scikit-plots/main.svg">
  </a>
  <a href="https://github.com/pre-commit/pre-commit" target="_blank" rel="noopener noreferrer">
    <img alt="pre-commit Status" height="17" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit">
  </a>
  <a href="https://github.com/astral-sh/ruff" target="_blank" rel="noopener noreferrer">
    <img alt="Ruff Version" height="17" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
  </a>
  <a href="https://github.com/astral-sh/ruff" target="_blank" rel="noopener noreferrer">
    <img alt="Ruff" height="17" src="https://img.shields.io/badge/code%20style-ruff-000000.svg">
  </a>
  <a href="https://codecov.io/gh/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img alt="Coverage Status" height="17" src="https://codecov.io/gh/scikit-plots/scikit-plots/branch/main/graph/badge.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.13367000" target="_blank" rel="noopener noreferrer">
    <img alt="Zenodo DOI" height="17" src="https://zenodo.org/badge/DOI/10.5281/zenodo.13367000.svg">
  </a>
  <a href="https://pypi.org/project/scikit-plots" target="_blank" rel="noopener noreferrer">
    <img alt="pyversions" height="17" src="https://img.shields.io/pypi/pyversions/scikit-plots.svg">
  </a>
</p>
</div>

<div>
  <a href="https://scikit-plots.github.io/dev/" target="_blank" rel="noopener noreferrer">
    <img alt="Scikit-plots" height="230" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/docs/source/logos/scikit-plots-logo-medium.png">
  </a>
</div>

<div>
<h3> Single line functions for detailed visualizations </h3>

<p> The quickest and easiest way to go from analysis... </p>

<h3> 📘 Documentation, Examples and Try|Install Scikit-plots </h3>

<h5>Explore the full features of Scikit-plots:
  <a href="https://scikit-plots.github.io/dev/devel/index.html" target="_blank" rel="noopener noreferrer">
    https://scikit-plots.github.io/dev/devel/index.html
  </a>
</h5>
</div>

<hr>

<h1> 🐛 User Installation: </h1>

<h2>
  📦 From
  <code>PIP</code>
  Installation by
  <a href="https://pypi.org/project/scikit-plots/" target="_blank" rel="noopener noreferrer">
    <code>pypi</code>
  </a>
  ,
  <a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>pypi.anaconda.org</code>
  </a>
  or
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>GITHUB</code>
  </a>
</h2>

<p> The easiest way to set up scikit-plots is to install it using pip with the following command: </p>

<h4> 🧠 Gotchas: </h4>

<ul>
  <li>⚠️ (Recommended): Use a Virtual Environmentt (like
    <a href="https://docs.python.org/3/library/venv.html" target="_blank" rel="noopener noreferrer">
      <code>venv</code>
    </a>
    <a href="https://pypi.org/project/pipenv/" target="_blank" rel="noopener noreferrer">
      <code>pipenv</code>
    </a>
    ) to Avoid Conflicts.</li>
  <li>🚫 Don't use conda <code>base</code> — it's prone to conflicts.</li>
  <li>✅ This avoids dependency issues and keeps your system stable.</li>
</ul>

```sh
# (conda or mamba) Create New Env and install ``scikit-plots``
# Create a new environment and install Python 3.11 with IPython kernel support
mamba create -n py311 python=3.11 ipykernel -y
```

```sh
# Activate the environment
conda activate py311
```

<hr>

<h3>
  - By
  <a href="https://pypi.org/project/scikit-plots/" target="_blank" rel="noopener noreferrer">
    <code>pypi</code>:
  </a>
</h3>

```sh
# Now Install scikit-plots (via pip, conda, or local source)
pip install scikit-plots
```

<hr>

<h3>
  - By
  <a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>pypi.anaconda.org</code>
  </a>
  (with
  <a href="https://scikit-plots.github.io/dev/install/dependencies.html#runtime-dependencies"
     target="_blank"
     rel="noopener noreferrer">
    required runtime dependencies
  </a>
  ):
</h3>

```sh
## (Optionally) Install the lost packages "Runtime dependencies"
## https://github.com/celik-muhammed/scikit-plots/tree/main/requirements
## wget https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
curl -O https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
pip install -r default.txt
```

```sh
## Try After Ensure all "Runtime dependencies" installed
pip install -U -i https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
```

<hr>

<h3>
  - By
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>GITHUB</code>:
  </a>
  <code>@&lt;branch&gt;</code>
  ,
  <code>@&lt;tag&gt;</code>
  or
  <code>Source Code Archive URLs</code>
  to specify a version
</h3>

<h4>
  - by
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>GITHUB</code>
  </a>
  Branches:
  <code>@&lt;branch&gt;</code>
</h4>

```bash
## pip install git+https://github.com/scikit-plots/scikit-plots.git@<branches>
## Latest in Development
pip install git+https://github.com/scikit-plots/scikit-plots.git@main
##
## (Added C, Cpp, Fortran Support) Works with standard Python (CPython)
pip install git+https://github.com/scikit-plots/scikit-plots.git@maintenance/0.4.x
##
## (Works with PyPy interpreter) Works with standard Python (CPython)
pip install git+https://github.com/scikit-plots/scikit-plots.git@maintenance/0.3.x
pip install git+https://github.com/scikit-plots/scikit-plots.git@maintenance/0.3.7
```

<h4>
  - by
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>GITHUB</code>
  </a>
  Tags:
  <code>@&lt;tag&gt;</code>
</h4>

```bash
## pip install git+https://github.com/scikit-plots/scikit-plots.git@<tags>
pip install git+https://github.com/scikit-plots/scikit-plots.git@v0.4.0rc5
pip install git+https://github.com/scikit-plots/scikit-plots.git@v0.3.9rc3
pip install git+https://github.com/scikit-plots/scikit-plots.git@v0.3.7
```

<hr>

<h2>
  📁 From Source
  Installation by
  <code>Archive</code>
  or
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>GIT Clone</code>
  </a>
</h2>

<h4>🐍 Pitfalls:</h4>

<ul>
  <li>💡 You can download
  <a href="https://github.com/scikit-plots/scikit-plots/tags" target="_blank" rel="noopener noreferrer">
    <code>GitHub Source Code Archives</code> (<code>.zip</code> or <code>.tar.gz</code>)
  </a>
  by specifying a <code>branch</code>, <code>tag</code>, or a specific <code>commit ID</code>.</li>
  <li>🛠️ After unzipping the <code>GitHub Source Code Archive</code> (similar to cloning), remember require to run <code>git submodule update</code> to initialize submodules.</li>
  <li>🔄 Alternatively, you can install <code>scikit-plots</code> directly from the <code>GitHub Source Code Repository</code> to access the latest updates.</li>
  <li>↔️ Alternatively,
  <a href="https://pypi.org/project/scikit-plots/#files" target="_blank" rel="noopener noreferrer">
    Source Distribution (<code>.tar.gz</code>)
  </a>
  are also available for direct installation via <code>PyPI (sdist)</code>, if applicable.</li>
</ul>

<h3>
  - By
  <a href="https://pypi.org/project/scikit-plots/#files" target="_blank" rel="noopener noreferrer">
    Source Distribution (<code>.tar.gz</code>)
  </a>
  (with/without
  <a href="https://scikit-plots.github.io/dev/install/dependencies.html#build-dependencies"
     target="_blank"
     rel="noopener noreferrer">
    required build dependencies
  </a>)
</h3>

```sh
## pip install package	Installs wheel (.whl) if available, else source
## pip install --no-binary=package package  # Forces source installation only the specified package
pip install --no-binary=scikit-plots scikit-plots
```

```sh
## pip install --no-binary=:all: package  # Forces source installation for Package + all dependencies
## This forces scikit-plots and all its dependencies to be installed from source (from .tar.gz).
pip install --no-binary=:all: scikit-plots
```

<h3>
  - By
  <a href="https://github.com/scikit-plots/scikit-plots/tags" target="_blank" rel="noopener noreferrer">
    <code>GITHUB Source Code</code>:
  </a>
  (with
  <a href="https://scikit-plots.github.io/dev/install/dependencies.html#build-dependencies"
     target="_blank"
     rel="noopener noreferrer">
    required build dependencies
  </a>)
</h3>

<h4>
  - by
  <a href="https://github.com/scikit-plots/scikit-plots/tags" target="_blank" rel="noopener noreferrer">
    <code>GITHUB Source Code Archive URLs</code>:
  </a>
  ( <code>.zip</code> or <code>.tar.gz</code> )
  (with
  <a href="https://scikit-plots.github.io/dev/install/dependencies.html#build-dependencies"
     target="_blank"
     rel="noopener noreferrer">
    required build dependencies
  </a>)
</h4>

<p>
  Source code archives are available at specific URLs for each repository.
  For example, consider the repository
  <a href="https://github.com/scikit-plots/scikit-plots/tags" target="_blank" rel="noopener noreferrer">
    <code>scikit-plots/scikit-plots</code>
  </a>
  .
</p>

<h4>
  - by
  <a href="https://github.com/scikit-plots/scikit-plots" target="_blank" rel="noopener noreferrer">
    <code>GitHub Source Code Repository Cloned</code>:
  </a>
  (with
  <a href="https://scikit-plots.github.io/dev/install/dependencies.html#build-dependencies"
     target="_blank"
     rel="noopener noreferrer">
    required build dependencies
  </a>)
</h4>

```sh
## Forked repo: https://github.com/scikit-plots/scikit-plots.git
git clone https://github.com/YOUR-USER-NAME/scikit-plots.git
cd scikit-plots
```

```sh
## (Optionally) Add safe directories for git
# bash docker/script/safe_dirs.sh
git config --global --add safe.directory '*'
```

```sh
## download submodules
git submodule update --init
```

```sh
# pip install -r ./requirements/all.txt
pip install -r ./requirements/build.txt
```

```sh
## Install development version
pip install --no-cache-dir -e . -v
```

<h2>
  🧊🔧 It is also possible to include optional dependencies:
</h2>

```sh
## https://github.com/celik-muhammed/scikit-plots/tree/main/requirements
## (Optionally) Try Development [build,dev,test,doc]
## For More in Doc: https://scikit-plots.github.io/
python -m pip install --no-cache-dir --no-build-isolation -e .[build,dev,test,doc] -v
```

```sh
## https://github.com/celik-muhammed/scikit-plots/tree/main/requirements
## [cpu] refer tensorflow-cpu, transformers, tf-keras
## [gpu] refer Cupy tensorflow lib require NVIDIA CUDA support
pip install "scikit-plots[cpu]"
```

<hr>

<h2 align=center>
  <a href="https://scikit-plots.github.io/dev/auto_examples/index.html" target="_blank" rel="noopener noreferrer">
    Sample Plots
  </a>
</h2>

<div>
<!-- GitHub-flavored Markdown (GFM) does not support inline CSS or HTML layout tags -->
<!-- <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; max-width: 580px; margin: auto;"> -->
<!-- <div style="display: flex; flex-direction: column; align-items: center; gap: 1.5em; max-width: 580px; margin: auto;"> -->
<!-- <div style="max-width: 1400px; margin: auto; display: flex; flex-direction: column; align-items: center; gap: 1.5em;">
  Header Row
  <div style="display: flex; width: 100%; justify-content: space-between; text-align: center; font-weight: bold;">
    <div style="width: 49.5%;">Sample Plot 1</div>
    <div style="width: 49.5%;">Sample Plot 2</div>
  </div>
  Image Rows
  <div style="display: flex; width: 100%; justify-content: space-between;">
    <img style="width: 49.5%;" alt="plot_learning_curve.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_learning_curve-1.png">
    <img style="width: 49.5%;" alt="plot_calibration_curve.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-metrics-plot_calibration-1.png">
  </div>

  <div style="display: flex; width: 100%; justify-content: space-between;">
    <img style="width: 49.5%;" alt="plot_classifier_eval.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_classifier_eval_script_001.png">
    <img style="width: 49.5%;" alt="plot_classifier_eval.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_classifier_eval_script_002.png">
    <img style="width: 49.5%;" alt="plot_feature_importances.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_feature_importances-1.png">
  </div>

  <div style="display: flex; width: 100%; justify-content: space-between;">
    <img style="width: 49.5%;" alt="plot_roc.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_roc_script_001.png">
    <img style="width: 49.5%;" alt="plot_precision_recall.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_precision_recall_script_001.png">
  </div>

  <div style="display: flex; width: 100%; justify-content: space-between;">
    <img style="width: 49.5%;" alt="plot_pca_component_variance.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-decomposition-plot_pca_component_variance-1.png">
    <img style="width: 49.5%;" alt="plot_pca_2d_projection.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-decomposition-plot_pca_2d_projection-1.png">
  </div>

  <div style="display: flex; width: 100%; justify-content: space-between;">
    <img style="width: 49.5%;" alt="plot_elbow.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_elbow-1.png">
    <img style="width: 49.5%;" alt="plot_silhouette.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_silhouette_script_001.png">
  </div>

  <div style="display: flex; width: 100%; justify-content: space-between;">
    <img style="width: 49.5%;" alt="plot_cumulative_gain.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_cumulative_gain_script_001.png">
    <img style="width: 49.5%;" alt="plot_lift.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_lift_script_001.png">
  </div>
</div> -->


<!-- GitHub- & PyPI-Compatible Grid -->
<!-- <p align="center"><strong>Sample Plot 1</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Sample Plot 2</strong></p> -->
<table align="center">
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_feature_importances-1.png" alt="plot_feature_importances.png" width="100%">
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_classifier_eval_script_001.png" alt="plot_classifier_eval.png" width="100%">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_classifier_eval_script_002.png" alt="plot_classifier_eval.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_roc_script_001.png" alt="plot_roc.png" width="100%">
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_precision_recall_script_001.png" alt="plot_precision_recall.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-decomposition-plot_pca_component_variance-1.png" alt="plot_pca_component_variance.png" width="100%">
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-decomposition-plot_pca_2d_projection-1.png" alt="plot_pca_2d_projection.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_elbow-1.png" alt="plot_elbow.png" width="100%">
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_silhouette_script_001.png" alt="plot_silhouette.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_cumulative_gain_script_001.png" alt="plot_cumulative_gain.png" width="100%">
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_lift_script_001.png" alt="plot_lift.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center" width="45%">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_learning_curve_script_001.png" alt="plot_learning_curve.png" width="100%">
    </td>
    <td align="center" width="45%">
      <img src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_plot_calibration_script_001.png" alt="plot_calibration_curve.png" width="100%">
    </td>
  </tr>
</table>
</div>

<hr>


Scikit-plots is the result of an unartistic data scientist's dreadful realization that *visualization is one of the most crucial components in the data science process, not just a mere afterthought*.

Gaining insights is simply a lot easier when you're looking at a colored heatmap of a confusion matrix complete with class labels rather than a single-line dump of numbers enclosed in brackets. Besides, if you ever need to present your results to someone (virtually any time anybody hires you to do data science), you show them visualizations, not a bunch of numbers in Excel.

That said, there are a number of visualizations that frequently pop up in machine learning. Scikit-plots is a humble attempt to provide aesthetically-challenged programmers (such as myself) the opportunity to generate quick and beautiful graphs and plots with as little boilerplate as possible.


### Okay then, prove it. Show us an example.

Say we use [Keras Classifier](https://keras.io/api/models/sequential/) in multi-class classification and decide we want to visualize the results of a common classification metric, such as sklearn's [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) with a [confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html).

Let’s start with a basic example where we use a Keras classifier to evaluate the digits dataset provided by Scikit-learn.

```python
# Before tf {'0':'All', '1':'Warnings+', '2':'Errors+', '3':'Fatal Only'} if any
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable GPU and force TensorFlow to use CPU
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
# Set TensorFlow's logging level to Fatal
import logging; tf.get_logger().setLevel(logging.CRITICAL)
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Loading the dataset
X, y = load_digits(
  return_X_y=True,
)
# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
  X, y, test_size=0.33, random_state=0
)
# Convert labels to one-hot encoding
Y_train = tf.keras.utils.to_categorical(y_train)
Y_val = tf.keras.utils.to_categorical(y_val)
# Define a simple TensorFlow model
tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input (Functional API)
    tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
# Train the model
model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=2,
    validation_data=(X_val, Y_val),
    verbose=0
)
# Predict probabilities on the validation set
y_probas = model.predict(X_val)
# Plot the data
import matplotlib.pyplot as plt
import scikitplot as sp
# sp.get_logger().setLevel(sp.sp_logging.WARNING)
sp.logger.setLevel(sp.logger.INFO)  # default WARNING
# Plot precision-recall curves
sp.metrics.plot_precision_recall(
  y_val, y_probas,
)
```

<div align=center>
  <img style="display:block;width:60%;height:auto;align:center;" alt="quick_start"
    src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/sphx_glr_quick_start_tf_001.png">
</div>

Pretty.


#### Maximum flexibility. Compatibility with non-scikit-learn objects.

Although Scikit-plot is loosely based around the scikit-learn interface, you don't actually need scikit-learn objects to use the available functions.
As long as you provide the functions what they're asking for, they'll happily draw the plots for you.

The possibilities are endless.


## Release Notes

See the [changelog](https://scikit-plots.github.io/dev/whats_new/index.html)
for a history of notable changes to scikit-plots.


## Contributing to Scikit-plots

**Reporting a bug? Suggesting a feature? Want to add your own plot to the library? Visit our.**

<!--
<a href="https://docs.astropy.org/en/latest/impact_health.html" target="_blank" rel="noopener noreferrer">
<img style="display:auto;width:auto;height:auto;" alt="User Statistics" src="https://github.com/astropy/repo_stats/blob/cache/cache/astropy_user_stats_light.png">
</a> -->

The Scikit-plots Project is made both by and for its users, so we welcome and
encourage contributions of many kinds. Our goal is to keep this a positive,
inclusive, successful, and growing community that abides by the
[Scikit-plots Community Code of Conduct](https://scikit-plots.github.io/dev/project/code_of_conduct.html).

For guidance on contributing to or submitting feedback for the Scikit-plots Project,
see the [contributions page](https://scikit-plots.github.io/dev/devel/index.html).
For contributing code specifically, the developer docs have a
[guide](https://scikit-plots.github.io/dev/devel/index.html) with a `quickstart`.
There's also a [summary of contribution guidelines](https://github.com/scikit-plots/scikit-plots/blob/main/CONTRIBUTING.md).


## Developing with Codespaces

GitHub Codespaces is a cloud development environment using Visual Studio Code
in your browser. This is a convenient way to start developing Scikit-plots, using
our [dev container](https://github.com/scikit-plots/scikit-plots/blob/main/.devcontainer/notebook_cpu/devcontainer.json) configured
with the required packages. For help, see the [GitHub Codespaces docs](https://docs.github.com/en/codespaces).

<div align=center>
  <a href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=889608023&skip_quickstart=true&machine=basicLinux32gb&devcontainer_path=.devcontainer%2Fnotebook_cpu%2Fdevcontainer.json&geo=EuropeWest" target="_blank" rel="noopener noreferrer">
  <img style="display:auto;width:auto;height:auto;" alt="Open in GitHub Codespaces" src="https://github.com/codespaces/badge.svg">
  </a><br>
</div>


## Acknowledging (Governance) and Citing Scikit-plots

See the [Acknowledgement](https://scikit-plots.github.io/dev/project/governance.html),
[Citation Guide](https://scikit-plots.github.io/dev/project/citing.html)
and the [CITATION.bib](https://github.com/scikit-plots/scikit-plots/blob/main/CITATION.bib),
[CITATION.cff](https://github.com/scikit-plots/scikit-plots/blob/main/CITATION.cff) file.

1. scikit-plots, “scikit-plots: vlatest”. Zenodo, Aug. 23, 2024.
   DOI: [10.5281/zenodo.13367000](https://doi.org/10.5281/zenodo.13367000).

2. scikit-plots, “scikit-plots: v0.3.8dev0”. Zenodo, Aug. 23, 2024.
   DOI: [10.5281/zenodo.13367001](https://doi.org/10.5281/zenodo.13367001).


## Supporting the Project (Upcoming)

<a href="https://numfocus.org" target="_blank" rel="noopener noreferrer">
<img style="display:auto;width:auto;height:auto;" alt="Powered by NumFOCUS" src="https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A">
</a>
<a href="https://numfocus.org/donate-to-scikit-plots" target="_blank" rel="noopener noreferrer">
<img style="display:auto;width:auto;height:auto;" alt="Donate" src="https://img.shields.io/badge/Donate-to%20Scikit-plots-brightgreen.svg">
</a>

NumFOCUS, a 501(c)(3) nonprofit in the United States.

<!-- The Scikit-plots Project is sponsored by NumFOCUS, a 501(c)(3) nonprofit in the
United States. You can donate to the project by using the link above, and this
donation will support our mission to promote sustainable, high-level code base
for the astronomy community, open code development, educational materials, and
reproducible scientific research. -->


## License

Scikit-plots is licensed under a 3-clause BSD style license - see the
[LICENSE](https://github.com/scikit-plots/scikit-plots/blob/main/LICENSE) file,
and [LICENSES](https://github.com/scikit-plots/scikit-plots/tree/main/LICENSES) files.
