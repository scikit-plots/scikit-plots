"""
MLflow
==========================================

.. currentmodule:: scikitplot.mlflow

An example showing the :py:mod:`~scikitplot.mlflow` submodule..
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause


# %%
# mlflow workflow helper
# ----------------------
# Adds a **project-level configuration** mechanism so multiple scripts
# (e.g., `train.py`, `hpo.py`, `predict.py`) share the exact same MLflow settings,
# regardless of current working directory.

# !pip install mlflow pyyaml
import scikitplot as sp

print(sp.mlflow.workflow.__doc__)

# %%

print(sp.mlflow.DEFAULT_PROJECT_MARKERS)

# %%
# Environment (best for CI)
# -------------------------
# Default marker file-folder for auto detection
# Walk upward from `start` until a directory containing any marker is found.

import os

# export SCIKITPLOT_PROJECT_MARKERS='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
os.environ["SCIKITPLOT_PROJECT_MARKERS"]='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'

# Check ROOT or base_dir is requested
sp.mlflow.find_project_root()

# %%
# 💡 Quiskstart Template: Beginner workflow demo
# -----------------------------------------------
# Demo save config from default settings then customize.

sp.mlflow.workflow(
    profile="local",
    open_ui_seconds=5,
    experiment_name="my-first-project",  # "scikitplot-project"
    fmt="toml",
    overwrite=True,  # Config already exists: ./configs/mlflow.toml (use overwrite=True).
)

# %%
# 🛠️ How to use customized settings?
# ----------------------------------

import time
import scikitplot as sp


ROOT = sp.mlflow.find_project_root(config_path=None)

with sp.mlflow.session_from_file(ROOT / "configs/mlflow.toml", profile="local") as mlflow:
    with mlflow.start_run():  # default_run_name + default tags apply automatically
        mlflow.log_param("phase", "train")

    # Overwrite new profile, If Needed
    sp.mlflow.dump_project_config_yaml(source_config_path=None)


# ROOT = sp.mlflow.find_project_root(config_path=None)

with sp.mlflow.session_from_file(ROOT / "configs/mlflow.yaml", profile="local") as mlflow:
    print("Open MLflow UI:", mlflow.ui_url)
    # do something
    time.sleep(5)


with sp.mlflow.session_from_file(ROOT / "configs/mlflow.yaml", profile="local") as mlflow:
    with mlflow.start_run(run_name="predict"):  # override default name if you want
        mlflow.log_param("phase", "predict")


# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model building
#    plot-type: mlflow
#    domain: mlflow
#    level: beginner
#    purpose: showcase
