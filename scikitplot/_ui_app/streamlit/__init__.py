# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
An interactive tool for exploring Machine Learning Exploratory Data Analysis (ML-EDA)
and interpretability plots from the `snsx` library using a Streamlit-based UI.

Modules:
- run_ui_app_st.py : CLI entrypoint to launch the Streamlit app.

Usage:
- `python -m run_ui_app_st` : Launch the UI.

This package is designed for both developers and end-users to browse, run, and
eventually visualize plotting functions dynamically.
"""  # noqa: D205

# __init__.py
from .snsx_catalog import snsx_catalog  # noqa: F401
from .template_ui_app_st_dataset_loader import get_sns_data  # noqa: F401
