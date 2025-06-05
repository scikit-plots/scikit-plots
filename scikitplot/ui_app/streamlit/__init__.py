"""
snsx_explorer package.

An interactive tool for exploring Machine Learning Exploratory Data Analysis (ML-EDA)
and interpretability plots from the `snsx` library using a Streamlit-based UI.

Modules:
- build_app.py          : PyInstaller automation to build executables.
- catalog.py            : Central registry of ML-EDA plotting functions with metadata.
- run_app.py            : CLI entrypoint to launch the Streamlit app.
- streamlit_app.py      : Streamlit-based UI logic for exploring the plotting catalog.

Usage:
- `python -m build_app` : Generate executable (requires PyInstaller).
- `python -m run_app`   : Launch the UI.

This package is designed for both developers and end-users to browse, run, and
eventually visualize plotting functions dynamically.
"""

# __init__.py
from .catalog import snsx_catalog  # noqa: F401
from .template_st_dataset_loader_ui import get_sns_data  # noqa: F401
