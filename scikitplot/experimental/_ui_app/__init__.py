# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
An interactive tool for exploring Machine Learning Exploratory Data Analysis (ML-EDA).

Modules:
- build_ui_app.py : PyInstaller automation to build executables.

Usage:
- `python -m build_ui_app`  : Generate executable (requires PyInstaller).
"""  # noqa: D205

from scikitplot import logger

logger.setLevel(logger.INFO)
del logger
