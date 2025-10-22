"""
Utility functions to reset global states for common ML and plotting libraries.

This module is useful when building galleries (e.g., with Sphinx-Gallery),
running tests, or maintaining consistent environments across scripts,
notebooks, and web apps (e.g., Streamlit, Dash).
"""

import numpy as np

__all__ = []

# ----------------------
# Numpy: capture global config
# ----------------------

_numpy_err_config = np.geterr()
_numpy_print_config = np.get_printoptions()
