"""
scikit-plots Functional API
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## scikit-plots Functional API modules
######################################################################

# Modules
from . import (
  decomposition,
  estimators,
  metrics,
)
# Deprecated namespaces, to be removed in v0.5.0
from .metrics import plotters  # noqa: F401,F403