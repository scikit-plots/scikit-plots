"""
This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
from . import (
    # model selection
    estimators,
    metrics,
    decomposition,
    cluster,
    # model analysis
    deciles,
    kds,
    modelplotpy,
)
from .utils import *  # noqa: F401,F403
from .utils.rcmod import *  # noqa: F401,F403
from .utils._show_versions import show_versions  # noqa: F401,F403

from scikitplot.classifiers import classifier_factory  # noqa: F401,F403
from scikitplot.clustering import clustering_factory  # noqa: F401,F403

# Capture the original matplotlib rcParams
import matplotlib as mpl
_orig_rc_params = mpl.rcParams.copy()


# __all__ = ['estimators']


# Define the scikitplot version
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
# https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers
__version__ = '0.3.9.rc3'