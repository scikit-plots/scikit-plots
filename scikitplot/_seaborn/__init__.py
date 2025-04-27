"""
seaborn: statistical data visualization

Seaborn is a Python visualization library based on matplotlib.
It provides a high-level interface for drawing attractive statistical graphics.

Documentation is available in the docstrings and
online at https://seaborn.pydata.org/.
"""

# scikitplot/_seaborn/__init__.py
# https://github.com/mwaskom/seaborn/blob/master/seaborn/__init__.py

# Import seaborn objects
from .rcmod import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .palettes import *  # noqa: F401,F403
from .relational import *  # noqa: F401,F403
from .regression import *  # noqa: F401,F403
from .categorical import *  # noqa: F401,F403
from .distributions import *  # noqa: F401,F403
from .matrix import *  # noqa: F401,F403
from .miscplot import *  # noqa: F401,F403
from .axisgrid import *  # noqa: F401,F403
from .widgets import *  # noqa: F401,F403
from .colors import xkcd_rgb, crayons  # noqa: F401
from . import cm  # noqa: F401

# Capture the original matplotlib rcParams
import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

# Define the seaborn version
__version__ = "0.14.0.dev0"

#######################################################################

del mpl

__author__ = "Michael Waskom"
__author_email__ = "mwaskom@gmail.com"

# Define the seaborn git hash
from .._build_utils.gitversion import git_remote_version

__git_hash__ = git_remote_version(url="https://github.com/scikit-plots/seaborn")[0]
del git_remote_version
