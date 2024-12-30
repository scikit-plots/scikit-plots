"""
seaborn: statistical data visualization

Seaborn is a Python visualization library based on matplotlib.
It provides a high-level interface for drawing attractive statistical graphics.

Documentation is available in the docstrings and
online at https://seaborn.pydata.org/.
"""
# scikitplot/_seaborn/__init__.py

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
_orig_rc_params = mpl.rcParams.copy(); del mpl;

# Define the seaborn version
# https://github.com/mwaskom/seaborn/blob/master/seaborn/__init__.py
__version__ = '0.14.0.dev0'

# Define the seaborn git hash
# scikitplot._build_utils.gitversion.git_remote_version(url='https://github.com/mwaskom/seaborn')[0]
__git_hash__ = '385e54676ca16d0132434bc9df6bc41ea8b2a0d4'