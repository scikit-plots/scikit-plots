"""
seaborn: statistical data visualization

Seaborn is a Python visualization library based on matplotlib.
It provides a high-level interface for drawing attractive statistical graphics.

Documentation is available in the docstrings and
online at https://seaborn.pydata.org/.
"""

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
__version__ = "0.14.0.dev0"