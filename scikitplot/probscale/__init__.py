"""
Real probability scales for matplotlib.
"""

# Copied module from matplotlib:
# https://github.com/matplotlib/mpl-probscale

# scikitplot/probscale/__init__.py

from matplotlib import scale

from .probscale import ProbScale
from .viz import *

scale.register_scale(ProbScale)
del scale

# Define the probscale version
# https://github.com/matplotlib/mpl-probscale/blob/master/probscale/__init__.py
__version__ = "0.2.6dev"
__author__ = "Paul Hobson (Herrera Environmental Consultants)"
__author_email__ = "phobson@herrerainc.com"

# Define the probscale git hash
from .._build_utils.gitversion import git_remote_version

# __git_hash__ = git_remote_version(url="https://github.com/scikit-plots/mpl-probscale")[0]
__git_hash__ = "be697c65ecaa223032ad2f7364ef350d684f73c0"
del git_remote_version
