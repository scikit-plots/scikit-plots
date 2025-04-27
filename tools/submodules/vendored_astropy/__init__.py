"""
Astropy is a package intended to contain core functionality and some
common tools needed for performing astronomy and astrophysics research with
Python. It also provides an index for other astronomy packages and tools for
managing them.

Documentation is available in the docstrings and
online at https://www.astropy.org/.
"""

# scikitplot/_astropy/__init__.py
# https://docs.astropy.org/en/latest/index.html

#######################################################################

# Define the astropy version
__version__ = "7.1.dev876+ge06963e7b"
__author__ = "Astropy Developers"
__author_email__ = "astropy.team@gmail.com"

# Define the astropy git hash
from .._build_utils.gitversion import git_remote_version

__git_hash__ = git_remote_version(url="https://github.com/scikit-plots/astropy")[0]
del git_remote_version
