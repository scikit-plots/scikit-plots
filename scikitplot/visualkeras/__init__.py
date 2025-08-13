# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the visualkeras project.
# https://github.com/paulgavrikov/visualkeras

# Your package/module initialization code goes here
# pip install tensorflow will also install the corresponding version of Keras
# pip install tf-keras keras Use If not compatibility
from .._testing._pytesttester import PytestTester  # Pytest testing
from .graph import *
from .layered import *
from .layer_utils import SpacingDummyLayer

test = PytestTester(__name__)
del PytestTester

# Define the visualkeras version
# https://github.com/paulgavrikov/visualkeras/blob/master/setup.py
__version__ = "0.1.4"
__author__ = "Paul Gavrikov"
__author_email__ = "paul.gavrikov@hs-offenburg.de"

# Define the visualkeras git hash
# from .._build_utils.gitversion import git_remote_version

# __git_hash__ = git_remote_version(url="https://github.com/scikit-plots/visualkeras")[0]
__git_hash__ = "8d42f3a9128373eac7b4d38c23a17edc9357e3c9"
# del git_remote_version

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    "SpacingDummyLayer",
    "graph",
    "graph_view",
    "layer_utils",
    "layered",
    "layered_view",
    "test",
    "utils",
]
