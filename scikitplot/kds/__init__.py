"""
Quick report for business analysis

The :py:mod:`~scikitplot.kds` KeyToDataScience module to Plot Decile Table, Lift, Gain
and KS Statistic charts with single line functions

Just input 'labels' and 'probabilities' to get quick report for analysis

kds is the result of a data scientist's humble effort to provide an easy way of
visualizing metrics. So that one can focus on the analysis rather than hassling
with copy/paste of various visialization functions.
"""

# scikitplot/kds/__init__.py

# Your package/module initialization code goes here
from ._kds import *

# Define the kds version
# https://github.com/tensorbored/kds/blob/master/setup.py
__version__ = "0.1.3"
__author__ = "Prateek Sharma"
__author_email__ = "s.prateek3080@gmail.com"

# Define the visualkeras git hash
# scikitplot._build_utils.gitversion.git_remote_version(url='https://github.com/tensorbored/kds')[0]
__git_hash__ = "18a2e90872f0dae8bb92a2eb13f637eeaa196fc4"
