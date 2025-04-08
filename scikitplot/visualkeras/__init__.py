# Your package/module initialization code goes here
# pip install tensorflow will also install the corresponding version of Keras
# pip install tf-keras keras Use If not compatibility
from .graph import *
from .layered import *
from .layer_utils import SpacingDummyLayer as SpacingDummyLayer

# Define the visualkeras version
# https://github.com/paulgavrikov/visualkeras/blob/master/setup.py
__version__ = "0.1.4"
__author__ = "Paul Gavrikov"
__author_email__ = "paul.gavrikov@hs-offenburg.de"

# Define the visualkeras git hash
# scikitplot._build_utils.gitversion.git_remote_version(url='https://github.com/paulgavrikov/visualkeras')[0]
__git_hash__ = "8d42f3a9128373eac7b4d38c23a17edc9357e3c9"
