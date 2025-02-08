import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath("."))

# Get the base directory of the current package
base_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure the base directory is not already in sys.path
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
