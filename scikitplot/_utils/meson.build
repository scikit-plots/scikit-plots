## scikit-plots/scikitplot/_utils/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# The Cython files that need to be processed, including dependencies and includes
# `_cython_tree` should have been defined earlier in the build script
# to make sure scikitplot/__init__.py is copied
experimental_cython_tree = [
  _root_cython_tree,                      # also _cython_tree
  fs.copyfile('__init__.py'),             # Ensure __init__.py is copied early
  fs.copyfile('_c_internal_utils.pyi'),    # Cython definition file for C/C++ code
]

######################################################################
##
######################################################################
