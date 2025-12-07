## scikitplot/annoy/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_annoy_cython_tree = [
  _root_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
  fs.copyfile('__init__.pyi'),  # Cython definition file for C/C++ code
  # fs.copyfile('annoylib.pyi'),
  fs.copyfile('_base.pyi'),
]
