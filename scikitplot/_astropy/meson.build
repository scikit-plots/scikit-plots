## scikit-plots/scikitplot/_astropy/meson.build

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_astropy_cython_tree = [
  _root_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
]

######################################################################
## fix_submodule_import (run_command, run_target)
# https://mesonbuild.com/Reference-manual_functions.html#run_command
# https://mesonbuild.com/Reference-manual_functions.html#run_target
######################################################################

# r = run_command(
#   py,
#   meson.current_source_dir() / '../_build_utils/fix_submodule_import.py',
#   '-m', 'astropy', '-r', meson.current_source_dir(), '--fix-type-hints',
#   # env: {'FOO': 'bar'},
#   check: true,
# )
# if r.returncode() != 0
#     error('Copying failed: ' + r.stderr().strip())
# endif

######################################################################
## Next
######################################################################

subdir('stats')
