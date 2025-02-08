## scikit-plots/scikitplot/_xp_core_lib/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# The Cython files that need to be processed, including dependencies and includes
# `_cython_tree` should have been defined earlier in the build script
# to make sure scikitplot/__init__.py is copied
_xp_core_lib_cython_tree = [
  _root_cython_tree,                              # also _cython_tree
  fs.copyfile('__init__.py'),                     # Ensure __init__.py is copied early
  fs.copyfile('_ccallback.pxd'),
  fs.copyfile('_ccallback_c.pxd'),
  fs.copyfile('_test_deprecation_def.pxd'),       # Cython definition file for C/C++ code
  fs.copyfile('messagestream.pxd'),
  # submodule stub
  fs.copyfile('array_api_compat.pyi'),
  fs.copyfile('array_api_extra.pyi'),
]

# _lib_pxd = [
#   fs.copyfile('__init__.py'),
#   fs.copyfile('_ccallback_c.pxd'),
#   fs.copyfile('ccallback.pxd'),
#   fs.copyfile('messagestream.pxd'),
# ]
#
# # Cython pyx -> c generator with _lib_pxd dependency
# lib_cython_gen = generator(cython,
#   arguments : cython_args,
#   output : '@BASENAME@.c',
#   depends : [_cython_tree, _lib_pxd])

dep_xp_core_lib = declare_dependency(include_directories: ['..', 'src'])

######################################################################
## create the corresponding module
######################################################################

py.extension_module('_test_deprecation_call',
  # cython_gen.process('_test_deprecation_call.pyx'),
  ['_test_deprecation_call.pyx'] + _xp_core_lib_cython_tree,
  include_directories: 'src',
  link_args: version_link_args,  # from root
  cython_args: cython_c_args,
  c_args: cython_c_flags,
  install: true,
  subdir: 'scikitplot/_xp_core_lib',
  install_tag: 'tests'
)

py.extension_module('_test_deprecation_def',
  # cython_gen.process('_test_deprecation_def.pyx'),
  ['_test_deprecation_def.pyx'] + _xp_core_lib_cython_tree,
  include_directories: 'src',
  link_args: version_link_args,  # from root
  cython_args: cython_c_args,
  c_args: cython_c_flags,
  install: true,
  subdir: 'scikitplot/_xp_core_lib',
  install_tag: 'tests'
)

######################################################################
## create the corresponding module
######################################################################

py.extension_module('_ccallback_c',
  # lib_cython_gen.process('_ccallback_c.pyx'),
  ['_ccallback_c.pyx'] + _xp_core_lib_cython_tree,
  include_directories: 'src',
  link_args: version_link_args,  # from root
  dependencies: dep_xp_core_lib,
  cython_args: cython_c_args,
  # disables warnings about discarded qualifiers (e.g., discarding const, volatile, or restrict qualifiers in function arguments or variables).
  c_args: cython_c_flags,
  install: true,
  subdir: 'scikitplot/_xp_core_lib',
  # install_tag: 'tests'
)

py.extension_module('_test_ccallback',
  ['src/_test_ccallback.c'],
  include_directories: 'src',
  link_args: version_link_args,  # from root
  dependencies: dep_xp_core_lib,
  install: true,
  subdir: 'scikitplot/_xp_core_lib',
  # install_tag: 'tests'
)

######################################################################
## create the corresponding module
######################################################################
