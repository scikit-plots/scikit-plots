## scikit-plots/scikitplot/experimental/meson.build

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
  fs.copyfile('_complexstuff.pxd'),       # Cython definition file for C/C++ code
  fs.copyfile('_cy_experimental.pxd'),    # Cython definition file for C/C++ code
  fs.copyfile('_cy_experimental.pyi'),    # Cython definition file for C/C++ code
]

######################################################################
## Include the headers directories
######################################################################

# include_directories for Specific Subfolders
experimental_inc_dir = [
  'src',
  'src/complexstuff',
  'include',
]

######################################################################
## Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_experimental = include_directories(experimental_inc_dir)

######################################################################
## library, If Neeed
######################################################################

# # Static library with C++ source file implementing bindings
# _st_lib_experimental = static_library('_st_lib_experimental', [],
#   include_directories: inc_dir_experimental,
#   dependencies: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'inlineshidden',
#   link_with: [],
#   link_args: [],
#   # install: true,
#   # install_dir: 'lightnumpy/_core',
# )
# # Shared (dynamic) library with C++ source file implementing bindings
# _dyn_lib_experimental = library('_dyn_lib_experimental', [],
#   include_directories: inc_dir_experimental,
#   dependencies: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'hidden',
#   gnu_symbol_visibility: 'inlineshidden',
#   # link_with: [_st_lib_experimental],  # Link with the static library
#   # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
#   install: true,
#   install_dir: 'lightnumpy/_core',
# )

######################################################################
## extension metadata
######################################################################

# Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
experimental_extension_metadata = {
  # Define Python Module Name
  '_cy_experimental':
  {
    # Ceyton bindings source .pyx extension with C/C++ support
    'sources': [
      # gen_cython_cpp.process('_cy_experimental.pyx'),     #  manually call a generator, if Needed
      '_cy_experimental.pyx'
    ],
    'dependencies': [
      dep_np,
    ],                                            # External libraries and dependencies
    'include_directories': [
      inc_dir_experimental,
      inc_dir_np,
    ],                                            # Include dirs for compilation
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'override_options': [
      'cython_language=cpp',                      # Ensure Cython knows to generate C++ code
      'optimization=3',                           # Optimization level '-O3'
    ],
    # 'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': cython_cpp_flags,
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/experimental',          # Path where the module is located
  },
  # Define Python Module Name
  '_py_experimental':
  {
    # Pybind11 bindings source .cpp extension with C/C++ support
    'sources': ['bindings/_py_experimental.cpp'],
    'dependencies': [
      dep_pybind11,
      dep_np,
    ],                                            # External libraries and dependencies
    'include_directories': [
      inc_dir_experimental,
      inc_dir_np,
    ],                                            # Include dirs for compilation
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'override_options': [
      'cython_language=cpp',                      # Ensure Cython knows to generate C++ code
      'optimization=3',                           # Optimization level '-O3'
    ],
    # 'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': cython_cpp_flags,
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/experimental',          # Path where the module is located
  },
}
######################################################################
## create the corresponding module
######################################################################
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : experimental_extension_metadata
  py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + experimental_cython_tree,            # Sources and dependencies
    include_directories: ext_dict.get('include_directories', []),  # Include directories
    link_with: ext_dict.get('link_with', []),                      # Libraries to link with
    dependencies: ext_dict.get('dependencies', []),                # Additional dependencies if any
    override_options : ext_dict.get('override_options', []),       # Options to override defaults
    cython_args: ext_dict.get('cython_args', []),                  # Use Cython specific arguments if any
    c_args: ext_dict.get('c_args', []),                            # Additional C compilation arguments
    cpp_args: ext_dict.get('cpp_args', []),                        # Additional C++ compilation arguments
    install: ext_dict.get('install', true),                        # Install the .so file this extension module
    subdir: ext_dict.get('subdir', '.'),                           # Subdirectory where the .so file module will be placed
    # install_dir: ext_dict.get('install_dir', '.'),                 # Subdirectory where the .so file module will be installed
  )
endforeach

######################################################################
##
######################################################################
