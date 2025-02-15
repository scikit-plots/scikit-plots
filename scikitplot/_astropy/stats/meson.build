## scikit-plots/scikitplot/_astropy/stats/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# The Cython files that need to be processed, including dependencies and includes
stats_cython_tree = [
  # `_cython_tree` should have been defined earlier in the build script
  # to make sure scikitplot/__init__.py is copied
  _astropy_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
  fs.copyfile('_stats.pxd'),   # Cython definition file for C/C++ code
]

######################################################################
## Include the headers directories
######################################################################

# include_directories for Specific Subfolders
stats_inc_dir = [
  # 'include',
  'src',
]

######################################################################
## Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_stats = include_directories(stats_inc_dir)

######################################################################
## Define dependencies and Compiler Flags
######################################################################

# Define dependencies
dep_list = [dep_np]

######################################################################
## extension metadata
######################################################################

# Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
stats_extension_metadata = {
  # Define Python Module Name
  '_fast_sigma_clip':
  {
    'sources': [                                  # C++ source file with Python/Pybind11 bindings
      'src/compute_bounds.c',
      'src/fast_sigma_clip.c',
      'src/wirth_select.c',
    ],
    'include_directories': [                      # Include dirs for compilation
      inc_dir_stats,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'override_options': [
      'cython_language=c',                        # Ensure Cython knows to generate C code
      'optimization=3',                           # Optimization level '-O3'
    ],
    'cython_args': cython_c_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    # 'cpp_args': cython_cpp_flags,                 # Additional C/C++ arguments
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/_astropy/stats',        # Path where the module is located
  },
  # Define Python Module Name
  '_stats':
  {
    'sources': [                                  # .pyx source file with Cython bindings
      # gen_cython_cpp.process('_stats.pyx'),     #  manually call a generator, if Needed
      '_stats.pyx'
    ],
    'include_directories': [                      # Include dirs for compilation
      inc_dir_stats,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'override_options': [
      'cython_language=c',                        # Ensure Cython knows to generate C code
      'optimization=3'                            # Optimization level '-O3'
    ],
    'cython_args': cython_c_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    # 'cpp_args': cython_cpp_flags,                 # Additional C/C++ arguments
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/_astropy/stats',        # Path where the module is located
  },
}
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : stats_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + stats_cython_tree,                   # Sources and dependencies
    include_directories: ext_dict.get('include_directories', []),  # Include directories
    dependencies: ext_dict.get('dependencies', []),                # Additional dependencies if any
    link_with: ext_dict.get('link_with', []),                      # Libraries to link with
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
## Load submodule files
######################################################################

## Load submodule files
# install_subdir(
#   '../',
#   install_dir: py.get_install_dir() / 'scikitplot/_astropy',
#   install_tag: 'python-runtime',
# )

######################################################################
##
######################################################################
