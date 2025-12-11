## scikitplot/nc/_version/meson.build

######################################################################
## cython tree
######################################################################

# Copy main "__init__.py"/"*.pxd" files into build dir (for Cython)
_version_cython_tree = [
  _nc_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
  fs.copyfile('_version.pxd'),
  fs.copyfile('_version.pxi'),
  fs.copyfile('_version.pyi'),
]

######################################################################
## Include the headers directories
######################################################################

_version_inc_dir = [
  # 'include',  # .hpp headers
  'src',  # .cpp pybind11 bindings files
]

######################################################################
## Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
inc_dir_version = include_directories(_version_inc_dir)

######################################################################
## Extension Module Metadata
######################################################################

## Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
## .c → plain C
## .cpp or .cc → C++
## .cxx or .C → also C++ (less common)
## Define Python Module Name
## Check: ELF 64-bit LSB shared object, dynamically linked
_version_extension_metadata = {
  # file builddir/scikitplot/nc/_version/_version.so
  '_version':  ## Build static NumCpp core library (C++)
  {
    'sources': [                                  # .pyx source file with NumCpp + Cython
      # gen_cython_cpp.process('_version.pyx'),   #  manually call a generator
      '_version.pyx'
    ],
    'include_directories': [                      # Include dirs for compilation
      # ⚠️ May Conflict develop first!
      # inc_dir_numcpp_develop,                   # develop first
      inc_dir_numcpp,                             # then the stable include folder
      inc_dir_version,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'link_args': [],                              # 👈 add this
    'override_options': [
      # 'cython_language=c',                        # Ensure Cython knows to generate C code
      'cython_language=cpp',                      # Ensure Cython knows to generate C code
      'optimization=3',                           # Optimization level '-O3'
    ],
    'cython_args': cython_cpp_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': [                                 # Additional C/C++ arguments
      # '-include', 'develop/NdArray/NdArrayCore.hpp',  # optional force include
      cython_cpp_flags,
      extra_compile_args,
    ],
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/nc/_version',           # Path where the module is located
  },
  # file builddir/scikitplot/nc/_version/__version.so
  '__version':  ## Build static NumCpp core library (C++)
  {
    # Only module your actual C++/Python/Pybind11 binding sources
    'sources': [                                  # C++ source file with Python/Pybind11 bindings entrypoint
      'src/module_version.cpp',
    ],
    'include_directories': [                      # Include dirs for compilation
      # ⚠️ May Conflict develop first!
      # inc_dir_numcpp_develop,                   # develop first
      inc_dir_numcpp,                             # then the stable include folder
      inc_dir_version,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'link_args': [                                # 👈 add this
      extra_link_args
    ],
    'override_options': [
      # 'cython_language=c',                        # Ensure Cython knows to generate C code
      'cython_language=cpp',                      # Ensure Cython knows to generate C code
      'optimization=3',                           # Optimization level '-O3'
    ],
    'cython_args': cython_cpp_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': [                                 # Additional C/C++ arguments
      # '-include', 'develop/NdArray/NdArrayCore.hpp',  # optional force include
      cython_cpp_flags,
      extra_compile_args,
    ],
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/nc/_version',           # Path where the module is located
  },
}
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : _version_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + _version_cython_tree,                # Sources and dependencies
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
## Notes:
## - Headers are installed into the Python site-packages tree, allowing
##   runtime Cython/Pybind11 extensions to include <NumCpp/...>.
## - If you use this as a Meson subproject, `dep_numcpp` can be imported
##   and reused for other modules.
## - Check your include chain if you see missing symbols like `size_` or
##   `endianess_` — these usually mean that only partial headers were
##   copied or the wrong version of NumCpp was vendored.
######################################################################

######################################################################
##
######################################################################
