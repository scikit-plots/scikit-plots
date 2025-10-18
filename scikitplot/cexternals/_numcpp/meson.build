## scikitplot/cexternals/_numcpp/meson.build
######################################################################
## NumCpp: C++ Header-Only Library + Python/Cython Integration
######################################################################

######################################################################
## Install Headers (Optional but Recommended for Python Integration)
######################################################################

# Resolve absolute include path robustly
_numcpp_root = meson.current_source_dir()
_numcpp_include_path = _numcpp_root / 'include'

# Safety check — prevent accidental directory traversal
if not fs.exists(_numcpp_include_path)
  error('❌ NumCpp include path not found: ' + _numcpp_include_path)
elif _numcpp_include_path.contains('..')
  error('❌ Unsafe include path traversal detected: ' + _numcpp_include_path)
endif

# Install all public headers under site-packages
# install_subdir(
#   'include',
#   install_dir: py.get_install_dir() / 'scikitplot/cexternals/_numcpp',
# )

######################################################################
## cython tree
######################################################################

# Copy main "__init__.py"/"*.pxd" files into build dir (for Cython)
_numcpp_cython_tree = [
  _root_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
]

######################################################################
## Include the headers directories
######################################################################

# include_directories for Specific Subfolders (header-only)
_ndarray_inc_dir = [
  'develop/NdArray',
]
_numcpp_inc_dir = [
  'include',
  # 'src',
]

######################################################################
## Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
# ⚠️ Make sure your develop headers take priority:
inc_dir_ndarray = include_directories(_ndarray_inc_dir)
inc_dir_numcpp = include_directories(_numcpp_inc_dir)

######################################################################
## Compiler Flags and Dependencies
######################################################################

extra_compile_args = [
  # Optional Boost Dependency
  # dpkg -s libboost-dev | grep 'Version'
  # Disable all NumCpp features that require Boost
  '-DNUMCPP_NO_USE_BOOST',        # Disable Boost features
  # Optional tbb Dependency
  # '-DNUMCPP_USE_MULTITHREAD',     # Enable multithreading support
  '-fpermissive',
]

# Optionally export for parent meson.build if used as subproject
dep_numcpp = declare_dependency(
  include_directories: inc_dir_numcpp,
  compile_args: extra_compile_args,
)

######################################################################
## Define dependencies and Compiler Flags
# meson introspect ./build/cp311/ --dependencies
# meson introspect ./build/cp311/ --buildoptions | grep python
######################################################################

# Define dependencies
dep_list = [
  dep_all,
  dep_numcpp,
]

######################################################################
## Optional: Build Standalone NumCpp Demo (C++ Executable)
######################################################################
# # Executable using the NumCpp headers (header-only library)
# message('🚀 Building NumCpp demo executable...')
# exe_numcpp_demo = executable(
#   'numcpp_demo',
#   'develop/main.cpp',  # only for standalone C++ demo file, not the library.
#   include_directories: [inc_dir_ndarray, inc_dir_numcpp],
#   dependencies: dep_list,
#   cpp_args: [cython_cpp_flags, extra_compile_args],
#   install: false,
# )
# # Optionally add a test run for CI
# test('NumCpp Demo Run', exe_numcpp_demo)

######################################################################
## Extension Module Metadata
######################################################################

# numcpp → core NumCpp Python module.
# ndarray → NdArray bindings module (if needed).

## Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
## .c → plain C
## .cpp or .cc → C++
## .cxx or .C → also C++ (less common)
_numcpp_extension_metadata = {
  ## Define Python Module Name
  ## Check: ELF 64-bit LSB shared object, dynamically linked
  # file builddir/scikitplot/cexternals/_numcpp/nc.so
  'nc':  ## Build static NumCpp core library (C++)
  {
    # Only module your actual C++/Python/Pybind11 binding sources
    'sources': [                                  # C++ source file with Python/Pybind11 bindings entrypoint
      'src/nc.cpp',
    ],
    'include_directories': [                      # Include dirs for compilation
      inc_dir_numcpp,   # then the stable include folder
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'override_options': [
      # 'cython_language=c',                        # Ensure Cython knows to generate C code
      'cython_language=cpp',                        # Ensure Cython knows to generate C code
      'optimization=3',                           # Optimization level '-O3'
    ],
    'cython_args': cython_cpp_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': [                                 # Additional C/C++ arguments
      cython_cpp_flags,
      extra_compile_args,
    ],
    'link_args': [],                              # 👈 add this
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/cexternals/_numcpp',     # Path where the module is located
  },
  # file builddir/scikitplot/cexternals/_numcpp/nc_develop.so
  'nc_develop':  ## Build static NumCpp core library (C++)
  {
    # Only your actual C++/Pybind11 binding sources
    'sources': [                                  # C++ source file with Python/Pybind11 bindings entrypoint
      'src/nc_develop.cpp',
    ],
    'include_directories': [                      # Include dirs for compilation
      # ⚠️ develop first!
      inc_dir_ndarray,  # develop first
      # inc_dir_numcpp,   # then the stable include folder
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
    ],
    'override_options': [
      # 'cython_language=c',                        # Ensure Cython knows to generate C code
      'cython_language=cpp',                        # Ensure Cython knows to generate C code
      'optimization=3',                             # Optimization level '-O3'
    ],
    'cython_args': cython_cpp_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': [                                 # Additional C/C++ arguments
      # '-include', 'develop/NdArray/NdArrayCore.hpp',  # optional force include
      cython_cpp_flags,
      extra_compile_args,
    ],
    'link_args': [],                              # 👈 add this
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/cexternals/_numcpp',     # Path where the module is located
  },
}
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : _numcpp_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + _numcpp_cython_tree,                 # Sources and dependencies
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
