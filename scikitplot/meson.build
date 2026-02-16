## scikit-plots/scikitplot/meson.build

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## Install actual python sources (.py)
## https://mesonbuild.com/Python-module.html#install_sources
######################################################################

# Not Needed we also copied before all files via "install_subdir"
# py_sources = [
#   '__init__.py',
# ]
# py.install_sources(
#   py_sources,
#   subdir: 'scikitplot'
#   install_tag: 'python-runtime', # default
# )

######################################################################
## Generate file 'version.py' for Sourde Distribution "--sdist"
## "--sdist" require '.git' if error use only "--wheel" packaging
######################################################################

# Generate version.py for sdist
meson.add_dist_script(
  '_build_utils/gitversion.py',        # The script to execute
  '--meson-dist',                      # Additional arguments
  '--write', 'scikitplot/version.py',  # Additional arguments
)
if not fs.exists('version.py')
  generate_version = custom_target(
    'generate-version',
    build_always_stale: true,
    build_by_default: true,
    input: '_build_utils/gitversion.py',
    output: 'version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install: true,
    install_dir: py.get_install_dir() / 'scikitplot',
    install_tag: 'python-runtime',
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir: 'scikitplot',
    install_tag: 'python-runtime', # default
  )
endif

######################################################################
## Root cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_root_cython_tree = [
  fs.copyfile('__init__.py'),             # Ensure __init__.py is copied early
  fs.copyfile('cexperimental.pxd'),       # Cython definition file for C/C++ code
  fs.copyfile('cexternals.pxd'),
  fs.copyfile('annoy.pxd'),
  fs.copyfile('nc.pxd'),

  # Python stub files.
  fs.copyfile('__init__.pyi'),
  fs.copyfile('_preprocess.pyi'),
  fs.copyfile('version.pyi'),
]

######################################################################
## NumCpp: C++ Header-Only Library + Python/Pybind11/Cython Integration
######################################################################

# Check/Install Headers (Optional but Recommended for Python Integration)
# Resolve absolute include path robustly
_here = meson.current_source_dir()
_numcpp_develop_path = _here / 'cexternals/_numcpp/develop/NdArray'
_numcpp_include_path = _here / 'cexternals/_numcpp/include'
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
## NumCpp: Include the headers directories
######################################################################

# include_directories for Specific Subfolders (header-only)
# develop nc_develop::NdArray
_numcpp_develop_inc_dir = [
  # 'develop/NdArray',
  'cexternals/_numcpp/develop/NdArray',
]
# stable nc::NdArray
_numcpp_inc_dir = [
  # 'include',
  'cexternals/_numcpp/include',
]

######################################################################
## NumCpp: Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
# ⚠️ Make sure your develop headers take priority can may conflict:
inc_dir_numcpp_develop = include_directories(_numcpp_develop_inc_dir)
inc_dir_numcpp = include_directories(_numcpp_inc_dir)

######################################################################
## NumCpp / pybind / Boost / Compiler Feature Flags and Dep (replicating setup.py logic)
######################################################################
# Optional disable all NumCpp features that require Boost
# dpkg -s libboost-dev | grep 'Version'
# Optional disable tbb Dependency

# Base compile and link flags
extra_compile_args = []
extra_link_args = []

extra_compile_args += [
  # '-DNUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE',  # Use Boost
  # '-DNUMCPP_USE_MULTITHREAD',  # Enable multithreading support
  '-DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE',
  '-DNUMCPP_NO_USE_BOOST',  # Disable Boost features
]

######################################################################
## Optional NumCpp threading (disabled by default)
######################################################################

# Uncomment only if TBB or threading dependency is satisfied
# if dependency('tbb', required: false).found()
#   extra_compile_args += ['-DNUMCPP_USE_MULTITHREAD']
# endif

######################################################################
## Compiler-specific flag handling
######################################################################

# -fpermissive equivalent handling
if is_msvc or is_clang_cl
  # MSVC does not support -fpermissive
  # Closest strict/standard-conforming behavior toggle
  if cc.has_argument('/permissive-')
    extra_compile_args += ['/permissive-']
  endif
elif is_gcc or is_clang
  if cc.has_argument('-fpermissive')
    extra_compile_args += ['-fpermissive']
  endif
endif

# --------------------------------------------------------------------
# macOS Minimum Target
# --------------------------------------------------------------------
# Architecture-specific tuning
os_name = host_machine.system()
cpu_arch = host_machine.cpu_family()

# #349: something with OS X Mojave causes libstd not to be found '-mmacosx-version-min=10.12'
# macOS minimum version and stdlib
# std::filesystem::path is part of C++17's <filesystem> library.
# On macOS, the <filesystem> implementation requires at least macOS 10.15 (Catalina) with Xcode 11+.

if os_name == 'darwin' and is_gnu_style
  if cc.has_argument('-mmacosx-version-min=10.15')
    extra_compile_args += ['-mmacosx-version-min=10.15']
    extra_link_args += [
      '-mmacosx-version-min=10.15',
      '-stdlib=libc++',
      # '-lomp',
    ]
  endif
endif

# Optionally export for parent meson.build if used as subproject
dep_numcpp = declare_dependency(
  include_directories: inc_dir_numcpp,
  compile_args: extra_compile_args,
  link_args: extra_link_args,
)

######################################################################
## Next
######################################################################

# Subpackages are mostly in alphabetical order except to handle Cython
# dependencies across subpackages
# Ordering of subdirs: because other submodules have dependencies on cython.
# After those, subdirs with the most heavy builds should come first
# (that parallelizes better)
# below can be needs to be to be before ... since ... cimport *.pxd
subdir('config')

# c submodules
subdir('_lib')
subdir('cexternals')
subdir('memmap')
subdir('random')
subdir('annoy')

subdir('nc')  # defined use in dep_numcpp

# submodules
subdir('cexperimental')
