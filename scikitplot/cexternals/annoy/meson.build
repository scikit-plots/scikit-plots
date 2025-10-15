## scikitplot/cexternals/annoy/meson.build

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

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
]

######################################################################
## Include the headers directories
######################################################################

# include_directories for Specific Subfolders
annoy_inc_dir = [
  # 'include',
  'src',
  # 'here/include',
]

######################################################################
## Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_annoy = include_directories(annoy_inc_dir)

######################################################################
## Define dependencies and Compiler Flags
# meson introspect ./build/cp311/ --dependencies
# meson introspect ./build/cp311/ --buildoptions | grep python
######################################################################

# Define dependencies
dep_list = dep_all

if host_machine.system() != 'windows'
  readline_dep = cc.find_library('readline', required: false)
	dep_list += [readline_dep]
endif

# https://mesonbuild.com/Release-notes-for-0-62-0.html#d-features-in-declare_dependency
foreach name : ['lua', 'lua53', 'lua5.3', 'lua-5.3']
  dep_lua = dependency(name, version: '>=5.1', required: false)
  if dep_lua.found()
	  dep_list += [dep_lua]
    break
  # else
  #   dep_lua = disabler()
  endif
endforeach
if not dep_lua.found()
  # error('Lua could not be found!')
  message(
    '\n' +
    BLUE + '[INFO]' + RESET +
    'Lua could not be found!'
  )
endif

######################################################################
## Platform and Compiler Flags (replicating setup.py logic)
######################################################################

# Architecture-specific tuning
os_name = host_machine.system()
cpu_arch = host_machine.cpu_family()

# Base compile and link flags
extra_compile_args = [
  '-D_CRT_SECURE_NO_WARNINGS',
  '-fpermissive',
]
extra_link_args = []

if cpu_arch == 'ppc64le'
  extra_compile_args += ['-mcpu=native']
elif cpu_arch == 'x86_64'
  if os_name != 'darwin'
    if cc.has_argument('-march=native')
      extra_compile_args += ['-march=native']
    endif
  endif
endif

# General optimization flags for non-Windows
if os_name != 'windows'
  extra_compile_args += [
    '-O3',
    '-ffast-math',
    '-fno-associative-math',
  ]
endif

# Multithreaded build flag (Python 3 or non-Windows Py2 equivalent)
# We can safely always enable this — modern environments all support threads
extra_compile_args += [
  '-DANNOYLIB_MULTITHREADED_BUILD',
]

# Use at least C++17
if cc.has_argument('-std=c++17')
  extra_compile_args += ['-std=c++17']
endif

# macOS-specific flags
# macOS minimum version and stdlib
if os_name == 'darwin'
  if cc.has_argument('-mmacosx-version-min=10.12')
    extra_compile_args += ['-mmacosx-version-min=10.12']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.12']
  endif
endif

## annoy-simd-cp313-win_amd64.whl → AVX baseline
## annoy-simd-cp313-win_amd64_avx512.whl → HPC optimized
## Optional AVX/AVX512 optimizations
## Check AVX512 first: if the CPU supports it, only the AVX512 block runs.
## Use elif: ensures AVX block is skipped if AVX512 was already applied.
## This avoids duplicate template instantiations, fixing the redefinition errors.
# if cc.has_argument('-mavx512f')
#   extra_compile_args += [
#     '-mavx512f',
#     '-DANNOYLIB_USE_AVX512',
#   ]
# el
if cc.has_argument('-mavx')
  extra_compile_args += [
    '-mavx',
    '-DANNOYLIB_USE_AVX',
  ]
else
  message('⚠️ No SIMD acceleration available, falling back to scalar mode')
endif

# Make sure we have the right definition when compiling as a shared library on Windows.
# if get_option('default_library') != 'static' and host_machine.system() == 'windows'
# 	add_project_arguments('-DLUA_BUILD_AS_DLL', language: 'c')
# 	compile_args = [ '-DLUA_BUILD_AS_DLL' ]
# endif

_env = environment()

message('Final compile args: ' + ' '.join(extra_compile_args))
message('Final link args: ' + ' '.join(extra_link_args))

annoy_core_sources = [
  'src/annoymodule.cc',  # (Python binding)
  # 'src/annoyluamodule.cc',  # (Lua binding)
]
annoy_static = static_library(
  'annoycore',
  annoy_core_sources,
  include_directories: inc_dir_annoy,
  dependencies: dep_list,
  cpp_args: cython_cpp_flags + extra_compile_args,
  link_args: extra_link_args,
  install: false
  # install_dir: np_dir / 'random/lib',
)

######################################################################
## extension metadata
######################################################################

## Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
## .c → plain C
## .cpp or .cc → C++
## .cxx or .C → also C++ (less common)
annoy_extension_metadata = {
  ## Define Python Module Name
  ## Check: ELF 64-bit LSB shared object, dynamically linked
  # file builddir/scikitplot/cexternals/annoy/annoylib.so
  'annoylib':  ## Build static Annoy core library (C++)
  {
    'sources': [                                  # C++ source file with Python/Pybind11 bindings
      'src/annoymodule.cc',  # (Python binding)
      # 'src/annoyluamodule.cc',  # (Lua binding)
    ],
    'include_directories': [                      # Include dirs for compilation
      inc_dir_annoy,
      python_inc,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
      annoy_static,
    ],
    'override_options': [
      # 'cython_language=c',                        # Ensure Cython knows to generate C code
      'cython_language=cpp',                        # Ensure Cython knows to generate C code
      'optimization=3',                           # Optimization level '-O3'
    ],
    'cython_args': cython_c_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': [                                 # Additional C/C++ arguments
      cython_cpp_flags,
      extra_compile_args,
    ],
    'link_args': [extra_link_args],                 # 👈 add this
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/cexternals/annoy',        # Path where the module is located
  },
}
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : annoy_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + _annoy_cython_tree,                   # Sources and dependencies
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
#   install_dir: py.get_install_dir() / 'scikitplot/cexternals/annoy',
#   install_tag: 'python-runtime',
# )

######################################################################
##
######################################################################
