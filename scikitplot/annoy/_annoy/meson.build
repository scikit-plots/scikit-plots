## scikitplot/annoy/_annoy/meson.build

######################################################################
## Tempita template generator
## Generates annoylib.pxd and annoylib.pyx from .in templates.
##
## Two usage patterns:
##   generator : for .pyx/.pxd files processed inline during the build
##   custom_target : when the generated .pxd must also be installed
##
## The CLI is:  python tempita.py <infile.in> -o <outfile>
######################################################################

# Path to the Tempita CLI helper (ships alongside this meson.build)
cli_tempita = find_program(
  meson.current_source_dir() + '/../../_build_utils/tempita.py',
  required: true,
)

# ---------------------------------------------------------------------------
# Generate annoylib.pxd from annoylib.pxd.in
#
# We use custom_target here (rather than generator) because the generated
# .pxd file must be copied to the build directory *before* the .pyx is
# compiled, so that Cython's cimport resolution works correctly.
# ---------------------------------------------------------------------------
_annoylib_pxd = custom_target('annoylib_pxd',
  output  : 'annoylib.pxd',
  input   : 'annoylib.pxd.in',
  command : [cli_tempita, '@INPUT@', '-o', '@OUTPUT@'],
  # Install the .pxd so downstream packages can cimport from us.
  install     : true,
  # install_dir : get_option('prefix') / get_option('python.install_dir') / 'scikitplot/annoy/_annoy',
  install_dir: py.get_install_dir() / 'scikitplot/annoy/_annoy',  # Folder relative to site-packages to install to
  install_tag : 'devel',
)

# ---------------------------------------------------------------------------
# Generate annoylib.pyx from annoylib.pyx.in
# ---------------------------------------------------------------------------
_annoylib_pyx = custom_target('annoylib_pyx',
  output  : 'annoylib.pyx',
  input   : 'annoylib.pyx.in',
  command : [cli_tempita, '@INPUT@', '-o', '@OUTPUT@'],
)

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_annoylib_cython_tree = [
  _root_cython_tree,
  fs.copyfile('__init__.py'),      # Ensure __init__.py is copied early

  # fs.copyfile('annoylib.pyi'),     # Cython definition file for C/C++ code
  fs.copyfile('annoylib.pxi'),     # Cython include file (static)

  # fs.copyfile('annoylib.pxd'),
  _annoylib_pxd,                # Generated .pxd from template
]


######################################################################
## Include the headers directories
######################################################################

# include_directories for Specific Subfolders
_annoylib_inc_dir = [
  # 'here/include',
  # 'include',
  # 'src',
  '.',
]

######################################################################
## Define include_directories for Source and Header files
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_annoylib = include_directories(_annoylib_inc_dir)

######################################################################
## extension metadata
######################################################################

## Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
## C library, the extension is `.c`, for C++ `.cc`, `.cpp` or `.cxx` are accepted.
## .c → plain C
## .cpp or .cc → C++
## .cxx or .C → also C++ (less common)
annoylib_extension_metadata = {
  ## Define Python Module Name
  ## Check: ELF 64-bit LSB shared object, dynamically linked
  # file builddir/scikitplot/annoy/_annoy/annoylib.so.p
  'annoylib':  ## Build static `annoylib` core library (Cython)
  {
    # Ceyton bindings source .pyx extension with C/C++ support
    'sources': [
      # 'annoylib.pyx',  # (Cython binding)
      _annoylib_pyx,        # Generated from annoylib.pyx.in via Tempita
    ],
    'include_directories': [                      # Include dirs for compilation
      inc_dir_annoylib,
      inc_dir_py,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
      # annoy_static,
    ],
    'override_options': [
      # 'cython_language=c',                      # Ensure Cython knows to generate C code
      'cython_language=cpp',                      # Ensure Cython knows to generate C code
      'optimization=3',                           # Optimization level '-O3'
    ],
    'cython_args': cython_cpp_args,
    'c_args': cython_c_flags,                     # Additional C/C++ arguments
    'cpp_args': [                                 # Additional C/C++ arguments
      cython_cpp_flags,
      extra_compile_args,
    ],
    'link_args': [                                # 👈 add this
      # cython_link_flags,
      # extra_link_args
    ],
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/annoy/_annoy',          # Path where the module is located
  },
}
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : annoylib_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + _annoylib_cython_tree,                   # Sources and dependencies
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
