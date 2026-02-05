## scikitplot/memmap/_memmap/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_memmap_cython_tree = [
  _root_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
  fs.copyfile('mem_map.pyi'),  # Cython definition file for C/C++ code

  fs.copyfile('mem_map.pxd'),
]


######################################################################
## Include the headers directories
######################################################################

# include_directories for Specific Subfolders
_memmap_inc_dir = [
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
inc_dir_memmap = include_directories(_memmap_inc_dir)

######################################################################
## extension metadata
######################################################################

## Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
## C library, the extension is `.c`, for C++ `.cc`, `.cpp` or `.cxx` are accepted.
## .c → plain C
## .cpp or .cc → C++
## .cxx or .C → also C++ (less common)
mem_map_extension_metadata = {
  ## Define Python Module Name
  ## Check: ELF 64-bit LSB shared object, dynamically linked
  # file builddir/scikitplot/memmap/_memmap/mem_map.so.p
  'mem_map':  ## Build static `mem_map` core library (Cython)
  {
    # Ceyton bindings source .pyx extension with C/C++ support
    'sources': [
      'mem_map.pyx',  # (Cython binding)
    ],
    'include_directories': [                      # Include dirs for compilation
      inc_dir_memmap,
      inc_dir_py,
    ],
    'dependencies': dep_list,                     # External libraries and dependencies
    'link_with': [                                # Link with the created static library
      # version_link_args
      # annoy_static,
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
    'link_args': [                                # 👈 add this
      # cython_link_flags,
      # extra_link_args
    ],
    'install': true,                              # Whether to install the .so file executable after building
    'subdir': 'scikitplot/memmap/_memmap',     # Path where the module is located
  },
}
# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : mem_map_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + _memmap_cython_tree,                   # Sources and dependencies
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
