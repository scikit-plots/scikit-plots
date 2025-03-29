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
## Generate file '_citation.py' for "--sdist" or use MANIFEST.in
## https://mesonbuild.com/Reference-manual_functions.html#configure_file
######################################################################

# Read bibtex from 'CITATION.bib'
_citation_file = meson.project_source_root() / 'CITATION.bib'
_citation_content = fs.read(_citation_file).strip()

# bib configure_file
conf_data_bib = configuration_data()
conf_data_bib.set('CITATION', _citation_content)

# Generate _CITATION.bib for sdist
configure_file(
  input: '_citation.py.in',
  output: '_citation.py',
  configuration: conf_data_bib,
  install: true,
  install_dir: py.get_install_dir() / 'scikitplot',
  install_tag: 'python-runtime',
)

######################################################################
## Templated file: '__config__.py.in'
## https://mesonbuild.com/Reference-manual_functions.html#configure_file
######################################################################

# OS configure_file
conf_data = configuration_data()
conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py.full_path())
conf_data.set('PYTHON_VERSION', py.language_version())

machines = {
  'HOST' : host_machine,
  'BUILD': build_machine,
}
# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

# Define the compilers dictionary
compilers = {
  'C': cc,
  'CPP': cpp,
  'CYTHON': cy,
}
compilers += {'FORTRAN': ff}

# Add `pythran` information if present
if get_option('pythran').enabled()
  conf_data.set(
    'PYTHRAN_VERSION', pythran.version(), description: 'pythran version')
  conf_data.set(
    'PYTHRAN_INCDIR', incdir_pythran, description: 'pythran path')
endif
# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Dependencies information, If Needed
dependency_map = {
  'PYBIND11': dep_pybind11,
}
# BLAS/LAPACK dependency info. Ensure we report dependencies correctly for
# `scikitplot.show_config()`; needs some special handling for the case BLAS was found
# but CBLAS not (and hence BLAS was also disabled)
dependency_map += {
  'LAPACK': lapack,
}
if have_blas
  dependency_map += {'BLAS': blas}
else
  conf_data.set('BLAS_NAME', blas_name)
  conf_data.set('BLAS_FOUND', false)
endif
# Dependencies information
foreach name, dep : dependency_map
  conf_data.set(name + '_NAME', dep.name())
  conf_data.set(name + '_FOUND', dep.found())
  if dep.found()
    conf_data.set(name + '_VERSION', dep.version())
    conf_data.set(name + '_TYPE_NAME', dep.type_name())
    # get_variable() results may be missing for a variety of reasons
    conf_data.set(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
    conf_data.set(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
    conf_data.set(name + '_OPENBLAS_CONFIG', dep.get_variable('openblas_config', default_value: 'unknown'))
    conf_data.set(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
  endif
endforeach

configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration: conf_data,
  install_dir: py.get_install_dir() / 'scikitplot',
  install: true,
  install_tag: 'python-runtime',
)

######################################################################
## Generate file 'scikitplot_config.h'
## https://mesonbuild.com/Reference-manual_functions.html#configure_file
######################################################################

# scikitplot_config configure_file
cdata = configuration_data()

# Test variable attribute to use for thread-local storage;
# Adapted from `numpy/_core/meson.build`.
check_tls_attrs = [
  ['thread_local', 'HAVE_THREAD_LOCAL'],    # C23
  ['_Thread_local', 'HAVE__THREAD_LOCAL'],  # C11/C17
  ['__thread', 'HAVE__THREAD'],
]
if is_windows and not is_mingw
  check_tls_attrs += ['__declspec(thread)', 'HAVE___DECLSPEC_THREAD_']
endif
f2py_tls_define = ''
foreach tls_attrs: check_tls_attrs
  attr = tls_attrs[0]
  code = f'''
    #pragma GCC diagnostic error "-Wattributes"
    #pragma clang diagnostic error "-Wattributes"

    int @attr@ foo;
  '''
  code += '''
    int
    main()
    {
      return 0;
    }
  '''
  if cc.compiles(code, name: tls_attrs[0])
    cdata.set10(tls_attrs[1], true)
    f2py_tls_define = tls_attrs[0]
  endif
endforeach

# Contains only TLS check results for now - name chosen for when more compiler
# checks need adding in the future.
scipy_config_h = configure_file(
  input: 'scikitplot_config.h.in',
  output: 'scikitplot_config.h',
  configuration: cdata,
  install: false
)

# Generate _CITATION.bib for sdist
configure_file(
  input: '_citation.py.in',
  output: '_citation.py',
  configuration: conf_data_bib,
  install: true,
  install_dir: py.get_install_dir() / 'scikitplot',
  install_tag: 'python-runtime',
)

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_root_cython_tree = [
  fs.copyfile('__init__.py'),       # Ensure __init__.py is copied early
  fs.copyfile('experimental.pxd'),  # Cython definition file for C/C++ code

  # Pure Python types.
  fs.copyfile('__init__.pyi'),
  fs.copyfile('_c_internal_utils.pyi'),
  fs.copyfile('_docstring.pyi'),
  fs.copyfile('_preprocess.pyi'),
  fs.copyfile('cbook.pyi'),
  fs.copyfile('version.pyi'),
]

######################################################################
## Next
######################################################################

# Subpackages are mostly in alphabetical order except to handle Cython
# dependencies across subpackages
# Ordering of subdirs: because other submodules have dependencies on cython.
# After those, subdirs with the most heavy builds should come first
# (that parallelizes better)
# below can be needs to be to be before ... since ... cimport *.pxd
subdir('_xp_core_lib')
# submodules
subdir('_seaborn')
subdir('_astropy')
subdir('experimental')
