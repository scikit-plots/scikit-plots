## scikit-plots/scikitplot/config/meson.build

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
  fs.copyfile('cbook.pyi'),               # Cython definition file for C/C++ code
]

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
  install_dir: py.get_install_dir() / 'scikitplot/config',
  install: true,
  install_tag: 'python-runtime',
)

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
  install_dir: py.get_install_dir() / 'scikitplot/config',
  install_tag: 'python-runtime',
)

######################################################################
## Generate file 'scikitplot_config.h'
## https://mesonbuild.com/Reference-manual_functions.html#configure_file
## https://github.com/scipy/scipy/blob/main/scipy/meson.build#L153
######################################################################

# Contains only TLS check results for now - name chosen for when more compiler
# checks need adding in the future.
# scipy_config_h
skplt_config_h = configure_file(
  input: 'scikitplot_config.h.in',
  output: 'scikitplot_config.h',
  configuration: cdata,
  install: false
)

######################################################################
##
######################################################################
