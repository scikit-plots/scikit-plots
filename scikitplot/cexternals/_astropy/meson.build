## scikitplot/cexternals/_astropy/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_astropy_cython_tree = [
  _root_cython_tree,
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
]

######################################################################
## Generate file '_citation.py' for "--sdist" or use MANIFEST.in
## https://mesonbuild.com/Reference-manual_functions.html#configure_file
######################################################################

# Read file
_file = meson.current_source_dir() / 'vendor.lock.json'
_file_content = fs.read(_file).strip().split('"')[7]

# configure_file
conf_ast = configuration_data()
conf_ast.set('VERSION', _file_content)

# Generate file
configure_file(
  input: 'version.py.in',
  output: 'version.py',
  configuration: conf_ast,
  install: true,
  install_dir: py.get_install_dir() / 'scikitplot/cexternals/_astropy',
  install_tag: 'python-runtime',
)

######################################################################
## Next
######################################################################

subdir('stats')
