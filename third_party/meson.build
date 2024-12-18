## scikit-plots/third_party/meson.build

######################################################################
## Submodule array-api-compat
######################################################################

if not fs.exists('array-api-compat/README.md')
  error(
  'Missing the `array-api-compat` submodule! ' + 
  'Run `git submodule update --init` to fix this.')
endif

# `array-api-compat` install to simplify import path;
# should be updated whenever new files are added to `array_api_compat`
# Copy all the .py files to the install dir, rather than using
# py.install_sources and needing to list them explicitely one by one
install_subdir(
  'array-api-compat/array_api_compat',
  install_dir: py.get_install_dir()/'scikitplot/_xp_core_lib',
  install_tag: 'python-runtime',
)
# Optionally add a message to confirm the installation
message(
  '\nArray API compatibility files installed successfully: ' +
  '@0@'.format(py.get_install_dir()/'scikitplot/_xp_core_lib')
)

######################################################################
## 
######################################################################