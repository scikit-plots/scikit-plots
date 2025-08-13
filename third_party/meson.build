## scikit-plots/third_party/meson.build

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## submodules
## ⚠️ Not Need anymore we copying all submodules before runtime from 'third_party'
######################################################################

# Check if Git is available, use 'which' for Linux/macOS, 'where' for Windows
if host_machine.system() == 'windows'
  git_check = run_command('where', 'git', check: false)
else
  # host_machine.system() == 'linux' or host_machine.system() == 'darwin'
  git_check = run_command('which', 'git', check: false)
endif

# Check if expected submodule folders exist
required_submodules = [
  'third_party/array-api-compat',
  'third_party/array-api-extra',
]

foreach submod : required_submodules
  if not (
      fs.exists(meson.current_source_dir() / submod / 'README.md') or
      fs.exists(meson.current_source_dir() / submod / 'README.rst')
  )
    # If git is found, attempt to update the submodules
    # If git is not found, print a message and skip the submodule update
    if git_check.returncode() != 0
      message(
        '\n' +
        BLUE + '[INFO]' + RESET +
        ' Git not found. Skipping submodule update.'
      )
    else
      # If git is found, try to update the submodules
      # Initialize and update submodules
      submodule_cmd = run_command(
        'git', 'submodule', 'update', '--init', '--recursive',
        check: false
      )
      if submodule_cmd.returncode() != 0
        # Optionally log the failure, but don't stop the build
        message(
          '\n' +
          BLUE + '[INFO]' + RESET +
          ' Git submodule update failed, but continuing without error.'
        )
      endif
    endif
  else
    message(green + '[FOUND]' + reset + ' ' + submod)
  endif
endforeach

######################################################################
## Submodule `array-api-compat` for Python array API standard
## https://data-apis.org/array-api/latest/index.html
######################################################################

if not fs.exists('array-api-compat/README.md')
  error(
  'Missing the `array-api-compat` submodule! ' +
  'Run `git submodule update --init` to fix this.')
endif

## `array-api-compat` install to simplify import path;
## should be updated whenever new files are added to `array_api_compat`
## Copy all the .py files to the install dir, rather than using
## py.install_sources and needing to list them explicitely one by one
install_subdir(
  'array-api-compat/array_api_compat',
  ## Load direct not need any changes
  install_dir: py.get_install_dir() / 'scikitplot/_lib',
  install_tag: 'python-runtime',
)
## Optionally add a message to confirm the installation
message(
  '\n' +
  BLUE + '[INFO]' + RESET +
  ' Array API compatibility files installed successfully: ' +
  '@0@'.format(py.get_install_dir() / 'scikitplot/_lib/array_api_compat')
)

######################################################################
## Submodule `array-api-extra` for Python array API standard
## https://data-apis.org/array-api-extra/
######################################################################

if not fs.exists('array-api-extra/README.md')
  error(
  'Missing the `array-api-extra` submodule! ' +
  'Run `git submodule update --init` to fix this.')
endif

## `array-api-extra` install to simplify import path;
## should be updated whenever new files are added to `array_api_extra`
## Copy all the .py files to the install dir, rather than using
## py.install_sources and needing to list them explicitely one by one
install_subdir(
  'array-api-extra/src/array_api_extra',
  ## Load direct not need any changes
  install_dir: py.get_install_dir() / 'scikitplot/_lib',
  install_tag: 'python-runtime',
)
## Optionally add a message to confirm the installation
message(
  '\n' +
  BLUE + '[INFO]' + RESET +
  ' Array API compatibility files installed successfully: ' +
  '@0@'.format(py.get_install_dir() / 'scikitplot/_lib/array_api_extra')
)

######################################################################
## Submodule `seaborn`
## https://github.com/mwaskom/seaborn
## Load before local then load modified compatipable version
######################################################################

if not fs.exists('seaborn/README.md')
  error(
  'Missing the `seaborn` submodule! ' +
  'Run `git submodule update --init` to fix this.')
endif

## `seaborn` currently not support vendoring
## Check tools/submodules for updating `seaborn`
## make copy
######################################################################

## `seaborn` install to simplify import path;
## should be updated whenever new files are added to `_seaborn`
# install_subdir(
#   'seaborn/seaborn',
#   ## Load before local then load modified compatipable version
#   install_dir: meson.project_source_root() / 'scikitplot/_seaborn',
#   ## Install directory contents
#   strip_directory : true,
#   install_tag: 'python-runtime',
# )

## Copy all the source files to the under `_seaborn`
# src_dir = meson.current_source_dir() / 'seaborn/seaborn'
# dest_dir = meson.current_source_dir() / '../scikitplot/_seaborn'
## Add -r for recursive copying
# r = run_command(
#   cli_copier, '-r',
#   src_dir, dest_dir,
#   check: true
# )
# if r.returncode() != 0
#     error('Copying failed: ' + r.stderr().strip())
# endif
## Add -r for recursive copying
# r = run_command(
#   cli_copier, '-r',
#   meson.current_source_dir() / '../scikitplot/_build_utils/vendored_seaborn',
#   meson.current_source_dir() / '../scikitplot/_seaborn',
#   check: true
# )
# if r.returncode() != 0
#     error('Copying failed: ' + r.stderr().strip())
# endif

## Optionally add a message to confirm the installation
# message(
#   '\n `seaborn` compatibility files installed successfully: ' +
#   '@0@'.format( meson.project_source_root() / 'scikitplot/_seaborn' )
# )

######################################################################
## Submodule `astropy`
## https://github.com/astropy/astropy
## Load before local then load modified compatipable version
######################################################################

if not fs.exists('astropy/README.rst')
  error(
  'Missing the `astropy` submodule! ' +
  'Run `git submodule update --init` to fix this.')
endif

## `astropy` currently not support vendoring
## Check tools/submodules for updating `astropy`
## make copy
######################################################################

## `astropy` install to simplify import path;
## should be updated whenever new files are added to `_astropy`
# install_subdir(
#   'astropy/astropy/stats',
#   ## Load before local then load modified compatipable version
#   install_dir: meson.project_source_root() / 'scikitplot/_astropy',
#   ## Install directory contents
#   # strip_directory : true,
#   install_tag: 'python-runtime',
# )

## Copy all the source files to the under `_astropy`
# src_dir = meson.current_source_dir() / 'astropy/astropy/stats'
# dest_dir = meson.current_source_dir() / '../scikitplot/_astropy/stats'
## Add -r for recursive copying
# r = run_command(
#   cli_copier, '-r',
#   src_dir,
#   dest_dir,
#   check: true
# )
# if r.returncode() != 0
#     error('Copying failed: ' + r.stderr().strip())
# endif
## Add -r for recursive copying
# r = run_command(
#   cli_copier, '-r',
#   meson.current_source_dir() / '../scikitplot/_build_utils/vendored_astropy',
#   meson.current_source_dir() / '../scikitplot/_astropy',
#   check: true
# )
# if r.returncode() != 0
#     error('Copying failed: ' + r.stderr().strip())
# endif

## Optionally add a message to confirm the installation
# message(
#   '\n `astropy` compatibility files installed successfully: ' +
#   '@0@'.format( meson.project_source_root()/'scikitplot/_astropy/stats' )
# )

######################################################################
##
######################################################################

# Optionally add a message to confirm the installation
message(
  '\n' +
  GREEN + '[SUCCESS]' + RESET +
  ' scikitplot submodules installed successfully.'
)
