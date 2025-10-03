## Makefile for Python Packaging Library
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# This Makefile defines various targets for project management tasks,
# including building the project, cleaning up build artifacts, running tests,
# creating Docker images, and more.
#
# Notes:
# - .ONESHELL ensures that all commands in a target run within a single shell,
#   so variables and environment changes persist across commands.
# - For cleaner multi-line logic, use .ONESHELL.
#   The '@' symbol only suppresses output for the first command.
# - To enable shell debugging, uncomment the following line:
#     SHELL = /bin/bash -x
#
# PHONY targets:
# These targets are not associated with files or directories. Declaring them as
# phony avoids conflicts with files or folders of the same name.
#
# Available phony targets: help, all, clean, publish
.ONESHELL:
.PHONY: help all clean publish

######################################################################
## Makefile Variable Assignment Styles
######################################################################
#
# +------------------+---------------------------------+------------------------------------------------------------------------------+
# | Syntax           | Name                            | Description                                                                  |
# +==================+=================================+==============================================================================+
# | `VAR := value`   | Simple (Immediate)              | Evaluates the value **immediately** and stores the result.                   |
# |                  | Cache the result now            | Useful when the value should not change, even if dependencies do later.      |
# +------------------+---------------------------------+------------------------------------------------------------------------------+
# | `VAR = value`    | Recursive (Lazy)                | Evaluates the value **each time** the variable is used.                      |
# |                  | Delay evaluation until later    | Useful when the value depends on something that may change later.            |
# +------------------+---------------------------------+------------------------------------------------------------------------------+
# | `VAR ?= value`   | Conditional                     | Assigns the value **only if** the variable is not already defined.           |
# |                  | Allow the user to override      | Allows overriding from the command line or environment variables.            |
# +------------------+---------------------------------+------------------------------------------------------------------------------+
# | `VAR += value`   | Append                          | Adds to the existing value of the variable (with a space separator).         |
# |                  | Add more values to a variable   | Works with both `=` and `:=` variables.                                      |
# +------------------+---------------------------------+------------------------------------------------------------------------------+
#
# Commands to display the project directory tree.
# - On Windows: uses 'tree' with files (/F) and ASCII formatting (/A).
# - On Unix-like systems: prefers 'tree -d' (directories only).
#   If 'tree' is not installed, falls back to 'find' + 'sed'.
#
ifeq ($(OS),Windows_NT)
    SYSTEM := Windows
    SHELL  := cmd.exe
    # TREE_CMD := dir /ad /b
    TREE_CMD := tree /F /A
else
    SYSTEM := Unix
    SHELL  := /bin/bash
    ## Check if 'tree' exists, otherwise use 'find' + 'sed'
    # TREE_CMD := bash -c 'command -v tree >/dev/null && tree || (find . -type d && find . | sed -e "s/[^-][^/]*\// |/g" -e "s/|\([^ ]\)/|-\1/")'
    ## Check if 'tree' exists, otherwise use 'find' + 'sed' (folders only)
    TREE_CMD := bash -c 'command -v tree >/dev/null && tree -d || find . -type d | sed -e "s|[^/]*/| |g" -e "s| |-|g"'
endif

# Simple (evaluated immediately)
NOW := $(shell date)

# Recursive (evaluated each time it’s used)
LATER = $(shell date)

# Conditional assignments (default values, can be overridden)
DEBUG ?= false
INFO ?= false

# Append values
# CFLAGS += -Wall

# Open browser using Python’s built-in module
BROWSER := python -mwebbrowser

######################################################################
## Makefile Syntax: action / command / target
######################################################################
#
# A "target" is the name of an action to be executed with `make <target>`.
# General syntax:
#
#   <target>: <dependencies>
#       @<command>   # Commands to execute (must be indented with a TAB, not spaces)
#
######################################################################
## Helper (always define this first)
######################################################################
#
# The `help` target provides an overview of available commands.
# -e enables interpretation of backslash escapes in echo.
#

help:
	@echo -e "Usage: make <target>\n"
	@echo "Available targets:"
	@echo ""
	@echo "  clean           Remove all 'build', 'test', 'coverage', and Python artifacts (e.g., .pyc files)."
	@echo "  clean-basic     Remove Jupyter and testing artifacts."
	@echo "  clean-build     Remove package build artifacts, Python cache files (.pyc), and temporary files."
	@echo "  clean-test      Remove test and coverage artifacts."
	@echo "  tree            Display the project file/folder structure."
	@echo ""
	@echo "  newm            Create a new Conda environment 'py311' using mamba."
	@echo "  conda           Create a new Conda environment 'py311'."
	@echo "  dep             Install build and runtime dependencies."
	@echo "  ins             Install the package into the active Python's site-packages."
	@echo "  dev             Install the package in development (editable) mode."
	@echo "  dist            Build distribution archives (e.g., .zip, .tar.gz, or .whl)."
	@echo "  update          Upgrade installed Python packages and linting tools."
	@echo ""
	@echo "  pkg-setup       Install the package using 'setup.py' (PyPI-compatible)."
	@echo "  pkg-build       Build the package via the 'build' library using 'setup.py' or 'pyproject.toml'."
	@echo "  comp-meson      Compile using the 'meson' build system with 'meson.build'."
	@echo ""
	@echo "  test            Run tests with the default Python version (e.g., 3.11)."
	@echo "  test-all        Run tests across multiple Python versions using tox."
	@echo "  coverage        Check code coverage with the default Python version."
	@echo "  examples        Execute Python scripts under the 'galleries/' folder."
	@echo ""
	@echo "  branch          Create a new branch locally. Usage: make branch BR=maintenance/x.x.x"
	@echo "  branch-del      Delete a local branch. Usage: make branch-del BR=maintenance/x.x.x"
	@echo "  branch-delr     Delete a remote branch. Usage: make branch-delr BR=maintenance/x.x.x"
	@echo "  branch-clean    Delete merged local branches to keep the workspace clean."
	@echo "  branch-push     Push a newly created stable branch to the remote repository (with -u flag)."
	@echo ""
	@echo "  tag             Tag the latest commit or use scikitplot version. Usage: make tag DEBUG=true"
	@echo "  tag-add         Add a tag locally. Usage: make tag"
	@echo "  tag-del         Delete a local tag. Usage: make tag-del GIT_TAG=1.0.0"
	@echo "  tag-delr        Delete a remote tag. Usage: make tag-delr GIT_TAG=1.0.0"
	@echo "  tag-pushr       Push the tag to the remote repository."
	@echo ""
	@echo "  check-release   Validate distribution files (e.g., README.md) for PyPI with twine."
	@echo "  release         Build and upload a release to PyPI."
	@echo ""

# Shortcuts
h:
	@echo "make -h"
	@make -h

## (Optional) Ensures that the project is rebuilt from a clean state.
all: clean publish
	@echo "all completed."

######################################################################
## ERROR HANDLING: Clock Skew Detected
######################################################################
#
# Example error:
#   ERROR: Clock skew detected. File
#   /home/jovyan/work/build/cp311/meson-private/coredata.dat
#   has a timestamp 7.9016s in the future.
#
# Simple solution:
# - First, close any open files.
# - If the issue persists, try synchronizing system time.
#
######################################################################
## sync-time
######################################################################
#
# This target attempts to fix clock skew issues that can cause build errors.
# Possible steps:
# 1. Use `timedatectl` to enable NTP synchronization (if available).
# 2. Use `ntpdate` to immediately sync the clock (if permitted).
# 3. As a fallback, refresh file timestamps or rebuild the project.
#
# Example (commented out):
#   @echo "Attempting time sync..."
#   @which timedatectl >/dev/null 2>&1 && sudo timedatectl set-ntp true || echo "timedatectl not available"
#   @which ntpdate >/dev/null 2>&1 && sudo ntpdate pool.ntp.org || echo "ntpdate failed or not permitted"
#
sync-time:
	@# 🔁 Step 1: Refresh file timestamps (inside container)
	@# find ./build -type f -exec touch {} +
	@# 🧹 Step 2: (Recommended) Delete and rebuild
	@rm -rf ./build

######################################################################
## Cleaning
######################################################################
#
# Commands for removing build artifacts, caches, and temporary files.
# Notes:
# - Command substitution: use $(...) instead of backticks (`...`) (its output in place of the backticks).
# - clean-basic is a safe cleanup that excludes third_party.
# - clean removes everything from clean-basic plus build artifacts.
# - clean-test only removes testing/coverage-related files.
#
## clean-basic
## Remove development caches and temporary files, excluding 'third_party'.
clean-basic:
	@echo ">> Starting basic cleaning..."

	@# Remove protobuf caches
	@#sudo -H rm -rf ./.cache/protobuf_cache || true

	@# Remove pip caches
	@rm -rf ~/.cache/pip
	@pip cache purge || true
	@echo "   - Removed pip cache files"

	@# Remove Jupyter checkpoints
	@# rm -rf `find -L . -type d -name ".ipynb_checkpoints" -not -path "./third_party/*"`
	@find . -name '.ipynb_checkpoints' -not -path './third_party/*' -exec rm -rf {} +
	@rm -rf "./third_party/.ipynb_checkpoints"
	@echo "   - Removed '.ipynb_checkpoints'"

	@# Remove Python cache directories
	@# rm -rf `find -L . -type d -name "__pycache__" -not -path "./third_party/*"`
	@find . -name '__pycache__' -not -path './third_party/*' -exec rm -rf {} +
	@echo "   - Removed '__pycache__'"

	@# Remove zip leftovers
	@# rm -rf `find -L . -type d -name "__MACOSX" -not -path "./third_party/*"`
	@find . -name '__MACOSX' -not -path './third_party/*' -exec rm -rf {} +
	@echo "   - Removed '__MACOSX'"

	@# Remove VSCode configs
	@# rm -rf `find -L . -type d -name ".vscode" -not -path "./third_party/*"`
	@find . -name '.vscode' -not -path './third_party/*' -exec rm -rf {} +
	@echo "   - Removed '.vscode'"

	@# Remove type checker/linter caches
	@rm -rf ".mypy_cache" ".ruff_cache"
	@echo "   - Removed mypy and ruff caches"

	@# Remove Gradio cache
	@rm -rf ".gradio"
	@echo "   - Removed Gradio cache"

	@# Remove matplotlib result images
	@rm -rf "result_images"
	@find . -name 'result_images' -not -path './third_party/*' -exec rm -rf {} +
	@echo "   - Removed 'result_images' from matplotlib builds 'matplotlib.sphinxext.plot_directive'"

	@# Remove pytest cache
	@# rm -rf `find -L . -type d -name ".pytest_cache" -not -path "./third_party/*"`
	@find . -name '.pytest_cache' -not -path './third_party/*' -exec rm -rf {} +
	@echo "   - Removed '.pytest_cache'"

	@echo ">> Basic cleaning completed."

## clean-test
## Remove testing and coverage artifacts.
clean-test:
	@echo ">> Starting test cleaning..."

	@# Remove pytest cache
	@# rm -rf `find -L . -type d -name ".pytest_cache" -not -path "./third_party/*"`
	@find . -name '.pytest_cache' -not -path './third_party/*' -exec rm -rf {} +
	@echo "   - Removed '.pytest_cache'"

	@rm -rf .tox/
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -rf htmlcov/
	@echo "   - Removed tox environments and coverage reports"

	@echo ">> Test cleaning completed."

## clean
## Perform a full cleanup: includes clean-basic + build artifacts.
clean: clean-basic
	@echo ">> Starting full cleanup..."

	@# Remove build directories and egg-info
	@rm -rf "build" "build_dir" "builddir" "dist" "scikit_plots.egg-info" *.egg-info*
	@echo "   - Removed build directories and egg-info files"

	@# Remove compiled shared objects
	@# rm -rf `find -L -type f -name "*.so" -path "*/build*"`
	@# find -L -type f -name *.so -path "*/build*" | xargs rm -rf
	@find -L -type f -name "*.so" -path "*/build*" -exec rm -rf {} +
	@echo "   - Removed '*.so' files in build directories"

	@# Remove Meson-related files
	@rm -rf .meson .mesonpy-*
	@echo "   - Removed Meson-related files (.meson, .mesonpy-*)"

	@# Uninstall local package
	@pip uninstall scikit-plots -y || true
	@echo "   - Uninstalled local 'scikit-plots' package (if present)"

	@echo ">> Full cleanup completed."

######################################################################
## Project Structure
######################################################################

## tree
## Print the project structure (directories, cross-platform).
tree:
	@echo ">> System detected: $(SYSTEM)"
	@$(TREE_CMD)

######################################################################
## Symbolic Links
######################################################################
#
# Create symbolic links for development container scripts and environment files.
# - Removes any existing links or directories before creating new ones.
# - Uses relative, forced links (-rsf) for safe updates.
#

## sym
## Create symbolic links
sym:
	@echo ">> Creating symbolic links..."

	@# Remove existing links or directories
	@# unlink ".devcontainer/scripts" "environment.yml"
	@rm -rf ".devcontainer/scripts" "environment.yml"

	@# Create symbolic links
	@ln -rsf "docker/scripts/" ".devcontainer/scripts"
	@ln -rsf "./docker/env_conda/environment.yml" "environment.yml"

	@echo ">> Symbolic links created successfully."

	@# Optional: list all symbolic links for verification
	@find . -type l -exec ls -l {} +

	@# find . -type l
	@# find . -type l -exec ls -l {} +
	@# find . -type l -exec readlink -f {} \; -exec ls -l {} \;

######################################################################
## Environment Management (conda / mamba)
######################################################################
#
# mamba is a drop-in replacement for conda.
# - Faster, parallelized dependency resolution
# - Same commands, configuration, and options
#
# References:
# - Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html
# - Mamba: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
#
## https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
## https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
## mamba is a drop-in replacement and uses the same commands and configuration options as conda.
## The --prune option causes conda to remove any dependencies that are no longer required from the environment.
# Useful conda commands:
# - conda info
# - conda info --envs
# - conda update conda
# - conda update anaconda
# - conda env list
# - conda search python
# - conda search --full-name python
# - conda update python
# - conda install python=3.11
# - conda env create -f environment.yml
# - conda env update --file environment.yml --prune
# - conda create --name <myenv> python=3.11
# - conda install -n <myenv> <package>
# - conda list -n <myenv>
# - conda env export > environment.yml
# - conda env export --from-history
# - conda remove --name <myenv> --all
#
## https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
## mamba is a drop-in replacement and uses the same commands and configuration options as conda.
## mamba create -n ... -c ... ...
## mamba install ...
## mamba list
## mamba deactivate && mamba remove -y --all --name py311
#

######################################################################
## Mamba
######################################################################

## newm
## Create a new environment with Python 3.11 (via mamba).
newm:
	# Example: create from environment.yml
	# mamba env create -f "./docker/environment.yml"

	# Other versions:
	# mamba create -n py38  python=3.8  ipykernel -y
	# mamba create -n py39  python=3.9  ipykernel -y
	# mamba create -n py310 python=3.10 ipykernel -y
	mamba create -n py311 python=3.11 ipykernel -y
	# mamba create -n py312 python=3.12 ipykernel -y
	# mamba create -n py313 python=3.13 ipykernel -y
	# mamba create -n py314 python=3.14 ipykernel -y

	# Activate/deactivate:
	# mamba activate py311
	# mamba deactivate

######################################################################
## Conda
######################################################################

## newc
## Create a new environment with Python 3.11 (via conda).
newc:
	conda create -n test python=3.11 ipykernel -y

	# conda activate test   # activate the environment

######################################################################
## Install scikit-plots & Dependencies
######################################################################
#
# Commands for installing dependencies, building packages,
# and development (editable) installs.
#

######################################################################
## Dependencies Installation
######################################################################

## dep
## Install required pip dependencies for build and runtime.
## Depends on 'clean' to ensure a fresh environment.
dep: clean
	@echo ">> Installing library pip dependencies..."
	@pip install --no-cache-dir -r ./requirements/build.txt
	@pip install --no-cache-dir -r ./requirements/cpu.txt

	@# pip install --no-cache-dir -r ./requirements/all.txt

######################################################################
## Packaging
######################################################################
#
# Reference links:
# - Setuptools / PyProject: https://setuptools.pypa.io/en/stable/userguide/pyproject_config.html
# - Meson Python tutorial: https://mesonbuild.com/meson-python/tutorials/introduction.html
# - Build library: https://build.pypa.io/en/stable/ or https://github.com/pypa/build
#

## ins-st
## Package using 'setup.py' (sdist + wheel).
## Depends on 'clean' to ensure no leftover build artifacts.
ins-st: clean
	@echo ">> Packaging with 'setup.py' using setuptools/wheel..."
	@python setup.py sdist bdist_wheel

	@# python setup.py sdist
	@# python setup.py bdist_wheel
	@# python setup.py build_ext --inplace --verbose

## build-st
## Package using 'build' library (supports setup.py or pyproject.toml)
## Depends on 'clean'.
build-st: clean
	@echo ">> Packaging with 'build' library (setup.py or pyproject.toml)..."
	@echo "   - Configuration can use (setuptools, wheel) or (meson, ninja)."
	@python -m build

	@# python -m build --sdist
	@# python -m build --wheel
	@# pip install --no-cache-dir build

######################################################################
## Development Mode / Editable Installation
######################################################################
#
# Reference links:
# - Setuptools / PyProject: https://setuptools.pypa.io/en/stable/userguide/development_mode.html
# - Meson Python tutorial: https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html
#

## ins
## Install package locally (non-editable), depends on 'clean' and 'dep'.
ins: clean dep
	@echo ">> Installing package locally (non-editable)..."
	@python -m pip install --no-build-isolation --no-cache-dir . -v

	@# python -m pip install --no-cache-dir .
	@# python -m pip install --no-cache-dir --use-pep517 .

## dev
## Install development version of scikit-plots (editable), depends on 'clean' and 'dep'.
## ➡️ Works only if "ninja" and "all build deps" are installed in the environment (--no-build-isolation).
dev: clean dep
	@echo ">> Installing package locally (editable, development mode)..."
	@# Requires ninja and all build dependencies installed
	@python -m pip install --no-build-isolation --no-cache-dir -e . -v

	@# python -m pip install --no-build-isolation --no-cache-dir --editable . -vvv
	@# python -m pip install --no-build-isolation --no-cache-dir -e .[build,dev,test,doc] -v

	@# find . -exec touch {} + && python -m pip install --no-build-isolation --no-cache-dir -e . -v

######################################################################
## Meson Step-by-Step Compilation
######################################################################

## build-me
## Compile the project using Meson for step-by-step debugging.
## Depends on 'clean'.
build-me: clean
	@echo ">> Compiling for debugging project step-by-step using Meson..."
	@# pip install --no-cache-dir mesonbuild meson ninja

	@echo "   - Cleaning previous Meson build artifacts..."
	@meson clean -C builddir

	@echo "   - Creating new Meson build directory..."
	@meson setup builddir

	@# Optional steps (commented out for reference)
	@# @meson setup --reconfigure builddir
	@# @meson setup --wipe builddir
	@# @meson compile -C builddir
	@# @meson compile --clean
	@# @ninja -C builddir
	@# @ninja -C builddir test

## sdist
## Build source distribution and install it.
sdist:
	@echo ">> Building source distribution..."
	@python -m build --sdist -Csetup-args=-Dallow-noblas=true

	@echo ">> Installing source distribution..."
	@python -m pip install --no-cache-dir -v dist/*.gz -Csetup-args=-Dallow-noblas=true

	@# meson setup builddir && meson dist -C builddir --allow-dirty --no-tests --formats gztar
	@# python -m pip install --no-cache-dir -v builddir/*dist/*.gz -Csetup-args=-Dallow-noblas=true

######################################################################
## Testing & Examples
######################################################################
#
# Commands for running unit tests and executing example scripts.
#

## test
## Run all project tests using pytest.
## Optional: depends on 'clean-basic' to ensure fresh caches.
# test: clean-basic
# 	@echo ">> Running project tests with pytest..."

# 	@cd scikitplot && pytest tests/
# 	@echo ">> pytest testing completed."

## examples
## Execute Python scripts to generate example plots/images.
## Optional: depends on 'clean-basic' to ensure a clean environment.
# examples: clean-basic
# 	@echo ">> Generating example plots/images..."

# 	@# Example: manually execute a script
# 	# @cd galleries/examples && python classification/plot_feature_importances_script.py

# 	@python auto_building_tools/discover_scripts.py --save-plots
# 	@echo ">> All example scripts executed."

######################################################################
## Publishing to PyPI (release)
######################################################################
#
# Targets to verify and upload your package to PyPI.
# - Uses 'twine' to check and upload distribution files.
# - Ensure that version tagging and build artifacts are ready before publishing.
#

## check-release
## Verify distribution files and readiness for PyPI upload.
check-release:
	@echo ">> Checking that tagging is complete before publishing..."
	@echo ">> Verifying distribution files with twine..."

	@# Install twine if needed
	@pip install --no-cache-dir twine

	@twine check dist/* || true
	@echo ">> Distribution files verification completed."

	@# twine upload dist/*
	@# echo "PyPI publish completed."

## release
## Upload the distribution files to PyPI.
## Depends on 'check-release'.
release: check-release
	@echo ">> Uploading distribution files to PyPI..."
	@# Example with API token:
	@# twine upload dist/* --username __token__ --password <your_api_token>
	@twine upload dist/*
	@echo ">> PyPI publish completed."

######################################################################
## Git from scratch
######################################################################

## git clone --mirror https://github.com/scikit-plots/scikit-plots.git  backup-scikit-plots
## git init
## git branch -m master main
## git branch
## git remote add origin https://github.com/scikit-plots/scikit-plots.git
## echo "# scikit-plots starting fresh with no history" > README.md
## git add .
## git commit -m "scikit-plots starting fresh with no history"
## git status
## git push -u origin main
## git push -u origin main --force  # Force Push to Overwrite Remote Repository

######################################################################
## Submodules
## git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party
## git submodule foreach --recursive git pull
## git submodule update --init --recursive --remote
######################################################################

######################################################################
## Git reset
######################################################################

# reset:
# 	@git status --short
# 	@echo "Discards any local uncommitted changes in tracked files."
# 	@echo "Warning: This is destructive—any unsaved changes will be lost."
# 	@echo "If unsure, use 'git status' before running this."
# 	@echo "Are you sure you want to reset? (y/N)"; \
# 	read confirm || exit 0; \
# 	if [ "$$confirm" = "y" ]; then \
# 	    git checkout -f; \
# 	else \
# 	    echo "Reset aborted."; \
# 	fi

######################################################################
## Git Reset Your Local Branch to the Remote Version
######################################################################

## Be sure you want to completely throw away any uncommitted changes before running this.
## git checkout develop             # Switch to the branch you want to reset
## git fetch origin                 # Fetch the latest changes from the remote repository
## git reset --hard origin/develop  # Reset the branch to match the remote
## git status
## git log --oneline

######################################################################
## Git Search and fix
######################################################################

grep:
	@#grep -rn "interp(" .
	@#Use git grep (faster in Git repos)
	@git grep -n "interp("

######################################################################
## Git Branch Management
######################################################################
#
# Targets to create a new branch with the latest changes from main
# and push commits to the remote repository.
# - 'origin' is your fork
# - 'upstream' is the original repository (if needed)
#

## newbr
## Create a new branch based on the latest main branch and push to remote.
newbr:
	@echo ">> Creating new branch 'subpackage-bug-fix' based on main..."
	@# Ensure you are on main and pull latest changes
	@git switch main \
	&& git pull \
	@# Delete old local branch if exists
	&& git branch -d subpackage-bug-fix || true \
	&& git branch -D subpackage-bug-fix || true \
	@# Create and switch to new branch
	&& git switch -c subpackage-bug-fix \
	&& git push --set-upstream origin subpackage-bug-fix \
	&& git branch
	@echo ">> New branch created and pushed successfully."

	@# git checkout main  # or the branch you want as a base \
	## If you're on 'feature-branch' but want to pull from 'main' \
	## same git fetch origin main && git merge origin/main \
	# git pull origin main  \
	# git branch -d subpackage-bug-fix || true \
	# git checkout -b subpackage-bug-fix

## push
## Stage all changes, commit with a message, and push to the current branch.
push:
	@echo ">> Committing and pushing changes..."
	@pre-commit install

	@git add . \
	&& git commit -m "fix dependency" \
	&& git push
	@echo ">> Changes pushed successfully."



######################################################################
## Git Tag
## Use := for immediate expansion consistently.
## Use $(shell ...) only when needed.
## Use $(if ...) for fallback logic cleanly.
######################################################################

ifeq ($(OS),Windows_NT)
  NULL_DEVICE := NUL
else
  NULL_DEVICE := /dev/null
endif

# Get scikitplot version if installed, else fallback to 0.0.0
# echo $(scikitplot -V | awk '{print $$3}' || echo 0.0.0)
# VERSION := $(shell scikitplot -V | awk '{print $$3}')
VERSION := $(shell command -v scikitplot >$(NULL_DEVICE) 2>&1 && scikitplot -V | awk '{print $$3}' || echo 0.0.0)

# VERSION := $(shell python -c "import importlib; print(getattr(importlib.import_module('scikitplot'), '__version__', '0.0.0'))")
# VERSION := $(shell python -c "import scikitplot; print(scikitplot.__version__) if hasattr(scikitplot, '__version__') else print('0.0.0')" 2>$(NULL_DEVICE) || echo "0.0.0")
# VERSION := $(shell python -c "try: import scikitplot; print(scikitplot.__version__) except Exception: print('0.0.0')" 2>$(NULL_DEVICE) || echo 0.0.0)

cv:
	@echo "$(VERSION)"

# Git info
# Check the short commit hash
LAST_COMMIT_ID := $(shell git rev-parse --short HEAD)
# Get last commit message (plain text)
LAST_COMMIT_MESSAGE := $(shell git log -1 --pretty=%B)
# Extract version-like string from commit message (v1.2.3 or 1.2.3) from the commit message using shell + grep
VERSION_EXT := $(shell echo "$(LAST_COMMIT_MESSAGE)" | grep -oE 'v?[0-9]+\.[0-9]+\.[0-9]+' | head -1)

## $(if condition,then-part[,else-part]) $(if $(VERSION),$(VERSION),)
# Decide which version to use:
# If VERSION != 0.0.0, use VERSION
# Else if VERSION_EXT exists, use VERSION_EXT
# Else empty (will be handled below)
# Use VERSION unless it's 0.0.0, in which case fallback to VERSION_EXT
VERSION_USED := $(if $(filter-out 0.0.0,$(VERSION)),$(VERSION),$(VERSION_EXT))

# $(patsubst pattern, replacement, text)
# UNQUOTED := $(strip $(patsubst %",$(patsubst "\"%,\
#     $(patsubst %',$(patsubst "'%,\
#     $(VAR)))))
# Remove leading quote (either " or ')
# # RESULT now has no leading/trailing quotes
# VERSION_USED := $(strip $(VERSION_USED))
# VERSION_USED := $(patsubst "v%,%,$(VERSION_USED))  # removes leading double quote
# VERSION_USED := $(patsubst "\"%,%,$(VERSION_USED))  # removes leading double quote
# VERSION_USED := $(patsubst "'%,%,$(VERSION_USED)) # removes leading single quote
# # Remove trailing quote (either " or ')
# VERSION_USED := $(patsubst %\" ,%,$(VERSION_USED))  # removes trailing double quote
# VERSION_USED := $(patsubst %',%,$(VERSION_USED))    # removes trailing single quote

# Ensure version has a 'v' prefix; default to v0.0.0 if undefined or  empty
GIT_TAG := v$(patsubst v%,%,$(VERSION_USED))

# If VERSION_USED is empty, default to "v0.0.0"
# If VERSION_USED starts with 'v', use as-is
# Else prefix with 'v'

# Force bash to run the shell command explicitly:
GIT_TAG := $(shell bash -c '\
  if [ -z "${VERSION_USED:-}" ]; then \
    echo "v0.0.0"; \
  elif echo "$(VERSION_USED)" | grep -q "^v"; then \
    echo "$(VERSION_USED)"; \
  else \
    echo "v$(VERSION_USED)"; \
  fi')

# using pure Make functions (no shell):
# GIT_TAG := $(if $(VERSION_USED), \
#              $(if $(filter v%,$(VERSION_USED)),\
#                  $(VERSION_USED),\
#                  v$(VERSION_USED)), \
#              v0.0.0)
# GIT_TAG := $(strip $(GIT_TAG))

# # Compose release message
GIT_TAG_MESSAGE := Release version $(GIT_TAG)

# Debug output if DEBUG is true or VERSION is 0.0.0 (scikitplot missing)
ifeq ($(or $(DEBUG),$(filter 0.0.0,$(VERSION))),true)
  $(info [DEBUG] Using scikitplot version: $(VERSION))
  # $(warning "Warning: scikitplot is not installed. VERSION is set to MISSING.")
endif
# Debug output if DEBUG is true
ifeq ($(DEBUG),true)
  $(info [DEBUG] scikitplot version: $(VERSION))
  $(info [DEBUG] Last commit ID: $(LAST_COMMIT_ID))
  $(info [DEBUG] Last commit message: $(LAST_COMMIT_MESSAGE))
  $(info [DEBUG] Extracted version from commit: $(VERSION_EXT))
  $(info [DEBUG] Version used: $(VERSION_USED))
  $(info [DEBUG] Final tag (with v prefix): $(GIT_TAG))
  $(info [DEBUG] Tag message: $(GIT_TAG_MESSAGE))
endif

# make tag               # Quiet mode
# make tag DEBUG=true    # Show debug messages
# 	@if [ "$(DEBUG)" = "true" ]; then \
# 		echo "[DEBUG] GIT_TAG is $(GIT_TAG)"; \
# 	fi
tag:
	@echo TAG: "$(GIT_TAG)"

# 💡 Rule of thumb:
# 	Tag before commit → freeze the old state.
# 	Tag after commit → mark the new state.
archive:
ifdef GIT_TAG
	@echo "Remove tag locally: "$(GIT_TAG)""
	@#git tag -d "v0.4.0rc0"
	@git tag -d "$(GIT_TAG)" || true

	@echo "Creates tag locally"
	@#git tag -a "v0.4.0.post3" -m "Release version 0.4.0.post3"
	git tag -a "$(GIT_TAG)" -m "Release version $(GIT_TAG)"

	@echo "Pushes the tag to your fork"
	@# git push --tags                   # then push all tags
	git push origin "$(GIT_TAG)"        # push just that tag

	@echo "Pushes tag to upstream (if allowed) "$(GIT_TAG)" (Month Dayth, Year)"
	@#git push upstream "v0.4.0.post3"
	git push upstream "$(GIT_TAG)"      # push just that tag

	@# git ls-remote --tags origin       # verify what's pushed by running
else
	@echo "GIT_TAG is not defined!"
endif

## Tagging the latest commit
## For larger projects or those requiring stability guarantees, tagging in stable is safer.
# tag-sample:
# 	@echo "Sample tag: '$(TAG_SAMPLE)' message: '$(TAG_MESSAGE)' by commit: '$(LAST_COMMIT_MESSAGE)'"

## Add a Tag to Stable Releases to the Local project
# tag:
# ifdef BR
# 	@## Tagging in the stable Branch (Stability-First Workflow)
# 	@#Best practice: Tag before PyPI publishing.
# 	@echo "Adding tag to branch: '$(BR)'"
# 	@git checkout "$(BR)"
# 	@echo "Existing tags:"
# 	@git tag
# 	@echo "Adding local tag: $(TAG_SAMPLE) message: $(TAG_MESSAGE)"
# 	@# git tag -a v0.4.0 -m "Release version 0.4.0"
# 	@git tag -a "$(TAG_SAMPLE)" -m $(TAG_MESSAGE)
# 	@echo "Local tagging completed."
# else
# 	@echo "BR is not defined"
# endif

## Delete the Tag locally use GIT_TAG Environment for del.
# tag-del:
# ifdef GIT_TAG
# 	@echo "Deleting Local tag: 'v$(GIT_TAG)'"
# 	@git tag -d "v$(GIT_TAG)"
# 	@git tag
# else
# 	@echo "GIT_TAG is not defined"
# endif

## Delete the Tag remotely use GIT_TAG Environment for del.
# tag-delr:
# ifdef GIT_TAG
# 	@echo "Deleting Remote tag: 'v$(GIT_TAG)'"
# 	@git push origin --delete "v$(GIT_TAG)"
# 	@git tag
# else
# 	@echo "GIT_TAG is not defined"
# endif

## Push the tag to the remote repository
# tag-push:
# 	@echo "Existing tags:"
# 	@git tag
# 	@echo "Adding to "remote repository" tag $(TAG_SAMPLE)..."
# 	@git push origin $(TAG_SAMPLE) || git push --tags
# 	@echo "Remote Repository Tagging completed."."

## Release combines tagging and pushing the tag to remote
# release: tag-sample tag tag-push
# 	@echo "Ready to Publish on PyPI"

######################################################################
## Git Branch
######################################################################

## Add a Branch to the Local project
## maintenance/0.3.x
# branch:
# ifdef BR
# 	@echo "Adding Local branch: '$(BR)' to main"
# 	@git checkout main
# 	@## git branch "$(BR)" && git checkout "$(BR)" || git switch "$(BR)"
# 	@## This command creates a new branch and switches to it immediately.
# 	@# git checkout -b "$(BR)"
# 	@git switch -c "$(BR)"
# 	@## Commit changes incrementally
# 	@cat Readme.md > Readme_$(BR).md
# 	@git add .
# 	@git commit -m "$(BR) initial commit (endpoints) for consistency"
# else
# 	@echo "BR is not defined"
# endif

## Delete the branch locally use BR Environment for del.
# branch-del:
# ifdef BR
# 	@echo "Deleting (safe) Local branch: '$(BR)'"
# 	@# git branch -D "$(BR)"
# 	@git branch -d "$(BR)"
# 	@git branch
# else
# 	@echo "BR is not defined"
# endif

## Delete the branch remotely use BR Environment for del.
# branch-delr:
# ifdef BR
# 	@echo "Deleting Remote branch: '$(BR)'"
# 	@git push origin --delete "$(BR)"
# 	@git branch
# else
# 	@echo "BR is not defined"
# endif

## Delete a feature branch after merging
## Periodically delete old local branches that have already been merged to keep your workspace clean.
# branch-clean:
# ifdef BR
# 	@echo "Deleting local old branches..."
# 	@git branch --merged main | grep -v "main" | xargs git branch -d
# 	@git branch
# else
# 	@echo "BR is not defined"
# endif

## Push the updated stable branch to the remote repository
## The -u (or --set-upstream) flag tells Git to link the current local branch to a branch on the remote (or create a new remote branch if it doesn't exist).
## Use git push -u origin <branch> When: Pushing a new branch to the remote for the first time.
## Use git push origin <branch> When: The branch already has an upstream tracking relationship, and you don't need to set it again.
# branch-push:
# ifdef BR
# 	@echo "To check if your branch is tracked upstream, use:"
# 	@git branch -vv
# 	@echo "Adding Remote branch: '$(BR)'"
# 	@git push -u origin "$(BR)"
# 	@echo "$(BR) You’ve finished the work and are ready to merge into the main branch."
# else
# 	@echo "BR is not defined"
# endif

######################################################################
##
######################################################################
