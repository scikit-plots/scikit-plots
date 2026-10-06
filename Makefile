## Makefile for Python Packaging Library
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# This Makefile defines various targets for project management tasks,
# including building the project, cleaning up build artifacts, running tests,
# creating Docker images, and more.
#
# Important:
# When you define a function that calls $(MAKE), you must add the @+ after the equality sign.
# So like: sub_make = @+$(MAKE) -C $(TOPDIR)/subdir/$(1) $(2).
# Otherwise make won't pass descriptors and options it ought to to submake calls,
# which would manifest itself as a warning: jobserver unavailable: using -j1.
# Add '+' to parent make rule.
#
# Notes:
# - .ONESHELL ensures that all commands in a target run within a single shell,
#   so variables and environment changes persist across commands.
# - For cleaner multi-line logic, use .ONESHELL.
#   The '@' symbol only suppresses output for the first command.
# - To enable shell debugging, uncomment the following line:
#     SHELL = /bin/bash -x
#
# Run each recipe in one Bash process.
# With .ONESHELL := true, you no longer need a backslash at the end of every recipe line.
# .ONESHELL := true
.ONESHELL:

# SHELL := /bin/bash
SHELL := bash

# set -Eeuxo pipefail || echo "pipefail not available"
# The flags mean:
# 	-E: inherit ERR traps in functions and subshells.
# 	-e: exit when a command fails.
# 	-u: fail on unset variables.
# 	-x: print commands before executing them.
# 	-o pipefail: fail a pipeline if any command in it fails.
# Apply strict mode to every Make recipe.
# .SHELLFLAGS := -Eeuxo pipefail -c
.SHELLFLAGS := -Eeuo pipefail -c

## Global logging configuration.
## Supported styles:
## 	plain: 1791247515.851565               # Epoch timestamps
## 	human: 2026-10-06T00:45:15+0000        # ISO-style timestamps
## 	github: Mon, 05 Oct 2026 18:21:15 GMT  # GitHub-style timestamps
## If you also export PS4 globally for commands executed by Make’s $(shell ...), make it safe under set -u:
## export PS4 := + [$${EPOCHREALTIME}] $${BASH_SOURCE[0]:-main}:$${LINENO:-0}:
## PS4='$${EPOCHREALTIME:-0} $${BASH_SOURCE[0]:-main}:$${LINENO:-0}: '
## PS4='$$(printf "%(%Y-%m-%dT%H:%M:%S%z)T" -1) $${BASH_SOURCE[0]:-main}:$${LINENO:-0}: '
PS4='$$(LC_ALL=C date -u "+%a, %d %b %Y %H:%M:%S GMT") $${BASH_SOURCE[0]:-main}:$${LINENO:-0}: '
export PS4

# Make command-line variables available to Bash.
TIME_STYLE ?= github
TRACE ?= 0
DRY_RUN ?= 0
VERIFY ?= 1
export TIME_STYLE
export TRACE
export DRY_RUN
export VERIFY

# Makefile
# tools/
# └── make-env.bash
# Bash automatically sources this file for every non-interactive shell.
# export BASH_ENV := $(CURDIR)/tools/make/make-env.bash >/dev/null 2>&1  || true

# PHONY targets:
# These targets are not associated with files or directories. Declaring them as
# phony avoids conflicts with files or folders of the same name.
#
# Available phony targets: help, all, clean, publish
.PHONY: check-tools help all clean publish newbr
# .DEFAULT_TARGET: all


REQUIRED_TOOLS := \
	bash \
	ln \
	realpath \
	readlink

check-tools:
	@for tool in $(REQUIRED_TOOLS); do \
		command -v "$$tool" >/dev/null 2>&1 || { \
			echo "ERROR: Required tool not found: $$tool" >&2; \
			exit 1; \
		}; \
	done

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
# clean-basic:
# 	@echo ">> Starting basic cleaning..."
# 	@# Remove protobuf caches
# 	@#sudo -H rm -rf ./.cache/protobuf_cache || true
# 	@# Remove pip caches
# 	@rm -rf ~/.cache/pip
# 	@pip cache purge || true
# 	@echo "   - Removed pip cache files"
# 	@# Remove Jupyter checkpoints
# 	@# rm -rf `find -L . -type d -name ".ipynb_checkpoints" -not -path "./third_party/*"`
# 	@find . -name '.ipynb_checkpoints' -not -path './third_party/*' -exec rm -rf {} +
# 	@rm -rf "./third_party/.ipynb_checkpoints"
# 	@echo "   - Removed '.ipynb_checkpoints'"
# 	@# Remove Python cache directories
# 	@# rm -rf `find -L . -type d -name "__pycache__" -not -path "./third_party/*"`
# 	@find . -name '__pycache__' -not -path './third_party/*' -exec rm -rf {} +
# 	@echo "   - Removed '__pycache__'"
# 	@# Remove zip leftovers
# 	@# rm -rf `find -L . -type d -name "__MACOSX" -not -path "./third_party/*"`
# 	@find . -name '__MACOSX' -not -path './third_party/*' -exec rm -rf {} +
# 	@echo "   - Removed '__MACOSX'"
# 	@# Remove VSCode configs
# 	@# rm -rf `find -L . -type d -name ".vscode" -not -path "./third_party/*"`
# 	@find . -name '.vscode' -not -path './third_party/*' -exec rm -rf {} +
# 	@echo "   - Removed '.vscode'"
# 	@# Remove type checker/linter caches
# 	@rm -rf ".mypy_cache" ".ruff_cache"
# 	@echo "   - Removed mypy and ruff caches"
# 	@# Remove Gradio cache
# 	@rm -rf ".gradio"
# 	@echo "   - Removed Gradio cache"
# 	@# Remove matplotlib result images
# 	@rm -rf "result_images"
# 	@find . -name 'result_images' -not -path './third_party/*' -exec rm -rf {} +
# 	@echo "   - Removed 'result_images' from matplotlib builds 'matplotlib.sphinxext.plot_directive'"
# 	@# Remove pytest cache
# 	@# rm -rf `find -L . -type d -name ".pytest_cache" -not -path "./third_party/*"`
# 	@find . -name '.pytest_cache' -not -path './third_party/*' -exec rm -rf {} +
# 	@echo "   - Removed '.pytest_cache'"
# 	@echo ">> Basic cleaning completed."

# Centralized patterns for index / ANN artifacts
ANN_INDEX_EXTS := -name '*.annoy' -o -name '*.tree' -o -name '*.ann' -o -name '*.voyager' -o -name '*.voy' -o -name '*.idx' -o -name '*.hdf5'
# \( -path './.git' -o -path './third_party' \) -prune -o \
# \( $(ANN_INDEX_EXTS) \) \

## clean-basic
## Remove development caches and temporary files, excluding 'third_party'.
clean-basic:
	@set -euo pipefail; \
	echo ">> Starting basic cleaning..."; \
	\
	echo ">> [1] Pip cache"; \
	rm -rf "$$HOME/.cache/pip" || true; \
	pip cache purge >/dev/null 2>&1 || true; \
	echo "   - Removed pip cache files"; \
	\
	echo ">> [2] Jupyter checkpoints"; \
	find . -name '.ipynb_checkpoints' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	rm -rf "./third_party/.ipynb_checkpoints" || true; \
	echo "   - Removed '.ipynb_checkpoints'"; \
	\
	echo ">> [3] Python bytecode caches"; \
	find . -name '__pycache__' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	echo "   - Removed '__pycache__'"; \
	\
	echo ">> [4] Zip leftovers"; \
	find . -name '__MACOSX' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	echo "   - Removed '__MACOSX'"; \
	\
	echo ">> [5] VSCode configs"; \
	find . -name '.vscode' \
		-not -path './.vscode' \
		-not -path './.git_clones/*' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	echo "   - Removed '.vscode'"; \
	\
	echo ">> [6] Type checker / linter caches"; \
	rm -rf ".mypy_cache" ".ruff_cache"; \
	echo "   - Removed mypy and ruff caches"; \
	\
	echo ">> [7] Gradio cache"; \
	rm -rf ".gradio"; \
	echo "   - Removed Gradio cache"; \
	\
	echo ">> [8] Matplotlib result images"; \
	rm -rf "result_images" || true; \
	find . -name 'result_images' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	echo "   - Removed 'result_images'"; \
	\
	echo ">> [9] Pytest cache"; \
	find . -name '.pytest_cache' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	echo "   - Removed '.pytest_cache'"; \
	\
	echo ">> [10] ANN/Voyager index artifacts"; \
	find . \
		\( $(ANN_INDEX_EXTS) \) \
		-not -path './.git/*' \
		-not -path './.git_clones/*' \
		-not -path './third_party/*' \
		-not -path './scikitplot/annoy/tests/test.tree' \
		-not -path './scikitplot/annoy/tests/test64.tree' \
		-not -path './galleries/examples/annoy/test.tree' \
		-not -path './galleries/examples/annoy/test64.tree' \
		-type f -print -delete || true; \
	echo "   - Removed: *.annoy *.tree *.ann *.voyager *.voy *.idx *.hdf5"; \
	\
	echo ">> Basic cleaning completed."

## clean-test
## Remove testing and coverage artifacts.
clean-test:
	@set -euo pipefail; \
	echo ">> Starting test cleaning..."; \
	\
	echo ">> [1] Pytest cache"; \
	find . -name '.pytest_cache' \
		-not -path './third_party/*' \
		-exec rm -rf {} +; \
	echo "   - Removed '.pytest_cache'"; \
	\
	echo ">> [2] Coverage artifacts"; \
	rm -rf .tox/ htmlcov/ || true; \
	rm -f .coverage coverage.xml || true; \
	echo "   - Removed tox environments and coverage reports"; \
	\
	echo ">> [3] ANN/Voyager index artifacts"; \delete || true; \
	find . \
		\( $(ANN_INDEX_EXTS) \) \
		-not -path './.git/*' \
		-not -path './.git_clones/*' \
		-not -path './third_party/*' \
		-not -path './scikitplot/annoy/tests/test.tree' \
		-not -path './scikitplot/annoy/tests/test64.tree' \
		-not -path './galleries/examples/annoy/test.tree' \
		-not -path './galleries/examples/annoy/test64.tree' \
		-type f -print -delete || true; \
	echo "   - Removed: *.annoy *.tree *.ann *.voyager *.voy *.idx *.hdf5"; \
	\
	echo ">> Test cleaning completed."

## clean
## Perform a full cleanup: includes clean-basic + build artifacts.
clean: clean-basic
	@set -euo pipefail; \
	echo ">> Starting full cleanup..."; \
	\
	echo ">> [1] Build directories and egg-info"; \
	rm -rf "build" "build_dir" "builddir" "dist" "scikit_plots.egg-info" *.egg-info* || true; \
	echo "   - Removed build directories and egg-info files"; \
	\
	echo ">> [2] Compiled shared objects in build dirs"; \
	find -L -type f -name "*.so" -path "*/build*" -exec rm -rf {} +; \
	echo "   - Removed '*.so' files in build directories"; \
	\
	echo ">> [3] Meson-related files"; \
	rm -rf .meson .mesonpy-* || true; \
	echo "   - Removed Meson-related files (.meson, .mesonpy-*)"; \
	\
	echo ">> [4] Uninstall local package"; \
	pip uninstall scikit-plots -y >/dev/null 2>&1 || true; \
	echo "   - Uninstalled local 'scikit-plots' package (if present)"; \
	\
	echo ">> [5] ANN/Voyager index artifacts"; \
	find . \
		\( $(ANN_INDEX_EXTS) \) \
		-not -path './.git/*' \
		-not -path './.git_clones/*' \
		-not -path './third_party/*' \
		-not -path './scikitplot/annoy/tests/test.tree' \
		-not -path './scikitplot/annoy/tests/test64.tree' \
		-not -path './galleries/examples/annoy/test.tree' \
		-not -path './galleries/examples/annoy/test64.tree' \
		-type f -print -delete || true; \
	echo "   - Removed: *.annoy *.tree *.ann *.voyager *.voy *.idx *.hdf5"; \
	\
	echo ">> Full cleanup completed."

######################################################################
## Project Structure
######################################################################

## tree
## Print the project structure (directories, cross-platform).
tree:
	@echo ">> System detected: $(SYSTEM)"
	@$(TREE_CMD)

######################################################################
## Repair
######################################################################

git-check:
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1

git_diff:
	@git diff 1bf588a 2c378ee -- meson.build scikitplot/config/meson.build > my.patc

git_reidx:
	@set -euo pipefail; \
	echo ">> Rebuilding Git pack indexes (*.idx) from existing *.pack files..."; \
	PACKS="$$(ls -1 .git/objects/pack/*.pack 2>/dev/null || true)"; \
	if [ -z "$$PACKS" ]; then \
		echo ">> No .git/objects/pack/*.pack files found. Nothing to re-index."; \
		exit 0; \
	fi; \
	echo "$$PACKS"; \
	for p in $$PACKS; do \
		echo ">> Re-indexing: $$p"; \
		git index-pack "$$p"; \
	done; \
	echo ">> Done: pack indexes rebuilt."

git_verify:
	@set -euo pipefail; \
	echo ">> Verifying repository integrity (git fsck --full)..."; \
	git fsck --full; \
	echo ">> OK: git fsck completed."

git_refs_check:
	@set -euo pipefail; \
	echo ">> Checking remote-tracking refs..."; \
	if git show-ref --verify --quiet refs/remotes/upstream/main; then \
		echo ">> OK: upstream/main -> $$(git rev-parse --short refs/remotes/upstream/main)"; \
	else \
		echo ">> WARN: refs/remotes/upstream/main not found"; \
	fi; \
	if git show-ref --verify --quiet refs/remotes/origin/main; then \
		echo ">> OK: origin/main   -> $$(git rev-parse --short refs/remotes/origin/main)"; \
	else \
		echo ">> WARN: refs/remotes/origin/main not found"; \
	fi

git_refs_verify:
	@set -euo pipefail; \
	echo ">> Verifying required remote-tracking refs exist..."; \
	git show-ref --verify refs/remotes/upstream/main >/dev/null; \
	git show-ref --verify refs/remotes/origin/main >/dev/null; \
	echo ">> OK: both refs exist."

git_repair: git_reidx git_verify git_refs_check
	@echo ">> Repair complete."

# just a zip at that commit
# git checkout 01641d8bb6a148d7d0d6754b086d970caffb7235
# (optional) create a working branch from that commit
# git switch -c annoy-src-01641d8
git_zip:
	curl -L -o scikit-plots-01641d8.zip \
	https://github.com/scikit-plots/scikit-plots/archive/01641d8bb6a148d7d0d6754b086d970caffb7235.zip
	unzip scikit-plots-01641d8.zip
	cd scikit-plots-01641d8bb6a148d7d0d6754b086d970caffb7235/scikitplot/cexternals/_annoy/src
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
## ln -srfT -- "$src" "$dst" -> can replace an existing regular file at "$dst".
sym:
	@echo ">> Creating symbolic links..."

	@# Remove existing links or directories
	@# unlink ".devcontainer/scripts" "environment.yml"
	@rm -rf ".devcontainer/scripts" "environment.yml"

	@# Create symbolic links
	@ln -srfT -- "docker/scripts/" ".devcontainer/scripts"
	@ln -srfT -- "./docker/env_conda/environment.yml" "environment.yml"

	@echo ">> Symbolic links created successfully."

	@# Optional: list all symbolic links for verification
	@find . -type l -exec ls -l {} +

	@# find . -type l
	@# find . -type l -exec ls -l {} +
	@# find . -type l -exec readlink -f {} \; -exec ls -l {} \;

# --------------------------------------------------------------------
# Partial distributions (libs/<name>)
# --------------------------------------------------------------------
#
# scikit-plots is also published as small, focused distributions built from
# this same source tree (scikit-plots-skinny, scikit-plots-rank-bm25, ...).
# See libs/README.md.
#
# Nothing is linked or copied into libs/ permanently. Each libs/<name>/setup.py
# stages the files its distribution owns when it builds and removes them when
# the build exits, so these targets work the same on Linux, macOS and Windows
# and need no symbolic links. (This replaces the former `sym-libs` target.)
#
#   make libs-list      distributions and what each one owns
#   make libs-gen       rewrite the generated files in libs/ (commit the result)
#   make libs-check     fail if a generated file is stale
#   make libs-test      run the tooling's own tests
#   make libs-build     build every sdist and wheel into $(LIBS_OUTDIR)
#   make libs-verify    build, then install and test them in clean environments
#   make libs-clean     remove staged files and build residue from libs/
#
# Select distributions with LIBS, Python versions with LIBS_PYTHONS:
#   make libs-build  LIBS="scikit-plots-rank-bm25 scikit-plots-corpus"
#   make libs-verify LIBS_PYTHONS="3.9 3.13"
#
# libs-build needs `python -m build`; libs-verify also needs `uv` on PATH.

LIBS_PYTHON ?= python
LIBS_OUTDIR ?= dist/libs
LIBS ?=
LIBS_PYTHONS ?=

.PHONY: libs-list libs-gen libs-check libs-test libs-build libs-verify libs-clean

libs-list:
	@$(LIBS_PYTHON) -m libs._tools list

libs-gen:
	@$(LIBS_PYTHON) -m libs._tools generate

libs-check:
	@$(LIBS_PYTHON) -m libs._tools check

libs-test:
	@$(LIBS_PYTHON) -m pytest libs/_tools/tests

libs-build: libs-check
	@$(LIBS_PYTHON) -m libs._tools build $(LIBS) --outdir "$(LIBS_OUTDIR)"

libs-verify: libs-check
	@$(LIBS_PYTHON) -m libs._tools verify $(LIBS) --outdir "$(LIBS_OUTDIR)" \
		$(foreach version,$(LIBS_PYTHONS),--python $(version))

libs-clean:
	@$(LIBS_PYTHON) -m libs._tools unstage

# python -m libs._tools check && python -m pytest libs/_tools/tests
libs-apply:
	@$(LIBS_PYTHON) -m libs._tools check && $(LIBS_PYTHON) -m pytest libs/_tools/tests

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
	# micromamba create -n py312 python=3.12 ipykernel -y
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
ins: clean
	@echo ">> Installing package locally (non-editable)..."
	@python -m pip install --no-build-isolation --no-cache-dir . -v

	@# python -m pip install --no-cache-dir .
	@# python -m pip install --no-cache-dir --use-pep517 .

## dev
## Install development version of scikit-plots (editable), depends on 'clean' and 'dep'.
## ➡️ Works only if "ninja" and "all build deps" are installed in the environment (--no-build-isolation).
dev: clean
	@echo ">> Installing package locally (editable, development mode)..."
	@# Requires ninja and all build dependencies installed
	@python -m pip install --no-build-isolation --no-cache-dir -e . -v

	@# python -m pip install --no-build-isolation --no-cache-dir --editable . -vvv
	@# python -m pip install --no-build-isolation --no-cache-dir -e .[build,dev,test,doc] -v

	@# find . -exec touch {} + && python -m pip install --no-build-isolation --no-cache-dir -e . -v

## If using bash/zsh and want cleaner syntax (|& = stdout + stderr)
# python -m pip install --no-build-isolation --no-cache-dir -e . -v |& tee build.log
## Append instead of overwrite
# python -m pip install --no-build-isolation --no-cache-dir -e . -v 2>&1 | tee -a pytest.log
## Keep colored output (optional)
# PYTHONUNBUFFERED=1 python -m pip install --no-build-isolation --no-cache-dir -e . -v 2>&1 | tee build.log
build:
	@python -m pip install --no-build-isolation --no-cache-dir -e . -v 2>&1 | tee build.log

######################################################################
## Meson Step-by-Step Compilation
######################################################################

## https://www.msys2.org/
## https://github.com/msys2
## https://github.com/msys2/msys2-docker
## https://github.com/msys2/MINGW-packages
## https://github.com/msys2/MSYS2-packages
#
## https://www.mingw-w64.org/
## https://github.com/mingw-w64/mingw-w64
#
## MinGW-W64 compiler binaries
## Release of 15.2.0-rt_v13-rev1 sha256:029bd02b5bce7c10fd9476165b3fe178239fe1838ad62516b5c3e0921bb283cf
## https://github.com/niXman/mingw-builds-binaries/releases/download/15.2.0-rt_v13-rev1/x86_64-15.2.0-release-posix-seh-ucrt-rt_v13-rev1.7z
## https://github.com/niXman/mingw-builds-binaries/releases
## https://github.com/niXman/mingw-builds
## https://github.com/digitalgust/mingw-w64/releases
#
## $env:CC = "gcc"; $env:FC = "gfortran";
## export CC=gcc; export FC=gfortran;
## mamba install -c conda-forge openblas libgfortran5 pkg-config ninja meson-python numpy scipy cython pybind11

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

# git log --oneline | grep -iE 'release|setup'              # regex
# git log --oneline | grep -iE '\b(release|setup)\b'        # regex
# git log --oneline | grep -i -e release -e setup           # multiple explicit patterns
# git log --oneline --grep='release' --grep='setup' -i      # Git-native filtering Compact, single-line summary per commit, Scanning history quickly
# git log --grep='release' --grep='setup' -i                # Git-native filtering Multi-line per commit, Full , verbose commit representation
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

# git ls-remote --heads origin | grep <branch_name>
# git clone --branch <branch_name> --single-branch https://github.com/<org>/<repo>.git
# git clone -b <branch_name> --single-branch --depth 1 <repo_url>  # Same, but shallow (faster / less history)
# git switch -c <branch_name> --track origin/<branch_name>
git_clone:
	git clone <repo_url>
	cd <repo>
	git fetch origin <branch_name>
	git checkout -b <branch_name> --track origin/<branch_name>

git_pre_push_check:
	@set -Eeuo pipefail; \
	echo ">> Pre-push checks..."; \
	echo ">> [1/4] Working tree clean?"; \
	git diff --quiet && git diff --cached --quiet || { echo "!! Dirty working tree. Commit/stash first."; exit 1; }; \
	echo ">> [2/4] Repo integrity (git fsck --full)"; \
	git fsck --full >/dev/null; \
	echo ">> [3/4] Required remote refs exist?"; \
	git show-ref --verify refs/remotes/origin/main >/dev/null; \
	git show-ref --verify refs/remotes/upstream/main >/dev/null; \
	echo ">> [4/4] Remotes reachable?"; \
	git ls-remote --exit-code origin HEAD >/dev/null; \
	echo ">> OK: pre-push checks passed."

infobr:
	@git status -sb; git log -1 --oneline;

## LOCAL:
##   main
##   subpackage-bug-fix
## REMOTE:
##   origin/main
##   origin/subpackage-bug-fix
##
## git checkout main
## git pull origin main
## 1. Try safe delete (-d) discard stderr messages
## 2. If that fails, force delete (-D) discard stderr messages
## 3. If that also fails, ignore error
## -d = safe delete
## -D = FORCE delete --delete --force
## git push origin --delete subpackage-bug-fix == git push origin :subpackage-bug-fix
# newbr:
# 	@set -Eeuo pipefail; \
# 	git switch main; \
# 	git pull origin main; \
# 	git branch -d subpackage-bug-fix 2>/dev/null || git branch -D subpackage-bug-fix 2>/dev/null || true; \
# 	git switch -c subpackage-bug-fix; \
# 	git push -u origin subpackage-bug-fix; \
# 	git branch && echo ">> New branch created and pushed successfully."

## -----------------------------------------------------------------------------
## Create a fresh feature branch from the latest origin/main
##
## Behavior
## --------
## 1. Switch to local main branch
## 2. Synchronize with origin/main using fast-forward only
## 3. Delete any existing local feature branch safely
## 4. Recreate the feature branch from updated main
## 5. Push branch to origin and set upstream tracking
##
## Properties
## ----------
## - Idempotent
## - Fail-fast
## - Safe against accidental merge commits
## - Safe against stale local state
## - Robust shell semantics
## - Explicit error handling
##
## Requirements
## ------------
## - Git >= 2.23 (for `git switch`)
## - Clean working tree recommended
##
## Usage
## -----
## make newbr
## -----------------------------------------------------------------------------

BRANCH_NAME ?= subpackage-bug-fix
BASE_BRANCH ?= main
REMOTE_NAME ?= origin

newbr:
	@set -Eeuo pipefail; \
	\
	echo ">> Verifying git repository ..."; \
	git rev-parse --is-inside-work-tree >/dev/null; \
	\
	echo ">> Fetching latest remote references ..."; \
	git fetch --prune "$(REMOTE_NAME)"; \
	\
	echo ">> Switching to $(BASE_BRANCH) ..."; \
	git switch "$(BASE_BRANCH)"; \
	\
	echo ">> Synchronizing local $(BASE_BRANCH) with $(REMOTE_NAME)/$(BASE_BRANCH) ..."; \
	git pull --ff-only "$(REMOTE_NAME)" "$(BASE_BRANCH)"; \
	\
	echo ">> Verifying clean working tree ..."; \
	test -z "$$(git status --porcelain)" || { \
		echo "ERROR: Working tree is not clean."; \
		echo "Commit, stash, or discard changes before running this target."; \
		exit 1; \
	}; \
	\
	if git show-ref --verify --quiet "refs/heads/$(BRANCH_NAME)"; then \
		echo ">> Removing existing local branch: $(BRANCH_NAME)"; \
		git branch -D "$(BRANCH_NAME)"; \
	fi; \
	\
	echo ">> Creating fresh branch: $(BRANCH_NAME)"; \
	git switch -c "$(BRANCH_NAME)"; \
	\
	if git ls-remote --exit-code --heads "$(REMOTE_NAME)" "$(BRANCH_NAME)" >/dev/null 2>&1; then \
		echo ">> Remote branch exists. Replacing remote branch safely ..."; \
		git push --force-with-lease --set-upstream "$(REMOTE_NAME)" "$(BRANCH_NAME)"; \
	else \
		echo ">> Publishing new remote branch ..."; \
		git push --set-upstream "$(REMOTE_NAME)" "$(BRANCH_NAME)"; \
	fi; \
	\
	echo ">> Current branches:"; \
	git branch --all; \
	\
	echo ">> Branch '$(BRANCH_NAME)' created and synchronized successfully."

REMOTE ?= origin
BASE   ?= main
BRANCH ?= subpackage-bug-fix
FORCE  ?= 0   # set FORCE=1 to push --force-with-lease

recreate-branch:
	@# Ensure we're inside a git repo
	git rev-parse --is-inside-work-tree >/dev/null

	@# Safeguard: require clean working tree
	if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "ERROR: Uncommitted changes detected. Commit/stash before running."; \
		exit 1; \
	fi

	@# Safeguard: ensure remote exists
	git remote get-url "$(REMOTE)" >/dev/null

	@# Update base branch safely
	git fetch "$(REMOTE)" --prune
	git switch "$(BASE)"
	git pull --ff-only "$(REMOTE)" "$(BASE)"

	@# Delete local branch if it exists (but never delete the checked-out branch)
	@if git show-ref --verify --quiet "refs/heads/$(BRANCH)"; then \
		if [ "$$(git symbolic-ref --short HEAD)" = "$(BRANCH)" ]; then \
			echo "ERROR: Currently on '$(BRANCH)'. Switch away before recreating."; \
			exit 1; \
		fi; \
		git branch -D "$(BRANCH)"; \
	fi

	@# Create branch from updated base
	@git switch -c "$(BRANCH)"

	@# Push (optionally force-with-lease)
	@if [ "$(FORCE)" = "1" ]; then \
		git push -u --force-with-lease "$(REMOTE)" "$(BRANCH)"; \
	else \
		git push -u "$(REMOTE)" "$(BRANCH)"; \
	fi

	@git status -sb
	@echo ">> Branch '$(BRANCH)' created from '$(BASE)' and pushed to '$(REMOTE)'."

## push
## Stage all changes, commit with a message, and push to the current branch.
push:
	@echo ">> Committing and pushing changes..."
	@pre-commit install

	@git add . \
	&& git commit -m "fix dependency" \
	&& git push
	@echo ">> Changes pushed successfully."

# SHELL := /bin/bash

git_push_force_branch:
	@set -euo pipefail; \
	BR="subpackage-bug-fix"; REMOTE="origin"; \
	echo ">> Force-pushing '$$BR' to '$$REMOTE' (force-with-lease)..."; \
	git push --force-with-lease "$$REMOTE" "$$BR"; \
	echo ">> Done."

git_verify_remote_head:
	@set -euo pipefail; \
	BR="subpackage-bug-fix"; REMOTE="origin"; \
	LOCAL="$$(git rev-parse HEAD)"; \
	REMOTE_SHA="$$(git ls-remote --heads "$$REMOTE" "$$BR" | awk '{print $$1}')"; \
	echo "local : $${LOCAL:0:12}"; \
	echo "remote: $${REMOTE_SHA:0:12}"; \
	if [ -z "$$REMOTE_SHA" ]; then \
		echo "NOT OK: remote branch '$$REMOTE/$$BR' not found"; \
		exit 1; \
	fi; \
	if [ "$$LOCAL" = "$$REMOTE_SHA" ]; then \
		echo "OK: remote matches local"; \
	else \
		echo "NOT OK: remote differs"; \
		exit 1; \
	fi

# cat -A Makefile | sed -n '940,960p'
yml:
	@python tools/scripts/check_yaml.py

######################################################################
## Git Branch
######################################################################

# ⛔ Merge error Head branch is out of date. Review and try the merge again.
# 1. Refresh / re-run mergeability check
empty:
	@git commit --allow-empty -m "Trigger PR refresh"
	@git push

######################################################################
## Git Tag
## Use := for immediate expansion consistently.
## Use $(shell ...) only when needed.
## Use $(if ...) for fallback logic cleanly.
######################################################################

# PYTHON ?= python3
PYTHON ?= python
DEBUG ?= false

# User-configurable values.
VERSION ?=
VERSION_FILE ?= pyproject.toml

# Development fallback.
# DEFAULT_VERSION ?= 0.0.dev0
DEFAULT_VERSION ?= 0.0.0

# Git availability.
GIT := $(shell command -v git 2>/dev/null)

ifeq ($(strip $(GIT)),)
	GIT_AVAILABLE := false
else
	GIT_AVAILABLE := true
endif

# --------------------------------------------------------------------
# Git metadata
# --------------------------------------------------------------------

ifeq ($(GIT_AVAILABLE),true)

GIT_ROOT := $(strip $(shell git rev-parse --show-toplevel 2>/dev/null))
GIT_COMMIT_ID := $(strip $(shell git rev-parse --short=12 HEAD 2>/dev/null))
GIT_COMMIT_SUBJECT := $(strip $(shell git log -1 --format=%s 2>/dev/null))

# Prefer the nearest existing version tag.
GIT_VERSION_TAG := $(strip \
	$(shell git describe --tags --match 'v[0-9]*.[0-9]*.[0-9]*' \
		--abbrev=0 2>/dev/null) \
)

else

GIT_ROOT :=
GIT_COMMIT_ID :=
GIT_COMMIT_SUBJECT :=
GIT_VERSION_TAG :=

endif

## Extract v1.2.3 or 1.2.3 from the commit subject.
##
## The Git output is piped directly into sed. It is not interpolated into
## another shell command, which avoids quoting and injection problems.
##
## The error is caused by the unescaped capture-group parentheses in the sed expression:
## 	(v?[0-9]+\.[0-9]+\.[0-9]+)
## make parses $(strip ...) and $(shell ...) before invoking the shell.
## It interprets those ( and ) characters as Make function syntax, so it thinks the strip call is unbalanced.
## Avoid capture groups and let grep extract the match:
# GIT_VERSION_FROM_SUBJECT := $(strip \
# 	$(shell \
# 		if command -v git >/dev/null 2>&1; then \
# 			git log -1 --format=%s 2>/dev/null \
# 			| sed -nE 's/.*(v?[0-9]+\.[0-9]+\.[0-9]+).*/\1/p' \
# 			| sed -n '1p'; \
# 		fi \
# 	)
# )
# Extract v1.2.3 or 1.2.3 from the commit subject.
GIT_VERSION_FROM_SUBJECT := $(strip \
	$(or \
		$(shell \
			if command -v git >/dev/null 2>&1; then \
				git log -1 --format=%s 2>/dev/null \
				| grep -oE 'v?[0-9]+\.[0-9]+\.[0-9]+' \
				| sed -n '1p'; \
			fi \
		), \
		$(GIT_VERSION_TAG), \
	) \
)

# --------------------------------------------------------------------
# Version selection
# --------------------------------------------------------------------

# Remove a leading v from a version.
strip-v = $(patsubst v%,%,$(strip $(1)))

# Add exactly one v prefix.
with-v = v$(call strip-v,$(strip $(1)))

# Treat VERSION=0.0.0 as unset so development metadata can be used.
VERSION_OVERRIDE := $(if \
	$(filter-out 0.0.0,$(strip $(VERSION))), \
	$(strip $(VERSION)), \
)

# Remove the v prefix from Git tag values before normalizing.
GIT_TAG_VERSION := $(call strip-v,$(GIT_VERSION_TAG))
GIT_SUBJECT_VERSION := $(call strip-v,$(GIT_VERSION_FROM_SUBJECT))

VERSION_USED := $(strip \
	$(or \
		$(VERSION_OVERRIDE), \
		$(GIT_TAG_VERSION), \
		$(GIT_SUBJECT_VERSION), \
		$(DEFAULT_VERSION) \
	) \
)

GIT_TAG := $(call with-v,$(VERSION_USED))
GIT_TAG_MESSAGE := Release version $(GIT_TAG)

# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------

# Semantic release versions:
#   1.2.3
#   v1.2.3
#
# Optional prerelease/build metadata can be enabled later if required.
VERSION_VALID := $(shell \
	printf '%s\n' "$(GIT_TAG)" \
	| sed -nE '/^v[0-9]+\.[0-9]+\.[0-9]+$$/p' \
	| grep -q . \
	&& echo true \
	|| echo false \
)

# --------------------------------------------------------------------
# Debug output
# --------------------------------------------------------------------

ifneq ($(filter true 1 yes,$(DEBUG)),)

$(info [DEBUG] Git available:       $(GIT_AVAILABLE))
$(info [DEBUG] Git root:            $(GIT_ROOT))
$(info [DEBUG] Commit ID:           $(GIT_COMMIT_ID))
$(info [DEBUG] Commit subject:      $(GIT_COMMIT_SUBJECT))
$(info [DEBUG] Git version tag:     $(GIT_VERSION_TAG))
$(info [DEBUG] Subject version:     $(GIT_VERSION_FROM_SUBJECT))
$(info [DEBUG] Explicit VERSION:    $(VERSION))
$(info [DEBUG] Selected version:    $(VERSION_USED))
$(info [DEBUG] Final Git tag:       $(GIT_TAG))
$(info [DEBUG] Version valid:       $(VERSION_VALID))
$(info [DEBUG] Tag message:         $(GIT_TAG_MESSAGE))

endif

# --------------------------------------------------------------------
# Public targets
# --------------------------------------------------------------------
# .PHONY: version tag tag-create check-version

# Show the calculated version:
# 	make version
# Show diagnostic information:
# 	make DEBUG=true version
# Override the version:
# 	make VERSION=0.4.0 version
version:
	@printf '%s\n' "$(GIT_TAG)"

# Validate it:
# 	make VERSION=0.4.0 check-version
check-version:
	@if [[ "$(VERSION_VALID)" != "true" ]]; then \
		printf 'ERROR: Invalid release version: %s\n' "$(GIT_TAG)" >&2; \
		exit 1; \
	fi
	@printf 'Valid release version: %s\n' "$(GIT_TAG)"

# The important distinction is:
# 	make tag
# 	make VERSION=0.4.0 tag
# 	make DEBUG=true VERSION=0.4.0 tag
# only displays the calculated tag, while:
# 	make tag-create
# actually creates it.
#
# make DEBUG=true VERSION=0.4.0 tag
# [DEBUG] Git available:       true
# [DEBUG] Git root:            /work
# [DEBUG] Commit ID:           76417b28daf1
# [DEBUG] Commit subject:      doc (#848)
# [DEBUG] Git version tag:     v0.4.0
# [DEBUG] Subject version:     v0.4.0
# [DEBUG] Explicit VERSION:    0.4.0
# [DEBUG] Selected version:    0.4.0
# [DEBUG] Final Git tag:       v0.4.0
# [DEBUG] Version valid:       true
# [DEBUG] Tag message:         Release version v0.4.0
# v0.4.0
tag: version

# Create an annotated Git tag:
# 	make VERSION=0.4.0 tag-create
tag-create: check-version
	@if [[ "$(GIT_AVAILABLE)" != "true" ]]; then \
		printf '%s\n' 'ERROR: Git is not available.' >&2; \
		exit 1; \
	fi
	@if [[ -n "$$(git status --porcelain)" ]]; then \
		printf '%s\n' 'ERROR: Working tree is not clean.' >&2; \
		exit 1; \
	fi
	@if git rev-parse --verify --quiet "refs/tags/$(GIT_TAG)" >/dev/null; then \
		printf 'ERROR: Tag already exists: %s\n' "$(GIT_TAG)" >&2; \
		exit 1; \
	fi
	git tag -a "$(GIT_TAG)" -m "$(GIT_TAG_MESSAGE)"
	printf 'Created tag: %s\n' "$(GIT_TAG)"

######################################################################
##
######################################################################

pytest1:
	@pytest \
		scikitplot/corpus/_similarity/tests/test__backends.py::TestAnnoyBackend::test_auto_falls_back_to_cython \
		-vv

# git show a240dc5692b7e547be1ec2e66ae04100124dadf6:scikitplot/_externals/_jupyter_ext/_jupyter_ai_assistant/__init__.py
# git diff -- scikitplot/_externals/_jupyter_ext/_jupyter_ai_assistant/__init__.py
# pre-commit run --all-files --show-diff-on-failure
precom:
	@pre-commit run --files \
		scikitplot/_externals/_jupyter_ext/_jupyter_ai_assistant/__init__.py \
		--show-diff-on-failure
