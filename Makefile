## Makefile for Python Packaging Library

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

## This Makefile contains various "targets" for project management tasks such as "compiling" the project,
## "cleaning" up build files, "running" tests, "building" Docker images, and more.

## ensures that all commands within a target run in one single shell,
## allowing variables and environment changes to persist across commands.
## better multi-line logic → use .ONESHELL: but @ only applies to the first command
## cleaner output → use @
.ONESHELL:

# Shell Debugging: If the issue persists, add debugging:
# SHELL = /bin/bash -x

## PHONY targets are used to avoid conflicts with files of the same name.
## Declare phony targets to indicate these are not files but commands to be executed.
.PHONY: help all clean publish

## "target" the name of the action to be execute as "make target". Syntax:
## <target>: <if-defined-previously-run-this-target>
##     @<command> The commands to execute, indented with a tab (not spaces).

## (Optional) Ensures that the project is rebuilt from a clean state.
all: clean publish
	@echo "all completed."

######################################################################
## helper
######################################################################

## helper for defined targets in Makefile
## -e enables interpretation of backslash escapes.
help:
	@echo -e "Please use one of below targets with 'make <target>' like: \nmake clean-basic"
	@echo "  clean-basic     to remove 'jupyter' artifacts and 'testing' artifacts files."
	@echo "  clean           to remove package 'build' artifacts and temporary files."
	@#echo "  reset           to discarding any local uncommitted changes in tracked files."
	@echo "  tree            Get project file/folder structure."
	@echo ""
	@echo "  newm            mamba create environment 'py311'."
	@echo "  conda           conda create environment 'py311'."
	@echo "  dep             Install scikit-plots build dependencies. and cpu dep pkgs."
	@echo "  ins             Install scikit-plots from local."
	@echo "  dev             Install the development version of scikit-plots."
	@echo ""
	@echo "  pkg-setup       Packaging: 'setup.py' build the pypi Packages, depends on 'clean'"
	@echo "  pkg-build       Packaging: 'build' library by 'setup.py' or 'pyproject.toml' for pypi Packages, depends on 'clean'"
	@echo "  comp-meson      Compiling: 'meson' library for step-by-step compiling, depends on 'clean'"
	@echo ""
	@echo "  test            to run unit tests after 'clean-basic'"
	@echo "  examples        to execute py scripts under 'galleries/' folder"
	@echo "  update-mod      Submodule Update. Usage: make update-sample [MOD=NumCpp]"
	@echo ""
	@echo "  branch          Add a Branch to the Local project. Usage: make branch BR=maintenance/x.x.x"
	@echo "  branch-del      Delete the branch locally use BR Environment for del. Usage: make branch-del BR=maintenance/x.x.x"
	@echo "  branch-delr     Delete the branch remotely use BR Environment for del. Usage: make branch-delr BR=maintenance/x.x.x"
	@echo "  branch-clean    Periodically delete old local branches that have already been merged to keep your workspace clean."
	@echo "  branch-push     Only new Creation stable branch Push to the remote repository. Including -u (or --set-upstream) flag."
	@echo ""
	@echo "  tag-sample      Tag sample by the latest commit or TAG Environment. Usage: make tag-sample [TAG=1.0.0]"
	@echo "  tag             Add a Tag to the Local project check 'tag-sample'. Usage: make tag BR=maintenance/x.x.x [TAG=1.0.0]"
	@echo "  tag-del         Delete the Local Tag use TAG Environment for del. Usage: make tag-del TAG=1.0.0"
	@echo "  tag-delr        Delete the Remote Tag use TAG Environment for del. Usage: make tag-delr TAG=1.0.0"
	@echo "  tag-push        Push the tag to the remote repository"
	@echo ""
	@echo "  check-publish   Checking the distribution files (Readme.md) for PyPI with twine."
	@echo "  publish         to publish the project on PyPI"

######################################################################
## Cleaning
######################################################################

## Clean up all the generated files, compiler cleaning without 'third_party'
clean-basic:
	@echo "Basic cleaning started ..."
	@# Command Substitution "$(...)" "(`...`)" (its output in place of the backticks):
	@rm -rf `find -L . -type d -name ".ipynb_checkpoints" -not -path "./third_party/*"`
	@rm -rf "./third_party/.ipynb_checkpoints"
	@echo "Removed all '.ipynb_checkpoints'"
	@rm -rf `find -L . -type d -name "__pycache__" -not -path "./third_party/*"`
	@echo "Removed all '__pycache__'"
	@rm -rf `find -L . -type d -name ".pytest_cache" -not -path "./third_party/*"`
	@echo "Removed all '.pytest_cache'"
	@rm -rf `find -L . -type d -name "__MACOSX" -not -path "./third_party/*"`
	@echo "Removed zip file leftovers '__MACOSX'"
	@rm -rf `find -L . -type d -name ".vscode" -not -path "./third_party/*"`
	@echo "Removed zip file leftovers '.vscode'"
	@echo "basic cleaning completed."

## pypi cleaning in 'build dirs'
clean: clean-basic
	@echo "Cleaning started..."
	@pip cache purge
	@echo "Removed all pip cache files"
	@rm -rf "result_images"
	@echo "Removed folder 'result_images' produced 'matplotlib.sphinxext.plot_directive'"
	@rm -rf "build" "build_dir" "builddir" "dist" "scikit_plots.egg-info" *.egg-info*
	@echo "Removed folder 'build, egg etc.'"
	@rm -rf rm -rf .meson .mesonpy-*
	@echo "Remove '.meson' and '.mesonpy-*' Files"
	@find -L -type f -name "*.so" -path "*/build*"
	@echo "Modules '*.so' files in 'build dirs'"
	@# find -L -type f -name "*.so" | xargs rm -rf
	@find -L -type f -name "*.so" -path "*/build*" -exec rm -rf {} +
	@echo "Removed all '*.so' files in 'build dirs'"
	@rm -rf ".mypy_cache" ".ruff_cache"
	@echo "Removed folder '.mypy_cache, .ruff_cache etc.'"
	@echo "pypi cleaning completed."

######################################################################
## project structure
######################################################################

## Get project structure
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

tree:
	@echo "System is: $(SYSTEM)"
	@$(TREE_CMD)

## https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
## mamba is a drop-in replacement and uses the same commands and configuration options as conda.
## mamba create -n ... -c ... ...
## mamba install ...
## mamba list
## mamba deactivate && mamba remove -y --all --name py311
newm:
	# mamba create -n py38 python=3.8 ipykernel -y
	# mamba create -n py39 python=3.9 ipykernel -y
	# mamba create -n py310 python=3.10 ipykernel -y
	mamba create -n py311 python=3.11 ipykernel -y
	# mamba create -n py312 python=3.12 ipykernel -y
	# mamba create -n py313 python=3.13 ipykernel -y
	# mamba create -n py314 python=3.14 ipykernel -y
	# mamba activate py311
	# mamba deactivate

## https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
## https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
## mamba is a drop-in replacement and uses the same commands and configuration options as conda.
## conda info
## conda info --envs
## conda update conda
## conda update anaconda
## conda env list
## conda search python
## conda search --full-name python
## conda update python
## conda install python=3.11
## conda env create -f environment.yml
## The --prune option causes conda to remove any dependencies that are no longer required from the environment.
## conda env update --file environment.yml  --prune
## conda create --name <my-env> python=3.11
## conda install -n myenv <my-env>
## conda list -n <my-env>
## conda env export > environment.yml
## conda env export --from-history
## conda remove --name myenv --all
newc:
	conda create -n test python=3.11 ipykernel
	# conda activate test  # activate our environment

## Install the development version of scikit-plots, depends on "clean"
dep: clean
	@echo "Installing library pip dependencies..."
	@# pip install -r ./requirements/all.txt
	@pip install -r ./requirements/build.txt
	@pip install -r ./requirements/cpu.txt

######################################################################
## Packaging
## https://setuptools.pypa.io/en/stable/userguide/pyproject_config.html
## https://mesonbuild.com/meson-python/tutorials/introduction.html
## https://build.pypa.io/en/stable/
## https://github.com/pypa/build
######################################################################

## Packaging by 'setup.py' build the pypi Packages, depends on "clean"
## 'setup.py' can also be use by 'build' or 'installer' libraries to packaging
ins-st: clean
	@echo "Packaging: 'setup.py' with setuptools, wheel..."
	@# python setup.py build_ext --inplace --verbose
	@# python setup.py sdist
	@# python setup.py bdist_wheel
	@python setup.py sdist bdist_wheel

## Package by 'build' library by 'setup.py' or 'pyproject.toml' for pypi Packages, depends on "clean"
build-st: clean
	@echo "Packaging: 'build' library by 'setup.py' or 'pyproject.toml' with own configuration..."
	@echo "Configuration libraries: can be (e.g. (setuptools, wheels) or (mesonbuild, meson, ninja))."
	@# pip install build
	@# python -m build --sdist
	@# python -m build --wheel
	@python -m build

######################################################################
## Installing Development Mode (a.k.a. “Editable Installs”)
## https://setuptools.pypa.io/en/stable/userguide/development_mode.html
## https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html
######################################################################

## Install Packages to local, depends on "clean"
ins: clean dep
	@echo "Installing Packages to local library (editable or not) for checking..."
	@# python -m pip install .
	@# python -m pip install --use-pep517 .
	@python -m pip install --no-build-isolation --no-cache-dir . -v

## Install the development version of scikit-plots, depends on "clean"
dev: clean dep
	@echo "Installing Packages to local library (editable or not) for checking..."
	@# python -m pip install .
	@# python -m pip install --use-pep517 .
	@# python -m pip install --no-build-isolation --no-cache-dir .
	@# python -m pip install --no-build-isolation --no-cache-dir --editable .
	@# python -m pip install --no-build-isolation --no-cache-dir -e . -vvv
	@python -m pip install --no-build-isolation --no-cache-dir -e .[build,dev,test,doc] -v

######################################################################
## Compiling by Meson step-by-step
######################################################################

## Compile by "meson" library for step-by-step, depends on "clean"
build-me: clean
	@echo "Compiling: 'meson' library step-by-step compiling for debugging..."
	@# pip install mesonbuild, meson, ninja

	@echo "meson cleaning previous build artifacts..."
	@meson clean -C builddir

	@echo "meson creating a 'build' directory..."
	@meson setup builddir

	@# echo "meson reconfiguring a 'build' directory..."
	@# meson setup --reconfigure builddir
	@# meson setup --wipe builddir

	@# echo "meson compiling a 'build' directory..."
	@# meson compile -C builddir
	@# meson compile --clean

	@# echo "ninja compiling a 'build' directory..."
	@# ninja -C builddir
	@## (Optional) Run tests
	@# ninja -C builddir test

sdist:
	@# meson setup builddir && meson dist -C builddir --allow-dirty --no-tests --formats gztar
	@# python -m pip install -v builddir/*dist/*.gz -Csetup-args=-Dallow-noblas=true
	@python -m build --sdist -Csetup-args=-Dallow-noblas=true
	@python -m pip install -v dist/*.gz -Csetup-args=-Dallow-noblas=true

######################################################################
## Testing
######################################################################

## Run this target to execute project tests.
# test: clean-basic
# 	@echo "Testing project started..."
# 	@cd scikitplot && pytest tests/
# 	@echo "pytest completed."

## Run this target to execute py script to save generating plots images.
# examples: clean-basic
# 	@echo "Generating plots images..."
# 	@#cd galleries/examples && python classification/plot_feature_importances_script.py
# 	@python auto_building_tools/discover_scripts.py --save-plots
# 	@echo "All py Script executed."

######################################################################
## Submodule
######################################################################

## Submodule Update
# update-mod:
# ifdef MOD
# 	@## NumCpp
# 	@echo "Submodule: '$(MOD)' Updating..."
# 	@git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/$(MOD)
# 	@git submodule sync
# 	@git submodule update --init --recursive
# 	@echo "'$(MOD)' Update completed."
# else
# 	@echo "MOD is not defined"
# endif

## git submodule foreach --recursive git pull
# git submodule update --init --recursive --remote

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
## Git Search and fix
######################################################################

## git grep "interp("

######################################################################
## Git Branch
######################################################################

## Add a Branch to the Local project
## maintenance/0.3.x
branch:
ifdef BR
	@echo "Adding Local branch: '$(BR)' to main"
	@git checkout main
	@## git branch "$(BR)" && git checkout "$(BR)" || git switch "$(BR)"
	@## This command creates a new branch and switches to it immediately.
	@# git checkout -b "$(BR)"
	@git switch -c "$(BR)"
	@## Commit changes incrementally
	@cat Readme.md > Readme_$(BR).md
	@git add .
	@git commit -m "$(BR) initial commit (endpoints) for consistency"
else
	@echo "BR is not defined"
endif

## Delete the branch locally use BR Environment for del.
branch-del:
ifdef BR
	@echo "Deleting (safe) Local branch: '$(BR)'"
	@# git branch -D "$(BR)"
	@git branch -d "$(BR)"
	@git branch
else
	@echo "BR is not defined"
endif

## Delete the branch remotely use BR Environment for del.
branch-delr:
ifdef BR
	@echo "Deleting Remote branch: '$(BR)'"
	@git push origin --delete "$(BR)"
	@git branch
else
	@echo "BR is not defined"
endif

## Delete a feature branch after merging
## Periodically delete old local branches that have already been merged to keep your workspace clean.
branch-clean:
ifdef BR
	@echo "Deleting local old branches..."
	@git branch --merged main | grep -v "main" | xargs git branch -d
	@git branch
else
	@echo "BR is not defined"
endif

## Push the updated stable branch to the remote repository
## The -u (or --set-upstream) flag tells Git to link the current local branch to a branch on the remote (or create a new remote branch if it doesn't exist).
## Use git push -u origin <branch> When: Pushing a new branch to the remote for the first time.
## Use git push origin <branch> When: The branch already has an upstream tracking relationship, and you don't need to set it again.
branch-push:
ifdef BR
	@echo "To check if your branch is tracked upstream, use:"
	@git branch -vv
	@echo "Adding Remote branch: '$(BR)'"
	@git push -u origin "$(BR)"
	@echo "$(BR) You’ve finished the work and are ready to merge into the main branch."
else
	@echo "BR is not defined"
endif

######################################################################
## Git Tag
######################################################################

## Generate a version based on the short commit hash and message
LAST_COMMIT_ID      = $(shell git rev-parse --short HEAD)
## $(if condition,then-part[,else-part]) $(if $(TAG),$(TAG),)
LAST_COMMIT_MESSAGE = "$(if $(TAG),$(TAG),$(shell git log -1 --pretty=%B))"
TAG_SAMPLE          = "v$(LAST_COMMIT_MESSAGE)"
TAG_MESSAGE         = "Release version $(LAST_COMMIT_MESSAGE)"
## Tagging the latest commit
## For larger projects or those requiring stability guarantees, tagging in stable is safer.
tag-sample:
	@echo "Sample tag: '$(TAG_SAMPLE)' message: '$(TAG_MESSAGE)' by commit: '$(LAST_COMMIT_MESSAGE)'"

## Add a Tag to Stable Releases to the Local project
tag:
ifdef BR
	@## Tagging in the stable Branch (Stability-First Workflow)
	@#Best practice: Tag before PyPI publishing.
	@echo "Adding tag to branch: '$(BR)'"
	@git checkout "$(BR)"
	@echo "Existing tags:"
	@git tag
	@echo "Adding local tag: $(TAG_SAMPLE) message: $(TAG_MESSAGE)"
	@# git tag -a v0.4.0 -m "Release version 0.4.0"
	@git tag -a "$(TAG_SAMPLE)" -m $(TAG_MESSAGE)
	@echo "Local tagging completed."
else
	@echo "BR is not defined"
endif

## Delete the Tag locally use TAG Environment for del.
tag-del:
ifdef TAG
	@echo "Deleting Local tag: 'v$(TAG)'"
	@git tag -d "v$(TAG)"
	@git tag
else
	@echo "TAG is not defined"
endif

## Delete the Tag remotely use TAG Environment for del.
tag-delr:
ifdef TAG
	@echo "Deleting Remote tag: 'v$(TAG)'"
	@git push origin --delete "v$(TAG)"
	@git tag
else
	@echo "TAG is not defined"
endif

## Push the tag to the remote repository
tag-push:
	@echo "Existing tags:"
	@git tag
	@echo "Adding to "remote repository" tag $(TAG_SAMPLE)..."
	@git push origin $(TAG_SAMPLE) || git push --tags
	@echo "Remote Repository Tagging completed."."

## Release combines tagging and pushing the tag to remote
# release: tag-sample tag tag-push
# 	@echo "Ready to Publish on PyPI"

######################################################################
## Publishing to PyPI
######################################################################

# Publish to PyPI (example for Python projects)
check-publish:
	@echo "Taaging is completed? Make sure before Publish on PyPI."
	@## *twine* for the PyPI upload
	@# pip install twine
	@echo "Checking the distribution files (Readme.md) for PyPI with twine."
	@twine check dist/*
	@echo "Uploading the distribution files to PyPI."
	@twine upload dist/*
	@echo "PyPI publish completed."

# Publish to PyPI (example for Python projects)
publish: check-publish
	@echo "Uploading the distribution files to PyPI."
	@# twine upload dist/* --username __token__ --password <your_api_token>
	@twine upload dist/*
	@echo "PyPI publish completed."

######################################################################
## Publishing to docker hub
######################################################################

publish-docker:
	# docker login
	## docker tag <existing_image_name>:<existing_tag> <new_image_name>:<new_tag>
	# docker tag jupyter_notebook-base_notebook:latest skplt/scikit-plots-dev:latest
	# docker push skplt/scikit-plots-dev:latest

######################################################################
## symbolic links
######################################################################

sym:
	@rm -rf .devcontainer/script
	@# mkdir -p .devcontainer/script
	@ln -rsf docker/script/ .devcontainer/script
	@echo "Created symbolic links..."

######################################################################
## New Branch with the Latest Changes
## forked repository 'origin', original repository 'upstream '
######################################################################

newbr:
	@# git checkout main  # or the branch you want as a base \
	## If you're on 'feature-branch' but want to pull from 'main' \
	## same git fetch origin main && git merge origin/main \
	# git pull origin main  \
	# git branch -d subpackage-bug-fix || true \
	# git checkout -b subpackage-bug-fix
	@git switch main \
	&& git pull \
	&& git branch -d subpackage-bug-fix || true \
	&& git branch -D subpackage-bug-fix || true \
	&& git switch -c subpackage-bug-fix \
	&& git push --set-upstream origin subpackage-bug-fix \
	&& git branch

push:
	@# pre-commit install
	@git add . && git commit -m "fix dependency"  && git push

release:
	@git tag -d "v0.4.0rc2" || true

	@## Creates the tag locally
	git tag -a "v0.4.0rc2" -m "Release version 0.4.0rc2"

	@## Pushes the tag to your fork
	git push origin "v0.4.0rc2"    # push just that tag

	@## Pushes all tags to your fork
	@## git push --tags            # then push all tags

	@## Pushes tag to upstream (if allowed) v0.4.0rc2 (APR 09th, 2025)
	git push upstream "v0.4.0rc2"    # push just that tag

	@# git ls-remote --tags origin  # verify what's pushed by running
