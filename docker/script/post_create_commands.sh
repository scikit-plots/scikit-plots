#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# set +e  # Disable 'exit on error' temporarily for debugging
set -e  # Exit script on error
set -x  # Enable debugging (prints commands as they run)

######################################################################
## safe_dirs.sh
######################################################################

# export GIT_CONFIG_GLOBAL=~/.gitconfig
# git config --global --list --show-origin
# git config --global --unset-all safe.directory
# git config --global --get-all safe.directory

## Directories to mark as safe
for DIR in \
  "$(realpath ./)" \
  "$(realpath ./third_party/array-api-compat)" \
  "$(realpath ./third_party/array-api-extra)" \
  "$(realpath ./third_party/astropy)" \
  "$(realpath ./third_party/seaborn)"
do
  # Check if the directory exists
  if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist"
  # else
  #   echo "Directory $DIR exists, attempting to add it to safe.directory..."
  fi
  ## Try adding the directory to the git safe.directory list
  #git config --global --add safe.directory "$DIR" 2>&1 | tee /dev/tty | grep -q "error" && { echo "Failed to add $DIR to safe.directory"; FALLBACK=1; }
  git config --global --add safe.directory "$DIR" 2>/dev/null || { echo "Failed to add $DIR to safe.directory"; FALLBACK=1; }
done

## If any command failed, allow all directories as safe
if [ "$FALLBACK" = "1" ]; then
  echo "Some directories failed. Allowing all directories as safe..."
  ## Alternative: Bypass Ownership Checks (If Safe)
  # sudo chown -R "$(whoami):$(id -gn whoami)" ~/.gitconfig || true
  git config --global --add safe.directory '*'
fi

echo -e "\033[1;32m## Safe directory configuration complete.\033[0m"

######################################################################
## post_create_commands
######################################################################

# to initialise local config file and fetch + checkout submodule (not needed every time)
echo -e "\033[1;32m## Initializing local configuration and fetching submodules...\033[0m"
git submodule update --init --recursive  # download submodules

# (Optionally) pulls changes from the upstream remote repo and merges them
# Check if user wants to pull changes from upstream
# echo -e "\033[1;34m## Pulling latest changes and merging...\033[0m"
# git submodule update --recursive --remote --merge # (not needed every time)

# (Optionally) Updating your submodule to the latest commit
# Check if user wants to update submodules to the latest commit
# echo -e "\033[1;34m## Updating submodules to the latest commit...\033[0m"
# git submodule update --remote # (not needed every time)

echo -e "\033[1;32m## Submodule update process completed!\033[0m"

# Add remote upstream if not already added
echo -e "\033[1;32m## Adding remote upstream repository...\033[0m"
git remote add upstream https://github.com/scikit-plots/scikit-plots.git || true

# Fetch tags from upstream
echo -e "\033[1;32m## Fetching tags from upstream...\033[0m"
git fetch upstream --tags

######################################################################
## env
######################################################################

# Initialize mamba (or conda)
mamba init --all || true

# Create a new environment with python 3.11 and ipykernel if it doesn't already exist
mamba create -n py311 python=3.11 ipykernel -y || true

# Activate the environment and install required packages
# Use `bash -i` to ensure the script runs in an interactive shell and respects environment changes
# Double quotes for the outer string and escaping the inner double quotes or use single
bash -i -c "
  mamba activate py311 || exit 1
  mamba info -e | grep '*' || exit 1

  echo -e '\033[1;32m## Installing development dependencies...\033[0m'
  # pip install -r ./requirements/build.txt
  pip install -r ./requirements/all.txt
  # pip install -r ./requirements/cpu.txt

  # Install pre-commit
  echo -e '\033[1;32m## Installing pre-commit hooks...\033[0m'
  pip install pre-commit

  # Install pre-commit hooks in the repository
  echo -e '\033[1;32m## Installing pre-commit hooks inside the repository...\033[0m'
  ( cd /workspaces/scikit-plots/ || true && pre-commit install )

  echo -e '\033[1;32m## Install the development version of scikit-plots...\033[0m'
  # Install the development version of scikit-plots
  pip install --no-build-isolation --no-cache-dir -e .[dev,build,test,docs] -v
"

# Show next steps to user
echo -e "\033[1;34m## Continue to the section below: 'Creating a Branch'\033[0m"

# Provide more information about the next steps
echo -e "\033[1;34m## Read more at: \033[0m\033[1;36mhttps://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch\033[0m"

# (Optionally) Open new terminal activate `py311`
echo -e "mamba info -e"
