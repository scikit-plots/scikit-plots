#!/usr/bin/env bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

set -e  # Exit script on error (Disable 'exit on error' temporarily for debugging)
set -x  # Enable debugging (prints commands as they run)
set -euxo pipefail

## Dynamically get shell name (bash, zsh, fish, etc.)
echo "shell_name=$(basename "$SHELL")"

# shellcheck disable=SC1090
source "$HOME/.$(basename "$SHELL")rc" || true

## Make sudo Passwordless for the User
sudo -n true && echo "Passwordless sudo ✅" || echo "Password required ❌"

## Ensure os packages installed
echo "📦 Installing dev tools (if sudo available)..."
(sudo -n true && sudo apt-get update -y \
    && sudo apt-get install -y sudo gosu git curl build-essential gfortran) \
    || echo "⚠️ Failed or skipped installing dev tools"

######################################################################
## git safe_dirs.sh
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
  ## Check if the directory exists
  if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist"
  ## else
  ##   echo "Directory $DIR exists, attempting to add it to safe.directory..."
  fi
  ## Try adding the directory to the git safe.directory list
  # git config --global --add safe.directory "$DIR" 2>&1 | tee /dev/tty | grep -q "error" && { echo "Failed to add $DIR to safe.directory"; FALLBACK=1; }
  git config --global --add safe.directory "$DIR" 2>/dev/null || { echo "Failed to add $DIR to safe.directory"; FALLBACK=1; }
done

## If any command failed, allow all directories as safe
if [ "${FALLBACK:-""}" = "1" ]; then
  echo "Some directories failed. Allowing all directories as safe..."
  ## Alternative: Bypass Ownership Checks (If Safe)
  # sudo chown -R "$(whoami):$(id -gn whoami)" ~/.gitconfig || true
  git config --global --add safe.directory '*' || true
fi

echo -e "\033[1;32m## Safe directory configuration complete.\033[0m"

######################################################################
## git fetching submodules
######################################################################

echo -e "\033[1;32m## Initializing local configuration and fetching submodules...\033[0m"
## Initialize and clone any missing submodules, set up the working tree
## Almost always used after cloning a repo with submodules.
git submodule update --init --recursive

# echo -e "\033[1;34m## Updating submodules to the latest commit...\033[0m"
## Update submodules to the latest commit on their configured remote branch
## Used when you want your submodules to move to their latest remote commit.
# git submodule update --remote --recursive # (not needed every time)

## For each submodule, fetch updates from its remote
## Used if you only want to fetch updates but not move HEAD or update the working directory yet.
# git submodule foreach git fetch # (Less Common)

## Same as above, but tries to merge if local submodule has uncommitted changes
## Only used if you already made changes inside submodules locally and you don't want to lose them
## — you want Git to merge updates instead of overwriting.
# git submodule update --remote --merge # (Rare)

# echo -e "\033[1;34m## Pulling latest changes and merging...\033[0m"
## Update recursively, in case submodules have submodules
## Submodules have nested submodules, and you have local changes inside those too,
## and you want to merge, not reset.
# git submodule update --remote --merge --recursive # (Very Rare (Edge Case))

echo -e "\033[1;32m## Submodule update process completed!\033[0m"

######################################################################
## git upstream configuration
######################################################################

## Add remote upstream if not already added
echo -e "\033[1;32m## Adding remote upstream repository...\033[0m"
git remote add upstream https://github.com/scikit-plots/scikit-plots.git || true

## Fetch tags from upstream
echo -e "\033[1;32m## Fetching tags from upstream...\033[0m"
git fetch upstream --tags

######################################################################
## Installing editable scikit-plots dev version to env "py311"
## Use micromamba See: env_micromamba.sh
# "conda" keyword compatipable Env (e.g., Conda, Miniconda, Mamba)
# Micromamba not "conda" keyword compatipable but same syntax
######################################################################

# env_conda py311 conda             # uses Python 3.11 (default)
# env_conda py38 micromamba 3.8     # uses Python 3.8 explicitly
env_conda() {
  local env_name=$1
  local conda_cmd=$2
  local python_version=${3:-3.11}   # Default to 3.11 if not specified

  # echo "Checking if environment '\''$env_name'\'' exists..."
  printf "Checking if environment '%s' exists...\n" "$env_name"

  if ! $conda_cmd env list | grep -qE "(^|[[:space:]])${env_name}([[:space:]]|$)"; then
    # echo "Creating '\''$env_name'\'' with Python 3.11 and ipykernel using $conda_cmd..."
    printf "Creating '%s' with Python %s and ipykernel using %s...\n" "$env_name" "$python_version" "$conda_cmd"
    $conda_cmd create -n "$env_name" python="$python_version" ipykernel -y || true
  else
    # echo "Environment '\''$env_name'\'' already exists. Skipping creation."
    printf "Environment '%s' already exists. Skipping creation.\n" "$env_name"
  fi

  # echo "Updating environment '\''$env_name'\'' with default.yml..."
  printf "Updating environment '%s' with default.yml...\n" "$env_name"
  # $conda_cmd env update -n "$env_name" -f "./docker/env_conda/default.yml" || { echo "Failed to update environment"; exit 0; }
  $conda_cmd env update -n "$env_name" -f "./docker/env_conda/default.yml" || { printf "Failed to update environment\n"; exit 0; }
}

## Activate the environment and install required packages
## Use bash -i to ensure the script runs in an interactive shell and respects environment changes
## Double quotes for the outer string and escaping the inner double quotes or use single
## awk '/"/ && !/\\"/ && !/".*"/ { print NR ": " $0 }' .devcontainer/scripts/post_create_commands.sh
bash -i -c "
set -euo pipefail

## ⚠️ If mamba isn't initialized in the shell (as often happens in Docker/CI)
## 👉 Some steps can be skipped when container creation due to storage size limitations
## Use || exit 0: exits cleanly if the command fails (stops the script).
## Use || true: absorbs the error, continues ( skip logic).
## source ${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh || true

## Try micromamba first (faster and more portable), then fallback to conda
## Choose micromamba if available, otherwise fallback to conda
# echo -e '\033[1;34m>> Checking and activating environment...\033[0m'
printf '\033[1;34m>> Checking and activating environment...\033[0m\n'

# >>> Conda/Mamba environment auto-activation >>>
# Only run in interactive shell
# Auto-activate py311 if it exists, otherwise fallback to base
# if micromamba env list | grep -qE '(^|[[:space:]])py311([[:space:]]|$)'; then
if [[ $- == *i* ]]; then
  if command -v micromamba >/dev/null 2>&1 && [[ -d '${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/py311' ]]; then
    micromamba activate py311
  elif command -v conda >/dev/null 2>&1 && [[ -d '/opt/conda/envs/py311' ]]; then
    conda activate py311
  elif command -v micromamba >/dev/null 2>&1; then
    micromamba activate base
  elif command -v conda >/dev/null 2>&1; then
    conda activate base
  else
    echo '❌ No compatible conda/mamba environment found.' >&2
    # Don't use exit 0 in .bashrc — it can break the shell
    exit 0
  fi
fi
# <<< Conda/Mamba environment auto-activation <<<

# echo -e '\033[1;32m## Installing development dependencies...\033[0m'
printf '\033[1;32m## Installing development dependencies...\033[0m\n'
# pip install -r ./requirements/all.txt || true
# pip install -r ./requirements/cpu.txt || true
pip install -r ./requirements/build.txt || true

# Install pre-commit
# echo -e '\033[1;32m## Installing pre-commit...\033[0m\n'
printf '\033[1;32m## Installing pre-commit...\033[0m\n'
pip install pre-commit || true

# Install pre-commit hooks in the repository
# echo -e '\033[1;32m## Installing pre-commit hooks...\033[0m\n'
printf '\033[1;32m## Installing pre-commit hooks...\033[0m\n'

set +u  # Temporarily disable unbound variable error
( cd /workspaces/scikit-plots || true && pre-commit install || true )
set -u  # Re-enable afterwards (if needed)

# echo -e '\033[1;32m## Installing editable scikit-plots dev version...\033[0m\n'
printf '\033[1;32m## Installing editable scikit-plots dev version...\033[0m\n'
# Install the development version of scikit-plots
pip install --no-build-isolation --no-cache-dir -e .[build,dev,test,doc] -v || true
"

######################################################################
## Provide more information
######################################################################

## Show next steps to user
echo -e "\033[1;34m## Continue to the section below: 'Creating a Branch'\033[0m"

## Provide more information about the next steps
echo -e "\033[1;34m## Read more at: \033[0m\033[1;36mhttps://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch\033[0m"

######################################################################
## info (if possible)
######################################################################

## (Optionally) Open new terminal activate py311
micromamba info -e || true
conda info -e || true
scikitplot -V || true

######################################################################
## . (if possible)
######################################################################
