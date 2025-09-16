#!/usr/bin/env bash
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
## $(eval echo ~...) breaks in Docker, CI, or Windows paths.
## Inside bash -c '...' string	\$p, if needed
# ( ... )  || fallback runs in a subshell — changes inside don't affect the parent script.
# { ...; } || fallback runs in current shell — can exit or affect current environment.

set -e  # Exit script on error (Disable 'exit on error' temporarily for debugging)
set -x  # Enable debugging (prints commands as they run)
set -euxo pipefail

cat /etc/os-release || echo "No /etc/os-release file found. Skipping OS release information."
cat uname -u || echo "No uname -u output available. Skipping system information."

## Dynamically get shell name (bash, zsh, fish, etc.)
echo "CWD_DIR=$PWD"
echo "REAL_DIR=$(realpath ./)"
echo "SCRIPT_DIR=$(cd -- $(dirname ${BASH_SOURCE[0]}) && pwd)"
echo "SHELL_DIR=$(cd -- $(dirname $0) && pwd)"
echo "SHELL_NAME=$(basename $SHELL)"

## Make sudo Passwordless for the User
sudo -n true && echo "Passwordless sudo ✅" || echo "Password required ❌"

## Ensure os packages installed
echo "📦 Installing dev tools (if sudo available)..."
{ sudo -n true && sudo apt-get update -y \
  && sudo apt-get install -y sudo gosu git curl build-essential gfortran ninja-build; } \
  || echo "⚠️ Failed or skipped installing dev tools"

# green
print_info() {
  echo -e "\033[1;32m$1\033[0m"
}
# yellow-orange
print_warn() {
  echo -e "\033[1;33m$1\033[0m"
}
# red
print_error() {
  echo -e "\033[1;31m$1\033[0m"
}
# blue
print_url() {
  echo -e "\033[1;34m$1\033[0m"
}
# purple
print_info2() {
  echo -e "\033[1;36m$1\033[0m"
}

######################################################################
## Git Safe Directories Configuration
# export GIT_CONFIG_GLOBAL=~/.gitconfig
# git config --global --list --show-origin
# git config --global --unset-all safe.directory
# git config --global --get-all safe.directory
######################################################################
print_info "## Configuring Git safe.directory..."

## Directories to mark as safe
# SAFE_DIRS=(
#   "$(realpath ./)"
#   "$(realpath ./third_party/array-api-compat)"
#   "$(realpath ./third_party/array-api-extra)"
#   "$(realpath ./third_party/astropy)"
#   "$(realpath ./third_party/seaborn)"
# )
# for DIR in "${SAFE_DIRS[@]}"; do
for DIR in \
  "$(realpath ./)" \
  "$(realpath ./third_party/array-api-compat)" \
  "$(realpath ./third_party/array-api-extra)" \
  "$(realpath ./third_party/astropy)" \
  "$(realpath ./third_party/seaborn)"
do
  ## Check if the directory exists
  if [ -d "$DIR" ]; then
  # git config --global --add safe.directory "$DIR" 2>&1 | tee /dev/tty | grep -q "error"
    git config --global --add safe.directory "$DIR" 2>/dev/null || {
      print_warn "⚠️ Failed to add $DIR to safe.directory";
      FALLBACK=1;
    }
  else
    print_warn "⚠️ Directory does not exist: $DIR"
  fi
done

## If any command failed, allow all directories as safe
if [ "${FALLBACK:-}" = "1" ]; then
  print_warn "Some directories failed. Marking all directories as safe."
  ## Alternative: Bypass Ownership Checks (If Safe)
  # sudo chown -R "$(whoami):$(id -gn whoami)" ~/.gitconfig || true
  git config --global --add safe.directory '*' || true
fi
print_info "✅ Safe directory configuration complete."

######################################################################
## Git Submodule Handling
######################################################################
print_info "## Initializing and fetching submodules..."

## Initialize and clone any missing submodules, set up the working tree
## Almost always used after cloning a repo with submodules.
git submodule update --init --recursive || print_warn "⚠️ Submodule init failed."

# Optional: Keep disabled unless you need bleeding-edge submodule versions
# print_info "## Updating submodules to latest from remote..."
# git submodule update --remote --recursive

print_info "✅ Submodule setup complete."

# echo -e "\033[1;34m## Updating submodules to the latest commit...\033[0m"
## Update submodules to the latest commit on their configured remote branch
## Used when you want your submodules to move to their latest remote commit.
# git submodule update --remote --recursive # (not needed every time)
#
## For each submodule, fetch updates from its remote
## Used if you only want to fetch updates but not move HEAD or update the working directory yet.
# git submodule foreach git fetch # (Less Common)
#
## Same as above, but tries to merge if local submodule has uncommitted changes
## Only used if you already made changes inside submodules locally and you don't want to lose them
## — you want Git to merge updates instead of overwriting.
# git submodule update --remote --merge # (Rare)
#
# echo -e "\033[1;34m## Pulling latest changes and merging...\033[0m"
## Update recursively, in case submodules have submodules
## Submodules have nested submodules, and you have local changes inside those too,
## and you want to merge, not reset.
# git submodule update --remote --merge --recursive # (Very Rare (Edge Case))

######################################################################
## Add and Sync Remote Upstream
######################################################################
print_info "## Configuring upstream remote..."

## Add remote upstream if not already added
if ! git remote | grep -q upstream; then
  git remote add upstream https://github.com/scikit-plots/scikit-plots.git || print_warn "⚠️ Failed to add upstream remote"
else
  print_info "✅ Upstream remote already exists."
fi
## Fetch tags from upstream
print_info "## Fetching upstream tags..."
git fetch upstream --tags || print_warn "⚠️ Failed to fetch upstream tags"

print_info "✅ Git configuration done!"

######################################################################
## Installing editable scikit-plots dev version to env "py311"
## Use micromamba See: env_micromamba.sh
# "micromamba" not "conda" keyword compatipable but same syntax
# "conda" keyword compatipable Env (e.g., Conda, Miniconda, Mamba)
######################################################################
# Re-source shell config to ensure activation takes effect
# shellcheck disable=SC1090
# . ~/."$(basename $SHELL)"rc || true  # ~/.bashrc or ~/.zshrc for zsh
source ~/."$(basename $SHELL)"rc || echo "⚠️ Failed to source $SHELL_RC"

# Disable unbound variable errors (for safer fallback defaults)
set +u   # Disable strict mode (for unset variables)

# Set default environment name if not provided
PY_VERSION="${PY_VERSION:-3.11}"  # Default Python version
ENV_NAME="${ENV_NAME:-py311}"  # Default environment name
# ──────────────────────────────────────────────────────────────
# Create environment "$ENV_NAME" if not already present
# Supports either micromamba or conda
# ──────────────────────────────────────────────────────────────
if command -v micromamba >/dev/null 2>&1; then
  echo "📦 Using micromamba to manage environment: $ENV_NAME"

  if ! micromamba env list | grep -q "$ENV_NAME"; then
    echo "🆕 Creating micromamba environment: $ENV_NAME"
    # micromamba create -n "$ENV_NAME" python="$PY_VERSION" ipykernel pip -y || true
    micromamba env create -f environment.yml --yes \
    && { micromamba clean --all -f -y || true; } \
    && { jupyter lab clean || true; } \
    && { rm -rf "${HOME}/.cache/yarn" || true; } \
    && { rm -rf ${HOME}/.cache || true; } \
    || { echo "Failed to creation Micromamba environment"; }
  else
    echo "✅ micromamba environment '$ENV_NAME' already exists."
  fi

elif command -v conda >/dev/null 2>&1; then
  echo "📦 Using conda to manage environment: $ENV_NAME"

  if ! conda env list | grep -q "$ENV_NAME"; then
    echo "🆕 Creating conda environment: $ENV_NAME"
    # conda create -n "$ENV_NAME" python="$PY_VERSION" ipykernel pip -y || true
    # conda env create -f base.yml || { echo "Failed to creation environment"; }
    # conda env update -n "$ENV_NAME" -f "./docker/env_conda/default.yml" || { echo "Failed to update environment"; }
    conda env create -f environment.yml --yes \
    && { conda clean --all -f -y || true; } \
    && { jupyter lab clean || true; } \
    && { rm -rf "${HOME}/.cache/yarn" || true; } \
    && { rm -rf ${HOME}/.cache || true; } \
    || { echo "Failed to creation Conda environment"; }
  else
    echo "✅ conda environment '$ENV_NAME' already exists."
  fi

else
  echo "❌ Neither micromamba nor conda found. Cannot create environment."
  # return 1 2>/dev/null || exit 1
fi

######################################################################
## 📦 starting a new interactive shell, and running some-command inside that new shell, not in the current shell.
# Use bash -i to ensure the script runs in an interactive shell and respects environment changes
# bash -i: starts a new interactive shell (reads .bashrc)
## Double quotes for the outer string and escaping the inner double quotes or use single
## awk '/"/ && !/\\"/ && !/".*"/ { print NR ": " $0 }' .devcontainer/scripts/post_create_commands.sh
######################################################################

## Activate the environment and install required packages in new interactive shell
bash -i -c "
## Use || exit 0: exits cleanly if the command fails (stops the script).
## Use || true: absorbs the error, continues ( skip logic).
## 👉 Some steps can be skipped when container creation due to storage size limitations
## ⚠️ If mamba isn't initialized in the shell (as often happens in Docker/CI)
## source \${MAMBA_ROOT_PREFIX:-~/micromamba}/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh || true

set -euo pipefail
set +u  # Temporarily disable unbound variable error
# set -u  # Re-enable afterwards (if needed)

## echo -e '\033[1;34m>> Checking and activating environment...\033[0m'
printf '\033[1;34m>> Checking and activating environment...\033[0m\n'
## Try micromamba first (faster and more portable), then fallback to conda
## Choose micromamba if available, otherwise fallback to conda
micromamba activate ${ENV_NAME:-py311} || conda activate ${ENV_NAME:-py311} || { . ~/.$(basename $SHELL)rc; } || true

## echo -e '\033[1;32m## Installing development dependencies...\033[0m'
printf '\033[1;32m## Installing development dependencies...\033[0m\n'
## pip install -r ./requirements/all.txt || true
## pip install -r ./requirements/cpu.txt || true
pip install -r ./requirements/build.txt || true

# ──────────────────────────────────────────────────────────────
# 📦 Installing scikit-plots version
# ──────────────────────────────────────────────────────────────
## echo -e '\033[1;32m## Installing editable scikit-plots dev version...\033[0m\n'
printf '\033[1;32m## Installing scikit-plots (Release or Dev)...\033[0m\n'

# Install logic
if [ -n \${SCIKITPLOT_VERSION:-} ]; then
  echo 📦 Installing scikit-plots version: \${SCIKITPLOT_VERSION}
  if ! pip install scikit-plots==\${SCIKITPLOT_VERSION}; then
    { echo ⚠️ Failed to install version \${SCIKITPLOT_VERSION}, trying latest release...; }
    pip install scikit-plots || echo ❌ Failed to install scikit-plots from PyPI
  fi

else
  echo 🛠️ Installing development version of scikit-plots from local source...
  if ! pip install --no-build-isolation --no-cache-dir -e .[build,dev,test,doc] -v; then
    echo ⚠️ Failed to install full dev extras, retrying with minimal setup...
    if ! pip install --no-build-isolation --no-cache-dir -e . -v; then
      { echo ❌ Failed to install local development version of scikit-plots; }
    fi
  fi
fi

# ──────────────────────────────────────────────────────────────
# 🧹 Install and set up pre-commit hooks
# ──────────────────────────────────────────────────────────────
## echo -e '\033[1;32m## Installing pre-commit...\033[0m\n'
printf '\033[1;32m## Installing pre-commit...\033[0m\n'

# Step 1: Install pre-commit
if ! pip install pre-commit; then
  { echo '❌ Failed to install pre-commit'; }
else
  printf '\033[1;32m## Installing pre-commit hooks...\033[0m\n'

  # Step 2: Attempt to cd into known project directories
  for p in /workspaces/scikit-plots /work/scikit-plots /home/jovyan/work/scikit-plots; do
    if [ -d \$p ]; then
      cd \$p || continue
      break
    fi
  done

  # Step 3: Install hooks only if pre-commit is available
  if command -v pre-commit >/dev/null 2>&1; then
    pre-commit install || { echo '⚠️ Failed to initialize pre-commit hooks'; }
  else
    { echo '⚠️ pre-commit command not found even after install'; }
  fi
fi
"

######################################################################
## ℹ️  Environment & Package Info (if available)
######################################################################
# Re-source shell config to ensure activation takes effect
# shellcheck disable=SC1090
# source ~/."$(basename $SHELL)"rc || true  # ~/.bashrc or ~/.zshrc for zsh
. ~/."$(basename $SHELL)"rc || true

print_info2 "🔍 Conda Environments:"
# conda info -e | grep -iw '\*' || true
conda info -e 2>/dev/null || echo "⚠️  Conda not found or not configured"

print_info2 "🔍 Micromamba Environments:"
micromamba info -e 2>/dev/null || echo "⚠️  Micromamba not found or not configured"

print_info2 "📦 scikit-plots version:"
scikitplot -V 2>/dev/null || echo "⚠️  scikitplot command not found"

######################################################################
## 📘 Next Steps and Contribution Guide
######################################################################
print_info2 "➡️  Continue to the section below: 'Creating a Branch'"
print_info2 "📖 Read more at: $(print_url https://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch)"

######################################################################
## . (if possible)
######################################################################
