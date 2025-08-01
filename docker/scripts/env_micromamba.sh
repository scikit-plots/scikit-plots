#!/usr/bin/env bash
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
## Inside bash -c '...' string	\$p, if needed
# { ...; } || fallback runs in current shell — can exit or affect current environment.
# ( ... )  || fallback runs in a subshell — changes inside don't affect the parent script.

set -e  # Exit script on error (Disable 'exit on error' temporarily for debugging)
set -x  # Enable debugging (prints commands as they run)
set -euxo pipefail

## Dynamically get shell name (bash, zsh, fish, etc.)
echo "shell_name=$(basename "$SHELL")"
echo "CWD_DIR=$PWD"
echo "REAL_DIR=$(realpath ./)"
echo "SHELL_DIR=$(cd -- "$(dirname "$0")" && pwd)"
echo "SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
## 📦 Install and Initialize Micromamba (or Conda compatible) Environment Setup Script
# "micromamba" not "conda" keyword compatipable but same syntax
# "conda" keyword compatipable Env (e.g., Conda, Miniconda, Mamba)
######################################################################

# Disable unbound variable errors (for safer fallback defaults)
set +u   # Disable strict mode (for unset variables)

# Allow override from environment
# e.g. SKIP_MICROMAMBA=true ./setup.sh or SKIP_MICROMAMBA=true . ./setup.sh
SKIP_MICROMAMBA="${SKIP_MICROMAMBA:-false}"
# Normalize to lowercase and handle multiple truthy values
# case "$(printf '%s' "$SKIP_CONDA" | tr '[:upper:]' '[:lower:]')" in
case "${SKIP_MICROMAMBA,,}" in
  true|1|yes|on)
    echo "Skipping conda activation"
    # Works whether script is sourced (returns) or executed (exits) directly
    return 0 2>/dev/null || exit 0
    ;;
esac
# ✅ Use POSIX-compatible:
# if [[ "$SKIP_MICROMAMBA" == "true" ]]; then
if [ "${SKIP_MICROMAMBA}" = "true" ]; then
  echo "Skipping conda activation"
  # handles both source (returns) and direct exec (exits).
  return 0 2>/dev/null || exit 0
fi

# Set default environment name if not provided
PY_VERSION="${PY_VERSION:-3.11}"  # Default Python version "3.11"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"  # Default environment name "py311"

# ──────────────────────────────────────────────────────────────
# 1. Install micromamba if not already available
# Need curl or fallback to wget and ps (usually from procps or procps-ng)
# ──────────────────────────────────────────────────────────────
# Install micromamba via official install script silently, only if not installed
echo "🔧 Installing or initializing micromamba..."
# if ! command -v micromamba &> /dev/null; then
if ! command -v micromamba >/dev/null 2>&1; then
  echo "➡️  micromamba not found, attempting install..."
  # if command -v curl &> /dev/null; then
  if command -v curl >/dev/null 2>&1; then
    # curl -Ls https://micro.mamba.pm/install.sh | bash
    # curl -Ls https://micro.mamba.pm/install.sh | "${SHELL}" || echo "⚠️ micromamba install failed"
    # "${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
    "${SHELL}" <(curl -Ls micro.mamba.pm/install.sh) < /dev/null
  # elif command -v wget &> /dev/null; then
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://micro.mamba.pm/install.sh | bash
  else
    echo "❌ ERROR: Neither curl nor wget is available. Please install one to proceed."
    # return 1 2>/dev/null || exit 0
    # exit 1
  fi
else
  echo "✅ micromamba is already installed."
fi

# ──────────────────────────────────────────────────────────────
# Apply shell config changes:
# ⚠️ Please restart your shell to activate micromamba or run the following:
# source ~/.bashrc (or ~/.zshrc, ~/.xonshrc, ~/.config/fish/config.fish, ...)
# ──────────────────────────────────────────────────────────────
SHELL_RC=~/."$(basename "$SHELL")"rc

if [ -f "$SHELL_RC" ]; then
  echo "📄 Sourcing shell config: $SHELL_RC"
  # shellcheck disable=SC1090
  # . ~/.bashrc or . ~/.zshrc for zsh
  # . ~/."$(basename "$SHELL")"rc || true  # ~/.bashrc or ~/.zshrc for zsh
  source ~/."$(basename "$SHELL")"rc || echo "⚠️ Failed to source $SHELL_RC"
else
  echo "⚠️ Shell config file not found: $SHELL_RC"
fi

# ──────────────────────────────────────────────────────────────
# 2. Initialize shell integration for conda/micromamba
# ~/.bashrc or ~/.zshrc for zsh
# ──────────────────────────────────────────────────────────────
# Optional: also initialize conda hooks (for compatibility with existing conda setups)
conda init --all || echo "⚠️ Failed to initialize conda hooks"
## Initialize micromamba for the current shell
## Initialize micromamba shell integration for bash (auto-detect install path)
## micromamba shell init -s bash -p ~/micromamba
micromamba shell init -s "$(basename "$SHELL")" || echo "⚠️ Failed to initialize micromamba hooks"

# ──────────────────────────────────────────────────────────────
# 3. Source updated shell configuration (Apply shell config changes)
# Note:
# - ⚠️ micromamba does NOT auto-activate environments on login.
# - ⚠️ You can activate manually: micromamba activate "$ENV_NAME"
# - 📄 Or add `micromamba activate "$ENV_NAME"` at the end of your .bashrc
# ──────────────────────────────────────────────────────────────
# Re-source shell config to ensure activation takes effect
# shellcheck disable=SC1090
# . ~/."$(basename "$SHELL")"rc || true  # ~/.bashrc or ~/.zshrc for zsh
source ~/."$(basename "$SHELL")"rc || echo "⚠️ Failed to source $SHELL_RC"

# ──────────────────────────────────────────────────────────────
# 4. Enable micromamba hook for current session
# Ensure micromamba shell hook command integration is active in the current session
# $ eval "$(micromamba shell hook --shell bash)"
# ──────────────────────────────────────────────────────────────
## echo micromamba shell hook --shell "$(basename "$SHELL")"
## Fallback to bash if SHELL is unset or unknown
eval "$(micromamba shell hook --shell "$(basename "${SHELL:-/bin/bash}")")"

# ──────────────────────────────────────────────────────────────
# 5. Ensure environment exists and is registered
# Create environment if missing and Configure "envs_dirs"
# ──────────────────────────────────────────────────────────────
## Also Configure base
micromamba install -n base python="$PY_VERSION" ipykernel pip -y || true

# ──────────────────────────────────────────────────────────────
# Create environment "$ENV_NAME" if not already present
# Supports either micromamba or conda
# ──────────────────────────────────────────────────────────────
if command -v micromamba >/dev/null 2>&1; then
  echo "📦 Using micromamba to manage environment: $ENV_NAME"

  if ! micromamba env list | grep -q "$ENV_NAME"; then
    echo "🆕 Creating micromamba environment: $ENV_NAME"
    # micromamba create -n py311 python=3.11 ipykernel pip -y
    # micromamba create -n "$ENV_NAME" python="$PY_VERSION" ipykernel pip -y || true
    micromamba env create -f environment.yml --yes \
    && { micromamba clean --all -f -y || true; } \
    && { jupyter lab clean || true; } \
    && { rm -rf "${HOME}/.cache/yarn" || true; } \
    && { rm -rf ${HOME}/.cache || true; } \
    || { echo "⚠️ Failed to creation Micromamba environment"; }
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
    || { echo "⚠️ Failed to creation Conda environment"; }
  else
    echo "✅ conda environment '$ENV_NAME' already exists."
  fi

else
  echo "❌ Neither micromamba nor conda found. Cannot create environment."
  # return 1 2>/dev/null || exit 1
fi

# Register envs directory to ".condarc" for better discovery
# Configure micromamba envs directory to simplify env discovery by conda/micromamba
# Enables users to activate environment without having to specify the full path
mkdir -p ~/micromamba/envs "/opt/conda" || true
# echo "envs_dirs:
#   - ${HOME:-~/}/micromamba/envs" > /opt/conda/.condarc
cat <<EOF > "/opt/conda/.condarc" || echo "⚠️ /opt/conda/.condarc: Permission denied"
envs_dirs:
  - ~/micromamba/envs
EOF

# ──────────────────────────────────────────────────────────────
# 6. Activate environment again for safety (e.g., py311), ignore failure if not present yet
# ──────────────────────────────────────────────────────────────

# Re-source shell config to ensure activation takes effect
# shellcheck disable=SC1090
# . ~/."$(basename "$SHELL")"rc || true  # ~/.bashrc or ~/.zshrc for zsh
source ~/."$(basename "$SHELL")"rc || echo "⚠️ Failed to source $SHELL_RC"

# Optional: auto-activate environment for current session
if command -v micromamba >/dev/null 2>&1; then
  micromamba activate "$ENV_NAME" || micromamba activate base || echo "⚠️ Failed to activate environment micromamba"
fi
# Note that `micromamba activate py311` doesn't work, it must be run by the
# user (same applies to `conda activate`) So try to bash activate to work:

######################################################################
## Clean up caches (if possible)
######################################################################

# Clean up caches and package manager artifacts to reduce disk usage
(sudo -n true && sudo apt-get clean) || true
pip cache purge || true
rm -rf ~/.cache/* || true

echo "✅ Setup complete. Restart your shell or run:"
echo "   source $SHELL_RC"
echo "to activate the micromamba environment in new sessions."

######################################################################
## . (if possible)
######################################################################
