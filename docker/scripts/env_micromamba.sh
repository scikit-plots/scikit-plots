#!/usr/bin/env bash
# Copied scipy project

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

## Inside bash -c '...' string	\$p
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
  && sudo apt-get install -y sudo gosu git curl build-essential gfortran; } \
  || echo "⚠️ Failed or skipped installing dev tools"

######################################################################
## 📦 Installing Conda/Mamba environment
# micromamba not "conda" keyword compatipable but same syntax
######################################################################
set +u   # Disable strict unbound mode

# Install micromamba via official install script silently, only if not installed
# if ! command -v micromamba &> /dev/null; then
echo "🔧 Installing micromamba or conda..."

## 1. Install micromamba via curl or fallback to wget
# if ! command -v micromamba &> /dev/null; then
if command -v curl &> /dev/null; then
  # curl -Ls https://micro.mamba.pm/install.sh | bash
  # "${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
  "${SHELL}" <(curl -Ls micro.mamba.pm/install.sh) < /dev/null
elif command -v wget &> /dev/null; then
  wget -qO- https://micro.mamba.pm/install.sh | bash
else
  echo "❌ Neither curl nor wget is available to download micromamba."
  echo "Please install curl or wget first."
  # exit 1
fi
# fi

## 2. Initialize micromamba shell support
# Initialize conda for all shells (optional if you use conda alongside micromamba)
# "conda" keyword compatipable Env (e.g., Conda, Miniconda, Mamba)
conda init --all || true
## Initialize micromamba shell integration for bash (auto-detect install path)
## micromamba shell init -s <shell_name> -p <micromamba_install_path>
## micromamba shell init -s bash -p ~/micromamba
micromamba shell init -s "$(basename "$SHELL")" || true

## 3. Source the shell profile (to apply init changes)
## Conda/Mamba environment auto-activation
add_auto_micromamba_env() {
  local rc_file=${1:-"$HOME/.bashrc"}  # default to user .bashrc
  local marker="# >>> Conda/Mamba environment auto-activation >>>"

  if grep -Fxq "$marker" "$rc_file"; then
    echo "✅ Auto-activation block already exists in $rc_file. Skipping..."
  else
    echo "🔧 Adding auto-activation block to $rc_file"
    cat << 'EOF' >> "$rc_file"

# >>> Conda/Mamba environment auto-activation >>>
# Auto-activate py311 if it exists, otherwise fallback to base
# if micromamba env list | grep -qE '(^|[[:space:]])py311([[:space:]]|$)'; then
# Only run in interactive shell, If Needed
# if [[ $- == *i* ]]; then
#
# Only set MAMBA_ROOT_PREFIX if it's not already set
# readonly locks it, this locks the variable so it cannot be reassigned accidentally later in the script.
# readonly MAMBA_ROOT_PREFIX
# echo "${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# : is a no-op command used here to safely assign a default.
: "${MAMBA_ROOT_PREFIX:=$HOME/micromamba}"
if command -v micromamba >/dev/null 2>&1 && [[ -d "$MAMBA_ROOT_PREFIX/envs/py311" ]]; then
  micromamba activate py311
elif command -v conda >/dev/null 2>&1 && [[ -d "/opt/conda/envs/py311" ]]; then
  conda activate py311
elif command -v micromamba >/dev/null 2>&1; then
  micromamba activate base
elif command -v conda >/dev/null 2>&1; then
  conda activate base
else
  echo "❌ No compatible conda/mamba environment found." >&2
  # Don't use exit 0 in .bashrc — it can break the shell
fi
# fi
# <<< Conda/Mamba environment auto-activation <<<
EOF
  fi
}
## Or to a global/system file
## add_auto_micromamba_env /etc/bash.bashrc
add_auto_micromamba_env "$HOME/.$(basename "$SHELL")rc"
## Source shell config to enable 'micromamba activate' command in current shell
## Note: The user must run 'micromamba activate <env>' manually after login
## or source shell config, automatic activation is not supported.
## source ~/micromamba/envs/py311/etc/profile.d/conda.sh  # Activate without shell hook (not recommended)
## Source shell config to enable activation (user must restart shell for persistent effect)
## Source shell config to ensure changes are loaded (optional, depends on context)
## This is safe in interactive or login scripts, but not always necessary in non-interactive CI scripts.
# if [ -f "$HOME/.$(basename "$SHELL")rc" ]; then
# shellcheck disable=SC1090
source "$HOME/.$(basename "$SHELL")rc" || true  # ~/.bashrc or ~/.zshrc for zsh
# fi

## 4. Enable micromamba command integration in the current session
## Ensure shell hook is active in current session
## $ eval "$(micromamba shell hook --shell bash)"
## echo micromamba shell hook --shell "$(basename "$SHELL")"
eval "$(micromamba shell hook --shell "$(basename "$SHELL")")"

## 5. Create environment if missing and Configure "envs_dirs"
## Also Configure base
micromamba install -n base python=3.11 ipykernel pip -y || true
# Create environment if not exists
ENV_NAME="py311"
if ! micromamba env list | grep -q "$ENV_NAME"; then
  # micromamba create -n "$ENV_NAME" python=3.11 ipykernel pip -y || true
  micromamba env create -f environment.yml --yes || true
fi
## Enables users to activate environment without having to specify the full path
## Configure micromamba envs directory to simplify env discovery by conda/micromamba
mkdir -p "$HOME/micromamba/envs" "/opt/conda" || true
echo "envs_dirs:
  - $HOME/micromamba/envs" > /opt/conda/.condarc
# Note that `micromamba activate scipy-dev` doesn't work, it must be run by the
# user (same applies to `conda activate`)

## 6. Ensure to Activate the environment (e.g., py311), ignore failure if not present yet
# shellcheck disable=SC1090
source "$HOME/.$(basename "$SHELL")rc" || true
micromamba activate "$ENV_NAME" || true

######################################################################
## Clean up caches (if possible)
######################################################################

# Clean up caches and package manager artifacts to reduce disk usage
(sudo -n true && sudo apt-get clean) || true
pip cache purge || true
rm -rf ~/.cache/* || true

echo "Please restart your shell or run 'source ~/.$(basename "$SHELL")rc' to enable micromamba environment activation."

######################################################################
## . (if possible)
######################################################################
