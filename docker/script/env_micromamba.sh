#!/bin/bash

# Copied scipy project
set -e  # Exit script on error (Disable 'exit on error' temporarily for debugging)
set -x  # Enable debugging (prints commands as they run)
set -euxo pipefail

# Initialize conda for all shells (optional if you use conda alongside micromamba)
# "conda" keyword compatipable Env (e.g., Conda, Miniconda, Mamba)
conda init --all || true

# Install micromamba via official install script silently, only if not installed
# micromamba not "conda" keyword compatipable but same syntax
# "${SHELL}" <(curl -Ls micro.mamba.pm/install.sh) < /dev/null
# if ! command -v micromamba &> /dev/null; then
echo "🔧 Installing micromamba..."

# Check for curl or fallback to wget
if command -v curl &> /dev/null; then
  # curl -Ls https://micro.mamba.pm/install.sh | bash
  "${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
elif command -v wget &> /dev/null; then
  wget -qO- https://micro.mamba.pm/install.sh | bash
else
  echo "❌ Neither curl nor wget is available to download micromamba."
  echo "Please install curl or wget first."
  exit 1
fi
# fi

# Dynamically get shell name (bash, zsh, fish, etc.)
shell_name=$(basename "$SHELL")
# Source shell config to enable 'micromamba activate' command in current shell
# Note: The user must run 'micromamba activate <env>' manually after login
# or source shell config, automatic activation is not supported.
# shellcheck disable=SC1090
# source ~/micromamba/envs/py311/etc/profile.d/conda.sh  # Activate without shell hook (not recommended)
# Source shell config to enable activation (user must restart shell for persistent effect)
# Source shell config to ensure changes are loaded (optional, depends on context)
# This is safe in interactive or login scripts, but not always necessary in non-interactive CI scripts.
if [ -f "$HOME/.${shell_name}rc" ]; then
    source "$HOME/.${shell_name}rc"  # ~/.bashrc or ~/.zshrc for zsh
fi

# Ensure shell hook is active in current session
# $ eval "$(micromamba shell hook --shell bash)"
eval "$(micromamba shell hook --shell \"$shell_name\")"

# Initialize micromamba shell integration for bash (auto-detect install path)
# micromamba shell init -s <shell_name> -p <micromamba_install_path>
# micromamba shell init -s bash -p ~/micromamba
micromamba shell init -s "$shell_name" || true

# Create env if not exists
if ! micromamba env list | grep -q py311; then
  micromamba env create -f environment.yml --yes || true
fi
# Note that `micromamba activate scipy-dev` doesn't work, it must be run by the
# user (same applies to `conda activate`)

# Try to activate env, ignore failure if not present yet
micromamba activate py311 || true

# Configure env dirs
# Enables users to activate environment without having to specify the full path
# Configure micromamba envs directory to simplify env discovery by conda/micromamba
mkdir -p "/opt/conda" "$HOME/micromamba/envs" || true
echo "envs_dirs:
  - $HOME/micromamba/envs" > /opt/conda/.condarc

# Clean up caches and package manager artifacts to reduce disk usage
sudo apt-get clean || true
pip cache purge || true
rm -rf ~/.cache/* || true

echo "Please restart your shell or run 'source ~/.${shell_name}rc' to enable micromamba environment activation."
