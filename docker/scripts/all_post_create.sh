#!/usr/bin/env bash
# Shebang Syntax Summary:
# ✅ #!/usr/bin/env bash — Recommended for portability, Open-source scripts, multi-platform devcontainers
# ✅ #!/bin/bash — Recommended for strict system environments, Controlled systems, Docker, CI/CD
# ✅ #!/bin/sh — POSIX-compliant shell — minimal and fast, but lacks many Bash features.
# ✅ #!/usr/bin/env python3 — For Python scripts using env.
# ✅ #!/usr/bin/env -S bash -e — Bash with options (modern env). Advanced with arguments (less common, Bash-only).

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# .devcontainer/scripts/all_post_create.sh

set -e  # Exit script on error (Disable 'exit on error' temporarily for debugging)
set -x  # Enable debugging (prints commands as they run)
set -euxo pipefail

## Dynamically get shell name (bash, zsh, fish, etc.)
echo "shell_name=$(basename "$SHELL")"
echo "SHELL_DIR=$(cd -- "$(dirname "$0")" && pwd)"
echo "SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$PWD"

# shellcheck disable=SC1090
source "$HOME/.$(basename "$SHELL")rc" || true

## Make sudo Passwordless for the User
sudo -n true && echo "Passwordless sudo ✅" || echo "Password required ❌"

## Ensure os base ppackages installed
echo "📦 Installing dev tools (if sudo available)..."
(sudo -n true && sudo apt-get update -y \
    && sudo apt-get install -y sudo gosu git curl build-essential gfortran) \
    || echo "⚠️ Failed or skipped installing dev tools"

######################################################################
## first-run notice (if possible)
######################################################################

echo "📝 Setting up first-run notice (if possible)..."
# Use sudo non-interactively if available
if sudo -n true 2>/dev/null; then
    sudo mkdir -p /usr/local/etc/vscode-dev-containers
    # Optional: Install notice file
    sudo cp "$PWD/.devcontainer/scripts/first-run-notice.txt" /usr/local/etc/vscode-dev-containers/first-run-notice.txt || echo "⚠️ Could not copy notice"
else
    echo "⚠️ Skipping first-run notice setup (sudo not available or no permission)"
fi

######################################################################
## micromamba env (if possible)
######################################################################

echo "🔁 Sourcing micromamba env setup..."
. "$PWD/.devcontainer/scripts/env_micromamba.sh" || echo "⚠️ Micromamba env setup failed or skipped"

######################################################################
## post-create steps (if possible)
######################################################################

echo "🚀 Running post-create steps..."
. "$PWD/.devcontainer/scripts/post_create_commands.sh" || echo "⚠️ Post-create steps failed or skipped"

######################################################################
## . (if possible)
######################################################################
