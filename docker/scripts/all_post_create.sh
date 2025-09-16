#!/usr/bin/env bash
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
## $(eval echo ~...) breaks in Docker, CI, or Windows paths.
## Inside bash -c '...' string	\$p, if needed
# ( ... )  || fallback runs in a subshell — changes inside don't affect the parent script.
# { ...; } || fallback runs in current shell — can exit or affect current environment.
#
# Shebang Syntax Summary:
# ✅ #!/usr/bin/env bash — Recommended for portability, Open-source scripts, multi-platform devcontainers
# ✅ #!/bin/bash — Recommended for strict system environments, Controlled systems, Docker, CI/CD
# ✅ #!/bin/sh — POSIX-compliant shell — minimal and fast, but lacks many Bash features.
# ✅ #!/usr/bin/env python3 — For Python scripts using env.
# ✅ #!/usr/bin/env -S bash -e — Bash with options (modern env). Advanced with arguments (less common, Bash-only).

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
## Micromamba env setup (if possible)
## env_micromamba.sh
######################################################################

MAMBA_ENV_SCRIPT="$(realpath ./docker/scripts/env_micromamba.sh)"
echo -e "\033[1;34m🔁 Sourcing micromamba env setup...\033[0m"

if [ -f "$MAMBA_ENV_SCRIPT" ]; then
  # shellcheck disable=SC1090
  . "$MAMBA_ENV_SCRIPT" || echo -e "\033[1;33m⚠️ Micromamba env script ran but failed\033[0m"
else
  echo -e "\033[1;33m⚠️ Micromamba env script not found at $MAMBA_ENV_SCRIPT\033[0m"
fi

######################################################################
## Conda env setup (if possible)
## env_conda.sh
######################################################################

CONDA_ENV_SCRIPT="$(realpath ./docker/scripts/env_conda.sh)"
echo -e "\033[1;34m🔁 Sourcing conda env setup...\033[0m"

if [ -f "$CONDA_ENV_SCRIPT" ]; then
  # shellcheck disable=SC1090
  . "$CONDA_ENV_SCRIPT" || echo -e "\033[1;33m⚠️ Conda env script ran but failed\033[0m"
else
  echo -e "\033[1;33m⚠️ Conda env script not found at $CONDA_ENV_SCRIPT\033[0m"
fi

######################################################################
## First-Run Notice and ASCII banner for scikit-plots (if possible)
## bash_first_run_notice.sh
######################################################################

FIRST_RUN_NOTICE_SCRIPT="$(realpath ./docker/scripts/bash_first_run_notice.sh)"
echo -e "\033[1;34m🚀 Running First-Run Notice setup...\033[0m"

if [ -f "$FIRST_RUN_NOTICE_SCRIPT" ]; then
  # shellcheck disable=SC1090
  . "$FIRST_RUN_NOTICE_SCRIPT" || echo -e "\033[1;33m⚠️ First-Run Notice script ran but failed\033[0m"
else
  echo -e "\033[1;33m⚠️ First-Run Notice script not found at $FIRST_RUN_NOTICE_SCRIPT\033[0m"
fi

######################################################################
## Post-create steps (if possible)
## post_create_commands.sh
######################################################################

POST_CREATE_SCRIPT="$(realpath ./docker/scripts/post_create_commands.sh)"
echo -e "\033[1;34m🚀 Running post-create steps...\033[0m"

if [ -f "$POST_CREATE_SCRIPT" ]; then
  # shellcheck disable=SC1090
  . "$POST_CREATE_SCRIPT" || echo -e "\033[1;33m⚠️ Post-create script ran but failed\033[0m"
else
  echo -e "\033[1;33m⚠️ Post-create script not found at $POST_CREATE_SCRIPT\033[0m"
fi

######################################################################
##
######################################################################
