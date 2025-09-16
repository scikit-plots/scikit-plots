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
## DevContainer First-Run Notice (if possible)
######################################################################

echo -e "\033[1;34m📝 Setting up first-run notice (if possible)...\033[0m"

NOTICE_SOURCE="$(realpath ./docker/scripts/bash-first-run-notice.txt)"
NOTICE_TARGET="/usr/local/etc/vscode-dev-containers/bash-first-run-notice.txt"

# Use sudo non-interactively if available
if sudo -n true 2>/dev/null; then
    if [ -f "$NOTICE_SOURCE" ]; then
        sudo mkdir -p "$(dirname "$NOTICE_TARGET")"
        sudo cp "$NOTICE_SOURCE" "$NOTICE_TARGET" && \
          echo -e "\033[1;32m✅ First-run notice installed at $NOTICE_TARGET\033[0m" || \
          echo -e "\033[1;33m\⚠️ Could not copy first-run notice file\033[0m"
    else
        echo -e "\033[1;33m⚠️ Source notice file not found: $NOTICE_SOURCE\033[0m"
    fi
else
    echo -e "\033[1;33m⚠️ Skipping first-run notice setup (sudo not available or permission denied)\033[0m"
fi

######################################################################
## 📝 ASCII banner for scikit-plots (bashrc.prefix → /etc/bash.bashrc)
######################################################################

echo -e "\033[1;34m📝 Setting up bash first-run notice (if possible)...\033[0m"

# [ -f "./docker/scripts/bash-first-run-notice.txt" ] && { cp ./docker/scripts/bash-first-run-notice.txt ~/.bash-first-run-notice.txt; } || true
NOTICE_SOURCE="$(realpath ./docker/scripts/bash-first-run-notice.txt)"
NOTICE_TARGET=~/.bash-first-run-notice.txt

# if [ -f "$NOTICE_SOURCE" ] && [ ! -f "$NOTICE_TARGET" ]; then
if [ -f "$NOTICE_SOURCE" ]; then
  cp -u "$NOTICE_SOURCE" "$NOTICE_TARGET" && \
  echo "✅ First-run notice installed to $NOTICE_TARGET" || \
  echo "⚠️ Could not copy notice to $NOTICE_TARGET"
else
  echo "⚠️ Notice file not found: $NOTICE_SOURCE"
fi

######################################################################
## 🧩 Setup bashrc.prefix (→ /etc/bash.bashrc) [System-wide]
######################################################################

BASHRC_PREFIX_SOURCE="$(realpath ./docker/scripts/bashrc.prefix)"
BASHRC_PREFIX_TARGET="/etc/bash.bashrc"  # to System wide initialization file

# sudo chmod a+rwx "$BASHRC_PREFIX_TARGET" && \
if [ -f "$BASHRC_PREFIX_SOURCE" ]; then
  if sudo -n true 2>/dev/null; then
    sudo cp "$BASHRC_PREFIX_SOURCE" "$BASHRC_PREFIX_TARGET" && \
    sudo chmod a+r "$BASHRC_PREFIX_TARGET" && \
    echo "✅ Global bashrc.prefix installed to $BASHRC_PREFIX_TARGET" || \
    echo "⚠️ Failed to install to $BASHRC_PREFIX_TARGET"
  else
    echo "⚠️ Skipped: sudo not available to install to $BASHRC_PREFIX_TARGET"
  fi
else
  echo "⚠️ Prefix source file not found: $BASHRC_PREFIX_SOURCE"
fi

######################################################################
## 👤 Setup bashrc.suffix (appended to ~/.bashrc if not already present)
######################################################################

BASHRC_SUFFIX_SOURCE="$(realpath ./docker/scripts/bashrc.suffix)"
BASHRC_SUFFIX_TARGET=~/.bashrc  # to the standard personal initialization file
BASHRC_SUFFIX_MARKER="# >>> (bashrc.suffix) scikit-plots personal initialization >>>"

## -q	Quiet, just return status	Basic existence check
## -F	Fixed (literal) string	Avoiding regex interpretation
## -x	Match entire line exactly	Ensuring full-line match, not partial
# grep -Fxq '# >>> (bashrc.suffix) scikit-plots personal initialization >>>' \"~/.bashrc\" || cat ./docker/scripts/bashrc >> \"~/.bashrc\" || true
if [ -f "$BASHRC_SUFFIX_SOURCE" ]; then
  if ! grep -Fq "$BASHRC_SUFFIX_MARKER" "$BASHRC_SUFFIX_TARGET"; then
    {
      echo ""
      echo "$BASHRC_SUFFIX_MARKER"
      cat "$BASHRC_SUFFIX_SOURCE"
    } >> "$BASHRC_SUFFIX_TARGET" && \
    echo "✅ Appended bashrc.suffix to $BASHRC_SUFFIX_TARGET"
  else
    echo "✅ bashrc.suffix already present in $BASHRC_SUFFIX_TARGET"
  fi
else
  echo "⚠️ Suffix source file not found: $BASHRC_SUFFIX_SOURCE"
fi
