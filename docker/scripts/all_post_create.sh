#!/usr/bin/env bash
# docker/scripts/all_post_create.sh
#
# bash docker/scripts/all_post_create.sh
# POST_CREATE_STRICT=1 bash docker/scripts/all_post_create.sh
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# ===============================================================
# Canonical post-create orchestrator (Bash)
# ===============================================================
# USER NOTES
# - Intended for Dev Containers / Docker "postCreateCommand" style hooks.
# - Strict by default (fails on real errors), but sub-steps can be optional.
# - Heavy OS installs are DISABLED by default.
#
# ENV VARS (explicit controls)
# - POST_CREATE_TRACE=1                 : enable xtrace
# - POST_CREATE_PRINTENV=1              : print environment variables
# - POST_CREATE_DIAGNOSTICS=1           : print OS/container diagnostics
# - POST_CREATE_STRICT=0|1              : if 1, missing/failed optional steps become fatal
# - POST_CREATE_RUN_CONDA=0|1           : default 1
# - POST_CREATE_RUN_MAMBA=0|1           : default 1
# - POST_CREATE_RUN_FIRST_NOTICE=0|1    : default 1
# - POST_CREATE_RUN_POST_CREATE=0|1     : default 1
#
# INSTALL (STRICT OPT-IN)
# - POST_CREATE_ALLOW_INSTALL=1         : allow installs (maps to COMMON_ALLOW_INSTALL=1)
# - POST_CREATE_INSTALL_DEV_TOOLS=1     : run install_dev_tools_apt (requires allow-install)
# - APT_GET_OPTS="..."                  : optional apt-get options override
# - DEBIAN_FRONTEND=noninteractive      : recommended in CI/Docker
# ===============================================================
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

# If available:
# - shellcheck (static correctness)
# - shfmt (format)
# shellcheck -s sh scripts/run.sh scripts/lib/common.sh
# shellcheck -s bash scripts/run.bash
# shfmt -w scripts/*.sh scripts/lib/*.sh

# Enable strict mode: fail on error (-e), undefined var (-u)
# POSIX does not have pipefail; avoid relying on it.
# Note: "pipefail" is NOT POSIX; we do not rely on it.
# set -e  # Exit immediately if a command exits with a non-zero status (Disable 'exit on error' temporarily for debugging)
# set -u  # Treat unset variables as an error
# set -x  # Enable debugging Print each command before executing it
# set -o pipefail  # Ensure pipeline errors are captured
set -Eeuxo pipefail

# The special shell variable IFS determines how Bash recognizes word boundaries while splitting a sequence of character strings.
# The default value of IFS is a three-character string comprising a space, tab, and newline:
echo "$IFS" | cat -et
# IFS=$'\n\t'

if [[ "${POST_CREATE_TRACE:-0}" == "1" ]]; then
  set -x
fi

# umask 022 is commonly used to prevent group and others from having write access, enhancing security.
umask 022
## Make sudo Passwordless for the User
sudo -n true && echo "Passwordless sudo ✅" || echo "Password required ❌"
uname -a >&2 || echo "No uname -a output available. Skipping system information."
cat /etc/os-release || echo "No /etc/os-release file found. Skipping OS release information."
# Printing all environment variables...
printenv

## Dynamically get shell name (bash, zsh, fish, etc.)
# echo "CWD_DIR=$PWD"
# echo "REAL_DIR=$(realpath ./)"
# echo "SCRIPT_DIR=$(cd -- $(dirname ${BASH_SOURCE[0]}) && pwd)"
# echo "SHELL_DIR=$(cd -- $(dirname $0) && pwd)"
# echo "SHELL_NAME=$(basename $SHELL)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

# Source the POSIX common library (works in bash). Re-apply bash strict after.
COMMON_SH="${COMMON_SH:-$SCRIPT_DIR/lib/common.sh}"
if [[ -f "$COMMON_SH" ]]; then
  # common.sh is POSIX; safe to source from bash.
  # It sets `set -eu` internally; we re-apply bash strict mode after.
  # shellcheck source=/dev/null
  . "$COMMON_SH"
  set -Eeuo pipefail
else
  # Minimal fallback logger if common.sh is missing.
  log() { printf '%s\n' "$*" >&2; }
  log_error_code() { local code="${1:-1}"; shift || true; log "[ERROR] $*"; exit "$code"; }
  log_error() { log "[ERROR] $*"; exit 1; }
  log_warning() { log "[WARNING] $*"; }
  log_info() { log "[INFO] $*"; }
fi

# ---------- Error trap (canonical bash) ----------
on_err() {
  local lineno="$1"
  local cmd="$2"
  log_error_code 1 "Failure at line ${lineno}: ${cmd}"
}
trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR

# ---------- Defaults (explicit) ----------
PY_VERSION="${PY_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"

POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"
POST_CREATE_RUN_CONDA="${POST_CREATE_RUN_CONDA:-1}"
POST_CREATE_RUN_MAMBA="${POST_CREATE_RUN_MAMBA:-1}"
POST_CREATE_RUN_FIRST_NOTICE="${POST_CREATE_RUN_FIRST_NOTICE:-1}"
POST_CREATE_RUN_POST_CREATE="${POST_CREATE_RUN_POST_CREATE:-1}"

# Map install allow flag into common.sh expected name
if [[ "${POST_CREATE_ALLOW_INSTALL:-0}" == "1" ]]; then
  export COMMON_ALLOW_INSTALL=1
fi

# ---------- Helpers ----------
maybe_fail() {
  # Usage: maybe_fail "message"
  if [[ "$POST_CREATE_STRICT" == "1" ]]; then
    log_error_code 1 "$1"
  else
    log_warning "$1"
  fi
}

source_step() {
  # Usage: source_step "Title" "/abs/path/to/script.sh"
  local title="$1"
  local path="$2"

  if [[ ! -f "$path" ]]; then
    maybe_fail "${title}: script not found: ${path}"
    return 0
  fi

  log_info "${title}: sourcing ${path}"
  # shellcheck disable=SC1090
  if ! . "$path"; then
    maybe_fail "${title}: script failed: ${path}"
    return 0
  fi

  log_info "${title}: done"
}

# ---------- Optional prints (strict opt-in) ----------
if [[ "${POST_CREATE_PRINTENV:-0}" == "1" ]]; then
  if command -v printenv >/dev/null 2>&1; then
    log_info "Printing environment variables (POST_CREATE_PRINTENV=1)"
    printenv
  else
    maybe_fail "printenv not available"
  fi
fi

if [[ "${POST_CREATE_DIAGNOSTICS:-0}" == "1" ]]; then
  # Prefer common.sh diagnostics if present
  if command -v system_info >/dev/null 2>&1; then
    system_info
  else
    log_info "OS=$(uname -s 2>/dev/null || echo unknown) ARCH=$(uname -m 2>/dev/null || echo unknown)"
    uname -a >&2 || true
  fi
fi

# ---------- Install dev tools (STRICT opt-in; NEVER implicit) ----------
if [[ "${POST_CREATE_INSTALL_DEV_TOOLS:-0}" == "1" ]]; then
  if command -v install_dev_tools_apt >/dev/null 2>&1; then
    log_info "Installing dev tools via apt (POST_CREATE_INSTALL_DEV_TOOLS=1)"
    install_dev_tools_apt
  else
    # Fallback to direct call if common.sh is absent (still strict)
    if [[ "${COMMON_ALLOW_INSTALL:-0}" != "1" ]]; then
      log_error_code 1 "Install blocked. Set POST_CREATE_ALLOW_INSTALL=1 (or COMMON_ALLOW_INSTALL=1)."
    fi
    if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
      log_error_code 1 "apt install is linux-only"
    fi
    command -v apt-get >/dev/null 2>&1 || log_error_code 1 "Missing apt-get"
    : "${DEBIAN_FRONTEND:=noninteractive}"; export DEBIAN_FRONTEND
    APT_GET_OPTS="${APT_GET_OPTS:--y --no-install-recommends}"
    # root or passwordless sudo only
    if [[ "$(id -u 2>/dev/null || echo 1)" == "0" ]]; then
      apt-get update -y
      # shellcheck disable=SC2086
      apt-get install $APT_GET_OPTS sudo gosu git curl build-essential gfortran ninja-build
    else
      command -v sudo >/dev/null 2>&1 || log_error_code 1 "Missing sudo"
      sudo -n true >/dev/null 2>&1 || log_error_code 1 "Passwordless sudo required (non-interactive)"
      sudo apt-get update -y
      # shellcheck disable=SC2086
      sudo apt-get install $APT_GET_OPTS sudo gosu git curl build-essential gfortran ninja-build
    fi
  fi
fi

# ---------- Canonical step paths (no realpath dependency) ----------
# MAMBA_ENV_SCRIPT="$(realpath ./docker/scripts/env_micromamba.sh)"
MAMBA_ENV_SCRIPT="$SCRIPT_DIR/env_micromamba.sh"
CONDA_ENV_SCRIPT="$SCRIPT_DIR/env_conda.sh"
FIRST_RUN_NOTICE_SCRIPT="$SCRIPT_DIR/bash_first_run_notice.sh"
POST_CREATE_SCRIPT="$SCRIPT_DIR/post_create_commands.sh"

# ---------- Execute steps (explicit enable flags) ----------
log_info "Repo root: $REPO_ROOT"
log_info "Script dir: $SCRIPT_DIR"

if [[ "$POST_CREATE_RUN_CONDA" == "1" ]]; then
  source_step "Conda env setup" "$CONDA_ENV_SCRIPT"
fi

if [[ "$POST_CREATE_RUN_MAMBA" == "1" ]]; then
  source_step "Micromamba env setup" "$MAMBA_ENV_SCRIPT"
fi

if [[ "$POST_CREATE_RUN_FIRST_NOTICE" == "1" ]]; then
  source_step "First-run notice" "$FIRST_RUN_NOTICE_SCRIPT"
fi

if [[ "$POST_CREATE_RUN_POST_CREATE" == "1" ]]; then
  source_step "Post-create commands" "$POST_CREATE_SCRIPT"
fi

log_info "all_post_create: complete"
