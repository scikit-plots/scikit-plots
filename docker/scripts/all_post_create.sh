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
# shellcheck -s sh scripts/run.sh scripts/common.sh
# shellcheck -s bash scripts/run.bash
# shfmt -w scripts/*.sh scripts/*.sh

# Enable strict mode: fail on error (-e), undefined var (-u)
# POSIX does not have pipefail; avoid relying on it.
# Note: "pipefail" is NOT POSIX; we do not rely on it.
# set -e  # Exit immediately if a command exits with a non-zero status (Disable 'exit on error' temporarily for debugging)
# set -u  # Treat unset variables as an error
# set -x  # Enable debugging Print each command before executing it
# set -o pipefail  # Ensure pipeline errors are captured
# The special shell variable IFS determines how Bash recognizes word boundaries while splitting a sequence of character strings.
# The default value of IFS is a three-character string comprising a space, tab, and newline:
# echo "$IFS" | cat -et
# IFS=$'\n\t'
# if [[ "${POST_CREATE_TRACE:-0}" == "1" ]]; then
#   set -x
# fi

# If user runs: sh all_post_create.sh  OR  zsh all_post_create.sh
# make it deterministic: re-exec into bash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

_on_err() {
  local rc="$?"
  printf '%s\n' "[ERROR] all_post_create.sh failed (exit=$rc) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
  exit "$rc"
}
trap _on_err ERR

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
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
PWD_ABS="$(pwd -P)"

_candidates=()
_add_candidate() {
  local p="$1"
  [[ -n "$p" ]] || return 0
  # de-dupe
  local x
  for x in "${_candidates[@]}"; do [[ "$x" == "$p" ]] && return 0; done
  _candidates+=("$p")
}
# 1) explicit override
_add_candidate "${COMMON_SH:-}"
# 2) relative to script
_add_candidate "$SCRIPT_DIR/common.sh"
# 3) relative to repo-root guess from script
_add_candidate "$REPO_ROOT/docker/scripts/common.sh"
# 4) relative to current working directory (you asked to add this)
_add_candidate "./docker/scripts/common.sh"
_add_candidate "$PWD_ABS/docker/scripts/common.sh"
# 5) common mounts/locations
_add_candidate "/work/docker/scripts/common.sh"
_add_candidate "/tmp/docker/scripts/common.sh"
COMMON_SH_RESOLVED=""
for _p in "${_candidates[@]}"; do
  if [[ -f "$_p" ]]; then
    COMMON_SH_RESOLVED="$_p"
    break
  fi
done

if [[ -z "$COMMON_SH_RESOLVED" ]]; then
  printf '%s\n' "[ERROR] common.sh not found. Checked:" >&2
  for _p in "${_candidates[@]}"; do
    printf '  - %s\n' "$_p" >&2
  done
  printf '%s\n' "[ERROR] Fix: ensure docker/scripts/common.sh is in the image/build context (not excluded by .dockerignore), or set COMMON_SH=/abs/path/to/common.sh" >&2
  exit 2
fi

export COMMON_SH="$COMMON_SH_RESOLVED"
# shellcheck source=/dev/null
. "$COMMON_SH"

# ---------- Defaults (explicit) ----------
# Strict/trace are controlled here (not inside common.sh)
POST_CREATE_TRACE="${POST_CREATE_TRACE:-0}"
common_enable_strict "$POST_CREATE_TRACE"
common_set_umask "${COMMON_UMASK:-022}"

POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"
POST_CREATE_PRINTENV="${POST_CREATE_PRINTENV:-0}"
POST_CREATE_DIAGNOSTICS="${POST_CREATE_DIAGNOSTICS:-0}"

POST_CREATE_RUN_CONDA="${POST_CREATE_RUN_CONDA:-0}"
POST_CREATE_RUN_MAMBA="${POST_CREATE_RUN_MAMBA:-1}"
POST_CREATE_RUN_FIRST_NOTICE="${POST_CREATE_RUN_FIRST_NOTICE:-1}"
POST_CREATE_RUN_POST_CREATE="${POST_CREATE_RUN_POST_CREATE:-1}"

PY_VERSION="${PY_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"
# MAMBA_ENV_SCRIPT="$(realpath ./docker/scripts/env_micromamba.sh)"

maybe_fail() {
  local msg="${1:-unknown failure}"
  if [[ "$POST_CREATE_STRICT" == "1" ]]; then
    log_error_code 1 "$msg"
  else
    log_warning "$msg"
  fi
}

# IMPORTANT CONTRACT for sourced step scripts:
# - MUST NOT call `exit`
# - MUST NOT change global shell options (`set -e/-u/...`)
# - MUST return non-zero on failure
source_step() {
  local title="${1:-step}"
  local path="${2:-}"

  if [[ -z "$path" || ! -f "$path" ]]; then
    maybe_fail "${title}: script not found: ${path}"
    return 0
  fi

  log_info "${title}: sourcing ${path}"

  # Temporarily relax errexit so we can capture rc deterministically.
  local rc=0
  set +e
  set +o pipefail
  # shellcheck source=/dev/null
  . "$path"
  rc=$?
  set -e
  set -o pipefail

  if [[ "$rc" -ne 0 ]]; then
    maybe_fail "${title}: failed (rc=$rc): ${path}"
    return 0
  fi

  log_info "${title}: done"
}

# Map install allow flag into common library expected name
if [[ "${POST_CREATE_ALLOW_INSTALL:-0}" == "1" ]]; then
  export COMMON_ALLOW_INSTALL=1
fi

# Optional controlled diagnostics (OFF by default)
if [[ "$POST_CREATE_DIAGNOSTICS" == "1" ]]; then
  log_env_summary || true
  if has_cmd uname; then uname -a >&2 || true; fi
  [[ -f /etc/os-release ]] && cat /etc/os-release >&2 || true
  if has_cmd sudo; then sudo -n true >/dev/null 2>&1 && log_info "sudo: passwordless ✅" || log_warning "sudo: passwordless ❌"; fi
fi

# Optional environment print (OFF by default; may contain secrets)
if [[ "$POST_CREATE_PRINTENV" == "1" ]]; then
  has_cmd printenv || maybe_fail "printenv not available"
  log_info "printenv (POST_CREATE_PRINTENV=1)"
  printenv
fi

CONDA_ENV_SCRIPT="$SCRIPT_DIR/env_conda.sh"
MAMBA_ENV_SCRIPT="$SCRIPT_DIR/env_micromamba.sh"
FIRST_RUN_NOTICE_SCRIPT="$SCRIPT_DIR/bash_first_run_notice.sh"
POST_CREATE_SCRIPT="$SCRIPT_DIR/post_create_commands.sh"

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

log_success "all_post_create: complete"
