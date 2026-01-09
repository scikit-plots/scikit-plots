#!/usr/bin/env bash
# docker/scripts/all_post_create.sh
#
# bash docker/scripts/all_post_create.sh
# POST_CREATE_RUN_CONDA_MAMBA=0 bash docker/scripts/all_post_create.sh
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# ===============================================================
# Canonical post-create orchestrator (Bash)
# ===============================================================
# USER NOTES
# - Intended for Dev Containers / Docker "postCreateCommand" style hooks.
# - Works when executed OR sourced.
# - Uses docker/scripts/common.sh as the ONLY common library location.
#
# CONFIG (strict precedence: existing ENV wins)
# - REPO_ROOT (optional)              : absolute repo root; default derived from script dir
# - PY_VERSION=3.12                  : default 3.12
# - ENV_NAME=py312                   : default derived from PY_VERSION
# - ENV_FILE=$REPO_ROOT/environment.yml
#
# CONTROLS
# - POST_CREATE_STRICT=0|1           : default 0 (warn + continue on step failures)
# - POST_CREATE_TRACE=0|1            : default 0 (set -x)
# - POST_CREATE_DIAGNOSTICS=0|1      : default 0 (safe system info)
# - POST_CREATE_PRINT_CONFIG=0|1     : default 1 (prints allowlisted config)
# - POST_CREATE_PRINTENV=0|1         : default 0 (DANGEROUS: may leak secrets)
#
# STEPS (toggles)
# - POST_CREATE_RUN_CONDA=0|1        : default 1
# - POST_CREATE_RUN_MICROMAMBA=0|1   : default 1
# - POST_CREATE_RUN_FIRST_NOTICE=0|1 : default 1
# - POST_CREATE_RUN_POST_CREATE=0|1  : default 1
#
# BACKWARD COMPAT
# - POST_CREATE_RUN_CONDA_MAMBA      : alias for POST_CREATE_RUN_CONDA (legacy)
#
# INSTALL (STRICT OPT-IN)
# - POST_CREATE_ALLOW_INSTALL=1      : maps to COMMON_ALLOW_INSTALL=1
# - POST_CREATE_INSTALL_DEV_TOOLS=1  : runs install_dev_tools_apt (requires allow-install)
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

## Dynamically get shell name (bash, zsh, fish, etc.)
# echo "CWD_DIR=$PWD"
# echo "REAL_DIR=$(realpath ./)"
# echo "SCRIPT_DIR=$(cd -- $(dirname ${BASH_SOURCE[0]}) && pwd)"
# echo "SHELL_DIR=$(cd -- $(dirname $0) && pwd)"
# echo "SHELL_NAME=$(basename $SHELL)"

# _candidates=()
# _add_candidate() {
#   local p="$1"
#   [[ -n "$p" ]] || return 0
#   # de-dupe
#   local x
#   for x in "${_candidates[@]}"; do [[ "$x" == "$p" ]] && return 0; done
#   _candidates+=("$p")
# }
# # 1) explicit override
# _add_candidate "${COMMON_SH:-}"
# # 2) relative to script
# _add_candidate "$SCRIPT_DIR/common.sh"
# # 3) relative to repo-root guess from script
# _add_candidate "$REPO_ROOT/docker/scripts/common.sh"
# # 4) relative to current working directory (you asked to add this)
# _add_candidate "./docker/scripts/common.sh"
# _add_candidate "$(pwd -P)/docker/scripts/common.sh"
# # 5) common mounts/locations
# _add_candidate "/work/docker/scripts/common.sh"
# _add_candidate "/tmp/docker/scripts/common.sh"
# COMMON_SH_RESOLVED=""
# for _p in "${_candidates[@]}"; do
#   if [[ -f "$_p" ]]; then
#     COMMON_SH_RESOLVED="$_p"
#     break
#   fi
# done

# if [[ -z "$COMMON_SH_RESOLVED" ]]; then
#   printf '%s\n' "[ERROR] common.sh not found. Checked:" >&2
#   for _p in "${_candidates[@]}"; do
#     printf '  - %s\n' "$_p" >&2
#   done
#   printf '%s\n' "[ERROR] Fix: ensure docker/scripts/common.sh is in the image/build context (not excluded by .dockerignore), or set COMMON_SH=/abs/path/to/common.sh" >&2
#   exit 2
# fi

# If user runs: sh all_post_create.sh  OR  zsh all_post_create.sh
# make it deterministic: re-exec into bash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# POST_CREATE_TRACE 1
# set -E  # or: set -o errtrace
set -Eeux
set -Eeuxo pipefail || echo "pipefail not available"

# Printing all environment variables...
printenv || echo "printenv not available"
# umask 022 is commonly used to prevent group and others from having write access, enhancing security.
umask 022 || echo "umask not available"

_on_err() {
  local rc="$?"
  printf '%s\n' "[ERROR] all_post_create.sh failed (exit=$rc) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
  exit "$rc"
}
trap _on_err ERR

## Make sudo Passwordless for the User
sudo -n true && echo "Passwordless sudo ✅" || echo "Password required ❌"
uname -a >&2 || echo "No uname -a output available. Skipping system information."
cat /etc/os-release || echo "No /etc/os-release file found. Skipping OS release information."

apc_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
apc_exit_or_return() { local rc="${1:-0}"; apc_is_sourced && return "$rc" || exit "$rc"; }

apc_main() {
  # Save caller state (safe for sourced mode; NO trap RETURN)
  local _OLD_SET _OLD_UMASK _OLD_TRAP_ERR _OLD_PWD
  _OLD_SET="$(set +o)"
  _OLD_UMASK="$(umask)"
  _OLD_TRAP_ERR="$(trap -p ERR || true)"
  _OLD_PWD="$(pwd -P 2>/dev/null || pwd)"

  apc_restore() {
    eval "$_OLD_SET"
    umask "$_OLD_UMASK" 2>/dev/null || true
    cd -- "$_OLD_PWD" 2>/dev/null || true
    if [[ -n "$_OLD_TRAP_ERR" ]]; then eval "$_OLD_TRAP_ERR"; else trap - ERR; fi
  }

  apc_on_err() {
    local rc="$?"
    printf '%s\n' "[ERROR] all_post_create.sh failed (exit=$rc) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
    apc_restore
    apc_exit_or_return "$rc"
  }

  trap 'apc_on_err' ERR
  set -Eeuo pipefail
  umask 022

  # -------- canonical paths --------
  local SCRIPT_DIR REPO_ROOT
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
  export SCRIPT_DIR REPO_ROOT

  # -------- source common (ONLY canonical path) --------
  : "${COMMON_SH:=$REPO_ROOT/docker/scripts/common.sh}"
  if [[ ! -f "$COMMON_SH" ]]; then
    printf '%s\n' "[ERROR] common.sh not found: $COMMON_SH" >&2
    printf '%s\n' "[ERROR] Fix: ensure docker/scripts/common.sh is included in build context and not excluded by .dockerignore" >&2
    apc_restore
    apc_exit_or_return 2
  fi
  export COMMON_SH
  # shellcheck source=/dev/null
  . "$COMMON_SH"

  # -------- global env normalization (single source of truth) --------
  default_var POST_CREATE_STRICT "0"
  default_var POST_CREATE_TRACE "0"
  default_var POST_CREATE_DIAGNOSTICS "0"
  default_var POST_CREATE_PRINT_CONFIG "1"
  default_var POST_CREATE_PRINTENV "0"

  default_var POST_CREATE_ALLOW_INSTALL "0"
  default_var POST_CREATE_INSTALL_DEV_TOOLS "0"

  # Core env
  default_var PY_VERSION "3.12"
  default_var ENV_NAME "py${PY_VERSION//./}"
  default_var ENV_FILE "$REPO_ROOT/environment.yml"

  # Legacy alias
  if [[ -z "${POST_CREATE_RUN_CONDA:-}" && -n "${POST_CREATE_RUN_CONDA_MAMBA:-}" ]]; then
    POST_CREATE_RUN_CONDA="$POST_CREATE_RUN_CONDA_MAMBA"
  fi

  default_var POST_CREATE_RUN_CONDA "1"
  default_var POST_CREATE_RUN_MICROMAMBA "1"
  default_var POST_CREATE_RUN_FIRST_NOTICE "1"
  default_var POST_CREATE_RUN_POST_CREATE "1"

  if [[ "$POST_CREATE_ALLOW_INSTALL" == "1" ]]; then
    export COMMON_ALLOW_INSTALL=1
  fi

  export \
    POST_CREATE_STRICT POST_CREATE_TRACE POST_CREATE_DIAGNOSTICS POST_CREATE_PRINT_CONFIG POST_CREATE_PRINTENV \
    POST_CREATE_ALLOW_INSTALL POST_CREATE_INSTALL_DEV_TOOLS \
    PY_VERSION ENV_NAME ENV_FILE \
    POST_CREATE_RUN_CONDA POST_CREATE_RUN_MICROMAMBA POST_CREATE_RUN_FIRST_NOTICE POST_CREATE_RUN_POST_CREATE

  if [[ "$POST_CREATE_TRACE" == "1" ]]; then
    set -x
  fi

  maybe_fail() {
    local msg="${1:-unknown failure}"
    if [[ "$POST_CREATE_STRICT" == "1" ]]; then
      log_error_code 1 "$msg"
    else
      log_warning "$msg"
      return 0
    fi
  }

  log_info "Repo root: $REPO_ROOT"
  log_info "Script dir: $SCRIPT_DIR"

  if [[ "$POST_CREATE_PRINT_CONFIG" == "1" ]]; then
    common_print_config_safe \
      REPO_ROOT SCRIPT_DIR COMMON_SH \
      PY_VERSION ENV_NAME ENV_FILE \
      POST_CREATE_STRICT POST_CREATE_TRACE POST_CREATE_DIAGNOSTICS POST_CREATE_PRINTENV \
      POST_CREATE_ALLOW_INSTALL POST_CREATE_INSTALL_DEV_TOOLS \
      POST_CREATE_RUN_CONDA POST_CREATE_RUN_MICROMAMBA POST_CREATE_RUN_FIRST_NOTICE POST_CREATE_RUN_POST_CREATE
  fi

  if [[ "$POST_CREATE_DIAGNOSTICS" == "1" ]]; then
    log_env_summary || true
    [[ -f /etc/os-release ]] && cat /etc/os-release >&2 || true
    if has_cmd sudo; then
      sudo -n true >/dev/null 2>&1 && log_info "sudo: passwordless ✅" || log_warning "sudo: passwordless ❌"
    fi
  fi

  # Never printenv unless explicitly enabled (secrets)
  if [[ "$POST_CREATE_PRINTENV" == "1" ]]; then
    if has_cmd printenv; then
      log_warning "printenv enabled; may leak secrets"
      printenv
    else
      maybe_fail "printenv not available"
    fi
  fi

  # Optional heavy installs (explicit opt-in)
  if [[ "$POST_CREATE_INSTALL_DEV_TOOLS" == "1" ]]; then
    if [[ "${COMMON_ALLOW_INSTALL:-0}" == "1" ]]; then
      install_dev_tools_apt || maybe_fail "install_dev_tools_apt failed"
    else
      maybe_fail "POST_CREATE_INSTALL_DEV_TOOLS=1 requires POST_CREATE_ALLOW_INSTALL=1"
    fi
  fi

  # Source steps with controlled error capture (no trap poisoning)
  source_step() {
    local title="$1" path="$2"
    if [[ -z "$path" || ! -f "$path" ]]; then
      maybe_fail "${title}: missing script: $path"
      return 0
    fi

    log_info "${title}: sourcing $path"

    local _STEP_SET _STEP_UMASK _STEP_PWD _TRAP_ERR _TRAP_EXIT _TRAP_INT _TRAP_TERM rc
    _STEP_SET="$(set +o)"
    _STEP_UMASK="$(umask)"
    _STEP_PWD="$(pwd -P 2>/dev/null || pwd)"
    _TRAP_ERR="$(trap -p ERR || true)"
    _TRAP_EXIT="$(trap -p EXIT || true)"
    _TRAP_INT="$(trap -p INT || true)"
    _TRAP_TERM="$(trap -p TERM || true)"

    # allow step to fail without aborting orchestrator; capture rc
    trap - ERR
    set +e

    # shellcheck source=/dev/null
    . "$path"
    rc=$?

    # restore shell state
    eval "$_STEP_SET"
    umask "$_STEP_UMASK" 2>/dev/null || true
    cd -- "$_STEP_PWD" 2>/dev/null || true
    if [[ -n "$_TRAP_ERR" ]]; then eval "$_TRAP_ERR"; else trap - ERR; fi
    if [[ -n "$_TRAP_EXIT" ]]; then eval "$_TRAP_EXIT"; else trap - EXIT; fi
    if [[ -n "$_TRAP_INT" ]]; then eval "$_TRAP_INT"; else trap - INT; fi
    if [[ -n "$_TRAP_TERM" ]]; then eval "$_TRAP_TERM"; else trap - TERM; fi

    if [[ "$rc" -ne 0 ]]; then
      maybe_fail "${title}: failed (rc=$rc): $path"
      return 0
    fi

    log_info "${title}: done"
    return 0
  }

  local CONDA_ENV_SCRIPT="$SCRIPT_DIR/env_conda.sh"
  local MICROMAMBA_ENV_SCRIPT="$SCRIPT_DIR/env_micromamba.sh"
  local FIRST_RUN_NOTICE_SCRIPT="$SCRIPT_DIR/bash_first_run_notice.sh"
  local POST_CREATE_SCRIPT="$SCRIPT_DIR/post_create_commands.sh"

  [[ "$POST_CREATE_RUN_CONDA" == "1" ]] && source_step "Conda env setup" "$CONDA_ENV_SCRIPT"
  [[ "$POST_CREATE_RUN_MICROMAMBA" == "1" ]] && source_step "Micromamba env setup" "$MICROMAMBA_ENV_SCRIPT"
  [[ "$POST_CREATE_RUN_FIRST_NOTICE" == "1" ]] && source_step "First-run notice" "$FIRST_RUN_NOTICE_SCRIPT"
  [[ "$POST_CREATE_RUN_POST_CREATE" == "1" ]] && source_step "Post-create commands" "$POST_CREATE_SCRIPT"

  log_success "all_post_create: complete"
  apc_restore
  return 0
}

apc_main "$@"
apc_exit_or_return $?
