#!/usr/bin/env bash
#
# docker/scripts/env_conda.sh
#
# bash docker/scripts/env_conda.sh
# CONDA_STRICT=1 bash docker/scripts/env_conda.sh
# CONDA_MANAGER=conda bash docker/scripts/env_conda.sh
# CONDA_MANAGER=mamba bash docker/scripts/env_conda.sh
# PY_VERSION=3.11 CONDA_ENV_NAME=py311 bash docker/scripts/env_conda.sh
# CONDA_USE_FILE_NAME=1 CONDA_ACTION=create bash docker/scripts/env_conda.sh
# # NOTE: ensure is intentionally blocked in this mode (cannot be done deterministically without guessing).
# CONDA_INIT_BASH=1 bash docker/scripts/env_conda.sh
# CONDA_ACTIVATE=1 bash -c '. docker/scripts/env_conda.sh'
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# ===============================================================
# Conda/Mamba environment setup (Bash)
# ===============================================================
# PURPOSE
# - Deterministic, non-interactive environment creation/update for Dev Containers / Docker.
# - Safe when SOURCED by an orchestrator (recommended) and safe when executed directly.
#
# ZERO SIDE EFFECTS BY DEFAULT
# - No apt installs, no conda init, no rc edits, no cache deletion unless explicitly enabled.
#
# STRICT CONTROLS (explicit env vars)
# - CONDA_SKIP=0|1                   : skip entirely (default 0)
# - CONDA_STRICT=0|1                 : fail hard on missing env file/manager (default inherits POST_CREATE_STRICT else 0)
# - CONDA_MANAGER=auto|conda|mamba   : default auto (conda if available else mamba)
# - CONDA_ACTION=none|ensure|create|update : default ensure
# - CONDA_ENV_FILE=/path/environment.yml   : default: $REPO_ROOT/environment.yml (if missing => warn/skip unless strict)
# - CONDA_ENV_NAME=NAME              : default: py${PY_VERSION//./}
# - PY_VERSION=3.11                  : used only for default CONDA_ENV_NAME
# - CONDA_USE_FILE_NAME=0|1          : if 1, do NOT pass -n/--name (use name inside env file)
# - CONDA_PRUNE=0|1                  : if 1, add --prune on env update (default 0; can be destructive)
# - CONDA_INIT_BASH=0|1              : run `conda init bash` (default 0; modifies ~/.bashrc)
# - CONDA_ACTIVATE=0|1               : activate env in current shell (default 0; only meaningful when sourced)
# - CONDA_CLEAN=0|1                  : run conda clean --all (default 0)
# - CONDA_RM_USER_CACHE=0|1          : remove ~/.cache (default 0)
#
# INSTALL (STRICT OPT-IN)
# - This script never installs OS packages by itself.
#   If you need installs, do it in all_post_create.sh via install_dev_tools_apt (COMMON_ALLOW_INSTALL=1).
# ===============================================================

# ---------- sourced vs executed ----------
__ENV_CONDA_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  __ENV_CONDA_SOURCED=1
fi

# If executed directly, enable strict bash behavior + helpful ERR trap.
if [[ "$__ENV_CONDA_SOURCED" == "0" ]]; then
  set -Eeuo pipefail
  IFS=$'\n\t'
  _env_conda_on_err() {
    local lineno="$1"
    local cmd="$2"
    printf '%s\n' "[ERROR] env_conda.sh failed at line ${lineno}: ${cmd}" >&2
    exit 1
  }
  trap '_env_conda_on_err "$LINENO" "$BASH_COMMAND"' ERR
fi

# ---------- locate script dir / repo root (deterministic, no realpath) ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

# ---------- optional: source common.sh if present (portable POSIX library) ----------
COMMON_SH="$SCRIPT_DIR/lib/common.sh"
if [[ -f "$COMMON_SH" ]]; then
  # shellcheck disable=SC1090
  . "$COMMON_SH"
fi

# ---------- local logging (never exits) ----------
_ec_log() { printf '%s\n' "$*" >&2; }
_ec_info() { _ec_log "[INFO] $*"; }
_ec_warn() { _ec_log "[WARN] $*"; }
_ec_err() { _ec_log "[ERROR] $*"; }

_env_conda_return_or_exit() {
  local code="${1:-1}"
  if [[ "$__ENV_CONDA_SOURCED" == "1" ]]; then
    return "$code"
  fi
  exit "$code"
}

_env_conda_fail() {
  _ec_err "$*"
  _env_conda_return_or_exit 1
}

# ---------- truthy parsing (bash) ----------
_is_truthy() {
  local v="${1:-0}"
  case "${v,,}" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off|"") return 1 ;;
    *) return 1 ;;
  esac
}

# ---------- defaults (explicit + canonical) ----------
CONDA_SKIP="${CONDA_SKIP:-0}"
CONDA_STRICT="${CONDA_STRICT:-${POST_CREATE_STRICT:-0}}"
CONDA_MANAGER="${CONDA_MANAGER:-auto}"
CONDA_ACTION="${CONDA_ACTION:-ensure}"
PY_VERSION="${PY_VERSION:-3.11}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-py${PY_VERSION//./}}"
CONDA_ENV_FILE="${CONDA_ENV_FILE:-$REPO_ROOT/environment.yml}"
CONDA_USE_FILE_NAME="${CONDA_USE_FILE_NAME:-0}"
CONDA_PRUNE="${CONDA_PRUNE:-0}"
CONDA_INIT_BASH="${CONDA_INIT_BASH:-0}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-0}"
CONDA_CLEAN="${CONDA_CLEAN:-0}"
CONDA_RM_USER_CACHE="${CONDA_RM_USER_CACHE:-0}"

if _is_truthy "$CONDA_SKIP"; then
  _ec_info "CONDA_SKIP=1 -> skipping conda environment setup"
  _env_conda_return_or_exit 0
fi

# If using file-defined env name, "ensure" cannot be implemented deterministically without parsing/guessing.
if _is_truthy "$CONDA_USE_FILE_NAME" && [[ "$CONDA_ACTION" == "ensure" ]]; then
  _env_conda_fail "CONDA_USE_FILE_NAME=1 is incompatible with CONDA_ACTION=ensure. Use CONDA_ACTION=create|update, or set CONDA_USE_FILE_NAME=0 and provide CONDA_ENV_NAME."
fi

# ---------- validate action ----------
case "$CONDA_ACTION" in
  none)
    _ec_info "CONDA_ACTION=none -> no action"
    _env_conda_return_or_exit 0
    ;;
  ensure|create|update) ;;
  *)
    _env_conda_fail "Invalid CONDA_ACTION='$CONDA_ACTION' (allowed: none|ensure|create|update)"
    ;;
esac

# ---------- select manager deterministically ----------
_has_cmd() { command -v "$1" >/dev/null 2>&1; }

_selected_mgr=""
case "$CONDA_MANAGER" in
  conda)
    _has_cmd conda || _env_conda_fail "CONDA_MANAGER=conda but 'conda' not found in PATH"
    _selected_mgr="conda"
    ;;
  mamba)
    _has_cmd mamba || _env_conda_fail "CONDA_MANAGER=mamba but 'mamba' not found in PATH"
    _selected_mgr="mamba"
    ;;
  auto)
    if _has_cmd conda; then
      _selected_mgr="conda"
    elif _has_cmd mamba; then
      _selected_mgr="mamba"
    else
      if _is_truthy "$CONDA_STRICT"; then
        _env_conda_fail "No conda-compatible manager found (need 'conda' or 'mamba')"
      else
        _ec_warn "No conda-compatible manager found; skipping (set CONDA_STRICT=1 to fail)"
        _env_conda_return_or_exit 0
      fi
    fi
    ;;
  *)
    _env_conda_fail "Invalid CONDA_MANAGER='$CONDA_MANAGER' (allowed: auto|conda|mamba)"
    ;;
esac

_ec_info "Manager: ${_selected_mgr}"
_ec_info "Repo root: ${REPO_ROOT}"

# ---------- env file check (deterministic) ----------
if [[ ! -f "$CONDA_ENV_FILE" ]]; then
  if _is_truthy "$CONDA_STRICT"; then
    _env_conda_fail "Environment file not found: $CONDA_ENV_FILE"
  else
    _ec_warn "Environment file not found: $CONDA_ENV_FILE (skipping; set CONDA_STRICT=1 to fail)"
    _env_conda_return_or_exit 0
  fi
fi

# ---------- list envs + check existence (exact match) ----------
_env_exists() {
  local mgr="$1"
  local name="$2"

  # conda/mamba output has comments and headers; filter those out and compare exact name.
  "$mgr" env list 2>/dev/null \
    | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -Fx -- "$name" >/dev/null 2>&1
}

# ---------- build create/update args ----------
# Non-interactive: always use --yes (canonical for CI/devcontainers)
_create_args=(env create -f "$CONDA_ENV_FILE" --yes)
_update_args=(env update -f "$CONDA_ENV_FILE" --yes)

if ! _is_truthy "$CONDA_USE_FILE_NAME"; then
  # Override env name deterministically
  _create_args+=(-n "$CONDA_ENV_NAME")
  _update_args+=(-n "$CONDA_ENV_NAME")
fi

if _is_truthy "$CONDA_PRUNE"; then
  # Can be destructive; explicit opt-in only
  _update_args+=(--prune)
fi

# ---------- action execution ----------
_exists=0
if ! _is_truthy "$CONDA_USE_FILE_NAME"; then
  if _env_exists "$_selected_mgr" "$CONDA_ENV_NAME"; then
    _exists=1
  fi
fi

case "$CONDA_ACTION" in
  create)
    if [[ "$_exists" == "1" ]]; then
      _ec_info "Environment already exists: $CONDA_ENV_NAME (CONDA_ACTION=create -> no-op)"
      _env_conda_return_or_exit 0
    fi
    _ec_info "Creating environment from: $CONDA_ENV_FILE"
    _ec_info "Command: $_selected_mgr ${_create_args[*]}"
    "$_selected_mgr" "${_create_args[@]}" || _env_conda_fail "Environment create failed"
    ;;
  update)
    if [[ "$_exists" == "0" ]] && ! _is_truthy "$CONDA_USE_FILE_NAME"; then
      _env_conda_fail "Environment not found: $CONDA_ENV_NAME (CONDA_ACTION=update). Use CONDA_ACTION=ensure or create."
    fi
    _ec_info "Updating environment from: $CONDA_ENV_FILE"
    _ec_info "Command: $_selected_mgr ${_update_args[*]}"
    "$_selected_mgr" "${_update_args[@]}" || _env_conda_fail "Environment update failed"
    ;;
  ensure)
    if [[ "$_exists" == "1" ]]; then
      _ec_info "Environment exists: $CONDA_ENV_NAME -> update"
      _ec_info "Command: $_selected_mgr ${_update_args[*]}"
      "$_selected_mgr" "${_update_args[@]}" || _env_conda_fail "Environment update failed"
    else
      _ec_info "Environment missing -> create"
      _ec_info "Command: $_selected_mgr ${_create_args[*]}"
      "$_selected_mgr" "${_create_args[@]}" || _env_conda_fail "Environment create failed"
    fi
    ;;
esac

# ---------- optional: conda init (modifies ~/.bashrc) ----------
if _is_truthy "$CONDA_INIT_BASH"; then
  _has_cmd conda || _env_conda_fail "CONDA_INIT_BASH=1 requires 'conda' command"
  _ec_warn "CONDA_INIT_BASH=1 -> running 'conda init bash' (modifies ~/.bashrc)"
  conda init bash || _env_conda_fail "conda init bash failed"
fi

# ---------- optional: activation (only meaningful when sourced) ----------
if _is_truthy "$CONDA_ACTIVATE"; then
  if [[ "$__ENV_CONDA_SOURCED" != "1" ]]; then
    _ec_warn "CONDA_ACTIVATE=1 has no persistent effect when executed. Source the script to activate in current shell."
  fi

  _has_cmd conda || _env_conda_fail "CONDA_ACTIVATE=1 requires 'conda' command"
  base="$(conda info --base 2>/dev/null)" || _env_conda_fail "conda info --base failed"
  conda_sh="${base}/etc/profile.d/conda.sh"
  [[ -f "$conda_sh" ]] || _env_conda_fail "Missing conda shell hook: $conda_sh"

  # shellcheck disable=SC1090
  . "$conda_sh"

  ## Also Configure base
  conda install -n base python="$PY_VERSION" ipykernel pip -y || true

  if _is_truthy "$CONDA_USE_FILE_NAME"; then
    _ec_warn "CONDA_USE_FILE_NAME=1 -> activation target is file-defined; set CONDA_ENV_NAME to activate explicitly."
  else
    conda activate "$CONDA_ENV_NAME" || _env_conda_fail "conda activate failed: $CONDA_ENV_NAME"
    _ec_info "Activated: $CONDA_ENV_NAME"
  fi
fi

# ---------- optional: cleanup ----------
if _is_truthy "$CONDA_CLEAN"; then
  if _has_cmd conda; then
    _ec_info "Running: conda clean --all -f -y"
    conda clean --all -f -y || _ec_warn "conda clean failed (ignored)"
  else
    _ec_warn "CONDA_CLEAN=1 but conda not found; skipping clean"
  fi
fi

if _is_truthy "$CONDA_RM_USER_CACHE"; then
  _ec_warn "Removing user cache: ${HOME}/.cache"
  rm -rf -- "${HOME}/.cache" || _ec_warn "Failed to remove ~/.cache (ignored)"
fi

_ec_info "env_conda: done"
_env_conda_return_or_exit 0
