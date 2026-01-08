#!/usr/bin/env bash
#
# docker/scripts/env_conda.sh
#
# bash docker/scripts/env_conda.sh
# CONDA_STRICT=1 bash docker/scripts/env_conda.sh
# CONDA_MANAGER=conda bash docker/scripts/env_conda.sh
# CONDA_MANAGER=mamba bash docker/scripts/env_conda.sh
# PY_VERSION=3.11 ENV_NAME=py311 bash docker/scripts/env_conda.sh
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
# - PY_VERSION=3.11                  : used only for default ENV_NAME
# - ENV_NAME=NAME                    : default: py${PY_VERSION//./}
# - ENV_FILE=/path/environment.yml   : default: $REPO_ROOT/environment.yml (if missing => warn/skip unless strict)
# - CONDA_SKIP=0|1                   : skip entirely (default 0)
# - CONDA_STRICT=0|1                 : fail hard on missing env file/manager (default inherits POST_CREATE_STRICT else 0)
# - CONDA_MANAGER=auto|conda|mamba   : default auto (conda if available else mamba)
# - CONDA_ACTION=none|ensure|create|update : default ensure
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

set -Eeuo pipefail

is_sourced() {
  # Deterministic bash check
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

exit_or_return() {
  local code="${1:-0}"
  if is_sourced; then
    return "$code"
  else
    exit "$code"
  fi
}

# ---------- truthy parsing (bash) ----------
## Normalize to lowercase and handle multiple truthy values
## value=$(echo "$SKIP_CONDA" | tr '[:upper:]' '[:lower:]')
## case "$(printf '%s' $SKIP_CONDA | tr '[:upper:]' '[:lower:]')" in
is_true() {
  # Usage: is_true "$VAL"
  # Returns 0 for: 1,true,yes,on  (case-insensitive)
  local v="${1:-}"
  v="${v,,}"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off|"") return 1 ;;
    *) return 1 ;;
  esac
}

# ---------- Error reporting ----------
_on_err() {
  local lineno="$1"
  local cmd="$2"
  printf '%s\n' "[ERROR] env_conda.sh failed at line ${lineno}: ${cmd}" >&2
  exit_or_return 1
}
# trap 'rc=$?; echo "[ERROR] env_conda.sh failed at line $LINENO: $BASH_COMMAND (exit=$rc)" >&2; exit $rc' ERR
trap '_on_err "$LINENO" "$BASH_COMMAND"' ERR

# ---------- locate script dir / repo root (deterministic, no realpath) ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
# Source the POSIX common library (works in bash). Re-apply bash strict after.
COMMON_SH="${COMMON_SH:-$REPO_ROOT/docker/scripts/lib/common.sh}"
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
  log_success() { printf '%s\n' "[SUCCESS] $*" >&2; }
  log_debug()   { :; }
  has_cmd() { command -v "$1" >/dev/null 2>&1; }
fi

# ---------- defaults (explicit + canonical) ----------
PY_VERSION="${PY_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"

CONDA_SKIP="${CONDA_SKIP:-0}"
CONDA_STRICT="${CONDA_STRICT:-${POST_CREATE_STRICT:-0}}"
CONDA_MANAGER="${CONDA_MANAGER:-auto}"
CONDA_ACTION="${CONDA_ACTION:-ensure}"
CONDA_USE_FILE_NAME="${CONDA_USE_FILE_NAME:-0}"
CONDA_PRUNE="${CONDA_PRUNE:-0}"
CONDA_INIT_BASH="${CONDA_INIT_BASH:-0}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-0}"
CONDA_CLEAN="${CONDA_CLEAN:-0}"
CONDA_RM_USER_CACHE="${CONDA_RM_USER_CACHE:-0}"

if is_true "$CONDA_SKIP"; then
  log_info "CONDA_SKIP=1 -> skipping conda environment setup"
  exit_or_return 0
fi

# If using file-defined env name, "ensure" cannot be implemented deterministically without parsing/guessing.
if is_true "$CONDA_USE_FILE_NAME" && [[ "$CONDA_ACTION" == "ensure" ]]; then
  log_error "CONDA_USE_FILE_NAME=1 is incompatible with CONDA_ACTION=ensure. Use CONDA_ACTION=create|update, or set CONDA_USE_FILE_NAME=0 and provide ENV_NAME."
fi

# ---------- validate action ----------
case "$CONDA_ACTION" in
  none)
    log_info "CONDA_ACTION=none -> no action"
    ;;
  ensure|create|update) ;;
  *)
    log_error "Invalid CONDA_ACTION='$CONDA_ACTION' (allowed: none|ensure|create|update)"
    ;;
esac

# ---------- select manager deterministically ----------
_selected_mgr=""
case "$CONDA_MANAGER" in
  conda)
    has_cmd conda || log_error "CONDA_MANAGER=conda but 'conda' not found in PATH"
    _selected_mgr="conda"
    ;;
  mamba)
    has_cmd mamba || log_error "CONDA_MANAGER=mamba but 'mamba' not found in PATH"
    _selected_mgr="mamba"
    ;;
  auto)
    if has_cmd conda; then
      _selected_mgr="conda"
    elif has_cmd mamba; then
      _selected_mgr="mamba"
    else
      if is_true "$CONDA_STRICT"; then
        log_error "No conda-compatible manager found (need 'conda' or 'mamba')"
      else
        log_warning "No conda-compatible manager found; skipping (set CONDA_STRICT=1 to fail)"
        exit_or_return 0
      fi
    fi
    ;;
  *)
    log_error "Invalid CONDA_MANAGER='$CONDA_MANAGER' (allowed: auto|conda|mamba)"
    ;;
esac

log_info "Manager: ${_selected_mgr}"
log_info "Repo root: ${REPO_ROOT}"

# ---------- env file check (deterministic) ----------
if [[ ! -f "$ENV_FILE" ]]; then
  if is_true "$CONDA_STRICT"; then
    log_error "Environment file not found: $ENV_FILE"
  else
    log_warning "Environment file not found: $ENV_FILE (skipping; set CONDA_STRICT=1 to fail)"
    exit_or_return 0
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
_create_args=(env create -f "$ENV_FILE" --yes)
_update_args=(env update -f "$ENV_FILE" --yes)

if ! is_true "$CONDA_USE_FILE_NAME"; then
  # Override env name deterministically
  _create_args+=(-n "$ENV_NAME")
  _update_args+=(-n "$ENV_NAME")
fi

if is_true "$CONDA_PRUNE"; then
  # Can be destructive; explicit opt-in only
  _update_args+=(--prune)
fi

# ---------- action execution ----------
_exists=0
if ! is_true "$CONDA_USE_FILE_NAME"; then
  if _env_exists "$_selected_mgr" "$ENV_NAME"; then
    _exists=1
  fi
fi

case "$CONDA_ACTION" in
  create)
    if [[ "$_exists" == "1" ]]; then
      log_info "Environment already exists: $ENV_NAME (CONDA_ACTION=create -> no-op)"
      exit_or_return 0
    fi
    log_info "Creating environment from: $ENV_FILE"
    log_info "Command: $_selected_mgr ${_create_args[*]}"
    "$_selected_mgr" "${_create_args[@]}" || log_error "Environment create failed"
    ;;
  update)
    if [[ "$_exists" == "0" ]] && ! is_true "$CONDA_USE_FILE_NAME"; then
      log_error "Environment not found: $ENV_NAME (CONDA_ACTION=update). Use CONDA_ACTION=ensure or create."
    fi
    log_info "Updating environment from: $ENV_FILE"
    log_info "Command: $_selected_mgr ${_update_args[*]}"
    "$_selected_mgr" "${_update_args[@]}" || log_error "Environment update failed"
    ;;
  ensure)
    if [[ "$_exists" == "1" ]]; then
      log_info "Environment exists: $ENV_NAME -> update"
      log_info "Command: $_selected_mgr ${_update_args[*]}"
      "$_selected_mgr" "${_update_args[@]}" || log_error "Environment update failed"
    else
      log_info "Environment missing -> create"
      log_info "Command: $_selected_mgr ${_create_args[*]}"
      "$_selected_mgr" "${_create_args[@]}" || log_error "Environment create failed"
    fi
    ;;
esac

# ---------- optional: conda init (modifies ~/.bashrc) ----------
if is_true "$CONDA_INIT_BASH"; then
  has_cmd conda || log_error "CONDA_INIT_BASH=1 requires 'conda' command"
  log_warning "CONDA_INIT_BASH=1 -> running 'conda init bash' (modifies ~/.bashrc)"
  conda init bash || log_error "conda init bash failed"
fi

# ---------- optional: activation (only meaningful when sourced) ----------
if is_true "$CONDA_ACTIVATE"; then
  if [[ "$__ENV_CONDA_SOURCED" != "1" ]]; then
    log_warning "CONDA_ACTIVATE=1 has no persistent effect when executed. Source the script to activate in current shell."
  fi

  has_cmd conda || log_error "CONDA_ACTIVATE=1 requires 'conda' command"
  base="$(conda info --base 2>/dev/null)" || log_error "conda info --base failed"
  conda_sh="${base}/etc/profile.d/conda.sh"
  [[ -f "$conda_sh" ]] || log_error "Missing conda shell hook: $conda_sh"

  # shellcheck disable=SC1090
  . "$conda_sh"

  ## Also Configure base
  conda install -n base python="$PY_VERSION" ipykernel pip -y || true

  if is_true "$CONDA_USE_FILE_NAME"; then
    log_warning "CONDA_USE_FILE_NAME=1 -> activation target is file-defined; set ENV_NAME to activate explicitly."
  else
    conda activate "$ENV_NAME" || log_error "conda activate failed: $ENV_NAME"
    log_info "Activated: $ENV_NAME"
  fi
fi

# ---------- optional: cleanup ----------
if is_true "$CONDA_CLEAN"; then
  if has_cmd conda; then
    log_info "Running: conda clean --all -f -y"
    conda clean --all -f -y || log_warning "conda clean failed (ignored)"
  else
    log_warning "CONDA_CLEAN=1 but conda not found; skipping clean"
  fi
fi

if is_true "$CONDA_RM_USER_CACHE"; then
  log_warning "Removing user cache: ${HOME}/.cache"
  rm -rf -- "${HOME}/.cache" || log_warning "Failed to remove ~/.cache (ignored)"
fi

log_info "env_conda: done"
