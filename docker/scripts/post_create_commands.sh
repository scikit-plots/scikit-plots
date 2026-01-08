#!/usr/bin/env bash
# docker/scripts/post_create_commands.sh
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# ===============================================================
# Post-create commands (Bash)
# ===============================================================
# USER NOTES
# - Intended for Dev Containers / Docker postCreate hooks.
# - This script may be *sourced* by an orchestrator (e.g., all_post_create.sh).
#   When sourced, it MUST NOT call `exit`. It returns non-zero on failure.
#
# STRICTNESS & HEAVY OPS
# - No OS package installs here. (Use all_post_create.sh + COMMON_ALLOW_INSTALL gate.)
# - No interactive shells. Prefer `micromamba run` / `conda run` (deterministic).
#
# ENV VARS (explicit controls)
# - POST_CREATE_STRICT=0|1                : if 1, optional failures become fatal (return non-zero)
# - POST_CREATE_GIT_SAFE_DIR=0|1          : default 1
# - POST_CREATE_GIT_SUBMODULES=0|1        : default 1
# - POST_CREATE_GIT_UPSTREAM=0|1          : default 1
# - POST_CREATE_UPSTREAM_URL=...          : default https://github.com/scikit-plots/scikit-plots.git
# - GIT_SAFE_DIR_ALLOW_ALL=0|1            : default 0 (if 1 and safe-dir add fails, adds '*')
# - POST_CREATE_ENV_TOOL=auto|micromamba|conda : default auto
# - POST_CREATE_ENV_NAME=...              : default from ENV_NAME/ENV_NAME else "py311"
# - POST_CREATE_ENV_REQUIRED=0|1          : default 0 (if 1, fail when no env tool found)
#
# Python/pip steps (explicit toggles)
# - POST_CREATE_PIP_REQUIREMENTS=0|1      : default 1
# - POST_CREATE_REQUIREMENTS_FILE=PATH    : default ./requirements/build.txt
# - POST_CREATE_INSTALL_PACKAGE=0|1       : default 1
# - POST_CREATE_INSTALL_EXTRAS=0|1        : default 1
# - POST_CREATE_LOCAL_EXTRAS=...          : default "build,dev,test,doc"
# - POST_CREATE_ALLOW_FALLBACK=0|1        : default 0 (if 1, fallback to minimal editable install)
# - SCIKITPLOT_VERSION=...                : if set, install that exact version from PyPI
# - POST_CREATE_INSTALL_PRECOMMIT=0|1     : default 1
# - POST_CREATE_SHOW_ENV_INFO=0|1         : default 0
# - POST_CREATE_PRINT_NEXT_STEPS=0|1      : default 1
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
  printf '%s\n' "[ERROR] post_create_commands.sh failed at line ${lineno}: ${cmd}" >&2
  exit_or_return 1
}
# trap 'rc=$?; echo "[ERROR] post_create_commands.sh failed at line $LINENO: $BASH_COMMAND (exit=$rc)" >&2; exit $rc' ERR
trap '_on_err "$LINENO" "$BASH_COMMAND"' ERR

# Resolve paths deterministically (no realpath dependency)
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

POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"

# ---------- Helpers ----------
maybe_fail() {
  # Usage: maybe_fail "message"
  if [[ "$POST_CREATE_STRICT" == "1" ]]; then
    log_error_code 1 "$1"
  else
    log_warning "$1"
  fi
}
_pc_optional_fail() {
  # Usage: _pc_optional_fail <code> <message...>
  local code="$1"; shift || true
  if [[ "$POST_CREATE_STRICT" == "1" ]]; then
    maybe_fail "$code" "$*"
  fi
  log_warning "$*"
  return 0
}

_pc_abspath_existing() {
  local p="$1"
  [[ -n "$p" ]] || maybe_fail "abspath_existing: missing path"
  [[ -e "$p" ]] || maybe_fail "Path does not exist: $p"
  if [[ -d "$p" ]]; then
    (cd -- "$p" && pwd -P)
  else
    local dir base
    dir="${p%/*}"
    base="${p##*/}"
    [[ "$dir" == "$p" ]] && dir="."
    (cd -- "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
  fi
}

# ===============================================================
# Git steps
# ===============================================================

######################################################################
## Git Safe Directories Configuration
# export GIT_CONFIG_GLOBAL=~/.gitconfig
# git config --global --list --show-origin
# git config --global --unset-all safe.directory
# git config --global --get-all safe.directory
######################################################################
pc_git_safe_directories() {
  local allow_all="${GIT_SAFE_DIR_ALLOW_ALL:-0}"
  local failed=0

  if ! has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping safe.directory config"
    return 0
  fi

  log_info "Configuring git safe.directory entries..."

  local -a dirs
  dirs=(
    "$REPO_ROOT"
    "$REPO_ROOT/third_party/array-api-compat"
    "$REPO_ROOT/third_party/array-api-extra"
    "$REPO_ROOT/third_party/astropy"
    "$REPO_ROOT/third_party/seaborn"
  )

  local d abs
  for d in "${dirs[@]}"; do
    if [[ -d "$d" ]]; then
      abs="$(_pc_abspath_existing "$d")"
      if ! git config --global --add safe.directory "$abs" >/dev/null 2>&1; then
        log_warning "Failed to add safe.directory: $abs"
        failed=1
      fi
    else
      log_warning "Directory missing (skip): $d"
    fi
  done

  if [[ "$failed" == "1" && "$allow_all" == "1" ]]; then
    log_warning "Some safe.directory entries failed; adding wildcard '*' (GIT_SAFE_DIR_ALLOW_ALL=1)"
    git config --global --add safe.directory '*' >/dev/null 2>&1 || _pc_optional_fail 0 "Failed to add safe.directory '*'"
  fi

  log_success "git safe.directory configuration complete"
}

pc_git_submodules_init() {
  if ! has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping submodules"
    return 0
  fi

  if [[ ! -f "$REPO_ROOT/.gitmodules" ]]; then
    log_info "No .gitmodules found; skipping submodule init"
    return 0
  fi

  log_info "Initializing/updating git submodules..."
  (cd -- "$REPO_ROOT" && git submodule update --init --recursive) || _pc_optional_fail 0 "Submodule init failed"
  log_success "Submodule setup complete"
}

pc_git_config_upstream() {
  if ! has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping upstream remote config"
    return 0
  fi

  local upstream_url="${POST_CREATE_UPSTREAM_URL:-https://github.com/scikit-plots/scikit-plots.git}"

  log_info "Configuring upstream remote (if needed)..."
  (cd -- "$REPO_ROOT" && git remote get-url upstream >/dev/null 2>&1) || {
    (cd -- "$REPO_ROOT" && git remote add upstream "$upstream_url") || _pc_optional_fail 0 "Failed to add upstream remote"
  }

  log_info "Fetching upstream tags..."
  (cd -- "$REPO_ROOT" && git fetch upstream --tags) || _pc_optional_fail 0 "Failed to fetch upstream tags"

  log_success "Git remote configuration complete"
}

# ===============================================================
# Environment runner (micromamba run / conda run)
# ===============================================================

_pc_env_kind=""  # micromamba|conda
pc_select_env_runner() {
  local tool="${POST_CREATE_ENV_TOOL:-auto}"

  case "$tool" in
    micromamba|conda|auto) ;;
    *) maybe_fail "Invalid POST_CREATE_ENV_TOOL: $tool (expected auto|micromamba|conda)" ;;
  esac

  if [[ "$tool" == "micromamba" ]]; then
    has_cmd micromamba || maybe_fail "micromamba required but not found"
    _pc_env_kind="micromamba"
    return 0
  fi

  if [[ "$tool" == "conda" ]]; then
    has_cmd conda || maybe_fail "conda required but not found"
    _pc_env_kind="conda"
    return 0
  fi

  # auto (deterministic priority: micromamba then conda)
  if has_cmd micromamba; then
    _pc_env_kind="micromamba"
    return 0
  fi
  if has_cmd conda; then
    _pc_env_kind="conda"
    return 0
  fi

  _pc_env_kind=""
  return 0
}

pc_env_run() {
  local env_name="$1"; shift || true
  [[ -n "$env_name" ]] || maybe_fail "pc_env_run: env name is empty"
  [[ -n "$_pc_env_kind" ]] || maybe_fail "pc_env_run: env runner not selected"

  case "$_pc_env_kind" in
    micromamba) micromamba run -n "$env_name" -- "$@" ;;
    conda) conda run --no-capture-output -n "$env_name" -- "$@" ;;
    *) maybe_fail "Unknown env runner kind: $_pc_env_kind" ;;
  esac
}

# ===============================================================
# Python / pip steps (explicit)
# ===============================================================

pc_pip_install_requirements() {
  local env_name="$1"
  local req_file="${POST_CREATE_REQUIREMENTS_FILE:-./requirements/build.txt}"

  local file_path="$req_file"
  if [[ -f "$REPO_ROOT/$req_file" ]]; then
    file_path="$REPO_ROOT/$req_file"
  fi

  if [[ ! -f "$file_path" ]]; then
    _pc_optional_fail 0 "Requirements file not found: $req_file"
    return 0
  fi

  log_info "Installing requirements: $file_path"
  pc_env_run "$env_name" python -m pip install --no-input -r "$file_path" || _pc_optional_fail 1 "pip requirements install failed"
  log_success "Requirements installed"
}

pc_install_scikit_plots() {
  local env_name="$1"

  if [[ -n "${SCIKITPLOT_VERSION:-}" ]]; then
    log_info "Installing scikit-plots from PyPI: scikit-plots==${SCIKITPLOT_VERSION}"
    pc_env_run "$env_name" python -m pip install --no-input "scikit-plots==${SCIKITPLOT_VERSION}" \
      || _pc_optional_fail 1 "Failed to install scikit-plots==${SCIKITPLOT_VERSION}"
    return 0
  fi

  local extras="${POST_CREATE_LOCAL_EXTRAS:-build,dev,test,doc}"
  local allow_fallback="${POST_CREATE_ALLOW_FALLBACK:-0}"
  local install_extras="${POST_CREATE_INSTALL_EXTRAS:-1}"

  log_info "Installing local scikit-plots (editable) from repo: $REPO_ROOT"
  if [[ "$install_extras" == "1" ]]; then
    if ! (cd -- "$REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e ".[${extras}]" -v); then
      if [[ "$allow_fallback" == "1" ]]; then
        log_warning "Editable install with extras failed; falling back to minimal editable install (POST_CREATE_ALLOW_FALLBACK=1)"
        (cd -- "$REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e . -v) \
          || _pc_optional_fail 1 "Minimal editable install failed"
      else
        _pc_optional_fail 1 "Editable install with extras failed (set POST_CREATE_ALLOW_FALLBACK=1 to fallback)"
      fi
    fi
  else
    (cd -- "$REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e . -v) \
      || _pc_optional_fail 1 "Minimal editable install failed"
  fi

  log_success "scikit-plots installation step complete"
}

pc_install_precommit() {
  local env_name="$1"

  if ! has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping pre-commit install"
    return 0
  fi

  log_info "Installing pre-commit..."
  pc_env_run "$env_name" python -m pip install --no-input pre-commit || _pc_optional_fail 1 "Failed to install pre-commit"

  log_info "Installing pre-commit hooks..."
  (cd -- "$REPO_ROOT" && pc_env_run "$env_name" pre-commit install) || _pc_optional_fail 0 "Failed to install pre-commit hooks"

  log_success "pre-commit setup complete"
}

pc_show_env_info() {
  local env_name="$1"
  log_info "Environment runner: ${_pc_env_kind:-none} env=${env_name}"

  if [[ -n "$_pc_env_kind" ]]; then
    pc_env_run "$env_name" python -c 'import sys; print("python:", sys.version)' || true
    pc_env_run "$env_name" python -m pip --version || true
    pc_env_run "$env_name" python -m pip show scikit-plots 2>/dev/null || true
  fi
}

pc_print_next_steps() {
  log_info "Next steps:"
  log_info " - Create a branch (see contribution guide):"
  log_info "   https://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch"
}

# ---------- defaults (explicit + canonical) ----------
PY_VERSION="${PY_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"

# ===============================================================
# Main
# ===============================================================

post_create_main() {
  local old_pwd="$PWD"
  cd -- "$REPO_ROOT" || maybe_fail "Failed to cd to repo root: $REPO_ROOT"

  # ----- Git -----
  if [[ "${POST_CREATE_GIT_SAFE_DIR:-1}" == "1" ]]; then
    pc_git_safe_directories || return $?
  fi
  if [[ "${POST_CREATE_GIT_SUBMODULES:-1}" == "1" ]]; then
    pc_git_submodules_init || return $?
  fi
  if [[ "${POST_CREATE_GIT_UPSTREAM:-1}" == "1" ]]; then
    pc_git_config_upstream || return $?
  fi

  # ----- Environment selection -----
  pc_select_env_runner || return $?
  local env_name="${POST_CREATE_ENV_NAME:-${ENV_NAME:-${ENV_NAME:-py311}}}"

  if [[ -z "$_pc_env_kind" ]]; then
    if [[ "${POST_CREATE_ENV_REQUIRED:-0}" == "1" ]]; then
      maybe_fail "No environment tool found (micromamba/conda)."
    fi
    log_warning "No environment tool found (micromamba/conda); skipping pip/package steps"
    cd -- "$old_pwd" || true
    return 0
  fi

  log_info "Using env tool: $_pc_env_kind (env: $env_name)"

  # ----- Python steps -----
  if [[ "${POST_CREATE_PIP_REQUIREMENTS:-1}" == "1" ]]; then
    pc_pip_install_requirements "$env_name" || return $?
  fi

  if [[ "${POST_CREATE_INSTALL_PACKAGE:-1}" == "1" ]]; then
    pc_install_scikit_plots "$env_name" || return $?
  fi

  if [[ "${POST_CREATE_INSTALL_PRECOMMIT:-1}" == "1" ]]; then
    pc_install_precommit "$env_name" || return $?
  fi

  if [[ "${POST_CREATE_SHOW_ENV_INFO:-0}" == "1" ]]; then
    pc_show_env_info "$env_name"
  fi

  if [[ "${POST_CREATE_PRINT_NEXT_STEPS:-1}" == "1" ]]; then
    pc_print_next_steps
  fi

  cd -- "$old_pwd" || true
  log_success "post_create_commands: complete"
  return 0
}

post_create_main

######################################################################
## 📦 starting a new interactive shell, and running some-command inside that new shell, not in the current shell.
# Use bash -i to ensure the script runs in an interactive shell and respects environment changes
# bash -i: starts a new interactive shell (reads .bashrc)
## Double quotes for the outer string and escaping the inner double quotes or use single
## awk '/"/ && !/\\"/ && !/".*"/ { print NR ": " $0 }' .devcontainer/scripts/post_create_commands.sh
######################################################################
## Activate the environment and install required packages in new interactive shell
# bash -i -c ""
