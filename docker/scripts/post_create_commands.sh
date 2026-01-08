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

__pc_is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  __pc_is_sourced=1
fi

# Strict only when executed directly (avoid mutating parent when sourced)
if [[ "$__pc_is_sourced" == "0" ]]; then
  set -Eeuo pipefail
  IFS=$'\n\t'
fi

# Resolve paths deterministically (no realpath dependency)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

# Source common.sh if present (idempotent)
COMMON_SH="$SCRIPT_DIR/lib/common.sh"
if [[ -f "$COMMON_SH" ]]; then
  # shellcheck disable=SC1090
  . "$COMMON_SH"
fi

# ---------- Logging wrappers (never exit when sourced) ----------
_pc_ts() {
  if command -v _common_ts >/dev/null 2>&1; then
    _common_ts
  else
    date -u +"%Y-%m-%dT%H:%M:%SZ"
  fi
}

_pc_log() { printf '%s\n' "$*" >&2; }

_pc_info()    { local ts; ts="$(_pc_ts)"; _pc_log "${ts} ${BOLD:-}${BLUE:-}[INFO]${RESET:-} $*"; }
_pc_warn()    { local ts; ts="$(_pc_ts)"; _pc_log "${ts} ${BOLD:-}${YELLOW:-}[WARNING]${RESET:-} $*"; }
_pc_success() { local ts; ts="$(_pc_ts)"; _pc_log "${ts} ${BOLD:-}${GREEN:-}[SUCCESS]${RESET:-} $*"; }
_pc_error()   { local ts; ts="$(_pc_ts)"; _pc_log "${ts} ${BOLD:-}${RED:-}[ERROR]${RESET:-} $*"; }

_pc_die() {
  local code="${1:-1}"; shift || true
  _pc_error "$*"
  if [[ "$__pc_is_sourced" == "1" ]]; then
    return "$code"
  fi
  exit "$code"
}

# If executed directly, add a canonical ERR trap
if [[ "$__pc_is_sourced" == "0" ]]; then
  trap '_pc_die 1 "Failure at line ${LINENO}: ${BASH_COMMAND}"' ERR
fi

# ---------- Strict controls ----------
POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"

_pc_optional_fail() {
  # Usage: _pc_optional_fail <code> <message...>
  local code="$1"; shift || true
  if [[ "$POST_CREATE_STRICT" == "1" ]]; then
    _pc_die "$code" "$*"
  fi
  _pc_warn "$*"
  return 0
}

_pc_has_cmd() { command -v "$1" >/dev/null 2>&1; }

_pc_abspath_existing() {
  local p="$1"
  [[ -n "$p" ]] || _pc_die 2 "abspath_existing: missing path"
  [[ -e "$p" ]] || _pc_die 2 "Path does not exist: $p"
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

  if ! _pc_has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping safe.directory config"
    return 0
  fi

  _pc_info "Configuring git safe.directory entries..."

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
        _pc_warn "Failed to add safe.directory: $abs"
        failed=1
      fi
    else
      _pc_warn "Directory missing (skip): $d"
    fi
  done

  if [[ "$failed" == "1" && "$allow_all" == "1" ]]; then
    _pc_warn "Some safe.directory entries failed; adding wildcard '*' (GIT_SAFE_DIR_ALLOW_ALL=1)"
    git config --global --add safe.directory '*' >/dev/null 2>&1 || _pc_optional_fail 0 "Failed to add safe.directory '*'"
  fi

  _pc_success "git safe.directory configuration complete"
}

pc_git_submodules_init() {
  if ! _pc_has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping submodules"
    return 0
  fi

  if [[ ! -f "$REPO_ROOT/.gitmodules" ]]; then
    _pc_info "No .gitmodules found; skipping submodule init"
    return 0
  fi

  _pc_info "Initializing/updating git submodules..."
  (cd -- "$REPO_ROOT" && git submodule update --init --recursive) || _pc_optional_fail 0 "Submodule init failed"
  _pc_success "Submodule setup complete"
}

pc_git_config_upstream() {
  if ! _pc_has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping upstream remote config"
    return 0
  fi

  local upstream_url="${POST_CREATE_UPSTREAM_URL:-https://github.com/scikit-plots/scikit-plots.git}"

  _pc_info "Configuring upstream remote (if needed)..."
  (cd -- "$REPO_ROOT" && git remote get-url upstream >/dev/null 2>&1) || {
    (cd -- "$REPO_ROOT" && git remote add upstream "$upstream_url") || _pc_optional_fail 0 "Failed to add upstream remote"
  }

  _pc_info "Fetching upstream tags..."
  (cd -- "$REPO_ROOT" && git fetch upstream --tags) || _pc_optional_fail 0 "Failed to fetch upstream tags"

  _pc_success "Git remote configuration complete"
}

# ===============================================================
# Environment runner (micromamba run / conda run)
# ===============================================================

_pc_env_kind=""  # micromamba|conda
pc_select_env_runner() {
  local tool="${POST_CREATE_ENV_TOOL:-auto}"

  case "$tool" in
    micromamba|conda|auto) ;;
    *) _pc_die 2 "Invalid POST_CREATE_ENV_TOOL: $tool (expected auto|micromamba|conda)" ;;
  esac

  if [[ "$tool" == "micromamba" ]]; then
    _pc_has_cmd micromamba || _pc_die 1 "micromamba required but not found"
    _pc_env_kind="micromamba"
    return 0
  fi

  if [[ "$tool" == "conda" ]]; then
    _pc_has_cmd conda || _pc_die 1 "conda required but not found"
    _pc_env_kind="conda"
    return 0
  fi

  # auto (deterministic priority: micromamba then conda)
  if _pc_has_cmd micromamba; then
    _pc_env_kind="micromamba"
    return 0
  fi
  if _pc_has_cmd conda; then
    _pc_env_kind="conda"
    return 0
  fi

  _pc_env_kind=""
  return 0
}

pc_env_run() {
  local env_name="$1"; shift || true
  [[ -n "$env_name" ]] || _pc_die 2 "pc_env_run: env name is empty"
  [[ -n "$_pc_env_kind" ]] || _pc_die 2 "pc_env_run: env runner not selected"

  case "$_pc_env_kind" in
    micromamba) micromamba run -n "$env_name" -- "$@" ;;
    conda) conda run --no-capture-output -n "$env_name" -- "$@" ;;
    *) _pc_die 2 "Unknown env runner kind: $_pc_env_kind" ;;
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

  _pc_info "Installing requirements: $file_path"
  pc_env_run "$env_name" python -m pip install --no-input -r "$file_path" || _pc_optional_fail 1 "pip requirements install failed"
  _pc_success "Requirements installed"
}

pc_install_scikit_plots() {
  local env_name="$1"

  if [[ -n "${SCIKITPLOT_VERSION:-}" ]]; then
    _pc_info "Installing scikit-plots from PyPI: scikit-plots==${SCIKITPLOT_VERSION}"
    pc_env_run "$env_name" python -m pip install --no-input "scikit-plots==${SCIKITPLOT_VERSION}" \
      || _pc_optional_fail 1 "Failed to install scikit-plots==${SCIKITPLOT_VERSION}"
    return 0
  fi

  local extras="${POST_CREATE_LOCAL_EXTRAS:-build,dev,test,doc}"
  local allow_fallback="${POST_CREATE_ALLOW_FALLBACK:-0}"
  local install_extras="${POST_CREATE_INSTALL_EXTRAS:-1}"

  _pc_info "Installing local scikit-plots (editable) from repo: $REPO_ROOT"
  if [[ "$install_extras" == "1" ]]; then
    if ! (cd -- "$REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e ".[${extras}]" -v); then
      if [[ "$allow_fallback" == "1" ]]; then
        _pc_warn "Editable install with extras failed; falling back to minimal editable install (POST_CREATE_ALLOW_FALLBACK=1)"
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

  _pc_success "scikit-plots installation step complete"
}

pc_install_precommit() {
  local env_name="$1"

  if ! _pc_has_cmd git; then
    _pc_optional_fail 0 "git not found; skipping pre-commit install"
    return 0
  fi

  _pc_info "Installing pre-commit..."
  pc_env_run "$env_name" python -m pip install --no-input pre-commit || _pc_optional_fail 1 "Failed to install pre-commit"

  _pc_info "Installing pre-commit hooks..."
  (cd -- "$REPO_ROOT" && pc_env_run "$env_name" pre-commit install) || _pc_optional_fail 0 "Failed to install pre-commit hooks"

  _pc_success "pre-commit setup complete"
}

pc_show_env_info() {
  local env_name="$1"
  _pc_info "Environment runner: ${_pc_env_kind:-none} env=${env_name}"

  if [[ -n "$_pc_env_kind" ]]; then
    pc_env_run "$env_name" python -c 'import sys; print("python:", sys.version)' || true
    pc_env_run "$env_name" python -m pip --version || true
    pc_env_run "$env_name" python -m pip show scikit-plots 2>/dev/null || true
  fi
}

pc_print_next_steps() {
  _pc_info "Next steps:"
  _pc_info " - Create a branch (see contribution guide):"
  _pc_info "   https://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch"
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
  cd -- "$REPO_ROOT" || _pc_die 1 "Failed to cd to repo root: $REPO_ROOT"

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
      _pc_die 1 "No environment tool found (micromamba/conda)."
    fi
    _pc_warn "No environment tool found (micromamba/conda); skipping pip/package steps"
    cd -- "$old_pwd" || true
    return 0
  fi

  _pc_info "Using env tool: $_pc_env_kind (env: $env_name)"

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
  _pc_success "post_create_commands: complete"
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
