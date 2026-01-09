#!/usr/bin/env bash
# docker/scripts/post_create_commands.sh
#
# Post-create commands (Bash-contract)
#
# Goals
# - Safe when SOURCED: no persistent set/trap changes; no `exit` when sourced.
# - Deterministic behavior via env vars (Dockerfile/devcontainer env wins).
# - Uses the environment tool selected by earlier env steps when available.
# - Supports micromamba, mamba, conda; prefers mamba over conda when configured.
#
# Key env vars (recommended)
# - POST_CREATE_STRICT=0|1
# - POST_CREATE_ENV_TOOL=auto|micromamba|conda|mamba
# - POST_CREATE_ENV_TOOL_SELECTED=micromamba|conda|mamba   (set by env_* scripts)
# - POST_CREATE_ENV_NAME=<name>                             (set by env_* scripts)
# - ENV_NAME=<name> / PY_VERSION=<x.y>                      (fallback)
#
# Conda-family runner selection (deterministic)
# - POST_CREATE_CONDA_RUNNER=auto|mamba|conda   (default auto)
#   - auto: if `mamba` exists, use it; else use `conda`.
# - POST_CREATE_CONDA_NO_CAPTURE=0|1           (default 1; only for conda)
#
# Python steps
# - POST_CREATE_PIP_REQUIREMENTS=0|1
# - POST_CREATE_REQUIREMENTS_FILES="requirements/build.txt,requirements/dev.txt"
# - POST_CREATE_PIP_UPGRADE_TOOLS=0|1
# - POST_CREATE_INSTALL_PACKAGE=0|1
# - POST_CREATE_PACKAGE_MODE=auto|local-editable|local|pypi|none
# - POST_CREATE_PACKAGE_EXTRAS="build,dev,test,doc"
# - POST_CREATE_ALLOW_FALLBACK=0|1
# - SCIKITPLOT_VERSION=<ver>                         (implies pypi in auto mode)
# - POST_CREATE_PIP_EXTRA_ARGS="--no-cache-dir"
#
# Git steps
# - POST_CREATE_GIT_SAFE_DIR=0|1
# - POST_CREATE_GIT_SAFE_DIRS="<newline-separated absolute or repo-relative paths>"
# - GIT_SAFE_DIR_ALLOW_ALL=0|1
# - POST_CREATE_GIT_SUBMODULES=0|1
# - POST_CREATE_GIT_UPSTREAM=0|1
# - POST_CREATE_UPSTREAM_URL=<url>

# Re-exec into bash if invoked by sh/zsh
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

# Re-source shell config to ensure activation takes effect
# shellcheck disable=SC1090
# . ~/."$(basename $SHELL)"rc || true  # ~/.bashrc or ~/.zshrc for zsh
source ~/."$(basename $SHELL)"rc || echo "⚠️ Failed to source $SHELL_RC"

# ---------------- sourced/executed helpers ----------------
pc_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
pc_exit_or_return() { local code="${1:-0}"; pc_is_sourced && return "$code" || exit "$code"; }

# ---------------- deterministic paths ----------------
PC_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PC_REPO_ROOT="${REPO_ROOT:-$(cd -- "$PC_SCRIPT_DIR/../.." && pwd -P)}"
PC_COMMON_SH="${COMMON_SH:-$PC_REPO_ROOT/docker/scripts/common.sh}"

# ---------------- logging (prefer common.sh; fallback must not hard-exit) ----------------
if [[ -f "$PC_COMMON_SH" ]]; then
  # shellcheck source=/dev/null
  . "$PC_COMMON_SH"
else
  _pc_log() { printf '%s\n' "$*" >&2; }
  log_info() { _pc_log "[INFO] $*"; }
  log_warning() { _pc_log "[WARNING] $*"; }
  log_success() { _pc_log "[SUCCESS] $*"; }
  log_debug() { :; }
  has_cmd() { command -v "$1" >/dev/null 2>&1; }
  default_var() { :; }  # no-op fallback
fi

POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"

pc_fail() {
  log_warning "$1"
  pc_exit_or_return "${2:-1}"
}

pc_maybe_fail() {
  local msg="$1"
  local code="${2:-1}"
  if [[ "$POST_CREATE_STRICT" == "1" ]]; then
    pc_fail "$msg" "$code"
  else
    log_warning "$msg"
    return 0
  fi
}

pc_optional_fail() {
  local code="$1"; shift || true
  local msg="$*"
  pc_maybe_fail "$msg" "$code"
}

pc_abspath_existing() {
  local p="$1"
  [[ -n "$p" ]] || pc_maybe_fail "abspath_existing: missing path" 2
  [[ -e "$p" ]] || pc_maybe_fail "Path does not exist: $p" 2

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
# Git steps (idempotent)
# ===============================================================

pc_git_safe_add_one() {
  local abs="$1"
  [[ -n "$abs" ]] || return 0

  local existing
  existing="$(git config --global --get-all safe.directory 2>/dev/null || true)"
  if printf '%s\n' "$existing" | grep -Fxq -- "$abs"; then
    log_debug "safe.directory already present: $abs"
    return 0
  fi

  git config --global --add safe.directory "$abs" >/dev/null 2>&1 || return 1
  return 0
}

pc_git_safe_directories() {
  local allow_all="${GIT_SAFE_DIR_ALLOW_ALL:-0}"
  local failed=0

  has_cmd git || { pc_optional_fail 0 "git not found; skipping safe.directory"; return 0; }

  log_info "Configuring git safe.directory entries..."

  local -a dirs=(
    "$PC_REPO_ROOT"
    "$PC_REPO_ROOT/third_party/array-api-compat"
    "$PC_REPO_ROOT/third_party/array-api-extra"
    "$PC_REPO_ROOT/third_party/astropy"
    "$PC_REPO_ROOT/third_party/seaborn"
  )

  # Optional additional dirs (newline-separated; absolute or repo-relative)
  if [[ -n "${POST_CREATE_GIT_SAFE_DIRS:-}" ]]; then
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -n "$line" ]] || continue
      if [[ "$line" == /* ]]; then
        dirs+=("$line")
      else
        dirs+=("$PC_REPO_ROOT/$line")
      fi
    done <<< "${POST_CREATE_GIT_SAFE_DIRS}"
  fi

  local d abs
  for d in "${dirs[@]}"; do
    if [[ -d "$d" ]]; then
      abs="$(pc_abspath_existing "$d")" || { failed=1; continue; }
      if ! pc_git_safe_add_one "$abs"; then
        log_warning "Failed to add safe.directory: $abs"
        failed=1
      fi
    else
      log_debug "safe.directory skip (missing): $d"
    fi
  done

  if [[ "$failed" == "1" && "$allow_all" == "1" ]]; then
    log_warning "Some safe.directory entries failed; adding wildcard '*' (GIT_SAFE_DIR_ALLOW_ALL=1)"
    pc_git_safe_add_one '*' || pc_optional_fail 0 "Failed to add safe.directory '*'"
  fi

  log_success "git safe.directory configuration complete"
}

pc_git_submodules_init() {
  has_cmd git || { pc_optional_fail 0 "git not found; skipping submodules"; return 0; }

  if [[ ! -f "$PC_REPO_ROOT/.gitmodules" ]]; then
    log_info "No .gitmodules found; skipping submodule init"
    return 0
  fi

  log_info "Initializing/updating git submodules..."
  (cd -- "$PC_REPO_ROOT" && git submodule update --init --recursive) || pc_optional_fail 0 "Submodule init failed"
  log_success "Submodule setup complete"
}

pc_git_config_upstream() {
  has_cmd git || { pc_optional_fail 0 "git not found; skipping upstream remote"; return 0; }

  local upstream_url="${POST_CREATE_UPSTREAM_URL:-https://github.com/scikit-plots/scikit-plots.git}"

  log_info "Configuring upstream remote (if needed)..."
  (cd -- "$PC_REPO_ROOT" && git remote get-url upstream >/dev/null 2>&1) || {
    (cd -- "$PC_REPO_ROOT" && git remote add upstream "$upstream_url") || pc_optional_fail 0 "Failed to add upstream remote"
  }

  log_info "Fetching upstream tags..."
  (cd -- "$PC_REPO_ROOT" && git fetch upstream --tags) || pc_optional_fail 0 "Failed to fetch upstream tags"

  log_success "Git remote configuration complete"
}

# ===============================================================
# Environment runner (micromamba run / mamba run / conda run)
# ===============================================================

_pc_env_kind=""  # micromamba|mamba|conda|empty

pc_resolve_env_name() {
  # Strict precedence: POST_CREATE_ENV_NAME -> ENV_NAME -> derive from PY_VERSION
  if [[ -n "${POST_CREATE_ENV_NAME:-}" ]]; then
    printf '%s\n' "${POST_CREATE_ENV_NAME}"
    return 0
  fi
  if [[ -n "${ENV_NAME:-}" ]]; then
    printf '%s\n' "${ENV_NAME}"
    return 0
  fi
  local py="${PY_VERSION:-3.12}"
  printf 'py%s\n' "${py//./}"
}

pc_select_conda_family_runner() {
  # Deterministic selection for conda-family commands.
  # - POST_CREATE_CONDA_RUNNER=auto|mamba|conda
  #   - auto: use mamba if present; else conda.
  local runner="${POST_CREATE_CONDA_RUNNER:-auto}"
  case "$runner" in
    auto|mamba|conda) ;;
    *) pc_maybe_fail "Invalid POST_CREATE_CONDA_RUNNER=$runner (expected auto|mamba|conda)" 2 ;;
  esac

  if [[ "$runner" == "mamba" ]]; then
    has_cmd mamba || pc_maybe_fail "mamba required but not found (POST_CREATE_CONDA_RUNNER=mamba)" 2
    _pc_env_kind="mamba"
    return 0
  fi

  if [[ "$runner" == "conda" ]]; then
    has_cmd conda || pc_maybe_fail "conda required but not found (POST_CREATE_CONDA_RUNNER=conda)" 2
    _pc_env_kind="conda"
    return 0
  fi

  # auto
  if has_cmd mamba; then _pc_env_kind="mamba"; return 0; fi
  if has_cmd conda; then _pc_env_kind="conda"; return 0; fi

  _pc_env_kind=""
  return 0
}

pc_select_env_runner() {
  # Strict precedence: POST_CREATE_ENV_TOOL_SELECTED -> POST_CREATE_ENV_TOOL -> auto
  local tool_selected="${POST_CREATE_ENV_TOOL_SELECTED:-}"
  local tool="${POST_CREATE_ENV_TOOL:-auto}"

  if [[ -n "$tool_selected" ]]; then
    tool="$tool_selected"
  fi

  # Normalize aliases
  case "$tool" in
    micromamba|conda|mamba|auto) ;;
    *) pc_maybe_fail "Invalid POST_CREATE_ENV_TOOL: $tool (expected auto|micromamba|mamba|conda)" 2 ;;
  esac

  if [[ "$tool" == "micromamba" ]]; then
    has_cmd micromamba || pc_maybe_fail "micromamba required but not found" 2
    _pc_env_kind="micromamba"
    return 0
  fi

  if [[ "$tool" == "mamba" ]]; then
    # Explicit mamba means conda-family runner must be mamba
    POST_CREATE_CONDA_RUNNER="mamba"
    pc_select_conda_family_runner
    [[ -n "$_pc_env_kind" ]] || pc_maybe_fail "mamba tool selection failed" 2
    return 0
  fi

  if [[ "$tool" == "conda" ]]; then
    # conda-family; choose runner (auto/mamba/conda)
    pc_select_conda_family_runner
    [[ -n "$_pc_env_kind" ]] || pc_maybe_fail "conda-family tool selection failed" 2
    return 0
  fi

  # auto: deterministic priority micromamba -> conda-family (mamba preferred by POST_CREATE_CONDA_RUNNER=auto)
  if has_cmd micromamba; then _pc_env_kind="micromamba"; return 0; fi
  pc_select_conda_family_runner
  return 0
}

pc_env_run() {
  local env_name="$1"; shift || true
  [[ -n "$env_name" ]] || pc_maybe_fail "pc_env_run: env name is empty" 2
  [[ -n "$_pc_env_kind" ]] || pc_maybe_fail "pc_env_run: env runner not selected" 2

  local conda_no_capture="${POST_CREATE_CONDA_NO_CAPTURE:-1}"

  case "$_pc_env_kind" in
    micromamba)
      micromamba run -n "$env_name" "$@"
      ;;
    mamba)
      # Keep flags minimal for maximum compatibility across mamba versions.
      mamba run -n "$env_name" -- "$@"
      ;;
    conda)
      if [[ "$conda_no_capture" == "1" ]]; then
        conda run --no-capture-output -n "$env_name" -- "$@"
      else
        conda run -n "$env_name" -- "$@"
      fi
      ;;
    *)
      pc_maybe_fail "Unknown env runner kind: $_pc_env_kind" 2
      ;;
  esac
}

pc_env_assert_python() {
  local env_name="$1"
  if ! pc_env_run "$env_name" python -c 'import sys; print(sys.version.split()[0])' >/dev/null 2>&1; then
    pc_optional_fail 1 "Python not available in env '$env_name'. Ensure environment.yml includes python and pip."
    return 1
  fi
  return 0
}

# ===============================================================
# Python / pip steps
# ===============================================================

pc_pip_upgrade_tools() {
  local env_name="$1"
  local extra="${POST_CREATE_PIP_EXTRA_ARGS:-}"
  log_info "Upgrading pip tooling (pip, setuptools, wheel)..."
  # shellcheck disable=SC2086
  pc_env_run "$env_name" python -m pip install --no-input -U pip setuptools wheel $extra || pc_optional_fail 1 "pip tooling upgrade failed"
}

pc_pip_install_requirements_files() {
  local env_name="$1"
  local files_csv="${POST_CREATE_REQUIREMENTS_FILES:-${POST_CREATE_REQUIREMENTS_FILE:-./requirements/build.txt}}"

  IFS=',' read -r -a files <<< "$files_csv"

  local req file_path
  for req in "${files[@]}"; do
    req="$(printf '%s' "$req" | xargs)"  # trim spaces
    [[ -n "$req" ]] || continue

    file_path="$req"
    [[ -f "$PC_REPO_ROOT/$req" ]] && file_path="$PC_REPO_ROOT/$req"

    if [[ ! -f "$file_path" ]]; then
      pc_optional_fail 0 "Requirements file not found: $req"
      continue
    fi

    log_info "Installing requirements: $file_path"
    local extra="${POST_CREATE_PIP_EXTRA_ARGS:-}"
    # shellcheck disable=SC2086
    if pc_env_run "$env_name" python -m pip install --no-input -r "$file_path" $extra; then
      log_success "Requirements installed: $req"
    else
      pc_optional_fail 1 "pip requirements install failed: $req"
      return 0
    fi
  done
}

pc_install_project() {
  local env_name="$1"

  local mode="${POST_CREATE_PACKAGE_MODE:-auto}"
  local extras="${POST_CREATE_PACKAGE_EXTRAS:-build,dev,test,doc}"
  local allow_fallback="${POST_CREATE_ALLOW_FALLBACK:-0}"

  # auto is deterministic:
  # - SCIKITPLOT_VERSION set  -> pypi
  # - else                    -> local-editable
  if [[ "$mode" == "auto" ]]; then
    if [[ -n "${SCIKITPLOT_VERSION:-}" ]]; then
      mode="pypi"
    else
      mode="local-editable"
    fi
  fi

  local extra_pip="${POST_CREATE_PIP_EXTRA_ARGS:-}"

  case "$mode" in
    none)
      log_info "Package install disabled (POST_CREATE_PACKAGE_MODE=none)"
      return 0
      ;;
    pypi)
      if [[ -n "${SCIKITPLOT_VERSION:-}" ]]; then
        log_info "Installing scikit-plots from PyPI: scikit-plots==${SCIKITPLOT_VERSION}"
        # shellcheck disable=SC2086
        pc_env_run "$env_name" python -m pip install --no-input "scikit-plots==${SCIKITPLOT_VERSION}" $extra_pip \
          || pc_optional_fail 1 "Failed to install scikit-plots==${SCIKITPLOT_VERSION}"
      else
        log_info "Installing scikit-plots from PyPI (latest)"
        # shellcheck disable=SC2086
        pc_env_run "$env_name" python -m pip install --no-input scikit-plots $extra_pip \
          || pc_optional_fail 1 "Failed to install scikit-plots from PyPI"
      fi
      log_success "PyPI install complete"
      return 0
      ;;
    local|local-editable)
      log_info "Installing local scikit-plots (editable) from repo: $PC_REPO_ROOT"
      if [[ "$mode" == "local-editable" ]]; then
        if ! (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation -e ".[${extras}]" -v); then
          if [[ "$allow_fallback" == "1" ]]; then
            log_warning "Editable install with extras failed; falling back to minimal editable install (POST_CREATE_ALLOW_FALLBACK=1)"
            (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation -e . -v) \
              || pc_optional_fail 1 "Minimal editable install failed"
          else
            pc_optional_fail 1 "Editable install with extras failed (set POST_CREATE_ALLOW_FALLBACK=1 to fallback)"
          fi
        fi
      else
        (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation -e . -v) \
          || pc_optional_fail 1 "Minimal editable install failed"
      fi
      log_success "Local install step complete"
      return 0
      ;;
    *)
      pc_maybe_fail "Invalid POST_CREATE_PACKAGE_MODE: $mode (expected auto|local-editable|local|pypi|none)" 2
      ;;
  esac
}

pc_install_precommit() {
  local env_name="$1"

  has_cmd git || { pc_optional_fail 0 "git not found; skipping pre-commit install"; return 0; }

  log_info "Installing pre-commit..."
  local extra="${POST_CREATE_PIP_EXTRA_ARGS:-}"
  # shellcheck disable=SC2086
  pc_env_run "$env_name" python -m pip install --no-input pre-commit $extra || pc_optional_fail 1 "Failed to install pre-commit"

  log_info "Installing pre-commit hooks..."
  (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" pre-commit install) || pc_optional_fail 0 "Failed to install pre-commit hooks"

  log_success "pre-commit setup complete"
}

pc_show_env_info() {
  local env_name="$1"
  log_info "Environment runner: ${_pc_env_kind:-none} env=${env_name}"

  [[ -n "$_pc_env_kind" ]] || return 0
  pc_env_run "$env_name" python -c 'import sys; print("python:", sys.version)' || true
  pc_env_run "$env_name" python -m pip --version || true
  pc_env_run "$env_name" python -m pip show scikit-plots 2>/dev/null || true
}

pc_write_tutorial() {
  local env_name="$1"
  local out="${POST_CREATE_TUTORIAL_PATH:-$HOME/.scikit-plots-postcreate.txt}"
  local tool="${_pc_env_kind:-}"

  log_info "Writing tutorial note: $out"
  mkdir -p -- "$(dirname -- "$out")" 2>/dev/null || true

  {
    printf '%s\n' "scikit-plots devcontainer / docker post-create"
    printf '%s\n' "Repo: $PC_REPO_ROOT"
    printf '%s\n' "Env tool: ${tool:-none}"
    printf '%s\n' "Env name: $env_name"
    printf '\n'
    printf '%s\n' "Quick checks:"
    printf '%s\n' "  python -c \"import scikitplot; print(scikitplot.__version__)\""
    printf '\n'
    printf '%s\n' "Run inside environment:"
    case "$tool" in
      micromamba) printf '%s\n' "  micromamba run -n $env_name python -V" ;;
      mamba)      printf '%s\n' "  mamba run -n $env_name -- python -V" ;;
      conda)      printf '%s\n' "  conda run -n $env_name -- python -V" ;;
      *)          printf '%s\n' "  (no env tool detected)" ;;
    esac
    printf '\n'
    printf '%s\n' "Install modes:"
    printf '%s\n' "  - Local editable: POST_CREATE_PACKAGE_MODE=local-editable"
    printf '%s\n' "  - PyPI pinned:    SCIKITPLOT_VERSION=0.4.0.post1 (or POST_CREATE_PACKAGE_MODE=pypi)"
    printf '\n'
    printf '%s\n' "Conda-family runner:"
    printf '%s\n' "  POST_CREATE_CONDA_RUNNER=auto  # uses mamba if available"
    printf '%s\n' "  POST_CREATE_CONDA_RUNNER=mamba # force mamba"
    printf '%s\n' "  POST_CREATE_CONDA_RUNNER=conda # force conda"
    printf '\n'
    printf '%s\n' "Common toggles:"
    printf '%s\n' "  POST_CREATE_STRICT=1"
    printf '%s\n' "  POST_CREATE_PIP_UPGRADE_TOOLS=1"
    printf '%s\n' "  POST_CREATE_REQUIREMENTS_FILES=requirements/build.txt,requirements/dev.txt"
  } > "$out" || pc_optional_fail 0 "Failed to write tutorial note: $out"

  log_success "Tutorial note written"
}

pc_print_next_steps() {
  log_info "Next steps:"
  log_info " - Create a branch (see contribution guide):"
  log_info "   https://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch"
}

# ===============================================================
# Main (strict inside; sourced-safe)
# ===============================================================

post_create_body() {
  set -Eeuo pipefail
  trap 'rc=$?; printf "%s\n" "[ERROR] post_create_commands.sh failed at line $LINENO: ${BASH_COMMAND-<cmd>} (exit=$rc)" >&2; return $rc' ERR

  cd -- "$PC_REPO_ROOT" || pc_maybe_fail "Failed to cd to repo root: $PC_REPO_ROOT" 2

  # ----- Git -----
  [[ "${POST_CREATE_GIT_SAFE_DIR:-1}" == "1" ]] && pc_git_safe_directories
  [[ "${POST_CREATE_GIT_SUBMODULES:-1}" == "1" ]] && pc_git_submodules_init
  [[ "${POST_CREATE_GIT_UPSTREAM:-1}" == "1" ]] && pc_git_config_upstream

  # ----- Environment selection -----
  pc_select_env_runner
  local env_name
  env_name="$(pc_resolve_env_name)"

  if [[ -z "$_pc_env_kind" ]]; then
    if [[ "${POST_CREATE_ENV_REQUIRED:-0}" == "1" ]]; then
      pc_maybe_fail "No environment tool found (micromamba/mamba/conda)." 2
    fi
    log_warning "No environment tool found (micromamba/mamba/conda); skipping pip/package steps"
    log_success "post_create_commands: complete (no env tool)"
    return 0
  fi

  log_info "Using env tool: $_pc_env_kind (env: $env_name)"

  # Sanity: python must exist in env for pip steps
  pc_env_assert_python "$env_name" || {
    log_warning "Skipping python/pip steps because python is not available in env '$env_name'"
    log_success "post_create_commands: complete (env missing python)"
    return 0
  }

  # ----- Python steps -----
  [[ "${POST_CREATE_PIP_UPGRADE_TOOLS:-0}" == "1" ]] && pc_pip_upgrade_tools "$env_name"
  [[ "${POST_CREATE_PIP_REQUIREMENTS:-1}" == "1" ]] && pc_pip_install_requirements_files "$env_name"
  [[ "${POST_CREATE_INSTALL_PACKAGE:-1}" == "1" ]] && pc_install_project "$env_name"
  [[ "${POST_CREATE_INSTALL_PRECOMMIT:-1}" == "1" ]] && pc_install_precommit "$env_name"
  [[ "${POST_CREATE_SHOW_ENV_INFO:-0}" == "1" ]] && pc_show_env_info "$env_name"
  [[ "${POST_CREATE_WRITE_TUTORIAL:-0}" == "1" ]] && pc_write_tutorial "$env_name"
  [[ "${POST_CREATE_PRINT_NEXT_STEPS:-1}" == "1" ]] && pc_print_next_steps

  log_success "post_create_commands: complete"
  return 0
}

# Run mode:
# - if sourced: isolate in subshell to avoid polluting caller
# - if executed: run directly
if pc_is_sourced; then
  ( post_create_body "$@" )
  pc_exit_or_return $?
else
  post_create_body "$@"
  pc_exit_or_return $?
fi
