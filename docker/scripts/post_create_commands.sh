#!/usr/bin/env bash
# docker/scripts/post_create_commands.sh
#
# Post-create commands (Bash-contract)
#
# - Safe when SOURCED: no global set/trap side effects; no `exit` when sourced.
# - Deterministic controls via env vars (no implicit installs).

# Re-exec into bash if invoked by sh/zsh
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

# ---------------- sourced/executed helpers ----------------
pc_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }

pc_exit_or_return() {
  local code="${1:-0}"
  if pc_is_sourced; then
    return "$code"
  else
    exit "$code"
  fi
}

# ---------------- truthy parsing (bash3-safe) ----------------
pc_is_true() {
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off|"") return 1 ;;
    *) return 1 ;;
  esac
}

# ---------------- deterministic paths ----------------
PC_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PC_REPO_ROOT="${REPO_ROOT:-$(cd -- "$PC_SCRIPT_DIR/../.." && pwd -P)}"
PC_COMMON_SH="${COMMON_SH:-$PC_REPO_ROOT/docker/scripts/common.sh}"

# ---------------- logging (prefer common.sh; never hard-exit here) ----------------
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
fi

POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"

pc_fail() {
  # Always fails (returns/exit depending on sourced)
  log_warning "$1"
  pc_exit_or_return "${2:-1}"
}

pc_maybe_fail() {
  # In strict -> fail, else warn+continue
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
  # Optional step failure:
  # - strict => fail with given code
  # - non-strict => warn and continue
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
# Git steps
# ===============================================================

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

  local d abs
  for d in "${dirs[@]}"; do
    if [[ -d "$d" ]]; then
      abs="$(pc_abspath_existing "$d")" || { failed=1; continue; }
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
    git config --global --add safe.directory '*' >/dev/null 2>&1 || pc_optional_fail 0 "Failed to add safe.directory '*'"
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
# Environment runner (micromamba run / conda run)
# ===============================================================

_pc_env_kind=""  # micromamba|conda|empty

pc_select_env_runner() {
  local tool="${POST_CREATE_ENV_TOOL:-auto}"

  case "$tool" in
    micromamba|conda|auto) ;;
    *) pc_maybe_fail "Invalid POST_CREATE_ENV_TOOL: $tool (expected auto|micromamba|conda)" 2 ;;
  esac

  if [[ "$tool" == "micromamba" ]]; then
    has_cmd micromamba || pc_maybe_fail "micromamba required but not found" 2
    _pc_env_kind="micromamba"
    return 0
  fi

  if [[ "$tool" == "conda" ]]; then
    has_cmd conda || pc_maybe_fail "conda required but not found" 2
    _pc_env_kind="conda"
    return 0
  fi

  # auto: deterministic priority micromamba -> conda
  if has_cmd micromamba; then _pc_env_kind="micromamba"; return 0; fi
  if has_cmd conda; then _pc_env_kind="conda"; return 0; fi

  _pc_env_kind=""
  return 0
}

pc_env_run() {
  local env_name="$1"; shift || true
  [[ -n "$env_name" ]] || pc_maybe_fail "pc_env_run: env name is empty" 2
  [[ -n "$_pc_env_kind" ]] || pc_maybe_fail "pc_env_run: env runner not selected" 2

  case "$_pc_env_kind" in
    micromamba) micromamba run -n "$env_name" "$@" ;;
    conda)      conda run --no-capture-output -n "$env_name" -- "$@" ;;
    *)          pc_maybe_fail "Unknown env runner kind: $_pc_env_kind" 2 ;;
  esac
}

# ===============================================================
# Python / pip steps
# ===============================================================

pc_pip_install_requirements() {
  local env_name="$1"
  local req_file="${POST_CREATE_REQUIREMENTS_FILE:-./requirements/build.txt}"

  local file_path="$req_file"
  [[ -f "$PC_REPO_ROOT/$req_file" ]] && file_path="$PC_REPO_ROOT/$req_file"

  [[ -f "$file_path" ]] || { pc_optional_fail 0 "Requirements file not found: $req_file"; return 0; }

  log_info "Installing requirements: $file_path"
  if pc_env_run "$env_name" python -m pip install --no-input -r "$file_path"; then
    log_success "Requirements installed"
  else
    pc_optional_fail 1 "pip requirements install failed"
    return 0
  fi
}

pc_install_scikit_plots() {
  local env_name="$1"

  if [[ -n "${SCIKITPLOT_VERSION:-}" ]]; then
    log_info "Installing scikit-plots from PyPI: scikit-plots==${SCIKITPLOT_VERSION}"
    pc_env_run "$env_name" python -m pip install --no-input "scikit-plots==${SCIKITPLOT_VERSION}" \
      || pc_optional_fail 1 "Failed to install scikit-plots==${SCIKITPLOT_VERSION}"
    return 0
  fi

  local extras="${POST_CREATE_LOCAL_EXTRAS:-build,dev,test,doc}"
  local allow_fallback="${POST_CREATE_ALLOW_FALLBACK:-0}"
  local install_extras="${POST_CREATE_INSTALL_EXTRAS:-1}"

  log_info "Installing local scikit-plots (editable) from repo: $PC_REPO_ROOT"

  if [[ "$install_extras" == "1" ]]; then
    if ! (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e ".[${extras}]" -v); then
      if [[ "$allow_fallback" == "1" ]]; then
        log_warning "Editable install with extras failed; falling back to minimal editable install (POST_CREATE_ALLOW_FALLBACK=1)"
        (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e . -v) \
          || pc_optional_fail 1 "Minimal editable install failed"
      else
        pc_optional_fail 1 "Editable install with extras failed (set POST_CREATE_ALLOW_FALLBACK=1 to fallback)"
      fi
    fi
  else
    (cd -- "$PC_REPO_ROOT" && pc_env_run "$env_name" python -m pip install --no-input --no-build-isolation --no-cache-dir -e . -v) \
      || pc_optional_fail 1 "Minimal editable install failed"
  fi

  log_success "scikit-plots installation step complete"
}

pc_install_precommit() {
  local env_name="$1"

  has_cmd git || { pc_optional_fail 0 "git not found; skipping pre-commit install"; return 0; }

  log_info "Installing pre-commit..."
  pc_env_run "$env_name" python -m pip install --no-input pre-commit || pc_optional_fail 1 "Failed to install pre-commit"

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

pc_print_next_steps() {
  log_info "Next steps:"
  log_info " - Create a branch (see contribution guide):"
  log_info "   https://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch"
}

# ---------- defaults ----------
PY_VERSION="${PY_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"

# ===============================================================
# Main (strict inside; no RETURN restore; sourced-safe)
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

  local env_name="${POST_CREATE_ENV_NAME:-${ENV_NAME:-py311}}"

  if [[ -z "$_pc_env_kind" ]]; then
    if [[ "${POST_CREATE_ENV_REQUIRED:-0}" == "1" ]]; then
      pc_maybe_fail "No environment tool found (micromamba/conda)." 2
    fi
    log_warning "No environment tool found (micromamba/conda); skipping pip/package steps"
    log_success "post_create_commands: complete (no env tool)"
    return 0
  fi

  log_info "Using env tool: $_pc_env_kind (env: $env_name)"

  # ----- Python steps -----
  [[ "${POST_CREATE_PIP_REQUIREMENTS:-1}" == "1" ]] && pc_pip_install_requirements "$env_name"
  [[ "${POST_CREATE_INSTALL_PACKAGE:-1}" == "1" ]] && pc_install_scikit_plots "$env_name"
  [[ "${POST_CREATE_INSTALL_PRECOMMIT:-1}" == "1" ]] && pc_install_precommit "$env_name"
  [[ "${POST_CREATE_SHOW_ENV_INFO:-0}" == "1" ]] && pc_show_env_info "$env_name"
  [[ "${POST_CREATE_PRINT_NEXT_STEPS:-1}" == "1" ]] && pc_print_next_steps

  log_success "post_create_commands: complete"
  return 0
}

# Run mode:
# - if sourced: run in subshell to avoid polluting caller (and avoids RETURN-restore traps)
# - if executed: run directly
if pc_is_sourced; then
  ( post_create_body "$@" )
  pc_exit_or_return $?
else
  post_create_body "$@"
  pc_exit_or_return $?
fi
