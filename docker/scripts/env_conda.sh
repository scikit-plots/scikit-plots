#!/usr/bin/env bash
# docker/scripts/env_conda.sh
# ===============================================================
# Conda/Miniforge environment setup (Bash)
# ===============================================================
# USER NOTES
# - Safe to EXECUTE:   bash docker/scripts/env_conda.sh
# - Safe to SOURCE:    .  docker/scripts/env_conda.sh
#
# GLOBAL INPUTS (recommended set by all_post_create.sh)
# - REPO_ROOT, COMMON_SH
# - PY_VERSION, ENV_NAME, ENV_FILE
# - POST_CREATE_STRICT=0|1
# - POST_CREATE_ENV_TOOL=auto|micromamba|conda           (optional)
# - POST_CREATE_ENV_LOCK=0|1                             (optional, default 1)
#
# CONDA CONFIG
# - POST_CREATE_RUN_CONDA=0|1                             (or orchestrator step toggle)
# - CONDA_SKIP=0|1                                        (legacy toggle)
# - CONDA_ALLOW_INSTALL=0|1                               (default: 1) installs Miniforge if missing
# - CONDA_INSTALL_PREFIX=/opt/conda|...                   (default: /opt/conda if writable else ~/.local/miniforge)
# - CONDA_INSTALL_URL=...                                 (optional override)
# - CONDA_MANAGER=auto|conda|mamba                        (default: auto)
# - CONDA_BOOTSTRAP_MAMBA=0|1                             (default: 0) install mamba into base if requested and missing
# - CONDA_ACTION=none|ensure|create|update                (default: ensure)
# - CONDA_USE_FILE_NAME=0|1                               (default: 0) allow env name from YAML "name:"
# - CONDA_PRUNE=0|1                                       (default: 0)
#
# OUTPUTS (exported on success)
# - POST_CREATE_ENV_TOOL_SELECTED=conda
# - POST_CREATE_ENV_READY=1
# - POST_CREATE_ENV_NAME=<resolved env name>
# ===============================================================

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

env_conda_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
env_conda_exit_or_return() { local rc="${1:-0}"; env_conda_is_sourced && return "$rc" || exit "$rc"; }

env_conda_main() {
  local _OLD_SET _OLD_TRAP_ERR _OLD_PWD
  _OLD_SET="$(set +o)"
  _OLD_TRAP_ERR="$(trap -p ERR || true)"
  _OLD_PWD="$(pwd -P 2>/dev/null || pwd)"

  env_conda_restore() {
    eval "$_OLD_SET"
    cd -- "$_OLD_PWD" 2>/dev/null || true
    if [[ -n "$_OLD_TRAP_ERR" ]]; then eval "$_OLD_TRAP_ERR"; else trap - ERR; fi
  }

  env_conda_on_err() {
    local rc="$?"
    printf '%s\n' "[ERROR] env_conda.sh failed (exit=$rc) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
    env_conda_restore
    return "$rc"
  }

  trap 'env_conda_on_err' ERR
  set -Eeuo pipefail

  local SCRIPT_DIR
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  export REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

  : "${COMMON_SH:=$REPO_ROOT/docker/scripts/common.sh}"
  if [[ ! -f "$COMMON_SH" ]]; then
    printf '%s\n' "[ERROR] common.sh not found: $COMMON_SH" >&2
    env_conda_restore
    return 2
  fi
  # shellcheck source=/dev/null
  . "$COMMON_SH"

  # ---- gating (tool selection + lock) ----
  default_var POST_CREATE_ENV_LOCK "1"
  if [[ -n "${POST_CREATE_ENV_TOOL_SELECTED:-}" && "${POST_CREATE_ENV_LOCK}" == "1" ]]; then
    if [[ "${POST_CREATE_ENV_TOOL_SELECTED}" != "conda" ]]; then
      log_info "Conda env: skipped (POST_CREATE_ENV_TOOL_SELECTED=${POST_CREATE_ENV_TOOL_SELECTED}, lock=1)"
      env_conda_restore
      return 0
    fi
  fi

  if [[ -n "${POST_CREATE_ENV_TOOL:-}" ]]; then
    common_validate_enum POST_CREATE_ENV_TOOL auto micromamba conda
    if [[ "${POST_CREATE_ENV_TOOL}" != "auto" && "${POST_CREATE_ENV_TOOL}" != "conda" ]]; then
      log_info "Conda env: skipped (POST_CREATE_ENV_TOOL=${POST_CREATE_ENV_TOOL})"
      env_conda_restore
      return 0
    fi
  fi

  # legacy skip toggle
  default_var CONDA_SKIP "0"
  if common_is_true "$CONDA_SKIP"; then
    log_info "CONDA_SKIP=1 -> skipping conda env"
    env_conda_restore
    return 0
  fi

  # ---- configuration ----
  default_var PY_VERSION "3.12"
  default_var ENV_NAME "py${PY_VERSION//./}"

  if [[ -z "${ENV_FILE:-}" ]]; then
    if [[ -f "$REPO_ROOT/environment.yml" ]]; then
      ENV_FILE="$REPO_ROOT/environment.yml"
    elif [[ -f "$REPO_ROOT/docker/env_conda/environment.yml" ]]; then
      ENV_FILE="$REPO_ROOT/docker/env_conda/environment.yml"
    else
      log_error "ENV_FILE not set and no default found (expected $REPO_ROOT/environment.yml or $REPO_ROOT/docker/env_conda/environment.yml)"
    fi
  fi
  export PY_VERSION ENV_NAME ENV_FILE

  default_var CONDA_ALLOW_INSTALL "1"
  default_var CONDA_MANAGER "auto"
  default_var CONDA_BOOTSTRAP_MAMBA "0"
  default_var CONDA_ACTION "ensure"
  default_var CONDA_USE_FILE_NAME "0"
  default_var CONDA_PRUNE "0"

  common_validate_enum CONDA_MANAGER auto conda mamba
  common_validate_enum CONDA_ACTION none ensure create update

  # ---- helpers ----
  _miniforge_installer_name() {
    local os arch
    os="$(detect_os)"
    arch="$(detect_arch)"
    case "$os" in
      linux) os="Linux" ;;
      macos) os="MacOSX" ;;
      *) log_error "Unsupported OS for Miniforge install: $os" ;;
    esac
    case "$arch" in
      x86_64) arch="x86_64" ;;
      arm64)  arch="aarch64" ;;
      *) log_error "Unsupported arch for Miniforge install: $arch" ;;
    esac
    printf 'Miniforge3-%s-%s.sh\n' "$os" "$arch"
  }

  _install_url() {
    if [[ -n "${CONDA_INSTALL_URL:-}" ]]; then
      printf '%s\n' "$CONDA_INSTALL_URL"
      return 0
    fi
    local name
    name="$(_miniforge_installer_name)"
    printf 'https://github.com/conda-forge/miniforge/releases/latest/download/%s\n' "$name"
  }

  _default_prefix() {
    if [[ -n "${CONDA_INSTALL_PREFIX:-}" ]]; then
      printf '%s\n' "$CONDA_INSTALL_PREFIX"
      return 0
    fi
    if [[ -w "/opt" ]]; then
      printf '%s\n' "/opt/conda"
    else
      printf '%s\n' "$HOME/.local/miniforge"
    fi
  }

  _ensure_conda_on_path() {
    if has_cmd conda; then return 0; fi
    local prefix
    prefix="$(_default_prefix)"
    if [[ -x "$prefix/bin/conda" ]]; then
      export PATH="$prefix/bin:$PATH"
      hash -r 2>/dev/null || true
      has_cmd conda && return 0
    fi
    return 1
  }

  _install_miniforge_if_needed() {
    _ensure_conda_on_path && return 0
    if has_cmd conda || has_cmd mamba; then return 0; fi

    if ! common_is_true "$CONDA_ALLOW_INSTALL"; then
      log_error "conda/mamba not found and CONDA_ALLOW_INSTALL=0"
    fi

    (has_cmd curl || has_cmd wget) || log_error "Miniforge install requires curl or wget"

    local url prefix tmp installer
    url="$(_install_url)"
    prefix="$(_default_prefix)"
    tmp="$(mktemp_dir miniforge)"
    setup_traps "$tmp"
    installer="$tmp/miniforge.sh"

    log_info "Installing Miniforge -> $prefix"
    log_info "Download: $url"

    if has_cmd curl; then
      curl -fsSL "$url" -o "$installer"
    else
      wget -qO "$installer" "$url"
    fi

    bash "$installer" -b -p "$prefix"
    export PATH="$prefix/bin:$PATH"
    hash -r 2>/dev/null || true

    has_cmd conda || log_error "conda still not available after Miniforge install (prefix=$prefix)"
  }

  _maybe_bootstrap_mamba() {
    if has_cmd mamba; then return 0; fi
    if ! common_is_true "$CONDA_BOOTSTRAP_MAMBA"; then return 0; fi

    # installing mamba is heavy; keep explicit
    log_info "Bootstrapping mamba into base env (conda-forge)"
    has_cmd conda || log_error "conda required to bootstrap mamba"
    conda install -n base -c conda-forge -y mamba
    hash -r 2>/dev/null || true
    has_cmd mamba || log_error "mamba bootstrap failed"
  }

  _yaml_env_name() {
    # strict: first "name:" line (YAML convention)
    awk -F: 'tolower($1)=="name" {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$ENV_FILE"
  }

  _resolve_env_name() {
    if common_is_true "$CONDA_USE_FILE_NAME"; then
      local n
      n="$(_yaml_env_name)"
      [[ -n "$n" ]] || log_error "CONDA_USE_FILE_NAME=1 but could not parse 'name:' from $ENV_FILE"
      if [[ -n "${ENV_NAME:-}" && "${ENV_NAME}" != "$n" ]]; then
        log_error "ENV_NAME='$ENV_NAME' conflicts with YAML name='$n' (set CONDA_USE_FILE_NAME=0 or align names)"
      fi
      ENV_NAME="$n"
      export ENV_NAME
    fi
  }

  _select_manager() {
    local mgr=""
    case "$CONDA_MANAGER" in
      conda) has_cmd conda || log_error "conda not found"; mgr="conda" ;;
      mamba) has_cmd mamba || log_error "mamba not found"; mgr="mamba" ;;
      auto)
        if has_cmd mamba; then mgr="mamba"
        elif has_cmd conda; then mgr="conda"
        else log_error "No conda-compatible manager found"
        fi
        ;;
    esac
    printf '%s\n' "$mgr"
  }

  _env_exists() {
    local mgr="$1" name="$2"
    "$mgr" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq -- "$name"
  }

  # ---- run ----
  [[ -f "$ENV_FILE" ]] || log_error "ENV_FILE not found: $ENV_FILE"

  _install_miniforge_if_needed
  _maybe_bootstrap_mamba
  _resolve_env_name

  local mgr exists=0
  mgr="$(_select_manager)"

  if _env_exists "$mgr" "$ENV_NAME"; then exists=1; fi

  case "$CONDA_ACTION" in
    none)
      log_info "CONDA_ACTION=none -> no env action"
      ;;
    create)
      if [[ "$exists" == "1" ]]; then
        log_info "Env exists: $ENV_NAME (create -> no-op)"
      else
        log_info "Creating env '$ENV_NAME' from $ENV_FILE"
        "$mgr" env create -n "$ENV_NAME" -f "$ENV_FILE" --yes
      fi
      ;;
    update)
      [[ "$exists" == "1" ]] || log_error "Env not found: $ENV_NAME (update)"
      log_info "Updating env '$ENV_NAME' from $ENV_FILE"
      if common_is_true "$CONDA_PRUNE"; then
        "$mgr" env update -n "$ENV_NAME" -f "$ENV_FILE" --yes --prune
      else
        "$mgr" env update -n "$ENV_NAME" -f "$ENV_FILE" --yes
      fi
      ;;
    ensure)
      if [[ "$exists" == "1" ]]; then
        log_info "Env exists -> update ($ENV_NAME)"
        if common_is_true "$CONDA_PRUNE"; then
          "$mgr" env update -n "$ENV_NAME" -f "$ENV_FILE" --yes --prune
        else
          "$mgr" env update -n "$ENV_NAME" -f "$ENV_FILE" --yes
        fi
      else
        log_info "Env missing -> create ($ENV_NAME)"
        "$mgr" env create -n "$ENV_NAME" -f "$ENV_FILE" --yes
      fi
      ;;
  esac

  export POST_CREATE_ENV_TOOL_SELECTED="conda"
  export POST_CREATE_ENV_READY="1"
  export POST_CREATE_ENV_NAME="$ENV_NAME"
  log_success "Conda env ready: $ENV_NAME (manager=$mgr)"

  env_conda_restore
  return 0
}

env_conda_main "$@"
env_conda_exit_or_return $?
