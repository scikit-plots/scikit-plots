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
# - PYTHON_VERSION, ENV_NAME, VARIANT
# - ENV_FILE_BASE, ENV_FILE_RUNTIME, ENV_FILE_DEVEL (set by all_post_create.sh)
# - ENV_FILE: legacy single-file fallback (treated as ENV_FILE_RUNTIME)
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
# - ENV_NAME=<resolved env name>
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
  # ── version / name ──
  default_var PYTHON_VERSION "3.12"
  if [[ -z "${PY_VERSION:-}" ]]; then PY_VERSION="$PYTHON_VERSION"; fi
  default_var ENV_NAME "py${PYTHON_VERSION//./}"

  # ── VARIANT ──
  default_var VARIANT "runtime"
  case "${VARIANT}" in
    runtime|devel) ;;
    *) log_error "Invalid VARIANT='${VARIANT}' (expected runtime|devel)" ;;
  esac

  # ── Layered env files ──
  local _env_dir_default="$REPO_ROOT/docker/env_conda"
  default_var ENV_DIR "$_env_dir_default"
  default_var ENV_FILE_BASE    "${ENV_DIR}/environment.base.yml"
  default_var ENV_FILE_RUNTIME "${ENV_DIR}/environment.yml"
  default_var ENV_FILE_DEVEL   "${ENV_DIR}/environment.devel.yml"

  # Legacy backward compat: if ENV_FILE set alone (old callers), treat as runtime file.
  if [[ -n "${ENV_FILE:-}" ]]; then
    ENV_FILE_RUNTIME="$ENV_FILE"
    log_info "env_conda: ENV_FILE override -> ENV_FILE_RUNTIME=$ENV_FILE_RUNTIME"
  fi

  # ── Resolve all env-file references to absolute, CWD-independent paths ──
  # ENV_DIR may arrive bare ("docker/env_conda") or absolute; the three ENV_FILE_*
  # may arrive as bare filenames (resolved under ENV_DIR), repo-root-relative paths,
  # or absolute paths. This bridges differing Dockerfile conventions safely and
  # makes the [[ -f ... ]] checks below independent of the current directory.
  ENV_DIR="$(common_resolve_path "$ENV_DIR")"
  ENV_FILE_BASE="$(common_resolve_path "$ENV_FILE_BASE" "$ENV_DIR")"
  ENV_FILE_RUNTIME="$(common_resolve_path "$ENV_FILE_RUNTIME" "$ENV_DIR")"
  ENV_FILE_DEVEL="$(common_resolve_path "$ENV_FILE_DEVEL" "$ENV_DIR")"
  # Ensure ENV_FILE points to runtime for any legacy code that reads it.
  ENV_FILE="$ENV_FILE_RUNTIME"

  export PYTHON_VERSION PY_VERSION ENV_NAME VARIANT ENV_DIR ENV_FILE_BASE ENV_FILE_RUNTIME ENV_FILE_DEVEL ENV_FILE

  default_var CONDA_ALLOW_INSTALL "1"
  default_var CONDA_MANAGER "auto"
  default_var CONDA_BOOTSTRAP_MAMBA "0"
  default_var CONDA_ACTION "ensure"
  default_var CONDA_USE_FILE_NAME "0"
  default_var CONDA_PRUNE "0"
  default_var CONDA_PIN_PYTHON "1"

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
    # strict: first "name:" line (YAML convention), read from the runtime layer
    awk -F: 'tolower($1)=="name" {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${ENV_FILE_RUNTIME:-$ENV_FILE}"
  }

  _resolve_env_name() {
    if common_is_true "$CONDA_USE_FILE_NAME"; then
      local n
      n="$(_yaml_env_name)"
      [[ -n "$n" ]] || log_error "CONDA_USE_FILE_NAME=1 but could not parse 'name:' from ${ENV_FILE_RUNTIME}"
      if [[ -n "${ENV_NAME:-}" && "${ENV_NAME}" != "$n" ]]; then
        log_error "ENV_NAME='$ENV_NAME' conflicts with YAML name='$n' in ${ENV_FILE_RUNTIME} (set CONDA_USE_FILE_NAME=0 or align names)"
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

  # ── Layered apply helpers ──────────────────────────────────────────────────
  # Apply all applicable yml layers in order to an EXISTING env named $2 via manager $1.
  # base (always) → runtime (always) → devel (only when VARIANT=devel).
  _conda_update_layered() {
    local mgr="$1" env_n="$2"
    local do_prune=0
    common_is_true "${CONDA_PRUNE:-0}" && do_prune=1
    local files=()
    [[ -f "$ENV_FILE_BASE" ]]    && files+=("$ENV_FILE_BASE")    || log_warning "ENV_FILE_BASE not found: $ENV_FILE_BASE"
    [[ -f "$ENV_FILE_RUNTIME" ]] && files+=("$ENV_FILE_RUNTIME") || log_warning "ENV_FILE_RUNTIME not found: $ENV_FILE_RUNTIME"
    if [[ "$VARIANT" == "devel" ]]; then
      [[ -f "$ENV_FILE_DEVEL" ]] && files+=("$ENV_FILE_DEVEL")   || log_warning "VARIANT=devel but ENV_FILE_DEVEL not found: $ENV_FILE_DEVEL"
    fi
    local n=${#files[@]}
    [[ $n -gt 0 ]] || log_error "No env files found for update (ENV_DIR=$ENV_DIR)"
    local i f prune_flag
    for ((i = 0; i < n; i++)); do
      f="${files[$i]}"
      prune_flag=""
      # Apply --prune ONLY on the final layer: pruning per layer would drop packages
      # introduced by earlier layers (base/runtime) that the current layer omits.
      [[ $do_prune -eq 1 && $i -eq $((n - 1)) ]] && prune_flag="--prune"
      log_info "Applying layer $((i + 1))/$n: $f${prune_flag:+  (--prune)}"
      # shellcheck disable=SC2086
      "$mgr" env update -n "$env_n" -f "$f" --yes $prune_flag
    done
  }

  # Create a brand-new env: optionally seed the requested Python version, then layer
  # base → runtime → (devel only when VARIANT=devel).
  _conda_create_layered() {
    local mgr="$1" env_n="$2"
    [[ -f "$ENV_FILE_BASE" ]] || log_error "ENV_FILE_BASE not found: $ENV_FILE_BASE"
    # Seed the env with the requested Python so ENV_NAME (pyNNN) matches the actual
    # interpreter. The yml layers omit a python pin, so without this the solver would
    # pick its own default and a "py311" env could ship a different version. Opt out
    # with CONDA_PIN_PYTHON=0.
    if common_is_true "${CONDA_PIN_PYTHON:-1}" && [[ -n "${PYTHON_VERSION:-}" ]]; then
      log_info "Creating env '$env_n' with python=${PYTHON_VERSION}, then layering base: $ENV_FILE_BASE"
      "$mgr" create -n "$env_n" "python=${PYTHON_VERSION}" --yes
      "$mgr" env update -n "$env_n" -f "$ENV_FILE_BASE" --yes
    else
      log_info "Creating env '$env_n' from base: $ENV_FILE_BASE"
      "$mgr" env create -n "$env_n" -f "$ENV_FILE_BASE" --yes
    fi

    if [[ -f "$ENV_FILE_RUNTIME" ]]; then
      log_info "Applying runtime layer: $ENV_FILE_RUNTIME"
      "$mgr" env update -n "$env_n" -f "$ENV_FILE_RUNTIME" --yes
    else
      log_warning "ENV_FILE_RUNTIME not found: $ENV_FILE_RUNTIME"
    fi

    if [[ "$VARIANT" == "devel" ]]; then
      if [[ -f "$ENV_FILE_DEVEL" ]]; then
        log_info "Applying devel layer (VARIANT=devel): $ENV_FILE_DEVEL"
        "$mgr" env update -n "$env_n" -f "$ENV_FILE_DEVEL" --yes
      else
        log_warning "VARIANT=devel but ENV_FILE_DEVEL not found: $ENV_FILE_DEVEL"
      fi
    fi
  }

  # ---- run ----

  _install_miniforge_if_needed

  # source ~/.bashrc (or ~/.zshrc, ~/.xonshrc, ~/.config/fish/config.fish, ...)
  # ──────────────────────────────────────────────────────────────────
  SHELL_RC=~/."$(basename $SHELL)"rc

  if [ -f "$SHELL_RC" ]; then
    echo "📄 Sourcing shell config: $SHELL_RC"
    # shellcheck disable=SC1090
    # . ~/.bashrc or . ~/.zshrc for zsh
    # . ~/."$(basename $SHELL)"rc || true  # ~/.bashrc or ~/.zshrc for zsh
    source ~/."$(basename $SHELL)"rc || echo "⚠️ Failed to source $SHELL_RC"
  else
    echo "⚠️ Shell config file not found: $SHELL_RC"
  fi

  # Optional: also initialize conda hooks (for compatibility with existing conda setups)
  conda init --all || echo "⚠️ Failed to initialize conda hooks"
  mamba init --all || echo "⚠️ Failed to initialize mamba hooks"

  # Re-source shell config to ensure activation takes effect
  # shellcheck disable=SC1090
  # . ~/."$(basename $SHELL)"rc || true  # ~/.bashrc or ~/.zshrc for zsh
  source ~/."$(basename $SHELL)"rc || echo "⚠️ Failed to source $SHELL_RC"

  _maybe_bootstrap_mamba

  _resolve_env_name

  local mgr exists=0
  mgr="$(_select_manager)"

  ## Also Configure base
  # "$mgr" install -n base python="$PY_VERSION" ipykernel pip -y || true

  if _env_exists "$mgr" "$ENV_NAME"; then exists=1; fi

  [[ -f "$ENV_FILE_BASE" ]] || log_error "ENV_FILE_BASE not found: $ENV_FILE_BASE"
  [[ -f "$ENV_FILE_RUNTIME" ]] || log_warning "ENV_FILE_RUNTIME not found: $ENV_FILE_RUNTIME"

  case "$CONDA_ACTION" in
    none)
      log_info "CONDA_ACTION=none -> no env action"
      ;;
    create)
      if [[ "$exists" == "1" ]]; then
        log_info "Env exists: $ENV_NAME (create -> no-op)"
      else
        _conda_create_layered "$mgr" "$ENV_NAME"
      fi
      ;;
    update)
      [[ "$exists" == "1" ]] || log_error "Env not found: $ENV_NAME (update)"
      log_info "Updating env '$ENV_NAME' (VARIANT=$VARIANT)"
      _conda_update_layered "$mgr" "$ENV_NAME"
      ;;
    ensure)
      if [[ "$exists" == "1" ]]; then
        log_info "Env exists -> update ($ENV_NAME, VARIANT=$VARIANT)"
        _conda_update_layered "$mgr" "$ENV_NAME"
      else
        _conda_create_layered "$mgr" "$ENV_NAME"
      fi
      ;;
  esac

  export POST_CREATE_ENV_TOOL_SELECTED="conda"
  export POST_CREATE_ENV_READY="1"
  export ENV_NAME="$ENV_NAME"
  log_success "Conda env ready: $ENV_NAME (manager=$mgr, VARIANT=$VARIANT)"

  env_conda_restore
  return 0
}

env_conda_main "$@"
env_conda_exit_or_return $?
