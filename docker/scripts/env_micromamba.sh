#!/usr/bin/env bash
# docker/scripts/env_micromamba.sh
# ===============================================================
# Micromamba environment setup (Bash)
# ===============================================================
# USER NOTES
# - Safe to EXECUTE:   bash docker/scripts/env_micromamba.sh
# - Safe to SOURCE:    .  docker/scripts/env_micromamba.sh
# - Intended to be sourced by docker/scripts/all_post_create.sh
#
# GLOBAL INPUTS (recommended set by all_post_create.sh)
# - REPO_ROOT, COMMON_SH
# - PY_VERSION, ENV_NAME, ENV_FILE
# - POST_CREATE_STRICT=0|1
# - POST_CREATE_ENV_TOOL=auto|micromamba|conda           (optional)
# - POST_CREATE_ENV_LOCK=0|1                             (optional, default 1)
#
# MICROMAMBA CONFIG
# - POST_CREATE_RUN_MICROMAMBA=0|1                       (or orchestrator step toggle)
# - SKIP_MICROMAMBA=0|1                                  (legacy toggle)
# - MICROMAMBA_ALLOW_INSTALL=0|1                         (default: 1)
# - MICROMAMBA_INSTALL_MODE=api|script                   (default: api)
# - MAMBA_EXE=/path                                      (default: /usr/local/bin if writable else ~/.local/bin)
# - MAMBA_ROOT_PREFIX=/path                              (default: ~/micromamba)
# - MICROMAMBA_ENV_ACTION=none|ensure|create|update      (default: ensure)
# - MICROMAMBA_PRUNE=0|1                                 (default: 0; applies to update/ensure(update))
#
# OUTPUTS (exported on success)
# - POST_CREATE_ENV_TOOL_SELECTED=micromamba
# - POST_CREATE_ENV_READY=1
# - ENV_NAME=<resolved env name>
# ===============================================================
# # >>> mamba initialize >>>
# # !! Contents within this block are managed by 'micromamba shell init' !!
# export MAMBA_EXE='/root/.local/bin/micromamba';
# export MAMBA_ROOT_PREFIX='/root/micromamba';
# __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__mamba_setup"
# else
#     alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
# fi
# unset __mamba_setup
# # <<< mamba initialize <<<
# ===============================================================
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

env_micromamba_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
env_micromamba_exit_or_return() { local rc="${1:-0}"; env_micromamba_is_sourced && return "$rc" || exit "$rc"; }

# -- - makes sure you’re not accidentally passing an extra argument to the command. For example, if you try to create a directory that starts with - (dash) without using -- the directory name will be interpreted as a command argument.
# && - ensures that the second command runs only if the first command is successful.
env_micromamba_main() {
  # ---- preserve caller state (important when sourced) ----
  local _OLD_SET _OLD_TRAP_ERR _OLD_PWD
  _OLD_SET="$(set +o)"
  _OLD_TRAP_ERR="$(trap -p ERR || true)"
  _OLD_PWD="$(pwd -P 2>/dev/null || pwd)"

  env_micromamba_restore() {
    eval "$_OLD_SET"
    cd -- "$_OLD_PWD" 2>/dev/null || true
    if [[ -n "$_OLD_TRAP_ERR" ]]; then eval "$_OLD_TRAP_ERR"; else trap - ERR; fi
  }

  env_micromamba_on_err() {
    local rc="$?"
    printf '%s\n' "[ERROR] env_micromamba.sh failed (exit=$rc) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
    env_micromamba_restore
    return "$rc"
  }

  trap 'env_micromamba_on_err' ERR
  set -Eeuo pipefail

  # ---- canonical paths + common ----
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  export REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

  : "${COMMON_SH:=$REPO_ROOT/docker/scripts/common.sh}"
  if [[ ! -f "$COMMON_SH" ]]; then
    printf '%s\n' "[ERROR] common.sh not found: $COMMON_SH" >&2
    env_micromamba_restore
    return 2
  fi
  # shellcheck source=/dev/null
  . "$COMMON_SH"

  # ---- gating (tool selection + lock) ----
  default_var POST_CREATE_ENV_LOCK "1"
  if [[ -n "${POST_CREATE_ENV_TOOL_SELECTED:-}" && "${POST_CREATE_ENV_LOCK}" == "1" ]]; then
    if [[ "${POST_CREATE_ENV_TOOL_SELECTED}" != "micromamba" ]]; then
      log_info "Micromamba env: skipped (POST_CREATE_ENV_TOOL_SELECTED=${POST_CREATE_ENV_TOOL_SELECTED}, lock=1)"
      env_micromamba_restore
      return 0
    fi
  fi

  if [[ -n "${POST_CREATE_ENV_TOOL:-}" ]]; then
    common_validate_enum POST_CREATE_ENV_TOOL auto micromamba conda
    if [[ "${POST_CREATE_ENV_TOOL}" != "auto" && "${POST_CREATE_ENV_TOOL}" != "micromamba" ]]; then
      log_info "Micromamba env: skipped (POST_CREATE_ENV_TOOL=${POST_CREATE_ENV_TOOL})"
      env_micromamba_restore
      return 0
    fi
  fi

  # Legacy skip toggle (keep behavior)
  default_var SKIP_MICROMAMBA "0"
  if common_is_true "$SKIP_MICROMAMBA"; then
    log_info "SKIP_MICROMAMBA=1 -> skipping micromamba"
    env_micromamba_restore
    return 0
  fi

  # ---- configuration (defaults consistent with orchestrator) ----
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
    log_info "env_micromamba: ENV_FILE override -> ENV_FILE_RUNTIME=$ENV_FILE_RUNTIME"
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

  # https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
  default_var MICROMAMBA_ALLOW_INSTALL "1"
  default_var MICROMAMBA_INSTALL_MODE "api"
  default_var MAMBA_EXE ""
  default_var MAMBA_ROOT_PREFIX "$HOME/micromamba"
  default_var MICROMAMBA_ENV_ACTION "ensure"
  default_var MICROMAMBA_PRUNE "0"
  default_var MICROMAMBA_PIN_PYTHON "1"

  common_validate_enum MICROMAMBA_ENV_ACTION none ensure create update
  common_validate_enum MICROMAMBA_INSTALL_MODE api script

  mkdir -p -- "$MAMBA_ROOT_PREFIX"
  export MAMBA_ROOT_PREFIX

  # ---- helpers ----
  # https://mamba.readthedocs.io/en/latest/user_guide/concepts.html#root-prefix
  # https://mamba.readthedocs.io/en/latest/user_guide/configuration.html#configuration
  # The location of the micromamba executable depends on your installation method:
  # Automatic installation script: The executable is typically installed to ~/.local/bin/micromamba (Linux/macOS)
  # or %LOCALAPPDATA%\micromamba\micromamba.exe (Windows).
  _bin_dir() {
    if [[ -n "${MAMBA_EXE:-}" ]]; then
      printf '%s\n' "$MAMBA_EXE"; return 0
    fi
    if [[ -w "/usr/local/bin" ]]; then
      printf '%s\n' "/usr/local/bin"
    else
      printf '%s\n' "$HOME/.local/bin"
    fi
  }

  _ensure_tools_for_install() {
    (has_cmd curl || has_cmd wget) || log_error "micromamba install requires curl or wget"
    has_cmd tar || log_error "micromamba install requires tar"
  }

  _install_via_api() {
    _ensure_tools_for_install
    local url platform bin_dir tmp
    platform="$(micromamba_api_platform)"
    # https://github.com/mamba-org/micromamba-releases/releases
    # https://github.com/mamba-org/micromamba-releases/releases/tag/2.0.5-0
    # RELEASE_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${VERSION}/micromamba-${PLATFORM}-${ARCH}"
    # https://github.com/mamba-org/micromamba-releases/releases/download/2.0.5-0/micromamba-linux-64
    # https://anaconda.org/conda-forge/micromamba/2.0.5/download/linux-64/micromamba-2.0.5-0.tar.bz2
    # curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvjf bin/micromamba
    url="$(micromamba_api_url)"
    bin_dir="$(_bin_dir)"
    mkdir -p -- "$bin_dir"

    tmp="$(mktemp_dir micromamba)"
    setup_traps "$tmp"

    log_info "Installing micromamba (${platform}) -> ${bin_dir}/micromamba"
    # -f — fail on HTTP errors (4xx/5xx), so a bad URL gives a real non-zero exit instead of writing an error page to disk.
    # -s — silent mode, suppresses the progress meter (good for keeping CI logs clean).
    # -S — show error anyway, even with -s on. This is the one I should have included before.
    # -L — follow redirects. GitHub release asset URLs redirect through object storage, so this is required (your original command already had it).
    if has_cmd curl; then
      (cd -- "$tmp" && curl -fsSL "$url" | tar -xvjf - "bin/micromamba")
    else
      (cd -- "$tmp" && wget -qO- "$url" | tar -xvjf - "bin/micromamba")
    fi

    [[ -f "$tmp/bin/micromamba" ]] || log_error "micromamba binary not found after extraction"

    if has_cmd install; then
      install -m 0755 "$tmp/bin/micromamba" "$bin_dir/micromamba"
    else
      cp "$tmp/bin/micromamba" "$bin_dir/micromamba"
      chmod 0755 "$bin_dir/micromamba"
    fi
    log_success "micromamba installed -> $bin_dir/micromamba"
  }

  _install_via_script() {
    # Official install.sh may prompt; keep it explicit by requiring MICROMAMBA_INSTALL_INTERACTIVE=1
    default_var MICROMAMBA_INSTALL_INTERACTIVE "1"
    common_is_true "$MICROMAMBA_INSTALL_INTERACTIVE" || log_error "script install requires MICROMAMBA_INSTALL_INTERACTIVE=1"
    has_cmd curl || log_error "script install requires curl"
    log_warning "Running interactive micromamba install script"
    # curl -Ls https://micro.mamba.pm/install.sh | bash
    # curl -Ls https://micro.mamba.pm/install.sh | "${SHELL}" || echo "⚠️ micromamba install failed"
    # "${SHELL}" <(curl -L https://micro.mamba.pm/install.sh) || echo "⚠️ micromamba install failed"
    # "${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
    "${SHELL}" <(curl -Ls micro.mamba.pm/install.sh) < /dev/null || bash <(curl -fsSL "https://micro.mamba.pm/install.sh") < /dev/null
  }

  # ──────────────────────────────────────────────────────────────
  # 1. Install micromamba if not already available
  # Need curl or fallback to wget and ps (usually from procps or procps-ng)
  # ──────────────────────────────────────────────────────────────
  # # Install micromamba via official install script silently, only if not installed
  # echo "🔧 Installing or initializing micromamba..."
  # # if ! command -v micromamba &> /dev/null; then
  # if ! command -v micromamba >/dev/null 2>&1; then
  #   echo "➡️  micromamba not found, attempting install..."
  #   # if command -v curl &> /dev/null; then
  #   if command -v curl >/dev/null 2>&1; then
  #     # curl -Ls https://micro.mamba.pm/install.sh | bash
  #     # curl -Ls https://micro.mamba.pm/install.sh | "${SHELL}" || echo "⚠️ micromamba install failed"
  #     # "${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
  #     "${SHELL}" <(curl -Ls micro.mamba.pm/install.sh) < /dev/null
  #   # elif command -v wget &> /dev/null; then
  #   elif command -v wget >/dev/null 2>&1; then
  #     wget -qO- https://micro.mamba.pm/install.sh | bash
  #   else
  #     echo "❌ ERROR: Neither curl nor wget is available. Please install one to proceed."
  #     # return 1 2>/dev/null || exit 0
  #     # exit 1
  #   fi
  # else
  #   echo "✅ micromamba is already installed."
  # fi
  _ensure_micromamba() {
    if has_cmd micromamba; then return 0; fi
    if [[ -x "$(_bin_dir)/micromamba" ]]; then
      export PATH="$(_bin_dir):$PATH"
      hash -r 2>/dev/null || true
      has_cmd micromamba && return 0
    fi

    if ! common_is_true "$MICROMAMBA_ALLOW_INSTALL"; then
      log_error "micromamba not found and MICROMAMBA_ALLOW_INSTALL=0"
    fi

    case "$MICROMAMBA_INSTALL_MODE" in
      api) _install_via_api ;;
      script) _install_via_script ;;
    esac

    export PATH="$(_bin_dir):$PATH"
    hash -r 2>/dev/null || true
    has_cmd micromamba || log_error "micromamba still not available after install"
  }

  _env_exists() {
    local name="$1"
    micromamba env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq -- "$name"
  }

  # ── Layered apply helpers ──────────────────────────────────────────────────
  # base (always) → runtime (always) → devel (only when VARIANT=devel).
  _micromamba_update_layered() {
    local env_n="$1"
    local do_prune=0
    common_is_true "${MICROMAMBA_PRUNE:-0}" && do_prune=1
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
      micromamba env update -n "$env_n" -f "$f" --yes $prune_flag
    done
  }

  # Create a brand-new env: optionally seed the requested Python version, then layer
  # base → runtime → (devel only when VARIANT=devel).
  _micromamba_create_layered() {
    local env_n="$1"
    [[ -f "$ENV_FILE_BASE" ]] || log_error "ENV_FILE_BASE not found: $ENV_FILE_BASE"
    # Seed the env with the requested Python so ENV_NAME (pyNNN) matches the actual
    # interpreter. The yml layers omit a python pin, so without this the solver would
    # pick its own default. Opt out with MICROMAMBA_PIN_PYTHON=0.
    if common_is_true "${MICROMAMBA_PIN_PYTHON:-1}" && [[ -n "${PYTHON_VERSION:-}" ]]; then
      log_info "Creating env '$env_n' with python=${PYTHON_VERSION}, then layering base: $ENV_FILE_BASE"
      micromamba create -n "$env_n" "python=${PYTHON_VERSION}" --yes
      micromamba env update -n "$env_n" -f "$ENV_FILE_BASE" --yes
    else
      log_info "Creating env '$env_n' from base: $ENV_FILE_BASE"
      micromamba env create -n "$env_n" -f "$ENV_FILE_BASE" --yes
    fi

    if [[ -f "$ENV_FILE_RUNTIME" ]]; then
      log_info "Applying runtime layer: $ENV_FILE_RUNTIME"
      micromamba env update -n "$env_n" -f "$ENV_FILE_RUNTIME" --yes
    else
      log_warning "ENV_FILE_RUNTIME not found: $ENV_FILE_RUNTIME"
    fi

    if [[ "$VARIANT" == "devel" ]]; then
      if [[ -f "$ENV_FILE_DEVEL" ]]; then
        log_info "Applying devel layer (VARIANT=devel): $ENV_FILE_DEVEL"
        micromamba env update -n "$env_n" -f "$ENV_FILE_DEVEL" --yes
      else
        log_warning "VARIANT=devel but ENV_FILE_DEVEL not found: $ENV_FILE_DEVEL"
      fi
    fi
  }

  # ---- run ----
  _ensure_micromamba

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

  ## ✅ Initialize micromamba shell integration for bash (auto-detect install path)
  ## micromamba shell init -s bash -p ~/micromamba
  micromamba shell init -s "$(basename $SHELL)" \
    || echo "⚠️ Failed to initialize micromamba hooks"

  ## ✅ Ensure available
  ## echo micromamba shell hook --shell "$(basename $SHELL)"
  ## Fallback to bash if SHELL is unset or unknown
  eval "$(micromamba shell hook --shell $(basename ${SHELL:-/bin/bash}))" \
    || echo "⚠️ Failed to enable micromamba shell hook"

  # Re-source shell config to ensure activation takes effect
  # shellcheck disable=SC1090
  # . ~/."$(basename $SHELL)"rc || true  # ~/.bashrc or ~/.zshrc for zsh
  source ~/."$(basename $SHELL)"rc || echo "⚠️ Failed to source $SHELL_RC"

  echo "Creating micromamba base environment: $ENV_NAME"
  micromamba install -n base python="${PYTHON_VERSION:-$PY_VERSION}" ipykernel pip -y || true

  [[ -f "$ENV_FILE_BASE" ]] || log_error "ENV_FILE_BASE not found: $ENV_FILE_BASE"
  [[ -f "$ENV_FILE_RUNTIME" ]] || log_warning "ENV_FILE_RUNTIME not found: $ENV_FILE_RUNTIME"

  case "$MICROMAMBA_ENV_ACTION" in
    none)
      log_info "MICROMAMBA_ENV_ACTION=none -> no env action"
      ;;
    create)
      if _env_exists "$ENV_NAME"; then
        log_info "Env exists: $ENV_NAME (create -> no-op)"
      else
        _micromamba_create_layered "$ENV_NAME"
      fi
      ;;
    update)
      _env_exists "$ENV_NAME" || log_error "Env not found: $ENV_NAME (update)"
      log_info "Updating env '$ENV_NAME' (VARIANT=$VARIANT)"
      _micromamba_update_layered "$ENV_NAME"
      ;;
    ensure)
      if _env_exists "$ENV_NAME"; then
        log_info "Env exists -> update ($ENV_NAME, VARIANT=$VARIANT)"
        _micromamba_update_layered "$ENV_NAME"
      else
        _micromamba_create_layered "$ENV_NAME"
      fi
      ;;
  esac

  export POST_CREATE_ENV_TOOL_SELECTED="micromamba"
  export POST_CREATE_ENV_READY="1"
  export ENV_NAME="$ENV_NAME"
  log_success "Micromamba env ready: $ENV_NAME (VARIANT=$VARIANT)"

  env_micromamba_restore

  # # Register envs directory to ".condarc" for better discovery
  # # Configure micromamba envs directory to simplify env discovery by conda/micromamba
  # # Enables users to activate environment without having to specify the full path
  # mkdir -p ~/micromamba/envs "/opt/conda" || true
  # # echo "envs_dirs:
  # #   - ${HOME:-~/}/micromamba/envs" > /opt/conda/.condarc
  # cat <<EOF > "/opt/conda/.condarc" || echo "⚠️ /opt/conda/.condarc: Permission denied"
  # envs_dirs:
  #   - ~/micromamba/envs
  # EOF

  return 0
}

env_micromamba_main "$@"
env_micromamba_exit_or_return $?
