#!/usr/bin/env bash
#
# docker/scripts/env_micromamba.sh
#
# bash docker/scripts/env_micromamba.sh
# MICROMAMBA_ALLOW_INSTALL=1 MICROMAMBA_INSTALL_MODE=api bash docker/scripts/env_micromamba.sh
# MICROMAMBA_ALLOW_INSTALL=1 MICROMAMBA_INSTALL_MODE=script MICROMAMBA_INSTALL_INTERACTIVE=1 \
# bash docker/scripts/env_micromamba.sh
# MICROMAMBA_SHELL_INIT=1 MICROMAMBA_ADD_AUTO_ACTIVATE=1 \
# bash docker/scripts/env_micromamba.sh
# MICROMAMBA_VERIFY=1 bash docker/scripts/env_micromamba.sh
#
# ===============================================================
# Micromamba environment setup (Bash)
# ===============================================================
# USER NOTES
# - This script is intended for Dev Containers / Docker postCreate usage.
# - It is STRICT by default:
#     - no OS package installs
#     - no micromamba install
#     - no shell rc modification
#   Everything "heavy" is opt-in via env vars.
#
# DEV NOTES
# - This is a Bash script (uses pipefail, arrays, ${var,,}, etc.)
# - It may be EXECUTED (postCreate) or SOURCED (interactive shell).
# - Activation only makes sense when sourced; executed scripts cannot
#   persist environment activation into the parent shell.
#
# CONFIG (explicit flags; no silent fallbacks)
# - SKIP_MICROMAMBA=1                : skip entire script
# - MICROMAMBA_ALLOW_INSTALL=1       : allow installing micromamba if missing
# - MICROMAMBA_INSTALL_MODE=api|script
#     api    : download tarball from micro.mamba.pm API (non-interactive)
#     script : run official install.sh (may prompt; requires MICROMAMBA_INSTALL_INTERACTIVE=1)
# - MICROMAMBA_INSTALL_INTERACTIVE=1 : allow running install.sh (prompts)
#
# - PY_VERSION=3.11                  : used only if you create env from packages (optional)
# - ENV_NAME=py311                   : expected env name (used for checks / optional activation)
# - ENV_FILE=environment.yml         : file used for env creation (default: ./environment.yml)
#
# - MAMBA_ROOT_PREFIX=$HOME/micromamba : root prefix for micromamba (default matches common installer)
# - MICROMAMBA_SHELL_INIT=1          : run `micromamba shell init ...` (modifies rc)
# - MICROMAMBA_RC_FILE=~/.bashrc     : rc file target for optional activation snippet
# - MICROMAMBA_ADD_AUTO_ACTIVATE=1   : append auto-activate ENV_NAME in rc (idempotent)
# - MICROMAMBA_VERIFY=1              : verify env by running `python -V` inside it
# - MICROMAMBA_CLEAN=1               : clean micromamba + caches (optional)
#
# - POST_CREATE_STRICT=1             : treat warnings as errors (optional)
# ===============================================================

set -Eeuo pipefail

# ---------- Source shared POSIX library (optional, but preferred) ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

COMMON_SH="${COMMON_SH:-$SCRIPT_DIR/lib/common.sh}"
if [[ -f "$COMMON_SH" ]]; then
  # common.sh is POSIX; safe to source from bash.
  # It sets `set -eu` internally; we re-apply bash strict mode after.
  # shellcheck source=/dev/null
  . "$COMMON_SH"
  set -Eeuo pipefail
else
  # Minimal fallback (keeps this script runnable standalone)
  log() { printf '%s\n' "$*" >&2; }
  log_info() { log "[INFO] $*"; }
  log_warning() { log "[WARNING] $*"; }
  log_error() { log "[ERROR] $*"; exit 1; }
  has_cmd() { command -v "$1" >/dev/null 2>&1; }
fi

STRICT="${POST_CREATE_STRICT:-0}"

maybe_die() {
  # Usage: maybe_die "message"
  if [[ "$STRICT" == "1" ]]; then
    log_error "$@"
  else
    log_warning "$@"
  fi
}

## Normalize to lowercase and handle multiple truthy values
## value=$(echo "$SKIP_CONDA" | tr '[:upper:]' '[:lower:]')
## case "$(printf '%s' $SKIP_CONDA | tr '[:upper:]' '[:lower:]')" in
is_true() {
  # Usage: is_true "$VAL"
  # Returns 0 for: 1,true,yes,on  (case-insensitive)
  local v="${1:-}"
  v="${v,,}"
  case "$v" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

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

# ---------- Skip gate ----------
SKIP_MICROMAMBA="${SKIP_MICROMAMBA:-0}"
if is_true "$SKIP_MICROMAMBA"; then
  log_status "SKIP_MICROMAMBA=1 -> skipping micromamba setup"
  exit_or_return 0
fi

# ---------- Config defaults (documented) ----------
PY_VERSION="${PY_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"

# Matches common micromamba installer default prefix ("~/micromamba")
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export MAMBA_ROOT_PREFIX

MICROMAMBA_ALLOW_INSTALL="${MICROMAMBA_ALLOW_INSTALL:-0}"
MICROMAMBA_INSTALL_MODE="${MICROMAMBA_INSTALL_MODE:-api}"  # api|script
MICROMAMBA_INSTALL_INTERACTIVE="${MICROMAMBA_INSTALL_INTERACTIVE:-0}"

MICROMAMBA_SHELL_INIT="${MICROMAMBA_SHELL_INIT:-0}"
MICROMAMBA_RC_FILE="${MICROMAMBA_RC_FILE:-$HOME/.bashrc}"
MICROMAMBA_ADD_AUTO_ACTIVATE="${MICROMAMBA_ADD_AUTO_ACTIVATE:-0}"

MICROMAMBA_VERIFY="${MICROMAMBA_VERIFY:-0}"
MICROMAMBA_CLEAN="${MICROMAMBA_CLEAN:-0}"

# ---------- Error diagnostics ----------
trap 'log_error "Failed at line $LINENO: $BASH_COMMAND"' ERR

# ===============================================================
# Helpers
# ===============================================================

ensure_dir() {
  local d="$1"
  [[ -n "$d" ]] || log_error "ensure_dir: empty path"
  [[ -d "$d" ]] || mkdir -p -- "$d"
}

append_line_once() {
  # Idempotently append a line to a file.
  # Usage: append_line_once FILE "line"
  local file="$1"
  local line="$2"
  ensure_dir "$(dirname -- "$file")"
  touch -- "$file"
  if ! grep -Fqx -- "$line" "$file" 2>/dev/null; then
    printf '%s\n' "$line" >> "$file"
  fi
}

install_micromamba_via_api() {
  # Non-interactive install:
  # - downloads tar.bz2 from micro.mamba.pm
  # - extracts micromamba binary into MICROMAMBA_BIN_DIR (default ~/.local/bin)
  #
  # Requires: curl or wget, tar
  # bzip2 is not pre-checked; we attempt extraction and fail with guidance if needed.

  local url bin_dir tmp tarball platform

  platform="$(micromamba_api_platform)"
  url="https://micro.mamba.pm/api/micromamba/${platform}/latest"

  has_cmd tar || log_error "tar is required for micromamba api install"

  bin_dir="${MICROMAMBA_BIN_DIR:-$HOME/.local/bin}"
  ensure_dir "$bin_dir"

  tmp="$(mktemp_dir micromamba 2>/dev/null || true)"
  [[ -n "$tmp" ]] || tmp="$(mktemp -d 2>/dev/null || true)"
  [[ -n "$tmp" ]] || log_error "Failed to create temp dir"

  tarball="$tmp/micromamba.tar.bz2"

  log_info "Downloading micromamba (${platform}): $url"
  if has_cmd curl; then
    curl -fsSL "$url" -o "$tarball"
  elif has_cmd wget; then
    wget -qO "$tarball" "$url"
  else
    log_error "Need curl or wget to download micromamba"
  fi

  # Try extraction; if it fails, tell the user what to install (strict).
  if ! tar -xjf "$tarball" -C "$tmp" 2>/dev/null; then
    log_error "Failed to extract micromamba tarball (tar -xjf). Install bzip2 support (e.g., apt-get install -y bzip2) or use a base image with bzip2 enabled."
  fi

  # Install binary (prefer `install`, fallback to cp+chmod)
  if [[ -f "$tmp/bin/micromamba" ]]; then
    if has_cmd install; then
      install -m 0755 "$tmp/bin/micromamba" "$bin_dir/micromamba"
    else
      cp "$tmp/bin/micromamba" "$bin_dir/micromamba"
      chmod 0755 "$bin_dir/micromamba"
    fi
  elif [[ -f "$tmp/micromamba" ]]; then
    if has_cmd install; then
      install -m 0755 "$tmp/micromamba" "$bin_dir/micromamba"
    else
      cp "$tmp/micromamba" "$bin_dir/micromamba"
      chmod 0755 "$bin_dir/micromamba"
    fi
  else
    log_error "micromamba binary not found after extraction"
  fi

  cleanup_dir "$tmp" 2>/dev/null || true
  export PATH="$bin_dir:$PATH"
  log_success "Installed micromamba to: $bin_dir/micromamba"
}

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
install_micromamba_via_script() {
  # Runs the official install.sh (may prompt).
  # This is intentionally blocked unless MICROMAMBA_INSTALL_INTERACTIVE=1.
  if ! is_true "$MICROMAMBA_INSTALL_INTERACTIVE"; then
    log_error "MICROMAMBA_INSTALL_MODE=script requires MICROMAMBA_INSTALL_INTERACTIVE=1 (install.sh prompts)"
  fi
  has_cmd curl || log_error "curl is required for script install"
  log_info "Running official micromamba installer (interactive)"
  bash <(curl -fsSL "https://micro.mamba.pm/install.sh")
  # Installer may place micromamba in ~/.local/bin; make it visible in this process.
  export PATH="$HOME/.local/bin:$PATH"
}

ensure_micromamba_available() {
  if has_cmd micromamba; then
    return 0
  fi

  if ! is_true "$MICROMAMBA_ALLOW_INSTALL"; then
    log_error "micromamba not found. Set MICROMAMBA_ALLOW_INSTALL=1 to allow install, or preinstall micromamba (recommended)."
  fi

  case "$MICROMAMBA_INSTALL_MODE" in
    api) install_micromamba_via_api ;;
    script) install_micromamba_via_script ;;
    *) log_error "Unknown MICROMAMBA_INSTALL_MODE: $MICROMAMBA_INSTALL_MODE (use api|script)" ;;
  esac

  has_cmd micromamba || log_error "micromamba install did not make micromamba available in PATH"
}

micromamba_env_exists() {
  local name="$1"
  micromamba env list | awk 'NF>0 && $1 !~ /^#/ {print $1}' | grep -Fxq -- "$name"
}

create_env_from_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || log_error "ENV_FILE not found: $env_file"
  log_info "Creating environment from file: $env_file"
  micromamba env create -f "$env_file" --yes
}

shell_init_optional() {
  if ! is_true "$MICROMAMBA_SHELL_INIT"; then
    return 0
  fi
  log_info "Initializing micromamba shell integration (may modify rc files)"
  micromamba shell init -s bash -p "$MAMBA_ROOT_PREFIX"
}

add_auto_activate_optional() {
  if ! is_true "$MICROMAMBA_ADD_AUTO_ACTIVATE"; then
    return 0
  fi

  # Canonical: append a small guarded snippet to rc.
  local marker_begin marker_end
  marker_begin="# >>> micromamba auto-activate (managed) >>>"
  marker_end="# <<< micromamba auto-activate (managed) <<<"

  log_info "Configuring auto-activation in: $MICROMAMBA_RC_FILE"
  # eval "$(micromamba shell hook --shell $(basename ${SHELL:-/bin/bash}))" || echo "⚠️ Failed to enable micromamba shell hook"
  append_line_once "$MICROMAMBA_RC_FILE" "$marker_begin"
  append_line_once "$MICROMAMBA_RC_FILE" 'if command -v micromamba >/dev/null 2>&1; then'
  append_line_once "$MICROMAMBA_RC_FILE" '  eval "$(micromamba shell hook --shell bash)"'
  append_line_once "$MICROMAMBA_RC_FILE" "  micromamba activate \"${ENV_NAME}\" >/dev/null 2>&1 || true"
  append_line_once "$MICROMAMBA_RC_FILE" 'fi'
  append_line_once "$MICROMAMBA_RC_FILE" "$marker_end"
}

verify_env_optional() {
  if ! is_true "$MICROMAMBA_VERIFY"; then
    return 0
  fi
  log_info "Verifying environment '$ENV_NAME' (python -V)"
  micromamba run -n "$ENV_NAME" python -V
}

clean_optional() {
  if ! is_true "$MICROMAMBA_CLEAN"; then
    return 0
  fi

  log_info "Cleaning micromamba caches"
  micromamba clean --all --yes || true

  # Optional general caches (only if tools exist)
  if has_cmd pip; then pip cache purge || true; fi
  rm -rf "${HOME}/.cache" 2>/dev/null || true
}

# ===============================================================
# Main
# ===============================================================

log_env_summary 2>/dev/null || true

ensure_dir "$MAMBA_ROOT_PREFIX"

ensure_micromamba_available

# Shell init is optional (modifies rc); keep opt-in
shell_init_optional

## Also Configure base
micromamba install -n base python="$PY_VERSION" ipykernel pip -y || true

# Env creation (strict)
if micromamba_env_exists "$ENV_NAME"; then
  log_success "Environment exists: $ENV_NAME"
else
  if [[ -f "$ENV_FILE" ]]; then
    create_env_from_file "$ENV_FILE"
  else
    # Strict behavior: fail unless caller explicitly opted to skip missing file.
    if is_true "${MICROMAMBA_ALLOW_MISSING_ENV_FILE:-0}"; then
      maybe_die "ENV_FILE missing ($ENV_FILE) and MICROMAMBA_ALLOW_MISSING_ENV_FILE=1 -> skipping env creation"
      exit_or_return 0
    else
      log_error "ENV_FILE not found: $ENV_FILE (set ENV_FILE=... or create it)"
    fi
  fi

  # After creation, enforce expected ENV_NAME
  if micromamba_env_exists "$ENV_NAME"; then
    log_success "Environment created: $ENV_NAME"
  else
    log_error "Environment file created an unexpected name. Expected ENV_NAME='$ENV_NAME'. Update ENV_NAME or environment.yml."
  fi
fi

# Optional: add auto-activate snippet to rc
add_auto_activate_optional

# Optional: verify
verify_env_optional

# Optional: clean
clean_optional

# Register envs directory to ".condarc" for better discovery
# Configure micromamba envs directory to simplify env discovery by conda/micromamba
# Enables users to activate environment without having to specify the full path
mkdir -p ~/micromamba/envs "/opt/conda" || true
# echo "envs_dirs:
#   - ${HOME:-~/}/micromamba/envs" > /opt/conda/.condarc
cat <<EOF > "/opt/conda/.condarc" || echo "⚠️ /opt/conda/.condarc: Permission denied"
envs_dirs:
  - ~/micromamba/envs
EOF

log_success "Micromamba setup complete."
log_info "Next steps:"
log_info "  - Open a new shell, or: source \"$MICROMAMBA_RC_FILE\""
log_info "  - Activate: micromamba activate \"$ENV_NAME\""
