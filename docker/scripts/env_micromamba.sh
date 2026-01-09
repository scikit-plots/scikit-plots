#!/usr/bin/env bash
# docker/scripts/env_micromamba.sh
# ===============================================================
# Micromamba environment setup (Bash-contract)
# ===============================================================
# USER NOTES
# - This script is intended for Dev Containers / Docker postCreate usage.
# - Default behavior is STRICT + LIGHTWEIGHT:
#     - will NOT install OS packages
#     - will NOT modify shell rc files
#     - will NOT install micromamba unless explicitly allowed
#     - will NOT install extra packages into base unless explicitly allowed
#
# - Recommended usage (postCreate):
#     bash docker/scripts/env_micromamba.sh
#
# - Optional installs:
#     MICROMAMBA_ALLOW_INSTALL=1 MICROMAMBA_INSTALL_MODE=api bash docker/scripts/env_micromamba.sh
#
# DEV NOTES
# - Bash-only by contract.
# - Safe to source: sourcing defines functions only; no traps/options are applied.
# - The "work" happens in env_micromamba_main; executed script calls it automatically.
#
# CONFIG (explicit flags)
# - SKIP_MICROMAMBA=1
# - MICROMAMBA_ALLOW_INSTALL=1
# - MICROMAMBA_INSTALL_MODE=api|script
# - MICROMAMBA_INSTALL_INTERACTIVE=1          (required for script mode)
# - MICROMAMBA_BIN_DIR=$HOME/.local/bin
# - MAMBA_ROOT_PREFIX=$HOME/micromamba
#
# - PY_VERSION=3.11
# - ENV_NAME=py311
# - ENV_FILE=$REPO_ROOT/environment.yml
#
# - MICROMAMBA_SHELL_INIT=1                   (modifies rc via micromamba)
# - MICROMAMBA_ADD_AUTO_ACTIVATE=1            (appends snippet to MICROMAMBA_RC_FILE)
# - MICROMAMBA_RC_FILE=$HOME/.bashrc
#
# - MICROMAMBA_VERIFY=1                       (runs python -V inside env)
# - MICROMAMBA_CLEAN=1                        (cleans caches)
#
# - MICROMAMBA_CONFIGURE_BASE=1               (installs PY_VERSION/ipykernel/pip into base)
# - MICROMAMBA_WRITE_CONDARC=1                (writes /opt/conda/.condarc if permitted)
#
# - POST_CREATE_STRICT=1                      (warnings become errors)
# ===============================================================

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

env_micromamba_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }

env_micromamba_exit_or_return() {
  local rc="${1:-0}"
  if env_micromamba_is_sourced; then return "$rc"; else exit "$rc"; fi
}

env_micromamba_is_true() {
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac
}

env_micromamba_has_cmd() { command -v "$1" >/dev/null 2>&1; }

env_micromamba_script_dir() { ( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P ); }

env_micromamba_detect_os() {
  local s; s="$(uname -s 2>/dev/null || printf '%s' unknown)"
  case "$s" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows-gitbash" ;;
    *) echo "unknown" ;;
  esac
}

env_micromamba_detect_arch() {
  local m; m="$(uname -m 2>/dev/null || printf '%s' unknown)"
  m="$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')"
  case "$m" in
    x86_64|amd64|*amd64*) echo "x86_64" ;;
    aarch64|arm64)        echo "arm64" ;;
    *)                    echo "unknown" ;;
  esac
}

env_micromamba_api_platform() {
  [[ -n "${MICROMAMBA_API_PLATFORM:-}" ]] && { printf '%s\n' "$MICROMAMBA_API_PLATFORM"; return 0; }

  local os arch
  os="$(env_micromamba_detect_os)"
  arch="$(env_micromamba_detect_arch)"

  [[ "$os" != "unknown" && "$arch" != "unknown" ]] || {
    echo "[ERROR] Cannot detect platform (os=$os arch=$arch). Set MICROMAMBA_API_PLATFORM." >&2
    return 2
  }

  case "$os/$arch" in
    linux/x86_64) echo "linux-64" ;;
    linux/arm64)  echo "linux-aarch64" ;;
    macos/x86_64) echo "osx-64" ;;
    macos/arm64)  echo "osx-arm64" ;;
    windows-gitbash/*) echo "win-64" ;;
    *) echo "[ERROR] No mapping for os=$os arch=$arch (set MICROMAMBA_API_PLATFORM)" >&2; return 2 ;;
  esac
}

env_micromamba_api_url() {
  local p; p="$(env_micromamba_api_platform)" || return $?
  # documented magic URL returns latest tarball stream :contentReference[oaicite:3]{index=3}
  printf '%s\n' "https://micro.mamba.pm/api/micromamba/${p}/latest"
}

env_micromamba_default_bin_dir() {
  if [[ -n "${MICROMAMBA_BIN_DIR:-}" ]]; then
    printf '%s\n' "$MICROMAMBA_BIN_DIR"
    return 0
  fi
  # deterministic: prefer /usr/local/bin if writable, else ~/.local/bin
  if [[ -w "/usr/local/bin" ]]; then
    printf '%s\n' "/usr/local/bin"
  else
    printf '%s\n' "$HOME/.local/bin"
  fi
}

env_micromamba_install_via_api() {
  # runs in a subshell so temp cleanup traps do not clobber caller traps
  (
    set -Eeuo pipefail
    local platform url bin_dir tmp
    platform="$(env_micromamba_api_platform)"
    url="$(env_micromamba_api_url)"
    bin_dir="$(env_micromamba_default_bin_dir)"
    mkdir -p -- "$bin_dir"

    env_micromamba_has_cmd tar || { echo "[ERROR] tar is required" >&2; exit 2; }
    (env_micromamba_has_cmd curl || env_micromamba_has_cmd wget) || { echo "[ERROR] need curl or wget" >&2; exit 2; }

    tmp="$(mktemp -d 2>/dev/null)" || { echo "[ERROR] mktemp -d failed" >&2; exit 2; }
    trap 'rm -rf -- "$tmp" 2>/dev/null || true' EXIT

    echo "[INFO] Downloading micromamba (${platform})" >&2

    # stream extract (official docs pattern) :contentReference[oaicite:4]{index=4}
    if env_micromamba_has_cmd curl; then
      (cd -- "$tmp" && curl -fsSL "$url" | tar -xvj "bin/micromamba")
    else
      (cd -- "$tmp" && wget -qO- "$url" | tar -xvj "bin/micromamba")
    fi

    if [[ ! -f "$tmp/bin/micromamba" ]]; then
      echo "[ERROR] extracted micromamba not found" >&2
      exit 2
    fi

    install -m 0755 "$tmp/bin/micromamba" "$bin_dir/micromamba" 2>/dev/null || {
      cp "$tmp/bin/micromamba" "$bin_dir/micromamba" && chmod 0755 "$bin_dir/micromamba"
    }

    echo "[SUCCESS] micromamba installed -> $bin_dir/micromamba" >&2
  )
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
  env_micromamba_has_cmd curl || echo "[ERROR] curl is required for script install"
  echo "[INFO] Running official micromamba installer (interactive)"
  bash <(curl -fsSL "https://micro.mamba.pm/install.sh")
  # Installer may place micromamba in ~/.local/bin; make it visible in this process.
  export PATH="$HOME/.local/bin:$PATH"
}

env_micromamba_ensure_available() {
  if env_micromamba_has_cmd micromamba; then
    return 0
  fi

  # default: install if missing (you requested that)
  if ! env_micromamba_is_true "${MICROMAMBA_ALLOW_INSTALL:-1}"; then
    echo "[ERROR] micromamba not found and MICROMAMBA_ALLOW_INSTALL=0" >&2
    return 2
  fi

  case "${MICROMAMBA_INSTALL_MODE:-api}" in
    api) env_micromamba_install_via_api ;;
    script) install_micromamba_via_script ;;
    *) echo "[ERROR] MICROMAMBA_INSTALL_MODE must be 'api|script' (or implement others)" >&2; return 2 ;;
  esac

  local bin_dir; bin_dir="$(env_micromamba_default_bin_dir)"
  export PATH="$bin_dir:$PATH"
  hash -r 2>/dev/null || true

  env_micromamba_has_cmd micromamba || { echo "[ERROR] micromamba still not in PATH after install" >&2; return 2; }
}

env_micromamba_env_exists() {
  local name="$1"
  micromamba env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq -- "$name"
}

env_micromamba_main() {
  # Save caller shell options/trap, restore inline (NO trap RETURN)
  local __old_set __old_trap_err
  __old_set="$(set +o)"
  __old_trap_err="$(trap -p ERR || true)"

  set -Eeuo pipefail
  trap 'rc=$?; echo "[ERROR] env_micromamba.sh failed at line $LINENO: ${BASH_COMMAND-<cmd>} (exit=$rc)" >&2; return $rc' ERR

  local SCRIPT_DIR REPO_ROOT
  SCRIPT_DIR="$(env_micromamba_script_dir)"
  REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

  if env_micromamba_is_true "${SKIP_MICROMAMBA:-0}"; then
    echo "[INFO] SKIP_MICROMAMBA=1 -> skipping micromamba" >&2
    eval "$__old_set"; [[ -n "$__old_trap_err" ]] && eval "$__old_trap_err" || trap - ERR
    return 0
  fi

  local PY_VERSION ENV_NAME ENV_FILE MAMBA_ROOT_PREFIX ACTION
  PY_VERSION="${PY_VERSION:-3.11}"
  ENV_NAME="${ENV_NAME:-py$(printf '%s' "$PY_VERSION" | tr -d '.')}"
  ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
  ACTION="${MICROMAMBA_ENV_ACTION:-ensure}"  # ensure|create|update|none
  export MAMBA_ROOT_PREFIX

  mkdir -p -- "$MAMBA_ROOT_PREFIX"

  env_micromamba_ensure_available

  case "$ACTION" in
    none)
      echo "[INFO] MICROMAMBA_ENV_ACTION=none -> skip env create/update" >&2
      ;;
    create|update|ensure)
      [[ -f "$ENV_FILE" ]] || { echo "[ERROR] ENV_FILE not found: $ENV_FILE" >&2; return 2; }

      if env_micromamba_env_exists "$ENV_NAME"; then
        if [[ "$ACTION" == "update" || "$ACTION" == "ensure" ]]; then
          echo "[INFO] Updating env '$ENV_NAME' from $ENV_FILE" >&2
          micromamba env update -n "$ENV_NAME" -f "$ENV_FILE" --yes
        else
          echo "[INFO] Env exists: $ENV_NAME (create -> no-op)" >&2
        fi
      else
        echo "[INFO] Creating env '$ENV_NAME' from $ENV_FILE" >&2
        micromamba env create -n "$ENV_NAME" -f "$ENV_FILE" --yes
      fi
      ;;
    *)
      echo "[ERROR] Invalid MICROMAMBA_ENV_ACTION=$ACTION (ensure|create|update|none)" >&2
      return 2
      ;;
  esac

  # Restore caller state
  eval "$__old_set"
  if [[ -n "$__old_trap_err" ]]; then eval "$__old_trap_err"; else trap - ERR; fi
  return 0
}

# IMPORTANT: run even when sourced (your orchestrator sources it)
env_micromamba_main "$@"
env_micromamba_exit_or_return $?
