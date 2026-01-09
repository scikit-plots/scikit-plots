#!/usr/bin/env bash
# docker/scripts/common.sh
# ===============================================================
# common.sh — Bash utilities (BASH-CONTRACT)
# ===============================================================
# USER NOTES
# - Source from bash scripts:
#     . "/abs/path/to/docker/scripts/common.sh"
# - This file MUST be sourced from bash (not sh/dash).
# - No side effects on source: it does NOT set -e/-u, does NOT set umask,
#   does NOT set traps, and does NOT install packages.
#
# DEV NOTES (contract)
# - Bash-only: may use [[ ]], local, printf -v, BASH_SOURCE, etc.
# - Provide functions only; orchestrators decide strict mode, traps, installs.
#
# ENV VARS (library controls)
# - COMMON_TRACE=1            : enable `set -x` (ONLY when caller uses common_enable_strict)
# - COMMON_DEBUG=1            : enable debug logs
# - COMMON_LOG_TIMESTAMP=0|1  : default 1 (UTC ISO8601)
# - COMMON_COLOR_MODE=never|auto|always : default "never"
# - NO_COLOR=1                : disable color (community standard)
# - FORCE_COLOR=1             : force color (common convention)
#
# INSTALL (STRICT OPT-IN)
# - COMMON_ALLOW_INSTALL=1    : allow install helpers to run
# - DEBIAN_FRONTEND=noninteractive (recommended in CI/Docker)
# - APT_GET_OPTS="..."        : override apt-get options (default: "-y --no-install-recommends")
# ===============================================================
# \033[1m\033[1;31m[ERROR / FAILURE / DANGER]\033[0m Build failed, exception
# \033[1m\033[1;32m[SUCCESS / PASS]\033[0m Compilation OK, done
# \033[1m\033[1;33m[WARNING / CAUTION]\033[0m Deprecated, risk detected
# \033[1m\033[1;34m[INFO / PROCESSING]\033[0m Current step, notice
# \033[1m\033[1;34m[HIGHLIGHT / SPECIAL]\033[0m Important feature or stage
# \033[1m\033[1;34m[ACTION / STATUS / QUERY]\033[0m User action, pending info
# RESET="${ESC}[0m"
# BOLD="${ESC}[1m"
# ITALIC="${ESC}[3m"
# UNDERLINE="${ESC}[4m"
# BLINK="${ESC}[5m"
# BLACK="${ESC}[1;30m"
# RED="${ESC}[1;31m"
# GREEN="${ESC}[1;32m"
# YELLOW="${ESC}[1;33m"
# BLUE="${ESC}[1;34m"
# MAGENTA="${ESC}[1;35m"
# CYAN="${ESC}[1;36m"
# WHITE="${ESC}[1;37m"
# BRIGHT_BLACK="${ESC}[1;90m"
# BRIGHT_RED="${ESC}[1;91m"
# BRIGHT_GREEN="${ESC}[1;92m"
# BRIGHT_YELLOW="${ESC}[1;93m"
# BRIGHT_BLUE="${ESC}[1;94m"
# BRIGHT_MAGENTA="${ESC}[1;95m"
# BRIGHT_CYAN="${ESC}[1;96m"
# BRIGHT_WHITE="${ESC}[1;97m"

# --- Require bash when sourced ---
if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "[ERROR] common.sh must be sourced from bash (BASH_VERSION is empty)." >&2
  return 2 2>/dev/null || exit 2
fi

# --- Idempotent load guard ---
if [[ "${COMMON_SH__LOADED:-0}" == "1" ]]; then
  return 0
fi
COMMON_SH__LOADED=1

# ===============================================================
# Exit/return semantics (nested-safe)
# ===============================================================

common_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }

common_is_direct_call_from_entry() {
  # True only when a common.sh function is called directly by the executed entry script ($0).
  # If the entry script is sourced, $0 is the interactive shell; we will return instead of exit.
  [[ "${BASH_SOURCE[1]:-}" == "$0" ]]
}

common_exit_or_return() {
  local code="${1:-0}"
  if common_is_direct_call_from_entry; then
    exit "$code"
  else
    return "$code"
  fi
}

# ===============================================================
# Strict mode controls (opt-in)
# ===============================================================

common_enable_strict() {
  # Usage: common_enable_strict [trace=0|1]
  local trace="${1:-0}"
  # set -E  # or: set -o errtrace
  set -Eeuo pipefail
  if [[ "$trace" == "1" ]]; then
    set -x
  fi
}

common_script_dir() {
  # Usage: common_script_dir "/path/to/script.sh"
  local src="${1:-}"
  [[ -n "$src" ]] || { echo "[ERROR] common_script_dir requires a script path" >&2; return 2; }
  ( cd -- "$(dirname -- "$src")" && pwd -P )
}

# ===============================================================
# Logging + Color
# ===============================================================

COMMON_COLOR_MODE="${COMMON_COLOR_MODE:-never}"
if [[ "${NO_COLOR:-0}" == "1" ]]; then COMMON_COLOR_MODE="never"; fi
if [[ "${FORCE_COLOR:-0}" == "1" ]]; then COMMON_COLOR_MODE="always"; fi

COMMON_COLOR=0
case "$COMMON_COLOR_MODE" in
  never)  COMMON_COLOR=0 ;;
  always) COMMON_COLOR=1 ;;
  auto)   [[ -t 2 ]] && COMMON_COLOR=1 || COMMON_COLOR=0 ;;
  *)      COMMON_COLOR=0 ;;
esac

ESC="$(printf '\033')"
RESET=""; BOLD=""
RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
if [[ "$COMMON_COLOR" == "1" ]]; then
  RESET="${ESC}[0m"
  BOLD="${ESC}[1m"
  RED="${ESC}[1;31m"
  GREEN="${ESC}[1;32m"
  YELLOW="${ESC}[1;33m"
  BLUE="${ESC}[1;34m"
  MAGENTA="${ESC}[1;35m"
  CYAN="${ESC}[1;36m"
fi

_common_ts() {
  if [[ "${COMMON_LOG_TIMESTAMP:-1}" == "1" ]]; then
    date -u +"%Y-%m-%dT%H:%M:%SZ"
  else
    printf '%s' ""
  fi
}

log() { printf '%s\n' "$*" >&2; }

log_error_code() {
  local code="${1:-1}"; shift || true
  local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "
  log "${ts}${BOLD}${RED}[ERROR]${RESET} $*"
  common_exit_or_return "$code"
}
log_error()   { log_error_code 1 "$@"; }
log_warning() { local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "; log "${ts}${BOLD}${YELLOW}[WARNING]${RESET} $*"; }
log_info()    { local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "; log "${ts}${BOLD}${BLUE}[INFO]${RESET} $*"; }
log_success() { local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "; log "${ts}${BOLD}${GREEN}[SUCCESS]${RESET} $*"; }
log_status()  { local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "; log "${ts}${BOLD}${CYAN}[STATUS]${RESET} $*"; }
log_highlight(){ local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "; log "${ts}${BOLD}${MAGENTA}[HIGHLIGHT]${RESET} $*"; }

log_debug() {
  if [[ "${COMMON_DEBUG:-0}" == "1" ]]; then
    local ts="$(_common_ts)"; [[ -n "$ts" ]] && ts="$ts "
    log "${ts}${BOLD}${CYAN}[DEBUG]${RESET} $*"
  fi
}

# ===============================================================
# Basic utilities (command/file/dir, truthy, enums)
# ===============================================================

has_cmd() { command -v "$1" >/dev/null 2>&1; }
need_cmd()  { has_cmd "$1" || log_error "Missing required command: $1"; }
need_file() { [[ -f "$1" ]] || log_error "Missing required file: $1"; }
need_dir()  { [[ -d "$1" ]] || log_error "Missing required dir: $1"; }

common_is_true() {
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off|"") return 1 ;;
    *) return 1 ;;
  esac
}

common_validate_enum() {
  local var="$1"; shift || true
  local val="${!var:-}"
  local a
  for a in "$@"; do
    [[ "$val" == "$a" ]] && return 0
  done
  log_error "Invalid $var='$val' (allowed: $*)"
}

common_print_config_safe() {
  # Usage: common_print_config_safe VAR1 VAR2 ...
  log "---------------- CONFIG (safe) ----------------"
  local k
  for k in "$@"; do
    printf '%s=%s\n' "$k" "${!k-}" >&2
  done
  log "----------------------------------------------"
}

# ===============================================================
# Env helpers (NO project defaults here!)
# ===============================================================

require_env() {
  local var="${1:-}"
  [[ -n "$var" ]] || log_error "require_env requires VAR_NAME"
  [[ "$var" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || log_error "Invalid env var name: $var"
  [[ -n "${!var:-}" ]] || log_error "Missing required env var: $var"
}

default_var() {
  # Bash-safe default assignment
  local var="${1:-}" def="${2:-}"
  [[ -n "$var" ]] || log_error "default_var requires VAR_NAME"
  [[ "$var" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || log_error "Invalid env var name: $var"
  if [[ -z "${!var:-}" ]]; then
    printf -v "$var" '%s' "$def"
  fi
}

load_env_kv() {
  # Strict KEY=VALUE loader (no eval).
  local env_file="${1:-}"
  [[ -n "$env_file" ]] || log_error "load_env_kv requires a path"
  need_file "$env_file"

  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in ""|\#*) continue ;; esac
    case "$line" in *=*) ;; *) log_error "Invalid .env line (must be KEY=VALUE): $line" ;; esac

    key=${line%%=*}
    val=${line#*=}

    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || log_error "Invalid env var name: $key"

    case "$val" in
      \"*\") val=${val#\"}; val=${val%\"} ;;
      \'*\')   val=${val#\'}; val=${val%\'} ;;
    esac

    export "$key=$val"
  done < "$env_file"
}

# ===============================================================
# OS / Arch / Distro
# ===============================================================

detect_os() {
  local s
  s="$(uname -s 2>/dev/null || printf '%s' unknown)"
  case "$s" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows-gitbash" ;;
    *) echo "unknown" ;;
  esac
}

detect_arch() {
  local m
  m="$(uname -m 2>/dev/null || printf '%s' unknown)"
  m="$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')"
  case "$m" in
    x86_64|amd64|*amd64*) echo "x86_64" ;;
    aarch64|arm64)        echo "arm64" ;;
    i386|i486|i586|i686|x86) echo "x86" ;;
    *) echo "unknown" ;;
  esac
}

os_release_get() {
  local key="${1:-}"
  [[ -n "$key" ]] || return 1
  [[ -f /etc/os-release ]] || return 1
  local val
  val="$(sed -n "s/^${key}=//p" /etc/os-release | sed -n '1{p;q;}')"
  [[ -n "$val" ]] || return 1
  case "$val" in
    \"*\") val=${val#\"}; val=${val%\"} ;;
    \'*\')   val=${val#\'}; val=${val%\'} ;;
  esac
  printf '%s\n' "$val"
}

detect_distro_id() { os_release_get ID 2>/dev/null || echo "unknown"; }
detect_distro_version_id() { os_release_get VERSION_ID 2>/dev/null || echo "unknown"; }

is_docker() {
  [[ -f /.dockerenv ]] && return 0
  [[ -r /proc/1/cgroup ]] || return 1
  has_cmd grep || return 1
  grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null
}

timestamp_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ===============================================================
# Path + FS
# ===============================================================

ensure_dir_writable() {
  local d="${1:-}"
  [[ -n "$d" ]] || log_error "ensure_dir_writable requires a path"
  [[ -d "$d" ]] || mkdir -p -- "$d" || log_error "Failed to create dir: $d"
  [[ -w "$d" ]] || log_error "Dir not writable: $d"
}

abspath_existing() {
  local p="${1:-}"
  [[ -n "$p" ]] || log_error "abspath_existing requires a path"
  [[ -e "$p" ]] || log_error "Path does not exist: $p"
  if [[ -d "$p" ]]; then
    (cd -- "$p" && pwd -P) || log_error "Failed to resolve dir: $p"
  else
    local dir base
    dir=${p%/*}
    base=${p##*/}
    [[ "$dir" == "$p" ]] && dir="."
    (cd -- "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base") || log_error "Failed to resolve file: $p"
  fi
}

script_dir_from() {
  local sp d
  sp="${1:-}"
  [[ -n "$sp" ]] || log_error "script_dir_from requires a script path (e.g., \$0)"
  d="$(dirname -- "$sp")" || log_error "dirname failed"
  (cd -- "$d" && pwd -P) || log_error "Failed to resolve script dir"
}

# ===============================================================
# Temp + traps (trap-safe, no locals in traps)
# ===============================================================

mktemp_dir() {
  local prefix="${1:-tmp}"
  local d ts
  if has_cmd mktemp; then
    d="$(mktemp -d "${TMPDIR:-/tmp}/${prefix}.XXXXXXXX" 2>/dev/null)" && { printf '%s\n' "$d"; return 0; }
    d="$(mktemp -d -t "$prefix" 2>/dev/null)" && { printf '%s\n' "$d"; return 0; }
    log_error "mktemp is present but failed to create temp dir"
  fi
  ts="$(date -u +%Y%m%d%H%M%S 2>/dev/null || echo 0)"
  d="${TMPDIR:-/tmp}/${prefix}.$$.$ts"
  mkdir -p -- "$d" || log_error "Failed to create temp dir: $d"
  printf '%s\n' "$d"
}

cleanup_dir() {
  local d="${1:-}"
  [[ -n "$d" && -d "$d" ]] || return 0
  rm -rf -- "$d"
}

common_trap_add() {
  # Usage: common_trap_add EXIT "cmd ..."
  local sig="${1:-}"; shift || true
  local cmd="$*"
  [[ -n "$sig" && -n "$cmd" ]] || return 0

  local prev line prev_cmd new_cmd
  line="$(trap -p "$sig" || true)"
  prev_cmd=""
  if [[ "$line" =~ ^trap\ --\ '(.*)\'\ ${sig}$ ]]; then
    prev_cmd="${BASH_REMATCH[1]}"
  fi

  if [[ -n "$prev_cmd" ]]; then
    new_cmd="${prev_cmd}; ${cmd}"
  else
    new_cmd="${cmd}"
  fi
  trap -- "$new_cmd" "$sig"
}

setup_traps() {
  # Backward-compatible helper.
  # Safe because the directory path is embedded (quoted) into the trap command.
  local d="${1:-}"
  [[ -n "$d" ]] || log_error "setup_traps requires a dir"
  common_trap_add EXIT "cleanup_dir $(printf %q "$d")"
  common_trap_add INT  "cleanup_dir $(printf %q "$d")"
  common_trap_add TERM "cleanup_dir $(printf %q "$d")"
}

# ===============================================================
# Micromamba helpers
# ===============================================================

micromamba_api_platform() {
  if [[ -n "${MICROMAMBA_API_PLATFORM:-}" ]]; then
    printf '%s\n' "$MICROMAMBA_API_PLATFORM"
    return 0
  fi

  local os arch
  os="$(detect_os)"
  arch="$(detect_arch)"

  [[ "$os" != "unknown" && "$arch" != "unknown" ]] || log_error "Cannot detect platform (os=$os arch=$arch). Set MICROMAMBA_API_PLATFORM."

  case "${os}/${arch}" in
    linux/x86_64) echo "linux-64" ;;
    linux/arm64)  echo "linux-aarch64" ;;
    macos/x86_64) echo "osx-64" ;;
    macos/arm64)  echo "osx-arm64" ;;
    windows-gitbash/*) echo "win-64" ;;
    *) log_error "No micromamba API mapping for OS=$os ARCH=$arch (set MICROMAMBA_API_PLATFORM)" ;;
  esac
}

micromamba_api_url() {
  local platform
  platform="$(micromamba_api_platform)"
  printf '%s\n' "https://micro.mamba.pm/api/micromamba/${platform}/latest"
}

# ===============================================================
# Execution wrapper
# ===============================================================

run_checked() {
  log_info "Running: $*"
  "$@" || log_error "Command failed: $*"
}

# ===============================================================
# Hashing
# ===============================================================

sha256_file() {
  local f
  f="${1:-}"
  [[ -n "$f" ]] || log_error "sha256_file requires a file path"
  need_file "$f"

  if has_cmd sha256sum; then
    sha256sum "$f" | cut -d ' ' -f 1
  elif has_cmd shasum; then
    shasum -a 256 "$f" | cut -d ' ' -f 1
  else
    log_error "No SHA256 utility found (need sha256sum or shasum)"
  fi
}

# ===============================================================
# Diagnostics (opt-in)
# ===============================================================

log_env_summary() {
  log "----------------------------------------"
  log "OS:      $(detect_os)"
  log "ARCH:    $(detect_arch)"
  log "TIME:    $(timestamp_iso)"
  log "USER:    ${USER:-unknown}"
  log "SHELL:   ${SHELL:-unknown}"
  log "PWD:     $(pwd -P 2>/dev/null || pwd)"
  log "----------------------------------------"
}

system_info() {
  log_info "OS=$(detect_os) ARCH=$(detect_arch) TIME=$(timestamp_iso)"
  if [[ "$(detect_os)" == "linux" && -f /etc/os-release ]]; then
    log_info "DISTRO_ID=$(detect_distro_id) DISTRO_VERSION_ID=$(detect_distro_version_id)"
  fi
  has_cmd uname && { log_info "uname:"; uname -a >&2 || true; }
  is_docker && log_info "container=docker"
}

# ===============================================================
# Optional heavy OS packages (STRICT opt-in)
# ===============================================================

sudo_nopass_available() {
  has_cmd sudo || return 1
  sudo -n true >/dev/null 2>&1
}

apt_install_packages() {
  [[ "${COMMON_ALLOW_INSTALL:-0}" == "1" ]] || log_error "Install blocked. Set COMMON_ALLOW_INSTALL=1."
  [[ "$(detect_os)" == "linux" ]] || log_error "apt_install_packages is linux-only"
  need_cmd apt-get

  : "${DEBIAN_FRONTEND:=noninteractive}"
  export DEBIAN_FRONTEND

  local apt_opts
  apt_opts="${APT_GET_OPTS:--y --no-install-recommends}"

  if has_cmd id && [[ "$(id -u 2>/dev/null || echo 1)" == "0" ]]; then
    apt-get update -y
    # shellcheck disable=SC2086
    apt-get install $apt_opts "$@"
    return 0
  fi

  sudo_nopass_available || log_error "Passwordless sudo not available (cannot install non-interactively)"
  sudo apt-get update -y
  # shellcheck disable=SC2086
  sudo apt-get install $apt_opts "$@"
}

install_dev_tools_apt() {
  apt_install_packages sudo gosu git curl build-essential gfortran ninja-build
}
