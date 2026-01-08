#!/usr/bin/env bash
# docker/scripts/bash_first_run_notice.sh
# ===============================================================
# bash_first_run_notice.sh — Dev Container first-run UX setup
# ===============================================================
# USER NOTES
# - This script is intended to be run inside a Dev Container / Docker
#   environment (postCreateCommand / postStartCommand), but is safe to run
#   on any Linux host: it performs no installs and has no background actions.
# - System-wide changes are performed ONLY when running as root or when
#   passwordless sudo is available (sudo -n).
#
# DEV NOTES
# - Bash script (uses BASH_SOURCE). Keep logic deterministic and explicit.
# - Avoid "realpath" dependency; use script-relative paths.
# - No package installation here. Keep installs in a dedicated script.
#
# Canonical Dev Containers first-run notice destination:
#   /usr/local/etc/vscode-dev-containers/first-run-notice.txt
# (used by devcontainers/images and commonly referenced by the community)
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
  printf '%s\n' "[ERROR] bash_first_run_notice.sh failed at line ${lineno}: ${cmd}" >&2
  exit_or_return 1
}
# trap 'rc=$?; echo "[ERROR] bash_first_run_notice.sh failed at line $LINENO: $BASH_COMMAND (exit=$rc)" >&2; exit $rc' ERR
trap '_on_err "$LINENO" "$BASH_COMMAND"' ERR

# ---------- Locate script + optional common library ----------
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

# ---------- Controls (explicit, deterministic) ----------
POST_CREATE_OVERWRITE="${POST_CREATE_OVERWRITE:-1}"
POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"

# ---------- Helpers ----------
_is_root() { [[ "$(id -u)" -eq 0 ]]; }

_sudo_nopass() {
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true >/dev/null 2>&1
}

_as_root() {
  if _is_root; then
    "$@"
  else
    _sudo_nopass || return 1
    sudo -n "$@"
  fi
}

_require_file_or_warn() {
  local p="$1"
  if [[ -f "$p" ]]; then
    return 0
  fi
  if [[ "$POST_CREATE_STRICT" = "1" ]]; then
    log_error "Required file not found: $p"
  fi
  log_warning "Skipping (source not found): $p"
  return 1
}

_copy_file() {
  local src="$1"
  local dst="$2"
  local mode="${3:-}"

  local parent
  parent="$(dirname -- "$dst")"

  if command -v install >/dev/null 2>&1; then
    if [[ -n "$mode" ]]; then
      install -m "$mode" -D "$src" "$dst"
    else
      install -D "$src" "$dst"
    fi
    return 0
  fi

  mkdir -p -- "$parent"
  cp -f -- "$src" "$dst"
  if [[ -n "$mode" ]]; then
    chmod "$mode" "$dst"
  fi
}

_copy_file_user() {
  local src="$1"
  local dst="$2"
  local mode="${3:-}"

  if [[ "$POST_CREATE_OVERWRITE" = "1" ]]; then
    _copy_file "$src" "$dst" "$mode"
  else
    [[ -e "$dst" ]] || _copy_file "$src" "$dst" "$mode"
  fi
}

_copy_file_system() {
  local src="$1"
  local dst="$2"
  local mode="${3:-}"

  if ! _is_root && ! _sudo_nopass; then
    log_warning "Skipping system install (need root or passwordless sudo): $dst"
    return 0
  fi

  if [[ "$POST_CREATE_OVERWRITE" = "1" ]]; then
    _as_root bash -lc "$(printf '%q ' _copy_file "$src" "$dst" "$mode")"
  else
    if _as_root test -e "$dst"; then
      log_info "System target already exists (overwrite disabled): $dst"
    else
      _as_root bash -lc "$(printf '%q ' _copy_file "$src" "$dst" "$mode")"
    fi
  fi
}

_append_once() {
  local marker="$1"
  local src="$2"
  local target="$3"

  [[ -f "$target" ]] || touch "$target"

  if grep -Fq "$marker" "$target"; then
    log_info "Already present: $marker -> $target"
    return 0
  fi

  {
    printf '\n%s\n' "$marker"
    cat "$src"
  } >> "$target"

  log_success "Appended to $target"
}

# ---------- Paths (script-relative sources) ----------
NOTICE_SRC_DEFAULT="${SCRIPT_DIR}/bash-first-run-notice.txt"
NOTICE_SRC="${POST_CREATE_NOTICE_SOURCE:-$NOTICE_SRC_DEFAULT}"

DEVCONTAINER_NOTICE_DST_DEFAULT="/usr/local/etc/vscode-dev-containers/first-run-notice.txt"
DEVCONTAINER_NOTICE_DST="${POST_CREATE_DEVCONTAINER_NOTICE_TARGET:-$DEVCONTAINER_NOTICE_DST_DEFAULT}"

USER_NOTICE_DST_DEFAULT="${HOME}/.bash-first-run-notice.txt"
USER_NOTICE_DST="${POST_CREATE_USER_NOTICE_TARGET:-$USER_NOTICE_DST_DEFAULT}"

BASHRC_PREFIX_SRC_DEFAULT="${SCRIPT_DIR}/bashrc.prefix"
BASHRC_PREFIX_SRC="${POST_CREATE_BASHRC_PREFIX_SOURCE:-$BASHRC_PREFIX_SRC_DEFAULT}"
BASHRC_PREFIX_DST_DEFAULT="/etc/bash.bashrc"
BASHRC_PREFIX_DST="${POST_CREATE_BASHRC_PREFIX_TARGET:-$BASHRC_PREFIX_DST_DEFAULT}"

BASHRC_SUFFIX_SRC_DEFAULT="${SCRIPT_DIR}/bashrc.suffix"
BASHRC_SUFFIX_SRC="${POST_CREATE_BASHRC_SUFFIX_SOURCE:-$BASHRC_SUFFIX_SRC_DEFAULT}"
BASHRC_SUFFIX_DST_DEFAULT="${HOME}/.bashrc"
BASHRC_SUFFIX_DST="${POST_CREATE_BASHRC_SUFFIX_TARGET:-$BASHRC_SUFFIX_DST_DEFAULT}"
BASHRC_SUFFIX_MARKER="${POST_CREATE_BASHRC_SUFFIX_MARKER:-# >>> (bashrc.suffix) scikit-plots personal initialization >>>}"

# ---------- Main ----------
log_info "Setting up Dev Containers first-run notice and bashrc customizations..."
log_debug "SCRIPT_DIR=$SCRIPT_DIR"

# 1) Dev Containers canonical first-run notice (system-wide)
if _require_file_or_warn "$NOTICE_SRC"; then
  _copy_file_system "$NOTICE_SRC" "$DEVCONTAINER_NOTICE_DST" "0644"
  log_success "Dev Containers notice -> $DEVCONTAINER_NOTICE_DST"
fi

# 2) User-level notice (home)
if _require_file_or_warn "$NOTICE_SRC"; then
  _copy_file_user "$NOTICE_SRC" "$USER_NOTICE_DST" "0644"
  log_success "User notice -> $USER_NOTICE_DST"
fi

# 3) System-wide bashrc prefix (Debian/Ubuntu typically uses /etc/bash.bashrc)
if _require_file_or_warn "$BASHRC_PREFIX_SRC"; then
  if [[ -f "$BASHRC_PREFIX_DST" ]]; then
    _copy_file_system "$BASHRC_PREFIX_SRC" "$BASHRC_PREFIX_DST" "0644"
    log_success "System bashrc prefix -> $BASHRC_PREFIX_DST"
  else
    log_warning "Skipping system bashrc prefix: target not found ($BASHRC_PREFIX_DST)"
  fi
fi

# 4) Append bashrc suffix once (user-level)
if _require_file_or_warn "$BASHRC_SUFFIX_SRC"; then
  _append_once "$BASHRC_SUFFIX_MARKER" "$BASHRC_SUFFIX_SRC" "$BASHRC_SUFFIX_DST"
fi

log_success "bash_first_run_notice.sh completed."
