#!/usr/bin/env bash
# docker/scripts/bash_first_run_notice.sh

# If invoked by sh/zsh, re-exec into bash deterministically
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

bf_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }

bf_exit_or_return() {
  local code="${1:-0}"
  if bf_is_sourced; then return "$code"; else exit "$code"; fi
}

bf_is_true() {
  # Bash-3 safe truthy parsing (no ${v,,})
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac
}

bf_script_dir() { ( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P ); }

# ---------------- privilege helpers ----------------
_bf_is_root() { [[ "$(id -u)" -eq 0 ]]; }

_bf_sudo_nopass() {
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true >/dev/null 2>&1
}

_bf_as_root() {
  if _bf_is_root; then
    "$@"
  else
    _bf_sudo_nopass || return 1
    sudo -n "$@"
  fi
}

# ---------------- core (strict) ----------------
bash_first_run_notice_body() {
  set -Eeuo pipefail
  trap 'rc=$?; printf "%s\n" "[ERROR] bash_first_run_notice.sh failed at line $LINENO: ${BASH_COMMAND-<cmd>} (exit=$rc)" >&2; return $rc' ERR

  local SCRIPT_DIR REPO_ROOT COMMON_SH
  SCRIPT_DIR="$(bf_script_dir)"
  REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
  COMMON_SH="${COMMON_SH:-$REPO_ROOT/docker/scripts/lib/common.sh}"

  # Logging: prefer common.sh; fallback must not hard-exit
  if [[ -f "$COMMON_SH" ]]; then
    # shellcheck source=/dev/null
    . "$COMMON_SH"
  else
    log() { printf '%s\n' "$*" >&2; }
    log_info() { log "[INFO] $*"; }
    log_warning() { log "[WARNING] $*"; }
    log_success() { log "[SUCCESS] $*"; }
    log_debug() { :; }
    log_error_code() { local code="${1:-1}"; shift || true; log "[ERROR] $*"; return "$code"; }
    log_error() { log_error_code 1 "$@"; }
  fi

  # Controls
  local POST_CREATE_OVERWRITE POST_CREATE_STRICT
  POST_CREATE_OVERWRITE="${POST_CREATE_OVERWRITE:-1}"
  POST_CREATE_STRICT="${POST_CREATE_STRICT:-0}"

  _require_file_or_warn() {
    local p="$1"
    if [[ -f "$p" ]]; then return 0; fi
    if [[ "$POST_CREATE_STRICT" == "1" ]]; then
      log_error "Required file not found: $p"
      return 2
    fi
    log_warning "Skipping (source not found): $p"
    return 1
  }

  _copy_file_user() {
    local src="$1" dst="$2" mode="${3:-0644}"
    mkdir -p -- "$(dirname -- "$dst")"
    if [[ "$POST_CREATE_OVERWRITE" == "1" || ! -e "$dst" ]]; then
      if command -v install >/dev/null 2>&1; then
        install -m "$mode" -D "$src" "$dst"
      else
        cp -f -- "$src" "$dst"
        chmod "$mode" "$dst" 2>/dev/null || true
      fi
    fi
  }

  _copy_file_system() {
    local src="$1" dst="$2" mode="${3:-0644}"

    if ! _bf_is_root && ! _bf_sudo_nopass; then
      log_warning "Skipping system write (need root or passwordless sudo): $dst"
      return 0
    fi

    if [[ "$POST_CREATE_OVERWRITE" != "1" ]]; then
      if _bf_as_root test -e "$dst" >/dev/null 2>&1; then
        log_info "System target exists (overwrite disabled): $dst"
        return 0
      fi
    fi

    local parent
    parent="$(dirname -- "$dst")"

    if command -v install >/dev/null 2>&1; then
      _bf_as_root install -m "$mode" -D "$src" "$dst"
    else
      _bf_as_root mkdir -p -- "$parent"
      _bf_as_root cp -f -- "$src" "$dst"
      _bf_as_root chmod "$mode" "$dst" 2>/dev/null || true
    fi
  }

  _append_once() {
    local marker="$1" src="$2" target="$3"
    [[ -f "$target" ]] || : > "$target"

    if grep -Fq -- "$marker" "$target"; then
      log_info "Already present: $marker -> $target"
      return 0
    fi

    { printf '\n%s\n' "$marker"; cat -- "$src"; } >> "$target"
    log_success "Appended to $target"
  }

  # ---------- Paths ----------
  local NOTICE_SRC_DEFAULT NOTICE_SRC
  local DEVCONTAINER_NOTICE_DST_DEFAULT DEVCONTAINER_NOTICE_DST
  local USER_NOTICE_DST_DEFAULT USER_NOTICE_DST
  local BASHRC_PREFIX_SRC_DEFAULT BASHRC_PREFIX_SRC BASHRC_PREFIX_DST_DEFAULT BASHRC_PREFIX_DST
  local BASHRC_SUFFIX_SRC_DEFAULT BASHRC_SUFFIX_SRC BASHRC_SUFFIX_DST_DEFAULT BASHRC_SUFFIX_DST
  local BASHRC_SUFFIX_MARKER

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

  # ---------- Actions ----------
  log_info "Setting up Dev Containers first-run notice and bashrc customizations..."

  if _require_file_or_warn "$NOTICE_SRC"; then
    _copy_file_system "$NOTICE_SRC" "$DEVCONTAINER_NOTICE_DST" "0644"
    log_success "Dev Containers notice -> $DEVCONTAINER_NOTICE_DST"
    _copy_file_user "$NOTICE_SRC" "$USER_NOTICE_DST" "0644"
    log_success "User notice -> $USER_NOTICE_DST"
  fi

  if _require_file_or_warn "$BASHRC_PREFIX_SRC"; then
    if [[ -f "$BASHRC_PREFIX_DST" ]]; then
      _copy_file_system "$BASHRC_PREFIX_SRC" "$BASHRC_PREFIX_DST" "0644"
      log_success "System bashrc prefix -> $BASHRC_PREFIX_DST"
    else
      log_warning "Skipping system bashrc prefix: target not found ($BASHRC_PREFIX_DST)"
    fi
  fi

  if _require_file_or_warn "$BASHRC_SUFFIX_SRC"; then
    _append_once "$BASHRC_SUFFIX_MARKER" "$BASHRC_SUFFIX_SRC" "$BASHRC_SUFFIX_DST"
  fi

  log_success "bash_first_run_notice: done"
  return 0
}

# ---------------- run mode ----------------
# When sourced: isolate in subshell to avoid polluting caller; eliminates restore/unbound bugs.
if bf_is_sourced; then
  ( bash_first_run_notice_body "$@" )
  bf_exit_or_return $?
else
  bash_first_run_notice_body "$@"
  bf_exit_or_return $?
fi
