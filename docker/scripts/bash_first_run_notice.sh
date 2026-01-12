#!/usr/bin/env bash
# docker/scripts/bash_first_run_notice.sh
#
# First-run notice + bashrc customizations (Bash-contract)
#
# ✅ Modular, ordered, non-root compatible bashrc customization
#
# Prefix/suffix design
# - Prefix and suffix are installed as ordered drop-ins (*.sh) and sourced by a loader.
# - Guarantees "prefix before suffix" by lexicographic filename order.
# - Allows customization before/after by changing the drop-in filenames (e.g. 00-..., 50-..., 99-...).
#
# Root/system behavior
# - If root (or passwordless sudo), install SYSTEM prefix drop-in:
#     /etc/bashrc.d/*prefix*
#   and ensure /etc/bash.bashrc sources /etc/bashrc.d/*.sh (idempotent loader).
#
# Non-root behavior
# - Always ensure USER loader in ~/.bashrc to source ~/.bashrc.d/*.sh (idempotent).
# - Install USER prefix drop-in:
#     ~/.bashrc.d/*prefix*
# - Install USER suffix drop-in:
#     ~/.bashrc.d/*suffix*   (default; you can rename to 99-... etc)
#
# Notes
# - Bash does NOT read "$HOME/bashrc" by default; the interactive config file is "~/.bashrc".
# - This script avoids overwriting /etc/bash.bashrc; it only appends a marked loader block once.
#
# Controls (all optional)
# - POST_CREATE_STRICT=0|1                 (default 0)
# - POST_CREATE_OVERWRITE=0|1              (default 1)
#
# System loader / prefix
# - POST_CREATE_SYSTEM_BASHRC=/etc/bash.bashrc
# - POST_CREATE_SYSTEM_BASHRC_D_DIR=/etc/bashrc.d
# - POST_CREATE_SYSTEM_LOADER_ENABLE=0|1   (default 1)
# - POST_CREATE_SYSTEM_PREFIX_ENABLE=0|1   (default 1)
# - POST_CREATE_SYSTEM_PREFIX_GLOB=*prefix*
#
# User loader / prefix / suffix
# - POST_CREATE_USER_BASHRC_D_DIR=$HOME/.bashrc.d
# - POST_CREATE_USER_LOADER_ENABLE=0|1     (default 1)
# - POST_CREATE_USER_PREFIX_ENABLE=0|1     (default 1)
# - POST_CREATE_USER_PREFIX_GLOB=*prefix*
# - POST_CREATE_USER_SUFFIX_ENABLE=0|1     (default 1)
# - POST_CREATE_USER_SUFFIX_GLOB=*suffix*
#
# Auto-activate env (optional; user-level)
# - POST_CREATE_BASHRC_AUTO_ACTIVATE=0|1 (default 1)
# - Uses POST_CREATE_ENV_TOOL_SELECTED / ENV_NAME when present

# If invoked by sh/zsh, re-exec into bash deterministically
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

# ---------- Error reporting ----------
_on_err() {
  local lineno="$1"
  local cmd="$2"
  printf '%s\n' "[ERROR] bash_first_run_notice.sh failed at line ${lineno}: ${cmd}" >&2
  exit_or_return 1
}
# trap 'rc=$?; echo "[ERROR] bash_first_run_notice.sh failed at line $LINENO: $BASH_COMMAND (exit=$rc)" >&2; exit $rc' ERR
trap '_on_err "$LINENO" "$BASH_COMMAND"' ERR

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

is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
exit_or_return() { local rc="${1:-0}"; is_sourced && return "$rc" || exit "$rc"; }
script_dir() { ( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P ); }

_is_root() { [[ "$(id -u)" -eq 0 ]]; }
_sudo_nopass() { command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; }
_can_system_write() {
  local allow_sudo="${POST_CREATE_ALLOW_SUDO:-0}"
  _is_root && return 0
  [[ "$allow_sudo" == "1" ]] && _sudo_nopass
}
_as_root() {
  if _is_root; then "$@"; return $?; fi
  local allow_sudo="${POST_CREATE_ALLOW_SUDO:-0}"
  [[ "$allow_sudo" == "1" ]] || return 1
  _sudo_nopass || return 1
  sudo -n "$@"
}

main() {
  local SCRIPT_DIR REPO_ROOT COMMON_SH
  SCRIPT_DIR="$(script_dir)"
  REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
  COMMON_SH="${COMMON_SH:-$REPO_ROOT/docker/scripts/common.sh}"

  if [[ -f "$COMMON_SH" ]]; then
    # shellcheck source=/dev/null
    . "$COMMON_SH"
  else
    log() { printf '%s\n' "$*" >&2; }
    # log_info(){ printf '%s\n' "[INFO] $*"; }
    log_info() { log "[INFO] $*"; }
    log_warning() { log "[WARNING] $*"; }
    log_success() { log "[SUCCESS] $*"; }
    log_debug() { :; }
    log_error_code() { local code="${1:-1}"; shift || true; log "[ERROR] $*"; return "$code"; }
    log_error() { log_error_code 1 "$@"; }
  fi

  local OVERWRITE STRICT
  OVERWRITE="${POST_CREATE_OVERWRITE:-1}"
  STRICT="${POST_CREATE_STRICT:-0}"

  local SRC_D_DIR NOTICE_SRC
  SRC_D_DIR="${POST_CREATE_BASHRC_D_SOURCE_DIR:-$SCRIPT_DIR/.bashrc.d}"
  NOTICE_SRC="${POST_CREATE_NOTICE_SOURCE:-$SCRIPT_DIR/bash-first-run-notice.txt}"

  local SYS_BASHRC SYS_D_DIR SYS_LOADER_ENABLE SYS_DROPINS_ENABLE
  SYS_BASHRC="${POST_CREATE_SYSTEM_BASHRC:-/etc/bash.bashrc}"
  SYS_D_DIR="${POST_CREATE_SYSTEM_BASHRC_D_DIR:-/etc/bashrc.d}"
  SYS_LOADER_ENABLE="${POST_CREATE_SYSTEM_LOADER_ENABLE:-1}"
  SYS_DROPINS_ENABLE="${POST_CREATE_SYSTEM_DROPINS_ENABLE:-1}"

  local USER_D_DIR USER_LOADER_ENABLE USER_DROPINS_ENABLE
  USER_D_DIR="${POST_CREATE_USER_BASHRC_D_DIR:-$HOME/.bashrc.d}"
  USER_LOADER_ENABLE="${POST_CREATE_USER_LOADER_ENABLE:-1}"
  USER_DROPINS_ENABLE="${POST_CREATE_USER_DROPINS_ENABLE:-__AUTO__}"

  # AUTO: if system loader+drop-ins are enabled and writable, default user drop-ins OFF (avoid double)
  if [[ "$USER_DROPINS_ENABLE" == "__AUTO__" ]]; then
    if [[ "$SYS_LOADER_ENABLE" == "1" && "$SYS_DROPINS_ENABLE" == "1" ]] \
       && _can_system_write && [[ -f "$SYS_BASHRC" ]]; then
      USER_DROPINS_ENABLE="0"
      log_info "User drop-ins default -> disabled (system drop-ins enabled); set POST_CREATE_USER_DROPINS_ENABLE=1 to force"
    else
      USER_DROPINS_ENABLE="1"
    fi
  fi

  require_dir() {
    local d="$1"
    [[ -d "$d" ]] && return 0
    [[ "$STRICT" == "1" ]] && { log_error "Missing dir: $d"; return 2; }
    log_warning "Skip (missing dir): $d"; return 1
  }
  require_file() {
    local f="$1"
    [[ -f "$f" ]] && return 0
    [[ "$STRICT" == "1" ]] && { log_error "Missing file: $f"; return 2; }
    log_warning "Skip (missing file): $f"; return 1
  }

  copy_user() {
    local src="$1" dst="$2" mode="${3:-0644}"
    mkdir -p -- "$(dirname -- "$dst")"
    if [[ "$OVERWRITE" == "1" || ! -e "$dst" ]]; then
      if command -v install >/dev/null 2>&1; then
        install -m "$mode" -D "$src" "$dst"
      else
        cp -f -- "$src" "$dst"
        chmod "$mode" "$dst" 2>/dev/null || true
      fi
    fi
  }
  copy_system() {
    local src="$1" dst="$2" mode="${3:-0644}"
    _can_system_write || return 1
    if [[ "$OVERWRITE" != "1" ]] && _as_root test -e "$dst" >/dev/null 2>&1; then
      return 0
    fi
    if command -v install >/dev/null 2>&1; then
      _as_root install -m "$mode" -D "$src" "$dst"
    else
      _as_root mkdir -p -- "$(dirname -- "$dst")"
      _as_root cp -f -- "$src" "$dst"
      _as_root chmod "$mode" "$dst" 2>/dev/null || true
    fi
  }

  append_block_once_user() {
    local begin="$1" end="$2" target="$3"; shift 3
    [[ -f "$target" ]] || : > "$target"
    grep -Fq -- "$begin" "$target" && return 0
    {
      printf '\n%s\n' "$begin"
      printf '%s\n' "$@"
      printf '%s\n' "$end"
    } >> "$target"
  }

  install_user_loader() {
    [[ "$USER_LOADER_ENABLE" == "1" ]] || return 0
    mkdir -p -- "$USER_D_DIR"
    append_block_once_user \
      "# >>> scikit-plots bashrc.d user loader >>>" \
      "# <<< scikit-plots bashrc.d user loader <<<" \
      "$HOME/.bashrc" \
      'case $- in *i*) ;; *) return ;; esac' \
      "if [[ -d \"${USER_D_DIR}\" ]]; then" \
      "  for __sp_f in \"${USER_D_DIR}\"/*.sh; do" \
      '    [[ -r "$__sp_f" ]] && . "$__sp_f"' \
      '  done' \
      '  unset __sp_f' \
      'fi'
  }

  install_system_loader() {
    [[ "$SYS_LOADER_ENABLE" == "1" ]] || return 0
    [[ -f "$SYS_BASHRC" ]] || { log_warning "Skip system loader (missing): $SYS_BASHRC"; return 0; }
    _can_system_write || { log_warning "Skip system loader (no privileges)"; return 0; }

    _as_root mkdir -p -- "$SYS_D_DIR"

    local begin end
    begin="# >>> scikit-plots bashrc.d system loader >>>"
    end="# <<< scikit-plots bashrc.d system loader <<<"
    # append once (safe way via temp)
    local tmp; tmp="$(mktemp)"
    cat -- "$SYS_BASHRC" > "$tmp"
    grep -Fq -- "$begin" "$tmp" && { rm -f -- "$tmp"; return 0; }

    {
      cat -- "$tmp"
      printf '\n%s\n' "$begin"
      printf '%s\n' 'case $- in *i*) ;; *) return ;; esac'
      printf '%s\n' "if [[ -d \"${SYS_D_DIR}\" ]]; then"
      printf '%s\n' "  for __sp_f in \"${SYS_D_DIR}\"/*.sh; do"
      printf '%s\n' '    [[ -r "$__sp_f" ]] && . "$__sp_f"'
      printf '%s\n' '  done'
      printf '%s\n' '  unset __sp_f'
      printf '%s\n' 'fi'
      printf '%s\n' "$end"
    } > "${tmp}.new"
    _as_root cp -f -- "${tmp}.new" "$SYS_BASHRC"
    rm -f -- "$tmp" "${tmp}.new"
  }

  log_info "Installing bashrc drop-ins bundle..."
  log_info "Repo drop-ins: $SRC_D_DIR"

  # Copy notices
  if require_file "$NOTICE_SRC"; then
    copy_user "$NOTICE_SRC" "$HOME/.bash-first-run-notice.txt" 0644
    copy_system "$NOTICE_SRC" "/usr/local/etc/vscode-dev-containers/first-run-notice.txt" 0644 || true
  fi

  install_user_loader
  install_system_loader

  # Copy drop-ins bundle
  if require_dir "$SRC_D_DIR"; then
    shopt -s nullglob
    local files=( "$SRC_D_DIR"/*.sh )
    shopt -u nullglob

    if [[ "${#files[@]}" -eq 0 ]]; then
      [[ "$STRICT" == "1" ]] && { log_error "No drop-ins in: $SRC_D_DIR"; return 2; }
      log_warning "No drop-ins found in: $SRC_D_DIR"
      return 0
    fi

    if [[ "$SYS_DROPINS_ENABLE" == "1" ]] && _can_system_write && [[ -f "$SYS_BASHRC" ]]; then
      _as_root mkdir -p -- "$SYS_D_DIR"
      for f in "${files[@]}"; do
        copy_system "$f" "$SYS_D_DIR/$(basename -- "$f")" 0644 || true
      done
      log_success "System drop-ins -> $SYS_D_DIR"
    fi

    if [[ "$USER_DROPINS_ENABLE" == "1" ]]; then
      mkdir -p -- "$USER_D_DIR"
      for f in "${files[@]}"; do
        copy_user "$f" "$USER_D_DIR/$(basename -- "$f")" 0644
      done
      log_success "User drop-ins -> $USER_D_DIR"
    else
      log_info "User drop-ins disabled"
    fi
  fi

  log_success "bash_first_run_notice complete"
}

main "$@"
exit_or_return $?
