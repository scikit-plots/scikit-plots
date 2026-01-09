#!/usr/bin/env bash
# docker/scripts/bash_first_run_notice.sh
#
# First-run notice + bashrc customizations (Bash-contract)
#
# Design goals
# - Works when executed OR sourced.
# - Fully compatible for NON-ROOT users:
#   - System-level writes (e.g. /etc/bash.bashrc, /usr/local/...) are attempted only if
#     running as root or passwordless sudo is available.
#   - Otherwise, we fall back to user-level drop-ins under $HOME.
#
# Optional feature: auto-activate env in interactive bash shells
# - POST_CREATE_BASHRC_AUTO_ACTIVATE=0|1 (default 0)
# - Uses POST_CREATE_ENV_TOOL_SELECTED / POST_CREATE_ENV_NAME when present
#
# Prefix strategy (system vs user)
# - POST_CREATE_BASHRC_PREFIX_ENABLE=0|1   (default 1)
# - POST_CREATE_BASHRC_PREFIX_MODE=auto|system|user (default auto)
# - POST_CREATE_BASHRC_PREFIX_FALLBACK_USER=0|1 (default 1 when mode=auto/system)
# - POST_CREATE_BASHRC_USER_DIR=$HOME/.bashrc.d (default)
#   - When user-mode, we write:
#       $POST_CREATE_BASHRC_USER_DIR/00-scikit-plots.prefix.sh
#     and ensure ~/.bashrc loads *.sh from that dir (idempotent).
#
# Suffix strategy
# - POST_CREATE_BASHRC_SUFFIX_ENABLE=0|1   (default 1)
# - Appends docker/scripts/bashrc.suffix to ~/.bashrc (idempotent marker)

# If invoked by sh/zsh, re-exec into bash deterministically
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

bf_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
bf_exit_or_return() { local code="${1:-0}"; bf_is_sourced && return "$code" || exit "$code"; }

bf_script_dir() { ( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P ); }

# ---------------- privilege helpers ----------------
_bf_is_root() { [[ "$(id -u)" -eq 0 ]]; }

_bf_sudo_nopass() {
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true >/dev/null 2>&1
}

_bf_can_system_write() {
  _bf_is_root || _bf_sudo_nopass
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
  COMMON_SH="${COMMON_SH:-$REPO_ROOT/docker/scripts/common.sh}"

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

  local PREFIX_ENABLE PREFIX_MODE PREFIX_FALLBACK_USER USER_DIR
  PREFIX_ENABLE="${POST_CREATE_BASHRC_PREFIX_ENABLE:-1}"
  PREFIX_MODE="${POST_CREATE_BASHRC_PREFIX_MODE:-auto}"
  PREFIX_FALLBACK_USER="${POST_CREATE_BASHRC_PREFIX_FALLBACK_USER:-1}"
  USER_DIR="${POST_CREATE_BASHRC_USER_DIR:-$HOME/.bashrc.d}"

  local SUFFIX_ENABLE
  SUFFIX_ENABLE="${POST_CREATE_BASHRC_SUFFIX_ENABLE:-1}"

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

    if ! _bf_can_system_write; then
      log_warning "Skipping system write (need root or passwordless sudo): $dst"
      return 1
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
    return 0
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

  _append_block_once() {
    # Append a dynamic snippet once (identified by begin marker)
    local begin="$1" end="$2" target="$3"
    shift 3
    [[ -f "$target" ]] || : > "$target"

    if grep -Fq -- "$begin" "$target"; then
      log_info "Already present: $begin -> $target"
      return 0
    fi

    {
      printf '\n%s\n' "$begin"
      printf '%s\n' "$@"
      printf '%s\n' "$end"
    } >> "$target"
    log_success "Appended block to $target"
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

  # Notice: system + user
  if _require_file_or_warn "$NOTICE_SRC"; then
    if _copy_file_system "$NOTICE_SRC" "$DEVCONTAINER_NOTICE_DST" "0644"; then
      log_success "Dev Containers notice -> $DEVCONTAINER_NOTICE_DST"
    else
      log_warning "Dev Containers notice skipped (no system privileges): $DEVCONTAINER_NOTICE_DST"
    fi

    _copy_file_user "$NOTICE_SRC" "$USER_NOTICE_DST" "0644"
    log_success "User notice -> $USER_NOTICE_DST"
  fi

  # Prefix: system vs user fallback
  if [[ "$PREFIX_ENABLE" == "1" ]] && _require_file_or_warn "$BASHRC_PREFIX_SRC"; then
    case "$PREFIX_MODE" in
      system|auto|user) ;;
      *) log_warning "Invalid POST_CREATE_BASHRC_PREFIX_MODE=$PREFIX_MODE (expected auto|system|user). Using auto." ; PREFIX_MODE="auto" ;;
    esac

    if [[ "$PREFIX_MODE" == "system" || "$PREFIX_MODE" == "auto" ]]; then
      # Only try system if target exists (avoid creating new system files unexpectedly)
      if [[ -f "$BASHRC_PREFIX_DST" ]]; then
        if _copy_file_system "$BASHRC_PREFIX_SRC" "$BASHRC_PREFIX_DST" "0644"; then
          log_success "System bashrc prefix -> $BASHRC_PREFIX_DST"
        else
          if [[ "$PREFIX_FALLBACK_USER" == "1" || "$PREFIX_MODE" == "auto" ]]; then
            PREFIX_MODE="user"
          else
            log_warning "System bashrc prefix skipped; user fallback disabled"
          fi
        fi
      else
        log_warning "System bashrc target not found ($BASHRC_PREFIX_DST); falling back to user prefix"
        PREFIX_MODE="user"
      fi
    fi

    if [[ "$PREFIX_MODE" == "user" ]]; then
      local user_prefix_file begin end
      mkdir -p -- "$USER_DIR"
      user_prefix_file="${USER_DIR}/00-scikit-plots.prefix.sh"
      _copy_file_user "$BASHRC_PREFIX_SRC" "$user_prefix_file" "0644"
      log_success "User bashrc prefix -> $user_prefix_file"

      # Ensure ~/.bashrc loads drop-ins (interactive-only)
      begin="# >>> scikit-plots bashrc.d loader >>>"
      end="# <<< scikit-plots bashrc.d loader <<<"
      _append_block_once "$begin" "$end" "$BASHRC_SUFFIX_DST" \
        'case $- in *i*) ;; *) return ;; esac' \
        "if [[ -d \"${USER_DIR}\" ]]; then" \
        "  for __sp_f in \"${USER_DIR}\"/*.sh; do" \
        '    [[ -r "$__sp_f" ]] && . "$__sp_f"' \
        '  done' \
        '  unset __sp_f' \
        'fi'
    fi
  fi

  # Suffix: append once to ~/.bashrc
  if [[ "$SUFFIX_ENABLE" == "1" ]] && _require_file_or_warn "$BASHRC_SUFFIX_SRC"; then
    _append_once "$BASHRC_SUFFIX_MARKER" "$BASHRC_SUFFIX_SRC" "$BASHRC_SUFFIX_DST"
  fi

  # Optional: auto-activate env in interactive shells (user-level)
  if [[ "${POST_CREATE_BASHRC_AUTO_ACTIVATE:-0}" == "1" ]]; then
    local tool env_name
    tool="${POST_CREATE_ENV_TOOL_SELECTED:-${POST_CREATE_ENV_TOOL:-}}"
    env_name="${POST_CREATE_ENV_NAME:-${ENV_NAME:-}}"

    if [[ -n "$tool" && -n "$env_name" ]]; then
      local begin end
      begin="# >>> scikit-plots auto-activate env >>>"
      end="# <<< scikit-plots auto-activate env <<<"

      if [[ "$tool" == "micromamba" ]]; then
        _append_block_once "$begin" "$end" "$BASHRC_SUFFIX_DST" \
          'case $- in *i*) ;; *) return ;; esac' \
          'if command -v micromamba >/dev/null 2>&1; then' \
          '  eval "$(micromamba shell hook -s bash 2>/dev/null)"' \
          "  micromamba activate \"${env_name}\" >/dev/null 2>&1 || true" \
          'fi'
      elif [[ "$tool" == "conda" ]]; then
        _append_block_once "$begin" "$end" "$BASHRC_SUFFIX_DST" \
          'case $- in *i*) ;; *) return ;; esac' \
          'if command -v conda >/dev/null 2>&1; then' \
          '  __conda_base="$(conda info --base 2>/dev/null)"' \
          '  if [[ -n "${__conda_base:-}" && -f "${__conda_base}/etc/profile.d/conda.sh" ]]; then' \
          '    . "${__conda_base}/etc/profile.d/conda.sh"' \
          "    conda activate \"${env_name}\" >/dev/null 2>&1 || true" \
          '  fi' \
          'fi'
      else
        log_warning "POST_CREATE_BASHRC_AUTO_ACTIVATE=1 but tool is not micromamba/conda (tool=$tool)"
      fi
    else
      log_warning "POST_CREATE_BASHRC_AUTO_ACTIVATE=1 but missing tool/env name (tool=$tool env_name=$env_name)"
    fi
  fi

  log_success "bash_first_run_notice: done"
  return 0
}

# When sourced: isolate in subshell to avoid polluting caller
if bf_is_sourced; then
  ( bash_first_run_notice_body "$@" )
  bf_exit_or_return $?
else
  bash_first_run_notice_body "$@"
  bf_exit_or_return $?
fi
