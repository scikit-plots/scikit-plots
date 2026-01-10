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
#     /etc/bashrc.d/00-scikit-plots.prefix.sh
#   and ensure /etc/bash.bashrc sources /etc/bashrc.d/*.sh (idempotent loader).
#
# Non-root behavior
# - Always ensure USER loader in ~/.bashrc to source ~/.bashrc.d/*.sh (idempotent).
# - Install USER prefix drop-in:
#     ~/.bashrc.d/00-scikit-plots.prefix.sh
# - Install USER suffix drop-in:
#     ~/.bashrc.d/10-scikit-plots.suffix.sh   (default; you can rename to 99-... etc)
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
# - POST_CREATE_SYSTEM_D_DIR=/etc/bashrc.d
# - POST_CREATE_SYSTEM_BASHRC=/etc/bash.bashrc
# - POST_CREATE_SYSTEM_PREFIX_ENABLE=0|1   (default 1)
# - POST_CREATE_SYSTEM_PREFIX_BASENAME=00-scikit-plots.prefix.sh
# - POST_CREATE_SYSTEM_LOADER_ENABLE=0|1   (default 1)
#
# User loader / prefix / suffix
# - POST_CREATE_BASHRC_USER_DIR=$HOME/.bashrc.d
# - POST_CREATE_USER_LOADER_ENABLE=0|1     (default 1)
# - POST_CREATE_USER_PREFIX_ENABLE=0|1     (default 1)
# - POST_CREATE_USER_PREFIX_BASENAME=00-scikit-plots.prefix.sh
# - POST_CREATE_USER_SUFFIX_ENABLE=0|1     (default 1)
# - POST_CREATE_USER_SUFFIX_BASENAME=10-scikit-plots.suffix.sh
#
# Auto-activate env (optional; user-level)
# - POST_CREATE_BASHRC_AUTO_ACTIVATE=0|1 (default 1)
# - Uses POST_CREATE_ENV_TOOL_SELECTED / ENV_NAME when present

# If invoked by sh/zsh, re-exec into bash deterministically
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

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

bf_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
bf_exit_or_return() { local code="${1:-0}"; bf_is_sourced && return "$code" || exit "$code"; }
bf_script_dir() { ( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P ); }

# ---------------- privilege helpers ----------------
_bf_is_root() { [[ "$(id -u)" -eq 0 ]]; }

_bf_sudo_nopass() {
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true >/dev/null 2>&1
}

_bf_can_system_write() { _bf_is_root || _bf_sudo_nopass; }

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

  # System controls
  local SYS_DIR SYS_BASHRC SYS_PREFIX_ENABLE SYS_LOADER_ENABLE SYS_PREFIX_BASENAME
  SYS_DIR="${POST_CREATE_SYSTEM_D_DIR:-/etc/bashrc.d}"
  SYS_BASHRC="${POST_CREATE_SYSTEM_BASHRC:-/etc/bash.bashrc}"
  SYS_PREFIX_ENABLE="${POST_CREATE_SYSTEM_PREFIX_ENABLE:-1}"
  SYS_LOADER_ENABLE="${POST_CREATE_SYSTEM_LOADER_ENABLE:-1}"
  SYS_PREFIX_BASENAME="${POST_CREATE_SYSTEM_PREFIX_BASENAME:-00-scikit-plots.prefix.sh}"

  # User controls
  local USER_DIR USER_LOADER_ENABLE USER_PREFIX_ENABLE USER_SUFFIX_ENABLE USER_PREFIX_BASENAME USER_SUFFIX_BASENAME
  USER_DIR="${POST_CREATE_BASHRC_USER_DIR:-$HOME/.bashrc.d}"
  USER_LOADER_ENABLE="${POST_CREATE_USER_LOADER_ENABLE:-1}"
  USER_PREFIX_ENABLE="${POST_CREATE_USER_PREFIX_ENABLE:-1}"
  USER_SUFFIX_ENABLE="${POST_CREATE_USER_SUFFIX_ENABLE:-1}"
  USER_PREFIX_BASENAME="${POST_CREATE_USER_PREFIX_BASENAME:-00-scikit-plots.prefix.sh}"
  USER_SUFFIX_BASENAME="${POST_CREATE_USER_SUFFIX_BASENAME:-10-scikit-plots.suffix.sh}"

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

  _append_block_once() {
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

  _install_loader_user() {
    [[ "$USER_LOADER_ENABLE" == "1" ]] || return 0

    local bashrc="${HOME}/.bashrc"
    local begin end
    begin="# >>> scikit-plots bashrc.d user loader >>>"
    end="# <<< scikit-plots bashrc.d user loader <<<"

    mkdir -p -- "$USER_DIR"

    _append_block_once "$begin" "$end" "$bashrc" \
      'case $- in *i*) ;; *) return ;; esac' \
      "if [[ -d \"${USER_DIR}\" ]]; then" \
      "  for __sp_f in \"${USER_DIR}\"/*.sh; do" \
      '    [[ -r "$__sp_f" ]] && . "$__sp_f"' \
      '  done' \
      '  unset __sp_f' \
      'fi'
  }

  _install_loader_system() {
    [[ "$SYS_LOADER_ENABLE" == "1" ]] || return 0

    if [[ ! -f "$SYS_BASHRC" ]]; then
      log_warning "System bashrc not found (skip system loader): $SYS_BASHRC"
      return 0
    fi

    if ! _bf_can_system_write; then
      log_warning "No system privileges; cannot install system loader into: $SYS_BASHRC"
      return 0
    fi

    _bf_as_root mkdir -p -- "$SYS_DIR"

    local begin end
    begin="# >>> scikit-plots bashrc.d system loader >>>"
    end="# <<< scikit-plots bashrc.d system loader <<<"

    # Put a minimal, interactive-only loader into /etc/bash.bashrc (idempotent).
    # We do NOT overwrite /etc/bash.bashrc; we append a clearly marked block.
    _bf_as_root bash -lc "$(printf '%q ' true)" >/dev/null 2>&1 || true  # no-op; keep sudo warm if needed
    _bf_as_root bash -c "true" >/dev/null 2>&1 || true

    # Append using root context (avoid writing as non-root).
    local tmp
    tmp="$(mktemp)"
    cat -- "$SYS_BASHRC" > "$tmp"
    if grep -Fq -- "$begin" "$tmp"; then
      log_info "Already present: $begin -> $SYS_BASHRC"
      rm -rf -- "$tmp"
      return 0
    fi
    {
      cat -- "$tmp"
      printf '\n%s\n' "$begin"
      printf '%s\n' 'case $- in *i*) ;; *) return ;; esac'
      printf '%s\n' "if [[ -d \"${SYS_DIR}\" ]]; then"
      printf '%s\n' "  for __sp_f in \"${SYS_DIR}\"/*.sh; do"
      printf '%s\n' '    [[ -r "$__sp_f" ]] && . "$__sp_f"'
      printf '%s\n' '  done'
      printf '%s\n' '  unset __sp_f'
      printf '%s\n' 'fi'
      printf '%s\n' "$end"
    } > "${tmp}.new"
    _bf_as_root cp -f -- "${tmp}.new" "$SYS_BASHRC"
    rm -rf -- "$tmp" "${tmp}.new"
    log_success "Installed system loader in $SYS_BASHRC"
  }

  # ---------- Source assets ----------
  local NOTICE_SRC NOTICE_SRC_DEFAULT
  local DEVCONTAINER_NOTICE_DST DEVCONTAINER_NOTICE_DST_DEFAULT
  local USER_NOTICE_DST USER_NOTICE_DST_DEFAULT
  local BASHRC_PREFIX_SRC BASHRC_PREFIX_SRC_DEFAULT BASHRC_PREFIX_DST_DEFAULT BASHRC_PREFIX_DST
  local BASHRC_SUFFIX_SRC BASHRC_SUFFIX_SRC_DEFAULT

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

  log_info "Setting up Dev Containers first-run notice and bashrc drop-ins..."

  # ---------- Notices ----------
  if _require_file_or_warn "$NOTICE_SRC"; then
    # 1) Dev Containers canonical first-run notice (system-wide)
    if _copy_file_system "$NOTICE_SRC" "$DEVCONTAINER_NOTICE_DST" "0644"; then
      log_success "Dev Containers notice -> $DEVCONTAINER_NOTICE_DST"
    else
      log_warning "Dev Containers notice skipped (no system privileges): $DEVCONTAINER_NOTICE_DST"
    fi

    # 2) User-level notice (home)
    _copy_file_user "$NOTICE_SRC" "$USER_NOTICE_DST" "0644"
    log_success "User notice -> $USER_NOTICE_DST"
  fi

  # ---------- Loaders ----------
  _install_loader_user
  _install_loader_system

  # ---------- Prefix ----------
  # 3) System-wide bashrc prefix (Debian/Ubuntu typically uses /etc/bash.bashrc)
  if _require_file_or_warn "$BASHRC_PREFIX_SRC"; then
    # System prefix (root only)
    if [[ "$SYS_PREFIX_ENABLE" == "1" ]] && _bf_can_system_write && [[ -f "$SYS_BASHRC" ]]; then
      local sys_prefix_dst="${SYS_DIR}/${SYS_PREFIX_BASENAME}"
      _bf_as_root mkdir -p -- "$SYS_DIR"
      if _copy_file_system "$BASHRC_PREFIX_SRC" "$sys_prefix_dst" "0644"; then
        log_success "System prefix drop-in -> $sys_prefix_dst"
      fi
    else
      log_debug "System prefix skipped (disabled or no privileges or missing $SYS_BASHRC)"
    fi

    # User prefix (always for non-root, optional for root)
    if [[ "$USER_PREFIX_ENABLE" == "1" ]]; then
      local user_prefix_dst="${USER_DIR}/${USER_PREFIX_BASENAME}"
      mkdir -p -- "$USER_DIR"
      _copy_file_user "$BASHRC_PREFIX_SRC" "$user_prefix_dst" "0644"
      log_success "User prefix drop-in -> $user_prefix_dst"
    fi
  fi

  # ---------- Suffix (user drop-in) ----------
  # 4) Append bashrc suffix once (user-level)
  if [[ "$USER_SUFFIX_ENABLE" == "1" ]] && _require_file_or_warn "$BASHRC_SUFFIX_SRC"; then
    local user_suffix_dst="${USER_DIR}/${USER_SUFFIX_BASENAME}"
    mkdir -p -- "$USER_DIR"
    _copy_file_user "$BASHRC_SUFFIX_SRC" "$user_suffix_dst" "0644"
    log_success "User suffix drop-in -> $user_suffix_dst"
  fi

  # ---------- Optional: auto-activate env (user-level; keep ordering by naming) ----------
  if [[ "${POST_CREATE_BASHRC_AUTO_ACTIVATE:-1}" == "1" ]]; then
    local tool env_name
    tool="${POST_CREATE_ENV_TOOL_SELECTED:-${POST_CREATE_ENV_TOOL:-}}"
    env_name="${ENV_NAME:-${ENV_NAME:-}}"

    if [[ -n "$tool" && -n "$env_name" ]]; then
      # Put this after prefix but before suffix by default; user can rename via basename.
      local auto_basename="${POST_CREATE_USER_AUTO_ACTIVATE_BASENAME:-05-scikit-plots.auto-activate.sh}"
      local auto_dst="${USER_DIR}/${auto_basename}"

      {
        printf '%s\n' '# scikit-plots auto-activate (generated)'
        printf '%s\n' 'case $- in *i*) ;; *) return ;; esac'
        if [[ "$tool" == "micromamba" ]]; then
          printf '%s\n' 'if command -v micromamba >/dev/null 2>&1; then'
          printf '%s\n' '  eval "$(micromamba shell hook -s bash 2>/dev/null)"'
          printf '%s\n' "  micromamba activate \"${env_name}\" >/dev/null 2>&1 || true"
          printf '%s\n' 'fi'
        elif [[ "$tool" == "conda" || "$tool" == "mamba" ]]; then
          printf '%s\n' 'if command -v conda >/dev/null 2>&1; then'
          printf '%s\n' '  __conda_base="$(conda info --base 2>/dev/null)"'
          printf '%s\n' '  if [[ -n "${__conda_base:-}" && -f "${__conda_base}/etc/profile.d/conda.sh" ]]; then'
          printf '%s\n' '    . "${__conda_base}/etc/profile.d/conda.sh"'
          printf '%s\n' "    conda activate \"${env_name}\" >/dev/null 2>&1 || true"
          printf '%s\n' '  fi'
          printf '%s\n' 'fi'
        fi
      } > "$auto_dst"

      chmod 0644 "$auto_dst" 2>/dev/null || true
      log_success "Auto-activate drop-in -> $auto_dst"
    else
      log_warning "POST_CREATE_BASHRC_AUTO_ACTIVATE=1 but missing tool/env name (tool=$tool env_name=$env_name)"
    fi
  fi

  log_success "bash_first_run_notice: done"
  return 0
}

# When sourced: isolate in subshell to avoid polluting caller
# if bf_is_sourced; then
#   ( bash_first_run_notice_body "$@" )
#   bf_exit_or_return $?
# else
#   bash_first_run_notice_body "$@"
#   bf_exit_or_return $?
# fi
bash_first_run_notice_body "$@"
bf_exit_or_return $?
