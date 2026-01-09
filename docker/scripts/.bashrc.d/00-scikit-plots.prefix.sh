#!/usr/bin/env bash
# >>> (bashrc.prefix) scikit-plots personal initialization >>>
#
# Purpose: early, ordered bash customizations (prompt, env display, minimal UX).
# This file is SOURCED (not executed) from /etc/bash.bashrc and/or ~/.bashrc via *.d loaders.
#
# Contract:
# - Must be safe to source multiple times.
# - Must not print on non-interactive shells.
# - Must not error if conda/micromamba/git are absent.
#
# Notes:
# - Bash reads ~/.bashrc for interactive non-login shells.
# - Login shells commonly source ~/.bashrc from ~/.bash_profile (we ensure that in bash_first_run_notice.sh when enabled).
#
# ====================================================================

# Do not run in non-interactive shells
case $- in
  *i*) ;;
  *) return ;;
esac

# ---------------- Prompt helpers (safe) ----------------
# Avoid errors when CONDA_DEFAULT_ENV is empty, and avoid running external programs unnecessarily.

if ! declare -F __sp_prompt_env >/dev/null 2>&1; then
  __sp_prompt_env() {
    local env=""
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
      env="${CONDA_DEFAULT_ENV##*/}"
    fi
    if [[ -n "$env" ]]; then
      # Magenta env name
      printf '(\[\e[35m\]%s\[\e[0m\]) ' "$env"
    fi
  }
fi

if ! declare -F __sp_prompt_git >/dev/null 2>&1; then
  __sp_prompt_git() {
    command -v git >/dev/null 2>&1 || return 0
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0

    local b=""
    b="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)" || true
    if [[ -n "$b" ]]; then
      # Cyan branch
      printf ' (\[\e[36m\]%s\[\e[m\])' "$b"
    fi
  }
fi

# Primary prompt:
# - env name (if any)
# - "sp-docker" label (red)
# - working dir (yellow)
# - git branch (cyan)
# - newline + $
export PS1='$(__sp_prompt_env)\[\e[31m\]sp-docker\[\e[m\] \[\e[33m\]\w\[\e[m\]$(__sp_prompt_git)\n$ '

# Terminal defaults (only if TERM is empty/dumb)
if [[ -z "${TERM:-}" || "${TERM:-}" == "dumb" ]]; then
  export TERM="xterm-256color"
fi

# Safe aliases (do not fail if commands missing)
alias grep="grep --color=auto" 2>/dev/null || true

# <<< (bashrc.prefix) scikit-plots personal initialization <<<
