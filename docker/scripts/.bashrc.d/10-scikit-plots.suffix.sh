#!/usr/bin/env bash
# >>> (bashrc.suffix) scikit-plots personal initialization >>>
#
# Purpose: late, ordered bash customizations (help text, optional UX).
# This file is SOURCED (not executed) from ~/.bashrc via ~/.bashrc.d loader.
#
# Contract:
# - Must be safe to source multiple times.
# - Must not print on non-interactive shells (unless user explicitly enables).
# - Must not perform environment activation (handled elsewhere).
#
# ====================================================================

case $- in
  *i*) ;;
  *) return ;;
esac

# Optional: show first-run hint (only if notice exists and flag enabled)
if [[ "${SP_SHOW_NOTICE:-0}" == "1" ]]; then
  if [[ -f "${HOME}/.bash-first-run-notice.txt" ]]; then
    cat -- "${HOME}/.bash-first-run-notice.txt" 2>/dev/null || true
  fi
fi

# Put additional user UX tweaks below (safe defaults).
# Example: enable persistent history settings (optional).
if [[ "${SP_TUNE_HISTORY:-0}" == "1" ]]; then
  export HISTSIZE="${HISTSIZE:-5000}"
  export HISTFILESIZE="${HISTFILESIZE:-20000}"
  shopt -s histappend 2>/dev/null || true
fi

# <<< (bashrc.suffix) scikit-plots personal initialization <<<
