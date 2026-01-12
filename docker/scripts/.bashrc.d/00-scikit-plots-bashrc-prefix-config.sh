#!/usr/bin/env bash
# docker/scripts/.bashrc.d/00-scikit-plots-bashrc-prefix-config.sh

# https://devtooleasy.com/cheat-sheet/bash

# >>> 00-scikit-plots-bashrc-prefix-config.sh scikit-plots personal initialization >>>
# ====================================================================

# Interactive shells only (must be first) $- contains shell flags (e.g. himBH).
# case $- in *i*) ;; *) return 0 ;; esac
# case $- in *i*) ;; *) return || true ;; esac
case $- in
  *i*) ;;  # Interactive shell → continue
  *) return || true ;;  # Not interactive → stop
esac

# Avoid double execution if prefix is sourced both from /etc/bashrc.d and ~/.bashrc.d
if [[ -n "${__SCIKIT_PLOTS_BASHRC_PREFIX_CONFIG_ONCE:-}" ]]; then
  return 0
fi
__SCIKIT_PLOTS_BASHRC_PREFIX_CONFIG_ONCE=1

# ====================================================================
# Set terminal type
# Set TERM to xterm-256color for better color support in terminals
# This is useful for terminals that support 256 colors, like modern terminal emulators.
# It allows for a wider range of colors in terminal applications.
# Colorize the prompt
# \[\e[0;32m\] - green
# \[\e[0;34m\] - blue
# \[\e[0;35m\] - magenta
# \[\e[0;36m\] - cyan
# \[\e[0;33m\] - yellow
# \[\e[0;31m\] - red
# \[\e[0m\] - reset color
export TERM=xterm-256color

# Set up aliases for common commands
# These aliases enhance the usability of common commands by adding options or changing behavior.
# For example, 'ls' with --color=auto enables colored output for better readability.
# alias grep="grep --color=auto"
# alias ls="ls --color=auto"
# alias l='ls -CF'
# alias la='ls -A'
# alias ll='ls -alF'

# -------------------------
# Convenience aliases (safe)
# -------------------------

# Only define aliases if not already defined by the user
__sp_alias() {
  local name="$1" value="$2"
  alias "$name" >/dev/null 2>&1 && return 0
  alias "$name=$value"
}

__sp_alias ll "'ls -alF'"
__sp_alias la "'ls -A'"
__sp_alias l  "'ls -CF'"
__sp_alias grep "'grep --color=auto'"

# Git shortcuts (only if git exists)
if command -v git >/dev/null 2>&1; then
  __sp_alias gs "'git status -sb'"
  __sp_alias gl "'git log --oneline --decorate -n 20'"
  __sp_alias gd "'git diff'"
fi

# unset -f __sp_alias 2>/dev/null || true

# ====================================================================
# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
# HISTCONTROL=ignoreboth
## append to the history file, don't overwrite it
# shopt -s histappend
## for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
# HISTSIZE=1000
# HISTFILESIZE=2000
## check the window size after each command and, if necessary,
## update the values of LINES and COLUMNS.
# shopt -s checkwinsize

# -------------------------
# Shell behavior (safe)
# -------------------------

# Keep terminal sizing correct for wrapped prompts
# shopt -s checkwinsize 2>/dev/null || true

# Append to history, keep duplicates under control
# shopt -s histappend 2>/dev/null || true
# export HISTSIZE="${HISTSIZE:-5000}"
# export HISTFILESIZE="${HISTFILESIZE:-10000}"
export HISTCONTROL="${HISTCONTROL:-ignoreboth:erasedups}"

# Timestamped history (opt-in friendly default)
# export HISTTIMEFORMAT="${HISTTIMEFORMAT:-%F %T  }"

# Safer globbing UX (optional; no errors if unsupported)
# shopt -s globstar 2>/dev/null || true
# shopt -s extglob 2>/dev/null || true

# -------------------------
# Tools / UX defaults
# -------------------------

# Safer pager defaults
# export LESS="${LESS:--FRSX}"
# export PAGER="${PAGER:-less}"

# Make pip quieter & deterministic (does not change install behavior)
# export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
# export PIP_NO_INPUT="${PIP_NO_INPUT:-1}"

# Prefer UTF-8 (only if not already set)
# export LANG="${LANG:-C.UTF-8}"
# export LC_ALL="${LC_ALL:-${LANG}}"

# Editor defaults (non-invasive)
# export EDITOR="${EDITOR:-nano}"
# export VISUAL="${VISUAL:-$EDITOR}"

# ====================================================================
## Alias definitions.
## You may want to put all your additions into a separate file like
## ~/.bash_aliases, instead of adding them here directly.
## See /usr/share/doc/bash-doc/examples in the bash-doc package.
## some more ls aliases
# if [ -f ~/.bash_aliases ]; then
#     . ~/.bash_aliases
# fi

# ====================================================================
# Load modular bash configs if interactive
# ~/.bashrc.d/ and name files like 10-myfeature.bashrc
# if [[ $- == *i* ]]; then
#   for f in ~/.bashrc.d/*.bashrc; do
#     [ -r "$f" ] && . "$f"
#   done
# fi

# ====================================================================
# Fix CUDA loading for running nvidia-smi
([ "$(id -u)" -eq 0 ] && command -v ldconfig >/dev/null && ldconfig) || \
true

# ====================================================================
# <<< 00-scikit-plots-bashrc-prefix-config.sh scikit-plots personal initialization <<<
