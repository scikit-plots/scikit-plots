#!/usr/bin/env bash
# docker/scripts/.bashrc.d/04-scikit-plots-bashrc-prefix-ps1.sh

# >>> 04-scikit-plots-bashrc-prefix-ps1.sh scikit-plots personal initialization >>>
# ====================================================================

# Interactive shells only (must be first) $- contains shell flags (e.g. himBH).
# case $- in *i*) ;; *) return 0 ;; esac
case $- in *i*) ;; *) return || true ;; esac

# Avoid double execution if prefix is sourced both from /etc/bashrc.d and ~/.bashrc.d
# if [[ -n "${__SCIKIT_PLOTS_BASHRC_PREFIX_PS1_ONCE:-}" ]]; then
#   return 0
# fi
# __SCIKIT_PLOTS_BASHRC_PREFIX_PS1_ONCE=1
[[ -n "${__SCIKIT_PLOTS_BASHRC_PREFIX_PS1_ONCE:-}" ]] && return 0
__SCIKIT_PLOTS_BASHRC_PREFIX_PS1_ONCE=1

# Stop env managers from also editing PS1 (prevents: (py311) (py311))
export CONDA_CHANGEPS1=false
export MAMBA_CHANGEPS1=false

# >>> PS1 attractive prompt >>>
# ====================================================================
# \! - history number of this command
# \# - command number of this command
# \a - alert (bell)
# \e - escape character
# \n - newline
# \u - username
# \h - hostname (up to the first dot)
# \H - hostname (full)
# \s - shell name
# \j - number of jobs currently managed by the shell
# \l - current terminal device name
# \p - current working directory, with tilde (~) for home
# \d - date in "Weekday Month Date" format
# \t - current time in 24-hour HH:MM:SS format
# \@ - current time in 12-hour am/pm format
# \w - current working directory
# \W - basename of current working directory
# \v - shell version
# \V - shell version with patch level
#
# Set up attractive prompt like tf
# export PS1="\[\e[31m\]tf-docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
# Dynamic PS1 with conda/mamba env and git branch (if available)
# Enhanced PS1 with conda env + git branch
# export PS1=''
# # Show conda/mamba environment name if active (if available)
# if command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1 || command -v micromamba >/dev/null 2>&1; then
#   export PS1+='(\[\e[35m\]$(basename "$CONDA_DEFAULT_ENV")\[\e[0m\]) \n'
# fi
# export PS1+="\[\e[31m\]sp-docker\[\e[m\] \[\e[33m\]\w\[\e[m\]"
# # Show git branch name or commit (detached head) if inside a Git repository
# export PS1+='$( \
# if git rev-parse --is-inside-work-tree &>/dev/null; then \
#   BRANCH_NAME=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)
#   echo -n " (\[\e[36m\]${BRANCH_NAME:-}\[\e[m\])"; \
# fi)'
# export PS1+=' \n$ '

__sp_env_name() {
  # Strict precedence
  # if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  #   printf '%s' "${CONDA_DEFAULT_ENV}"
  #   return 0
  # fi
  [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && { printf '%s' "$CONDA_DEFAULT_ENV"; return 0; }
  [[ -n "${VIRTUAL_ENV:-}" ]] && { printf '%s' "${VIRTUAL_ENV##*/}"; return 0; }
  return 1
}

__sp_git_branch() {
  command -v git >/dev/null 2>&1 || return 0
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  local b
  b="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || true)"
  [[ -n "$b" ]] && printf ' (%s)' "$b"
}

__sp_ps1_render() {
  local env=""
  if env="$(__sp_env_name)"; then
    printf '(\[\e[35m\]%s\[\e[0m\]) ' "$env"
  fi
  printf '\n\[\e[31m\]sp-docker\[\e[0m\] \[\e[33m\]\w\[\e[0m\]'
  printf '\[\e[36m\]%s\[\e[0m\]' "$(__sp_git_branch)"
  printf '\n$ '
}

__sp_prompt_update() { PS1="$(__sp_ps1_render)"; }

# Prepend our update (don’t clobber other PROMPT_COMMAND logic)
if [[ -n "${PROMPT_COMMAND:-}" ]]; then
  PROMPT_COMMAND="__sp_prompt_update; ${PROMPT_COMMAND}"
else
  PROMPT_COMMAND="__sp_prompt_update"
fi

__sp_prompt_update

# Single source of truth for prompt
# Primary prompt:
# - env name (if any)
# - "sp-docker" label (red)
# - working dir (yellow)
# - git branch (cyan)
# - newline + $
# export PS1="$(__sp_ps1)"
# export PROMPT_COMMAND="PS1=\"$(__sp_ps1)\""

# ====================================================================
# <<< PS1 attractive prompt <<<

# ====================================================================
# <<< 04-scikit-plots-bashrc-prefix-ps1.sh scikit-plots personal initialization <<<
