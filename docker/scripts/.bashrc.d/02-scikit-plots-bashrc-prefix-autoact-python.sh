#!/usr/bin/env bash
# docker/scripts/.bashrc.d/02-scikit-plots-bashrc-prefix-autoact-python.sh
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

# >>> 02-scikit-plots-bashrc-prefix-autoact-python.sh scikit-plots personal initialization >>>
# ====================================================================

# Interactive shells only (must be first) $- contains shell flags (e.g. himBH).
# case $- in *i*) ;; *) return 0 ;; esac
case $- in *i*) ;; *) return || true ;; esac

# Run-once guard
# if [[ -n "${__SCIKIT_PLOTS_BASHRC_PREFIX_AUTOACT_PYTHON_ONCE:-}" ]]; then
#   return 0
# fi
# __SCIKIT_PLOTS_BASHRC_PREFIX_AUTOACT_PYTHON_ONCE=1
[[ -n "${__SCIKIT_PLOTS_BASHRC_PREFIX_AUTOACT_PYTHON_ONCE:-}" ]] && return 0
__SCIKIT_PLOTS_BASHRC_PREFIX_AUTOACT_PYTHON_ONCE=1

# logname, An alternative to whoami is id -u -n. id -u will return the user id (e.g. 0 for root).
# export MAMBA_ROOT_PREFIX="/c/Users/$(whoami)/AppData/Roaming/mamba";
# export MAMBA_EXE="/c/Users/$(whoami)/AppData/Local/micromamba/micromamba.exe";

# Controls (strict defaults)
: "${SP_AUTO_ACTIVATE:=1}"          # set to 1 to enable

: "${SP_ENV_FALLBACK:=base}"        # fallback env
: "${ENV_NAME:=py311}"              # Default to py311 if ENV_NAME is not set

: "${MAMBA_ROOT_PREFIX:=${HOME}/micromamba}"
: "${MAMBA_EXE:=micromamba}"

: "${CONDA_ROOT_PREFIX:=/opt/conda}"
: "${CONDA_EXE:=${CONDA_ROOT_PREFIX}/bin/conda}"

# Avoid Prevent duplicate "(env)" (your PS1 owns env display)
export CONDA_CHANGEPS1="${CONDA_CHANGEPS1:-false}"
export MAMBA_CHANGEPS1="${MAMBA_CHANGEPS1:-false}"

__sp_has() { command -v "$1" >/dev/null 2>&1; }

__sp_exe_ok() {
  local exe="$1"
  # if [[ "$exe" == */* ]]; then [[ -f "$exe" && -r "$exe" ]]; else command -v "$exe" >/dev/null 2>&1; fi
  if [[ "$exe" == */* || "$exe" == *\\* ]]; then
    [[ -f "$exe" && -r "$exe" ]]
  else
    command -v "$exe" >/dev/null 2>&1
  fi
}

__sp_mm_env_exists() {
  local name="$1"
  [[ "$name" == "base" ]] && return 0
  [[ -d "${MAMBA_ROOT_PREFIX}/envs/${name}" ]]
}

__sp_conda_env_exists() {
  local name="$1"
  [[ "$name" == "base" ]] && return 0
  [[ -d "${CONDA_ROOT_PREFIX}/envs/${name}" ]] || [[ -d "${HOME}/.conda/envs/${name}" ]]
}

__sp_mm_activate_eval() {
  local name="$1"
  __sp_exe_ok "$MAMBA_EXE" || return 1
  local code
  # initialize the current bash
  # eval "$(micromamba.exe shell hook --shell bash)"
  # automatically initialize all future (bash)
  # micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
  # reinitialize your shell with:
  # micromamba shell reinit --shell bash
  # eval "$("$MAMBA_EXE" shell hook --shell bash)"
  # eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
  # "$MAMBA_EXE" shell init --shell bash --root-prefix="$MAMBA_ROOT_PREFIX"
  code="$("$MAMBA_EXE" shell activate -s bash -n "$name" 2>/dev/null)" || return 1
  [[ -n "$code" ]] || return 1
  eval "$code" >/dev/null 2>&1 || true
}

__sp_conda_activate_eval() {
  local name="$1"
  # if ! __sp_exe_ok "$CONDA_EXE"; then
  #   command -v conda >/dev/null 2>&1 || return 1
  #   CONDA_EXE="conda"
  # fi
  __sp_exe_ok "$CONDA_EXE" || return 1
  local code
  code="$("$CONDA_EXE" shell.bash activate "$name" 2>/dev/null)" || return 1
  [[ -n "$code" ]] || return 1
  eval "$code" >/dev/null 2>&1 || true
}

sp_activate() {
  local target="${1:-}"
  [[ -n "$target" ]] || target="${ENV_NAME:-}"
  [[ -n "$target" ]] || target="${SP_ENV_FALLBACK}"

  # Already active
  [[ "${CONDA_DEFAULT_ENV:-}" == "$target" ]] && return 0

  # 1) micromamba
  if __sp_exe_ok "$MAMBA_EXE"; then
    if [[ -n "${ENV_NAME:-}" ]] && __sp_mm_env_exists "$target"; then
      __sp_mm_activate_eval "$target" || true
    else
      __sp_mm_activate_eval "$SP_ENV_FALLBACK" || true
    fi
    return 0
  fi

  # 2) conda-family
  if __sp_exe_ok "$CONDA_EXE"; then
    if [[ -n "${ENV_NAME:-}" ]] && __sp_conda_env_exists "$target"; then
      __sp_conda_activate_eval "$target" || true
    else
      __sp_conda_activate_eval "$SP_ENV_FALLBACK" || true
    fi
    return 0
  fi
}

# legacy-friendly replacement for: source activate
# activate() { sp_activate "$@"; }

# Auto-activate
# if [[ "$SP_AUTO_ACTIVATE" == "1" ]]; then
#   sp_activate "${ENV_NAME:-}"
# fi
[[ "$SP_AUTO_ACTIVATE" == "1" ]] && sp_activate "${ENV_NAME:-}"

# unset -f __sp_has __sp_exe_ok __sp_mm_env_exists __sp_conda_env_exists \
#   __sp_mm_activate_eval __sp_conda_activate_eval 2>/dev/null || true
# ---------------------------------------------------------------------

# ====================================================================
# <<< 02-scikit-plots-bashrc-prefix-autoact-python.sh scikit-plots personal initialization <<<
