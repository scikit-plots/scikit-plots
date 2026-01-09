#!/usr/bin/env bash
#
# docker/scripts/env_conda.sh

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

env_conda_is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }

env_conda_exit_or_return() {
  local rc="${1:-0}"
  if env_conda_is_sourced; then return "$rc"; else exit "$rc"; fi
}

env_conda_is_true() {
  local v="${1:-}"
  v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
  case "$v" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac
}

env_conda_has_cmd() { command -v "$1" >/dev/null 2>&1; }

env_conda_detect_os() {
  local s; s="$(uname -s 2>/dev/null || printf '%s' unknown)"
  case "$s" in
    Linux) echo "Linux" ;;
    Darwin) echo "MacOSX" ;;
    *) echo "unknown" ;;
  esac
}

env_conda_detect_arch() {
  local m; m="$(uname -m 2>/dev/null || printf '%s' unknown)"
  m="$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')"
  case "$m" in
    x86_64|amd64|*amd64*) echo "x86_64" ;;
    aarch64|arm64)        echo "aarch64" ;;
    *)                    echo "unknown" ;;
  esac
}

env_conda_miniforge_installer_name() {
  local os arch
  os="$(env_conda_detect_os)"
  arch="$(env_conda_detect_arch)"
  [[ "$os" != "unknown" && "$arch" != "unknown" ]] || return 2
  printf 'Miniforge3-%s-%s.sh\n' "$os" "$arch"
}

env_conda_install_url() {
  if [[ -n "${CONDA_INSTALL_URL:-}" ]]; then
    printf '%s\n' "$CONDA_INSTALL_URL"
    return 0
  fi
  local name; name="$(env_conda_miniforge_installer_name)" || return 2
  # canonical latest download URL :contentReference[oaicite:7]{index=7}
  printf 'https://github.com/conda-forge/miniforge/releases/latest/download/%s\n' "$name"
}

env_conda_default_prefix() {
  if [[ -n "${CONDA_INSTALL_PREFIX:-}" ]]; then
    printf '%s\n' "$CONDA_INSTALL_PREFIX"
    return 0
  fi
  # deterministic: prefer /opt/conda if writable, else ~/.local/miniforge
  if [[ -w "/opt" ]]; then
    printf '%s\n' "/opt/conda"
  else
    printf '%s\n' "$HOME/.local/miniforge"
  fi
}

env_conda_install_miniforge() {
  if env_conda_has_cmd conda; then return 0; fi

  if ! env_conda_is_true "${CONDA_ALLOW_INSTALL:-1}"; then
    echo "[ERROR] conda not found and CONDA_ALLOW_INSTALL=0" >&2
    return 2
  fi

  (env_conda_has_cmd curl || env_conda_has_cmd wget) || { echo "[ERROR] need curl or wget to install conda" >&2; return 2; }

  local url prefix tmp installer
  url="$(env_conda_install_url)" || { echo "[ERROR] cannot build conda install URL; set CONDA_INSTALL_URL" >&2; return 2; }
  prefix="$(env_conda_default_prefix)"
  tmp="$(mktemp -d 2>/dev/null)" || { echo "[ERROR] mktemp -d failed" >&2; return 2; }
  installer="$tmp/miniforge.sh"

  echo "[INFO] Installing Miniforge -> $prefix" >&2
  echo "[INFO] Download: $url" >&2

  if env_conda_has_cmd curl; then
    curl -fsSL "$url" -o "$installer"
  else
    wget -qO "$installer" "$url"
  fi

  bash "$installer" -b -p "$prefix"
  rm -rf -- "$tmp" 2>/dev/null || true

  export PATH="$prefix/bin:$PATH"
  hash -r 2>/dev/null || true

  env_conda_has_cmd conda || { echo "[ERROR] conda still not available after install (prefix=$prefix)" >&2; return 2; }
}

env_conda_body() {
  # Save caller state; restore inline after (NO trap RETURN)
  local __old_set __old_trap_err
  __old_set="$(set +o)"
  __old_trap_err="$(trap -p ERR || true)"

  set -Eeuo pipefail
  trap 'rc=$?; printf "%s\n" "[ERROR] env_conda.sh failed at line $LINENO: ${BASH_COMMAND-<cmd>} (exit=$rc)" >&2; return $rc' ERR

  local SCRIPT_DIR REPO_ROOT
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"

  local PY_VERSION ENV_NAME ENV_FILE
  PY_VERSION="${PY_VERSION:-3.11}"
  ENV_NAME="${ENV_NAME:-py${PY_VERSION//./}}"
  ENV_FILE="${ENV_FILE:-$REPO_ROOT/environment.yml}"

  local CONDA_SKIP CONDA_MANAGER CONDA_ACTION CONDA_USE_FILE_NAME CONDA_PRUNE
  CONDA_SKIP="${CONDA_SKIP:-0}"
  CONDA_MANAGER="${CONDA_MANAGER:-auto}"      # auto|conda|mamba
  CONDA_ACTION="${CONDA_ACTION:-ensure}"      # ensure|create|update|none
  CONDA_USE_FILE_NAME="${CONDA_USE_FILE_NAME:-0}"
  CONDA_PRUNE="${CONDA_PRUNE:-0}"

  if env_conda_is_true "$CONDA_SKIP"; then
    echo "[INFO] CONDA_SKIP=1 -> skipping conda env" >&2
    eval "$__old_set"; [[ -n "$__old_trap_err" ]] && eval "$__old_trap_err" || trap - ERR
    return 0
  fi

  case "$CONDA_ACTION" in
    none) echo "[INFO] CONDA_ACTION=none -> no action" >&2; return 0 ;;
    ensure|create|update) ;;
    *) echo "[ERROR] Invalid CONDA_ACTION=$CONDA_ACTION" >&2; return 2 ;;
  esac

  if env_conda_is_true "$CONDA_USE_FILE_NAME" && [[ "$CONDA_ACTION" == "ensure" ]]; then
    echo "[ERROR] CONDA_USE_FILE_NAME=1 incompatible with CONDA_ACTION=ensure" >&2
    return 2
  fi

  # ensure conda exists (install if missing)
  if ! env_conda_has_cmd conda && ! env_conda_has_cmd mamba; then
    env_conda_install_miniforge
  fi

  # select manager deterministically
  local mgr=""
  case "$CONDA_MANAGER" in
    conda) env_conda_has_cmd conda || { echo "[ERROR] conda not found" >&2; return 2; }; mgr="conda" ;;
    mamba) env_conda_has_cmd mamba || { echo "[ERROR] mamba not found" >&2; return 2; }; mgr="mamba" ;;
    auto)
      if env_conda_has_cmd conda; then mgr="conda"
      elif env_conda_has_cmd mamba; then mgr="mamba"
      else echo "[ERROR] no conda-compatible manager found even after install" >&2; return 2
      fi
      ;;
    *) echo "[ERROR] Invalid CONDA_MANAGER=$CONDA_MANAGER" >&2; return 2 ;;
  esac

  [[ -f "$ENV_FILE" ]] || { echo "[ERROR] ENV_FILE not found: $ENV_FILE" >&2; return 2; }

  _env_exists() {
    local m="$1" name="$2"
    "$m" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq -- "$name"
  }

  local create_args update_args
  create_args=(env create -f "$ENV_FILE" --yes)
  update_args=(env update -f "$ENV_FILE" --yes)

  if ! env_conda_is_true "$CONDA_USE_FILE_NAME"; then
    create_args+=(-n "$ENV_NAME")
    update_args+=(-n "$ENV_NAME")
  fi
  if env_conda_is_true "$CONDA_PRUNE"; then
    update_args+=(--prune)
  fi

  local exists=0
  if ! env_conda_is_true "$CONDA_USE_FILE_NAME"; then
    _env_exists "$mgr" "$ENV_NAME" && exists=1 || exists=0
  fi

  case "$CONDA_ACTION" in
    create)
      [[ "$exists" == "1" ]] && { echo "[INFO] Env exists: $ENV_NAME (create -> no-op)" >&2; return 0; }
      echo "[INFO] Creating env '$ENV_NAME' from $ENV_FILE" >&2
      "$mgr" "${create_args[@]}"
      ;;
    update)
      [[ "$exists" == "0" ]] && { echo "[ERROR] Env not found: $ENV_NAME (update)" >&2; return 2; }
      echo "[INFO] Updating env '$ENV_NAME' from $ENV_FILE" >&2
      "$mgr" "${update_args[@]}"
      ;;
    ensure)
      if [[ "$exists" == "1" ]]; then
        echo "[INFO] Env exists -> update ($ENV_NAME)" >&2
        "$mgr" "${update_args[@]}"
      else
        echo "[INFO] Env missing -> create ($ENV_NAME)" >&2
        "$mgr" "${create_args[@]}"
      fi
      ;;
  esac

  # restore caller state
  eval "$__old_set"
  if [[ -n "$__old_trap_err" ]]; then eval "$__old_trap_err"; else trap - ERR; fi
  return 0
}

env_conda_body "$@"
env_conda_exit_or_return $?
