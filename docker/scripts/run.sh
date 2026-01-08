#!/usr/bin/env bash
# Strict bash entrypoint

set -Eeuo pipefail
umask 022

log()  { printf '%s\n' "$*" >&2; }
die()  { log "[ERROR] $*"; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing cmd: $1"; }

cleanup() {
  # dev note: cleanup temp resources if you created them
  :
}
trap cleanup EXIT
trap 'die "Failed at line $LINENO: $BASH_COMMAND"' ERR

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run.bash <command> [--config PATH] [--env PATH] [--] [extra args...]
EOF
}

cmd="${1:-}"; [[ -n "$cmd" ]] || { usage; exit 2; }
shift || true

config="configs/base.yaml"
env_file=""

extra_args=()  # keeps quotes exactly

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || die "--config needs value"; config="$2"; shift 2 ;;
    --env)    [[ $# -ge 2 ]] || die "--env needs value"; env_file="$2"; shift 2 ;;
    --) shift; extra_args+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

[[ -f "$config" ]] || die "Missing config: $config"

# Strict .env parsing (no source)
load_env_kv() {
  local f="$1"
  [[ -f "$f" ]] || die "Missing env file: $f"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || die "Invalid .env line: $line"
    local key="${line%%=*}"
    local val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid env var: $key"
    if [[ "$val" == \'*\' ]]; then val="${val:1:-1}"; fi
    if [[ "$val" == \"*\" ]]; then val="${val:1:-1}"; fi
    export "$key=$val"
  done < "$f"
}

if [[ -n "$env_file" ]]; then
  load_env_kv "$env_file"
fi

need_cmd python
python_bin="${PYTHON_BIN:-python}"

"$python_bin" -m project.cli "$cmd" --config "$config" "${extra_args[@]}"
