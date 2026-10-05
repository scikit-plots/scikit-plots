#!/usr/bin/env bash
# docker/scripts/install_spacy.sh
# ===============================================================
# install_spacy.sh — install spaCy pipelines (models) and check they load
# ===============================================================
# USER NOTES
# - Run it anywhere a shell runs: a GitHub or CircleCI job, a Dockerfile RUN,
#   a devcontainer post-create step, a laptop.
#
#     bash docker/scripts/install_spacy.sh
#     SPACY_MODELS="en_core_web_sm de_core_news_sm" bash docker/scripts/install_spacy.sh
#     bash docker/scripts/install_spacy.sh --list
#     bash docker/scripts/install_spacy.sh --verify-only
#
# - Running it again is cheap: a pipeline that is already installed and
#   loads is left alone.
# - It exits non-zero if any pipeline cannot be loaded at the end.
#
# DEV NOTES
# - `python -m spacy download <name>` picks the pipeline release that matches
#   the installed spaCy, so the name alone is enough and stays correct when
#   spaCy is upgraded.
# - A pipeline is checked by loading it and processing one sentence; that it
#   is importable does not prove it is compatible with the installed spaCy.
#
# ENV VARS
# - SPACY_MODELS            : space-separated pipeline names (default: en_core_web_sm)
# - SPACY_INSTALL_PACKAGE=1 : pip-install spaCy itself when it is missing
# - SPACY_FORCE=1           : download even when the pipeline already loads
# - SPACY_RETRIES           : download attempts per pipeline (default: 3)
# - PIP_EXTRA_ARGS          : extra arguments for pip, e.g. "--no-cache-dir"
# - PYTHON                  : interpreter to use (default: python3, then python)
# ===============================================================

set -Eeuo pipefail

log() { printf '[install_spacy] %s\n' "$*" >&2; }
die() { printf '[install_spacy] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
install_spacy.sh — install spaCy pipelines (models) and check they load.

Usage: install_spacy.sh [--list] [--verify-only] [--help]

  --list         print the pipelines that would be installed and exit
  --verify-only  check that the pipelines load; download nothing
  --help         show this text

Configuration is by environment variable (SPACY_MODELS,
SPACY_INSTALL_PACKAGE, ...); see the header of this file.
EOF
}

MODE="install"
for argument in "$@"; do
  case "$argument" in
    --list) MODE="list" ;;
    --verify-only) MODE="verify" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $argument (try --help)" ;;
  esac
done

SPACY_MODELS="${SPACY_MODELS:-en_core_web_sm}"
SPACY_RETRIES="${SPACY_RETRIES:-3}"

# --- validate the configuration before touching anything ------------------
[[ "$SPACY_RETRIES" =~ ^[1-9][0-9]*$ ]] || die "SPACY_RETRIES must be a positive whole number, got '$SPACY_RETRIES'"
read -r -a MODELS <<<"$SPACY_MODELS"
[[ "${#MODELS[@]}" -gt 0 ]] || die "SPACY_MODELS is empty"
for model in "${MODELS[@]}"; do
  # A pipeline name is a Python package name: nothing pip could read as an
  # option, a path or a URL.
  [[ "$model" =~ ^[a-z][a-z0-9_]*$ ]] || die "not a spaCy pipeline name: '$model'"
done

if [[ "$MODE" == "list" ]]; then
  printf '%s\n' "${MODELS[@]}"
  exit 0
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then PYTHON="python"
  else die "no python interpreter found; set PYTHON"
  fi
fi
command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON='$PYTHON' is not runnable"

read -r -a PIP_ARGS <<<"${PIP_EXTRA_ARGS:-}"

loads() {
  # loads <pipeline>: true when spaCy can load it and process a sentence.
  "$PYTHON" - "$1" >/dev/null 2>&1 <<'PY'
import sys

import spacy

pipeline = spacy.load(sys.argv[1])
document = pipeline("Ada Lovelace wrote the first program.")
sys.exit(0 if len(document) > 0 else 1)
PY
}

if ! "$PYTHON" -c 'import spacy' >/dev/null 2>&1; then
  if [[ "$MODE" == "install" && "${SPACY_INSTALL_PACKAGE:-0}" == "1" ]]; then
    log "installing spaCy"
    "$PYTHON" -m pip install ${PIP_ARGS[@]+"${PIP_ARGS[@]}"} spacy
  else
    die "spaCy is not installed for $PYTHON (set SPACY_INSTALL_PACKAGE=1 to install it)"
  fi
fi

if [[ "$MODE" == "install" ]]; then
  for model in "${MODELS[@]}"; do
    if [[ "${SPACY_FORCE:-0}" != "1" ]] && loads "$model"; then
      log "present: $model"
      continue
    fi
    attempt=1
    until "$PYTHON" -m spacy download "$model" ${PIP_ARGS[@]+"${PIP_ARGS[@]}"}; do
      if [[ "$attempt" -ge "$SPACY_RETRIES" ]]; then
        die "could not download $model after $SPACY_RETRIES attempt(s)"
      fi
      log "download of $model failed (attempt $attempt of $SPACY_RETRIES); retrying"
      attempt=$((attempt + 1))
      sleep $((attempt * 2))
    done
  done
fi

failed=()
for model in "${MODELS[@]}"; do
  loads "$model" || failed+=("$model")
done
if [[ "${#failed[@]}" -gt 0 ]]; then
  die "spaCy cannot load: ${failed[*]}"
fi
version="$("$PYTHON" -c 'import spacy; print(spacy.__version__)')"
log "ok: ${#MODELS[@]} pipeline(s) load with spaCy $version"
