#!/usr/bin/env bash
# docker/scripts/install_open_clip.sh
# ===============================================================
# install_open_clip.sh — install open_clip models (Hugging Face Hub cache), and prove they load offline
# ===============================================================
# USER NOTES
# - Run it anywhere a shell runs: a GitHub or CircleCI job, a Dockerfile RUN,
#   a devcontainer post-create step, a laptop.
#
#     bash docker/scripts/install_open_clip.sh
#     OPEN_CLIP_MODELS="<model> <model>" bash docker/scripts/install_open_clip.sh
#     bash docker/scripts/install_open_clip.sh --list
#     bash docker/scripts/install_open_clip.sh --verify-only
#
# - Running it again is cheap: a model that already loads offline is not
#   fetched.
# - It exits non-zero if any model does not load with the network switched
#   off (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1), and prints the library's
#   own error for it.
#
# DEV NOTES
# - A model is "<architecture>::<pretrained tag>", as scikitplot.corpus takes
#   it (MultimodalEmbeddingEngine); the tag defaults to "openai".
# - The work is done by install_models.py beside this file (one engine for
#   sentence-transformers, open_clip, openai-whisper and faster-whisper);
#   every fetch and check runs in a child interpreter.
#
# ENV VARS
# - OPEN_CLIP_MODELS : space-separated model names (default: ViT-B-32::openai)
# - MODELS_RETRIES : fetch attempts per model (default: 3)
# - MODELS_FORCE=1 : fetch even when a model already loads offline
# - MODELS_TIMEOUT : seconds per fetch or check (default: 1800)
# - HF_HOME / HF_HUB_CACHE / XDG_CACHE_HOME : where the libraries keep models
# - PYTHON         : interpreter to use (default: python3, then python)
# ===============================================================

set -Eeuo pipefail

HERE="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then PYTHON="python"
  else printf '[install_open_clip.sh] ERROR: no python interpreter found; set PYTHON\n' >&2; exit 1
  fi
fi

args=(--library open_clip "--models=${OPEN_CLIP_MODELS:-ViT-B-32::openai}"
      "--retries=${MODELS_RETRIES:-3}" "--timeout=${MODELS_TIMEOUT:-1800}")
[[ "${MODELS_FORCE:-0}" == "1" ]] && args+=(--force)
for argument in "$@"; do
  case "$argument" in
    --list|--verify-only) args+=("$argument") ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf '[install_open_clip.sh] ERROR: unknown argument: %s (try --help)\n' "$argument" >&2; exit 2 ;;
  esac
done

exec "$PYTHON" "$HERE/install_models.py" "${args[@]}"
