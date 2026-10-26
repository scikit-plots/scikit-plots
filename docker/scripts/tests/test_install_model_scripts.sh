#!/usr/bin/env bash
# docker/scripts/tests/test_install_model_scripts.sh
#
# bash docker/scripts/tests/test_install_model_scripts.sh
# PYTHON=/path/to/venv/bin/python bash docker/scripts/tests/test_install_model_scripts.sh
#
# Offline tests for install_models.py and its four wrappers
# (install_sentence_transformers.sh, install_open_clip.sh,
# install_openai_whisper.sh, install_faster_whisper.sh). Nothing is
# downloaded: argument handling for all four; with sentence-transformers
# installed for $PYTHON, a missing model must be reported, and a tiny model
# built here must be proved to load offline. Nothing outside a temporary
# directory is written.

set -Eeuo pipefail

HERE="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS="$HERE/.."
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/install_models_tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

passed=0
skipped=0
ok() { passed=$((passed + 1)); printf 'ok %d - %s\n' "$passed" "$1"; }
skip() { skipped=$((skipped + 1)); printf 'ok - # SKIP %s\n' "$1"; }
fail() { printf 'not ok - %s\n' "$1" >&2; exit 1; }

# expect_failure <exit status> <text the error must contain> <description> <command...>
expect_failure() {
  local status="$1" needle="$2" label="$3" output code=0
  shift 3
  output="$("$@" 2>&1)" || code=$?
  [[ "$code" -eq "$status" ]] || fail "$label: exit status $code, expected $status: $output"
  [[ "$output" == *"$needle"* ]] || fail "$label: error did not mention '$needle': $output"
  ok "$label"
}

declare -A DEFAULTS=(
  [install_sentence_transformers.sh]="paraphrase-multilingual-mpnet-base-v2"
  [install_open_clip.sh]="ViT-B-32::openai"
  [install_openai_whisper.sh]="base"
  [install_faster_whisper.sh]="base"
)
declare -A VARIABLES=(
  [install_sentence_transformers.sh]="SENTENCE_TRANSFORMERS_MODELS"
  [install_open_clip.sh]="OPEN_CLIP_MODELS"
  [install_openai_whisper.sh]="WHISPER_MODELS"
  [install_faster_whisper.sh]="FASTER_WHISPER_MODELS"
)

# --- argument handling, every wrapper --------------------------------------
for script in "${!DEFAULTS[@]}"; do
  variable="${VARIABLES[$script]}"
  [[ "$(PYTHON="$PYTHON" bash "$SCRIPTS/$script" --list)" == "${DEFAULTS[$script]}" ]] \
    || fail "$script --list did not print its default"
  ok "$script --list prints its default model"
  [[ "$(env "$variable=m1 m2" PYTHON="$PYTHON" bash "$SCRIPTS/$script" --list | tr '\n' ' ')" == "m1 m2 " ]] \
    || fail "$script --list did not print the configured models"
  ok "$script takes its models from $variable"
  expect_failure 2 "not a model name" "$script refuses a traversal" \
    env "$variable=../etc/passwd" PYTHON="$PYTHON" bash "$SCRIPTS/$script" --list
  expect_failure 2 "not a model name" "$script refuses an option as a model name" \
    env "$variable=--index-url=http://x" PYTHON="$PYTHON" bash "$SCRIPTS/$script" --list
  expect_failure 2 "not a model name" "$script refuses shell syntax" \
    env "$variable=a;b" PYTHON="$PYTHON" bash "$SCRIPTS/$script" --list
  expect_failure 2 "--retries must be" "$script refuses zero retries" \
    env MODELS_RETRIES=0 PYTHON="$PYTHON" bash "$SCRIPTS/$script" --list
  expect_failure 2 "unknown argument" "$script refuses an unknown argument" \
    env PYTHON="$PYTHON" bash "$SCRIPTS/$script" --nope
  PYTHON="$PYTHON" bash "$SCRIPTS/$script" --help | grep -q "$script" \
    || fail "$script --help printed nothing useful"
  ok "$script --help describes the script"
done

# --- a library that is not installed ---------------------------------------
module_present() { "$PYTHON" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('$1') is None)"; }
if module_present open_clip; then
  skip "open_clip is installed for $PYTHON; the not-installed message is not exercised"
else
  expect_failure 1 "pip install open_clip_torch" "a missing library is named with its install command" \
    env PYTHON="$PYTHON" bash "$SCRIPTS/install_open_clip.sh" --verify-only
fi

# --- sentence-transformers: absent and present, offline --------------------
if ! module_present sentence_transformers; then
  skip "sentence-transformers is not installed for $PYTHON (set PYTHON to an environment that has it)"
else
  export HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  expect_failure 1 "MISSING: paraphrase-multilingual-mpnet-base-v2 does not load offline" \
    "a model that is not in the cache is reported with the library's error" \
    env PYTHON="$PYTHON" bash "$SCRIPTS/install_sentence_transformers.sh" --verify-only

  # A two-layer BERT with a ten-word vocabulary, saved as a sentence-transformers
  # model directory: no network, a second or two.
  "$PYTHON" - "$WORK/tiny" >/dev/null 2>&1 <<'PY'
import os
import sys

from sentence_transformers import SentenceTransformer, models
from transformers import BertConfig, BertModel, BertTokenizerFast

root = sys.argv[1]
base = os.path.join(root, "bert")
os.makedirs(base)
words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "first", "second", "one", "sentence"]
with open(os.path.join(base, "vocab.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(words) + "\n")
BertTokenizerFast(os.path.join(base, "vocab.txt")).save_pretrained(base)
config = BertConfig(vocab_size=len(words), hidden_size=8, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=16, max_position_embeddings=32)
BertModel(config).save_pretrained(base)
transformer = models.Transformer(base)
pooling = models.Pooling(transformer.get_word_embedding_dimension())
SentenceTransformer(modules=[transformer, pooling]).save(os.path.join(root, "model"))
PY
  [[ -d "$WORK/tiny/model" ]] || fail "could not build the tiny sentence-transformers model"
  output="$(SENTENCE_TRANSFORMERS_MODELS="$WORK/tiny/model" PYTHON="$PYTHON" \
    bash "$SCRIPTS/install_sentence_transformers.sh" --verify-only 2>&1)" \
    || fail "a model on disk was not proved to load offline: $output"
  [[ "$output" == *"present: $WORK/tiny/model (loads offline; dimension 8)"* ]] \
    || fail "the proof did not report the model's dimension: $output"
  ok "a model on disk is proved to load offline, with its dimension"
fi

printf '1..%d\nall %d checks passed, %d skipped\n' "$passed" "$passed" "$skipped"
