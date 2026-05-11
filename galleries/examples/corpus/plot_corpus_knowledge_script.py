"""
corpus Knowledge and Information local .png with examples
=========================================================

.. currentmodule:: scikitplot.corpus

Examples related to the :py:mod:`~scikitplot.corpus` submodule.
Demonstrates all four chunkers (WordChunker-by-document, WordChunker-by-sentence,
SentenceChunker, FixedWindowChunker-chars, FixedWindowChunker-tokens) on an
image file containing multi-script text extracted via OCR.

Notes
-----
**User note:** Run from any working directory — paths are resolved relative
to this script's location, not the caller's CWD.

**Developer note:** ``FileLink`` / ``FileLinks`` (IPython display utilities)
are guarded behind ``_IN_JUPYTER`` so this script executes correctly in
plain Python, pytest, Docker CI, and notebook contexts alike.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import os
import sys
from pathlib import Path
from pprint import pprint

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

import scikitplot as sp  # noqa: F401  (kept for side effects / version logging)
from scikitplot.corpus import (
    CorpusPipeline,
    ExportFormat,
    FixedWindowChunker,
    FixedWindowChunkerConfig,
    NLPEnricher,
    EnricherConfig,
    SentenceBackend,
    SentenceChunker,
    SentenceChunkerConfig,
    SourceType,
    StemmingBackend,
    StopwordSource,
    TokenizerBackend,
    LemmatizationBackend,
    WindowUnit,
    WordChunker,
    WordChunkerConfig,
)

# ---------------------------------------------------------------------------
# Path resolution — always relative to this file, not caller's CWD.
# ---------------------------------------------------------------------------

# _SCRIPT_DIR: Path = Path(__file__).resolve().parent
_SCRIPT_DIR = Path.cwd()
_DATA_DIR: Path = _SCRIPT_DIR / "data"
_OUTPUT_DIR: Path = _SCRIPT_DIR / "output"
_IMAGE_PATH: Path = _DATA_DIR / "echo_of_the_wise" / "AI_Generated_Image_1ix.png"

# Detect Jupyter environment once — used to guard IPython display utilities.
_IN_JUPYTER: bool = "ipykernel" in sys.modules

# ---------------------------------------------------------------------------
# Helper: build a pipeline and run it on the shared image path.
# ---------------------------------------------------------------------------


def _run(chunker: object, label: str) -> object:
    """Build a CorpusPipeline, run it, print head, return result.

    Parameters
    ----------
    chunker : object
        An instantiated chunker (WordChunker, SentenceChunker, etc.).
    label : str
        Human-readable label printed before the CSV head.

    Returns
    -------
    object
        The pipeline run result (carries ``output_path`` and ``input_path``).
    """
    pipeline = CorpusPipeline(
        chunker=chunker,
        output_path=_OUTPUT_DIR,
        export_format=ExportFormat.CSV,
    )
    result = pipeline.run(_IMAGE_PATH)

    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)

    # Guard 1: pipeline produced no documents — CSV is header-only.
    # result.n_documents is always 0 in this case; skip pd.read_csv()
    # so we never hit EmptyDataError even if the caller does not have
    # the header-only fix deployed on the exporter side.
    if result.n_documents == 0:
        print("[WARNING] Pipeline produced 0 documents — CSV contains no data rows.")
        return result

    # Guard 2: output_path may be None when export is skipped (e.g. no
    # output_path supplied to CorpusPipeline).  Should not happen in this
    # script, but fail fast with a clear message rather than AttributeError.
    if result.output_path is None:
        print("[WARNING] No output_path in result — export was skipped.")
        return result

    # Guard 3: catch residual EmptyDataError for any edge-case where the
    # exporter writes a zero-byte file (e.g. older exporter version).
    try:
        df = pd.read_csv(result.output_path)
    except pd.errors.EmptyDataError:
        print(
            f"[WARNING] CSV at {result.output_path!s} is empty — "
            "no rows to display.  Check exporter version."
        )
        return result

    pprint(df.head().to_dict())
    return result


# %%
# 1. Word chunker — chunk_by="document"
# ---------------------------------------
# One chunk per image (all OCR text joined as a single document).
# Demonstrates PORTER stemming + BUILTIN stopwords.

result_word_doc = _run(
    WordChunker(
        WordChunkerConfig(
            chunk_by="document",
            stemmer=StemmingBackend.PORTER,
            nltk_language="english",
            tokenizer=TokenizerBackend.NLTK,
            lemmatizer=LemmatizationBackend.NLTK_WORDNET,
            stopwords=StopwordSource.BUILTIN,
            lowercase=True,
            remove_punctuation=False,
            min_token_length=2,
            ngram_range=(1, 1),
        )
    ),
    label="Word chunker — chunk_by='document' (PORTER stemming)",
)

# %%
# 2. Word chunker — chunk_by="sentence"
# ----------------------------------------
# One chunk per sentence, each tokenised separately.
# Demonstrates SNOWBALL stemming on English text.

result_word_sent = _run(
    WordChunker(
        WordChunkerConfig(
            chunk_by="sentence",
            stemmer=StemmingBackend.SNOWBALL,
            nltk_language="english",
            tokenizer=TokenizerBackend.SIMPLE,
            lemmatizer=LemmatizationBackend.NLTK_WORDNET,
            stopwords=StopwordSource.BUILTIN,
            lowercase=True,
            remove_punctuation=False,
            min_token_length=2,
            ngram_range=(1, 1),
        )
    ),
    label="Word chunker — chunk_by='sentence' (SNOWBALL stemming)",
)

# %%
# 3. Sentence chunker (NLTK backend)
# ------------------------------------
# Splits OCR text into individual sentences; preserves raw text with offsets.

result_sentence = _run(
    SentenceChunker(
        SentenceChunkerConfig(
            backend=SentenceBackend.NLTK,
            nltk_language="english",
            strip_whitespace=True,
            include_offsets=True,
        )
    ),
    label="Sentence chunker (NLTK backend)",
)

# %%
# 4. Fixed Window chunker — unit=CHARS
# --------------------------------------
# Splits by character count regardless of word/sentence boundaries.

result_fw_chars = _run(
    FixedWindowChunker(
        FixedWindowChunkerConfig(
            unit=WindowUnit.CHARS,
            window_size=512,
            step_size=256,
            min_length=10,
        )
    ),
    label="Fixed Window chunker — unit=CHARS (window=512, step=256)",
)

# %%
# 5. Fixed Window chunker — unit=TOKENS
# ----------------------------------------
# Splits by whitespace-delimited token count.
# CJK text is auto-handled via character-level fallback.

result_fw_tokens = _run(
    FixedWindowChunker(
        FixedWindowChunkerConfig(
            unit=WindowUnit.TOKENS,
            window_size=64,
            step_size=32,
            min_length=10,
        )
    ),
    label="Fixed Window chunker — unit=TOKENS (window=64, step=32)",
)

# %%
# 6. Semantic Chunker with MultilangConfig
# ----------------------------------------
# Every chunk carries chunk.metadata["multilang"] with:
#   script, script_direction, is_rtl, grapheme_count, codepoint_count,
#   token_count, stopword_count, unique_token_count, avg_token_length,
#   char_count, chunking_duration_ms, preprocessing_duration_ms,
#   created_at_utc, layer2_strategy,
#   semantemes[{surface, morphemes, lemma, stem, pos_tag, ...}],
#   preprocessing_trace[{steps, raw_text, pipeline_fingerprint}]

from scikitplot.corpus._chunkers import (
    MultilangConfig,
    SemanticChunker,
    SemanticChunkerConfig,
    SemanticBackend,
)

# Build MultilangConfig with all enhanced features enabled.
ml = MultilangConfig(
    include_raw_text=True,              # preserve pre-NFC raw text per chunk
    include_preprocessing_trace=True,   # full audit trail: BOM strip, control strip, NFC
    include_semantemes=True,            # SemantemeInfo per token
    include_grapheme_counts=True,       # UAX #29 grapheme cluster counts
    include_script_spans=True,          # per-script span list for mixed-script chunks
)

# Bug fix A: pass multilang_config=ml so the SemanticChunker uses the
# configured feature flags, not its own default MultilangConfig.
result_semantic = _run(                 # Bug fix B: renamed from result_fw_tokens
    SemanticChunker(
        SemanticChunkerConfig(
            backend=SemanticBackend.HYBRID,
            model_name="paraphrase-multilingual-mpnet-base-v2",
            multilang_config=ml,        # <-- was missing: ml was built but discarded
        )
    ),
    label="Semantic chunker (HYBRID backend, multilang enriched)",
)


# %%
# Display the source image
# --------------------------
# Renders inline in Jupyter; opens a matplotlib window otherwise.

print(f"\nSource image: {result_fw_tokens.input_path}")

if _IN_JUPYTER:
    # IPython display utilities — only import inside Jupyter to avoid
    # ImportError in plain Python / CI environments.
    from IPython.display import FileLink  # noqa: PLC0415

    display(FileLink(str(result_fw_tokens.input_path)))  # noqa: F821

plt.figure(figsize=(4, 4), dpi=150)
img = mpimg.imread(result_fw_tokens.input_path)
plt.imshow(img)
plt.axis("off")
plt.title("Source image (OCR input)", fontsize=12)
plt.tight_layout()
plt.show()

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
