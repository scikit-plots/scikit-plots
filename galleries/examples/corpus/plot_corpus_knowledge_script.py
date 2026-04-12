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
        The pipeline run result (carries ``output_path`` and ``source``).
    """
    pipeline = CorpusPipeline(
        chunker=chunker,
        output_dir=_OUTPUT_DIR,
        export_format=ExportFormat.CSV,
    )
    result = pipeline.run(_IMAGE_PATH)

    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    df = pd.read_csv(result.output_path)
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
# Display the source image
# --------------------------
# Renders inline in Jupyter; opens a matplotlib window otherwise.

print(f"\nSource image: {result_fw_tokens.source}")

if _IN_JUPYTER:
    # IPython display utilities — only import inside Jupyter to avoid
    # ImportError in plain Python / CI environments.
    from IPython.display import FileLink  # noqa: PLC0415

    display(FileLink(str(result_fw_tokens.source)))  # noqa: F821

# plt.figure(figsize=(8, 8), dpi=150)
# img = mpimg.imread(result_fw_tokens.source)
# plt.imshow(img)
# plt.axis("off")
# plt.title("Source image (OCR input)", fontsize=12)
# plt.tight_layout()
# plt.show()

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
