"""
corpus A Tale of Two Cities .mp3 with examples
==============================================

.. currentmodule:: scikitplot.corpus

Examples related to the :py:mod:`~scikitplot.corpus` submodule.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

import os
import json
import sys
import textwrap
from pathlib import Path

import scikitplot as sp
from scikitplot import corpus
from scikitplot.corpus import (
    DocumentReader,
    CorpusPipeline,
    SentenceChunker,
    SentenceChunkerConfig,
    ExportFormat,
    CorpusDocument,
    SourceType,
    SentenceBackend,
    EnricherConfig,
    NLPEnricher,
)

# %%
# Audio URL transcription
# -----------------------
# Pass ``transcribe=True`` and ``whisper_model`` via ``reader_kwargs``.
# These kwargs are forwarded by the pipeline to the :class:`AudioReader`
# constructor — for both :meth:`~CorpusPipeline.run` (local files) and
# :meth:`~CorpusPipeline.run_url` (remote URLs).

pipeline = CorpusPipeline(
    chunker=SentenceChunker(SentenceChunkerConfig(backend=SentenceBackend.NLTK)),
    output_path=Path("output/"),
    export_format=ExportFormat.CSV,
    reader_kwargs={
        "transcribe": True,
        "whisper_model": "base",  # "tiny" | "base" | "small" | "medium" | "large"
    },
)
pipeline

# %%

# https://archive.org/compress/tale_two_cities_librivox/formats=128KBPS%20MP3&file=/tale_two_cities_librivox.zip
# https://archive.org/details/tale_two_cities_librivox/tale_of_two_cities_01_dickens.mp3
# result = pipeline.run_url(
#     "https://archive.org/details/tale_two_cities_librivox/tale_of_two_cities_01_dickens.mp3"
# )
result = pipeline.run(
    # "https://archive.org/details/tale_two_cities_librivox/tale_of_two_cities_01_dickens.mp3"
    "data/tale_of_two_cities_01_dickens_64kb.mp3"
)
result

# %%

print(result.documents)

# %%

print(result.documents[0].text, result.documents[1].text)

# %%

rich1 = NLPEnricher(
    EnricherConfig("nltk", lemmatizer="nltk", stemmer="snowball")
).enrich_documents(result.documents[:1])
rich1

# %%

print(rich1[0].keywords, rich1[0].lemmas, rich1[0].stems)

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
