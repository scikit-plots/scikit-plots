"""
corpus WHO European Region YouTube shorts with examples
=======================================================

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

pipeline = CorpusPipeline(
    chunker=SentenceChunker(SentenceChunkerConfig(backend=SentenceBackend.NLTK)),
    output_path=Path("output/"),
    export_format=ExportFormat.CSV,
)
pipeline

# %%

# Unfortunately, most IPs from cloud providers are blocked by YouTube.
# result = pipeline.run_url("https://www.youtube.com/shorts/VMZ40dVugAk")

# Richard Feynman - The Character of Physical Law (1964) - Complete - Better Audio
# https://www.youtube.com/watch?v=kEx-gRfuhhk
# result = pipeline.run("https://www.youtube.com/shorts/VMZ40dVugAk")
# result

# %%

# print(result.documents)

# %%

# print(result.documents[0].text, result.documents[1].text)

# %%

# rich1 = NLPEnricher(EnricherConfig("nltk", lemmatizer="nltk", stemmer="snowball")).enrich_documents(result.documents[:1])
# rich1

# %%

# print(rich1[0].keywords, rich1[0].lemmas, rich1[0].stems)

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
