"""
corpus WHO European Region local .zip with examples
===================================================

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
# ZIP archive with per-extension kwargs
# --------------------------------------
# Pass a nested ``"reader_kwargs"`` key to configure individual member types
# inside the archive independently.  Global kwargs go alongside it; per-extension
# values always win when both specify the same key.
#
# The pattern mirrors the :class:`~scikitplot.corpus._readers.ZipReader`
# constructor signature — the pipeline threads the outer dict straight through.

# zip_to_doc = list(
#     DocumentReader.create(
#         "data/WHO-EURO-2025-12555-52329-80560-eng.zip",
#         reader_kwargs={
#             ".mp3": {"transcribe": True, "whisper_model": "small"},
#             # ".jpg": {"backend": "easyocr"},   # uncomment to enable OCR on images
#         },
#     ).get_documents()
# )
# zip_to_doc

# %%
# The same ZIP via :class:`CorpusPipeline`
# ----------------------------------------

pipeline_zip = CorpusPipeline(
    chunker=SentenceChunker(SentenceChunkerConfig(backend=SentenceBackend.NLTK)),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
    reader_kwargs={
        "reader_kwargs": {
            ".mp3": {"transcribe": True, "whisper_model": "small"},
            # ".jpg": {"backend": "easyocr"},
        },
    },
)
result_zip = pipeline_zip.run(Path("data/WHO-EURO-2025-12555-52329-80560-eng.zip"))
result_zip

# %%

import pandas as pd
from pprint import pprint

pprint(pd.read_csv(result_zip.output_path).head().to_dict())

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
