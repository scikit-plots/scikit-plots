"""
corpus Knowledge and Information local .png with examples
=========================================================

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
# via :class:`CorpusPipeline`
# ----------------------------------------

pipeline_zip = CorpusPipeline(
    chunker=SentenceChunker(SentenceChunkerConfig(backend=SentenceBackend.NLTK)),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
)
result_zip = pipeline_zip.run(Path("data/echo_of_the_wise/Gemini_Generated_Image_1ix.png"))
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
