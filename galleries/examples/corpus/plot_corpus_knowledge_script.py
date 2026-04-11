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
    SourceType,
    DocumentReader,
    CorpusDocument,
    CorpusPipeline,
    SentenceBackend,
    SentenceChunker,
    SentenceChunkerConfig,
    ExportFormat,
    EnricherConfig,
    NLPEnricher,
    WordChunkerConfig,
    StemmingBackend,
    LemmatizationBackend,
    WordChunker,
    FixedWindowChunkerConfig,
    FixedWindowChunker,
    StopwordSource,
    WindowUnit,
    TokenizerBackend,
)

# %%
# 1. Word chunker by document
# ----------------------------------------
# via :class:`CorpusPipeline`

pipeline_zip = CorpusPipeline(
    chunker=WordChunker(
        WordChunkerConfig(
            chunk_by="document",
            stemmer=StemmingBackend.PORTER,
            nltk_language="english",
            tokenizer=TokenizerBackend.NLTK,
            lemmatizer=LemmatizationBackend.NLTK_WORDNET,
            stopwords=StopwordSource.BUILTIN,
            lowercase=True,
            remove_punctuation=True,
            min_token_length=2,
            ngram_range=(1,1),
        )
    ),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
)
result_zip = pipeline_zip.run(Path("data/echo_of_the_wise/AI_Generated_Image_1ix.png"))
result_zip

# %%

import pandas as pd
from pprint import pprint

print("Word chunker by document")
pprint(pd.read_csv(result_zip.output_path).head().to_dict())

# %%
# 1. Word chunker by sentence
# ----------------------------------------
# via :class:`CorpusPipeline`

pipeline_zip = CorpusPipeline(
    chunker=WordChunker(
        WordChunkerConfig(
            chunk_by="sentence",
            stemmer=StemmingBackend.SNOWBALL,
            nltk_language="english",
            tokenizer=TokenizerBackend.SIMPLE,
            lemmatizer=LemmatizationBackend.NLTK_WORDNET,
            stopwords=StopwordSource.BUILTIN,
            lowercase=True,
            remove_punctuation=True,
            min_token_length=2,
            ngram_range=(1,1),
        )
    ),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
)
result_zip = pipeline_zip.run(Path("data/echo_of_the_wise/AI_Generated_Image_1ix.png"))
result_zip

# %%

import pandas as pd
from pprint import pprint

print("Word chunker by sentence")
pprint(pd.read_csv(result_zip.output_path).head().to_dict())

# %%
# 2. Sentence chunker
# ----------------------------------------
# via :class:`CorpusPipeline`

pipeline_zip = CorpusPipeline(
    chunker=SentenceChunker(
        SentenceChunkerConfig(
            backend=SentenceBackend.NLTK,
            nltk_language="english",
            strip_whitespace=True,
            include_offsets=True,
        ),
    ),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
)
result_zip = pipeline_zip.run(Path("data/echo_of_the_wise/AI_Generated_Image_1ix.png"))
result_zip

# %%

import pandas as pd
from pprint import pprint

print("Sentence chunker")
pprint(pd.read_csv(result_zip.output_path).head().to_dict())

# %%
# 3. Fixed Window chunker by chars
# ----------------------------------------
# via :class:`CorpusPipeline`

pipeline_zip = CorpusPipeline(
    chunker=FixedWindowChunker(
        FixedWindowChunkerConfig(
            unit=WindowUnit.CHARS,
            min_length=10,
        )
    ),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
)
result_zip = pipeline_zip.run(Path("data/echo_of_the_wise/AI_Generated_Image_1ix.png"))
result_zip

# %%

import pandas as pd
from pprint import pprint

print("Fixed Window chunker by chars")
pprint(pd.read_csv(result_zip.output_path).head().to_dict())

# %%
# 3. Fixed Window chunker by tokens
# ----------------------------------------
# via :class:`CorpusPipeline`

pipeline_zip = CorpusPipeline(
    chunker=FixedWindowChunker(
        FixedWindowChunkerConfig(
            unit=WindowUnit.TOKENS,
            min_length=10,
        )
    ),
    output_dir=Path("output/"),
    export_format=ExportFormat.CSV,
)
result_zip = pipeline_zip.run(Path("data/echo_of_the_wise/AI_Generated_Image_1ix.png"))
result_zip

# %%

import pandas as pd
from pprint import pprint

print("Fixed Window chunker by tokens")
pprint(pd.read_csv(result_zip.output_path).head().to_dict())

# %%

from IPython.display import FileLink, FileLinks

# Replace 'path/to/your_file.csv' with your actual file path
FileLink(result_zip.source)

# %%

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# plt.figure(dpi=300)  # Set DPI to 150
# img = mpimg.imread(result_zip.source)
# plt.imshow(img)
# plt.axis('off')  # hides axes
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
