"""
corpus with examples
====================

.. currentmodule:: scikitplot.corpus

Examples related to the :py:mod:`~scikitplot.corpus` submodule.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Download the related packages
# #############################
#
# First we download the media preproccess libraries (text, image, audio or video).
# pip install nltk gensim langdetect faster-whisper openai-whisper pytesseract youtube-transcript-api
# sudo apt-get install tesseract-ocr

# .. seealso::
#    * galleries/examples/00-jupyter_notebooks/corpus/plot_corpus_from_any_media_notebook.ipynb

# import faster_whisper
# import whisper  # openai-whisper
# import youtube_transcript_api

# %%

import scikitplot as sp
from scikitplot import corpus

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: bar
#    level: beginner
#    purpose: showcase
