.. docs/source/user_guide/corpus/index.rst
..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # https://documatt.com/restructuredtext-reference/element/admonition.html
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.corpus

.. _corpus-index:

======================================================================
Corpus (Remarks) Generation
======================================================================

.. image:: /_static/user_guide/scikitplot_corpus_architecture.svg
    :align: center
    :alt: Corpus Architecture

Quick start
-----------

.. rubric:: Examples

.. .. code-block:: python
.. .. code-block:: bash
.. .. jupyter-execute
.. .. prompt:: python >>>
.. prompt:: python >>>

    # First we download the media preproccess libraries (text, image, audio or video).
    # pip install nltk gensim langdetect faster-whisper openai-whisper pytesseract youtube-transcript-api
    # sudo apt-get install tesseract-ocr
    # pip install scikit-plots[corpus]
    from scikitplot import corpus

    print(corpus.__doc__)


.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_corpus_plot_corpus_who_per_file_script.py`: Example notebook.
