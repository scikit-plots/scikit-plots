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

.. currentmodule:: scikitplot.random

.. _random-index:

Random
======================================================================

Examples relevant to the :py:mod:`~.random` module.

.. seealso::
   * https://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
   * :download:`"Download GoodPracticeRNG.pdf" <./GoodPracticeRNG.pdf>`.

.. rubric:: Examples

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

   from scikitplot.random import default_rng, kiss_context
   rng = default_rng(42)
   data = rng.random(1000)

Context manager

.. prompt:: python >>>

   with default_rng(42) as rng:
      data = rng.random(1000)

   with kiss_context(42) as rng:
      data = rng.random(1000)

Serialization

.. prompt:: python >>>

   import pickle
   state = pickle.dumps(rng)
   restored = pickle.loads(state)

JSON export

.. prompt:: python >>>

   import json
   json_str = json.dumps(rng.serialize())
   restored = KissGenerator.deserialize(json.loads(json_str))
