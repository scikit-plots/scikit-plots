.. currentmodule:: scikitplot.corpus

.. _corpus-retrieval-similarity:

Retrieval and similarity
========================

Corpus supports its existing lexical/vector/hybrid retrieval stack and can also
consume user scorers through :class:`CustomRetrievalIndex`.

Levenshtein facade
------------------

:mod:`scikitplot.levenshtein` is intentionally independent from Corpus and is
always importable. The automatic backend preference is:

.. code-block:: text

   bundled scikitplot.cexternals._editdistance
       -> RapidFuzz when installed
       -> dependency-free Python fallback

The separately distributed ``Levenshtein``/``python-Levenshtein`` backend is
supported only when explicitly requested; it is not silently selected by the
automatic chain.

.. code-block:: python

   from scikitplot.levenshtein import distance, rank

   distance("kitten", "sitting")
   # 3

   rank(
       "colour",
       ["color", "collar", "cloud"],
       score_cutoff=0.60,
   )

Corpus compatibility is lazy:

.. code-block:: python

   from scikitplot.levenshtein import make_corpus_scorer
   from scikitplot.corpus import CustomRetrievalIndex

   index = CustomRetrievalIndex(
       custom_scorer_fn=make_corpus_scorer(
           backend="auto",
           score_cutoff=0.70,
       )
   )

Importing :mod:`scikitplot.levenshtein` alone does not import Corpus.

``score_cutoff`` uses the same normalized similarity scale ``[0, 1]`` for every
backend, so switching between bundled, RapidFuzz and pure-Python implementations
does not change the filtering contract. ``top_k`` is therefore an upper bound:
a cutoff can intentionally return fewer hits when the remaining candidates are
too dissimilar.

Choosing a metric
-----------------

Levenshtein distance is a character/sequence edit metric, not a semantic
embedding model. It is useful for spelling variants, OCR noise, identifiers,
near-duplicate labels and fuzzy lexical matching. Do not present edit-distance
similarity as semantic equivalence.
