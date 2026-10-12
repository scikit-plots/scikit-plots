.. _levenshtein-corpus:

Corpus integration
==================

.. currentmodule:: scikitplot.corpus

The Levenshtein facade is independently importable and never imports Corpus.
Corpus owns the adapter: :func:`make_levenshtein_ranker` returns a ranker for
:class:`CustomRetrievalIndex`.

Create a ranker
---------------

.. code-block:: python

   from scikitplot.corpus import CustomRetrievalIndex, make_levenshtein_ranker

   index = CustomRetrievalIndex(
       custom_scorer_fn=make_levenshtein_ranker(
           backend="auto",
           score_cutoff=0.6,
       )
   )

:mod:`scikitplot.levenshtein` is imported when the ranker is built; a Corpus
installation without it raises ``ImportError`` there, with
:func:`make_pairwise_ranker` and :func:`edit_similarity` named as the
dependency-free alternative.

Use Corpus documents
--------------------

.. code-block:: python

   from scikitplot.corpus import CorpusDocument, RetrievalConfig, make_levenshtein_ranker

   docs = [
       CorpusDocument.create("a.txt", 0, "kitten"),
       CorpusDocument.create("b.txt", 0, "sitting"),
       CorpusDocument.create("c.txt", 0, "bitten"),
   ]

   ranker = make_levenshtein_ranker(backend="python")
   hits = ranker("kitten", docs, RetrievalConfig(top_k=2))

   assert [hit.doc.text for hit in hits] == ["kitten", "bitten"]

Retrieval metadata
------------------

The ranker records:

- ``match_mode="levenshtein"``;
- normalized similarity as the score;
- the actual Levenshtein backend in ``backend`` (``"levenshtein:<name>"``),
  which differs from the requested one after a runtime fallback;
- ``native_metric="normalized_levenshtein_similarity"``.

Normalized versus original text
-------------------------------

By default the ranker prefers ``doc.normalized_text`` when present and falls
back to ``doc.text``.

Pass ``use_normalized_text=False`` when the original document text is the
retrieval contract.

Any pairwise function
---------------------

:func:`make_pairwise_ranker` accepts any ``score_func(query, text)``. Pass
``greater_is_better=False`` for a distance, which is negated so that higher is
better, as :func:`sklearn.metrics.make_scorer` does for a loss:

.. code-block:: python

   from scikitplot import levenshtein
   from scikitplot.corpus import make_pairwise_ranker

   ranker = make_pairwise_ranker(levenshtein.distance, greater_is_better=False)

Measuring retrieval quality
---------------------------

To compare this ranker with BM25 or dense retrieval, run
:func:`evaluate_retrieval` on a query set with relevance judgments; see
:ref:`corpus-evaluation`.

When to use it
--------------

Levenshtein retrieval is useful for:

- short names and identifiers;
- misspelling-tolerant lookup;
- small/medium candidate sets;
- deterministic local matching.

It is usually not the best first-stage retriever for large natural-language
corpora. Use lexical/vector retrieval to narrow the candidate set, then edit
distance as a reranker when appropriate.
