# scikitplot/rank_bm25/__init__.py
#
# Authors: D. Brown
# SPDX-License-Identifier: Apache License 2.0

"""
Various BM25 algorithms for document ranking.

A collection of algorithms for querying a set of documents and returning the ones most relevant to the query.
The most common use case for these algorithms is, as you might have guessed, to create search engines.

So far the algorithms that have been implemented are:

* ▣ Okapi BM25
* ▣ BM25L
* ▣ BM25+
* ▢ BM25-Adpt
* ▢ BM25T

.. seealso::
  * https://github.com/dorianbrown/rank_bm25

Examples
--------
Build an index over tokenised documents:

>>> from scikitplot.rank_bm25 import BM25Okapi
>>> corpus = [
...     "Hello there good man!",
...     "It is quite windy in London",
...     "How is the weather today?",
... ]
>>> tokenized_corpus = [doc.split(" ") for doc in corpus]
>>> bm25 = BM25Okapi(tokenized_corpus)
>>> bm25.corpus_size
3

Score every document, which is a diagnostic rather than a retrieval path:

>>> tokenized_query = "windy London".split(" ")
>>> scores = bm25.get_scores(tokenized_query)
>>> [round(float(score), 4) for score in scores]
[0.0, 0.9373, 0.0]

Retrieve the best documents. ``get_top_n`` maps row offsets into the sequence
you pass, so it is correct only while that sequence is in corpus order:

>>> bm25.get_top_n(tokenized_query, corpus, n=1)
['It is quite windy in London']

Prefer binding results to identities the index holds, which has no such
precondition:

>>> bm25 = BM25Okapi(tokenized_corpus, doc_ids=["greeting", "weather", "question"])
>>> [doc_id for doc_id, _score in bm25.get_top_ids(tokenized_query, n=1)]
['weather']

The scoring formula is named and versioned rather than implied by the class:

>>> bm25.recipe.identifier
'okapi-epsilon-floor/1'

"""

from __future__ import annotations

from . import _rank_bm25
from ._rank_bm25 import *  # noqa: F403

__all__ = []
__all__ += _rank_bm25.__all__
