"""
Sparse Retrieval and Dense BM25 Diagnostics
===========================================

.. currentmodule:: scikitplot.rank_bm25

Rank-BM25 exposes two deliberately different views of the same query.
:meth:`BM25.get_top_ids` is the retrieval path: it scores documents that contain
at least one query term.  :meth:`BM25.get_scores` is a dense diagnostic: it
returns one score for every corpus row, including zeros for non-matches.

Use :meth:`BM25.candidate_count` when you need to see how much of the corpus a
query actually touches.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from scikitplot.rank_bm25 import BM25Okapi

# %%
# 1. Build a corpus with an obvious candidate set
# -----------------------------------------------
# Only three rows contain ``retrieval``; the filler rows make the difference
# between candidate count and corpus size visible without creating a large
# example.

corpus = [
    ["retrieval", "bm25", "lexical"],
    ["retrieval", "hybrid"],
    ["retrieval", "evaluation"],
]
corpus.extend([["filler", str(i)] for i in range(12)])

doc_ids = [f"doc-{i:02d}" for i in range(len(corpus))]
index = BM25Okapi(corpus, doc_ids=doc_ids)
query = ["retrieval"]

print("corpus size:", index.corpus_size)
print("candidate count:", index.candidate_count(query))

assert index.corpus_size == 15
assert index.candidate_count(query) == 3

# %%
# 2. Sparse retrieval returns only matching rows
# ----------------------------------------------

hits = index.get_top_ids(query, n=10)
print("sparse hits:", [doc_id for doc_id, _score in hits])

assert len(hits) == 3

# %%
# 3. Dense diagnostics preserve row alignment
# --------------------------------------------
# ``get_scores`` is useful for evaluation code that wants a score aligned with
# every input row.  It intentionally has different output cardinality from the
# sparse retrieval call above.

scores = index.get_scores(query)
nonzero_positions = [i for i, score in enumerate(scores) if float(score) != 0.0]

print("dense vector length:", len(scores))
print("nonzero positions:", nonzero_positions)

assert len(scores) == index.corpus_size
assert nonzero_positions == [0, 1, 2]

# %%
# 4. The two views agree on scores for actual candidates
# ------------------------------------------------------

sparse_by_id = dict(hits)
for row, doc_id in enumerate(doc_ids[:3]):
    assert math.isclose(sparse_by_id[doc_id], float(scores[row]), rel_tol=1e-12)

print("sparse and dense candidate scores agree:", True)

# %%
# Do not read a dense zero as a retrieved result.  If the application wants
# documents, use ``get_top_ids``; if it wants a row-aligned diagnostic vector,
# use ``get_scores``.
#
# .. tags::
#
#    model-workflow: retrieval
#    plot-type: text
#    level: intermediate
#    purpose: explain
