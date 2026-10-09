"""
Compare the Three Rank-BM25 Scoring Recipes
===========================================

.. currentmodule:: scikitplot.rank_bm25

Scikit-Plots exposes three concrete BM25 recipes: :class:`BM25Okapi`,
:class:`BM25L`, and :class:`BM25Plus`.  This example runs one corpus and query
through all three, shows each versioned recipe identifier, and compares result
ordering without assuming that scores from different formulas share one scale.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from scikitplot.rank_bm25 import BM25L, BM25Okapi, BM25Plus

# %%
# 1. One corpus, one analysis policy
# ----------------------------------
# Keep the corpus and query fixed so that the scoring recipe is the only thing
# changing.

documents = {
    "short": "python plotting library",
    "focused": "python statistical plotting plotting plotting",
    "broad": "python machine learning visualization metrics plotting",
    "other": "database transaction storage engine",
}


def analyze(text: str) -> list[str]:
    return text.lower().split()


doc_ids = list(documents)
corpus = [analyze(documents[doc_id]) for doc_id in doc_ids]
query = analyze("python plotting")

# %%
# 2. Construct each public recipe
# -------------------------------
# The defaults are intentionally left intact.  ``recipe.identifier`` records a
# named/versioned formula choice instead of asking callers to infer semantics
# from the class name alone.

indexes = [
    BM25Okapi(corpus, doc_ids=doc_ids),
    BM25L(corpus, doc_ids=doc_ids),
    BM25Plus(corpus, doc_ids=doc_ids),
]

for index in indexes:
    print(type(index).__name__)
    print("  recipe:", index.recipe.identifier)
    for rank, (doc_id, score) in enumerate(index.get_top_ids(query, n=3), start=1):
        print(f"  {rank}. {doc_id:7s} score={score:.6f}")

# %%
# 3. Compare rankings, not score units
# ------------------------------------
# Each formula is free to produce a different numeric scale.  A robust
# comparison records the ordering and evaluates it against your retrieval task
# rather than comparing a ``0.8`` from one recipe with a ``0.8`` from another.

rankings = {
    type(index).__name__: [
        doc_id for doc_id, _score in index.get_top_ids(query, n=len(doc_ids))
    ]
    for index in indexes
}

print("rankings:")
for name, ranking in rankings.items():
    print(f"  {name:10s}: {ranking}")

assert set(rankings) == {"BM25Okapi", "BM25L", "BM25Plus"}
assert all(ranking for ranking in rankings.values())

# %%
# Parameter domains are validated when an index is built.  Tune a recipe only
# with an evaluation set that represents your application; the gallery avoids
# presenting one parameter choice as universally optimal.
#
# .. tags::
#
#    model-workflow: retrieval
#    plot-type: text
#    level: intermediate
#    purpose: compare
