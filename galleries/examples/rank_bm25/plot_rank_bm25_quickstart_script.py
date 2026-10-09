"""
Rank-BM25 Quick Start with Stable Document IDs
===============================================

.. currentmodule:: scikitplot.rank_bm25

A useful lexical-search result should name the document it belongs to, not ask
its caller to remember which list happened to occupy the same row order as the
index.  This example builds :class:`BM25Okapi` with ``doc_ids`` and retrieves
``(document_id, score)`` pairs with :meth:`BM25.get_top_ids`.

It also checks two small but important contracts: an out-of-vocabulary query
returns no sparse hits, and equal scores are returned in original corpus order.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from scikitplot.rank_bm25 import BM25Okapi

# %%
# 1. Tokenize the corpus once
# ---------------------------
# BM25 consumes token sequences.  Keep the same normalization policy for the
# corpus and every query; here that policy is deliberately just lowercase plus
# whitespace splitting so the example stays transparent.

documents = [
    "A search index maps terms to documents",
    "BM25 is a lexical ranking function",
    "Vector search ranks documents by embedding similarity",
    "Hybrid retrieval can combine lexical and vector evidence",
]
doc_ids = ["indexing", "bm25", "vectors", "hybrid"]


def analyze(text: str) -> list[str]:
    """Apply the same tiny analysis policy to documents and queries."""
    return text.lower().split()


corpus = [analyze(text) for text in documents]
index = BM25Okapi(corpus, doc_ids=doc_ids)

print("corpus rows:", index.corpus_size)
print("bound document IDs:", index.doc_ids)

# %%
# 2. Retrieve by identity
# -----------------------
# ``get_top_ids`` only considers rows containing at least one query term and
# returns identities held by the index itself.

query = analyze("lexical BM25 ranking")
hits = index.get_top_ids(query, n=3)

for position, (doc_id, score) in enumerate(hits, start=1):
    print(f"{position}. {doc_id:8s} score={score:.6f}")

assert hits
assert hits[0][0] == "bm25"

# %%
# 3. Unknown terms do not manufacture results
# --------------------------------------------
# A dense score vector has a zero for every non-match, but sparse retrieval has
# no reason to return arbitrary zero-score rows.  A query with no indexed term
# therefore returns an empty list.

missing = index.get_top_ids(analyze("quasar-neutrino"), n=3)
print("out-of-vocabulary hits:", missing)
assert missing == []

# %%
# 4. Ties are deterministic
# -------------------------
# Equal scores preserve corpus row order.  This makes repeated runs stable and
# keeps the identity path aligned with the positional path.

tied = BM25Okapi(
    [["same"], ["same"], ["same"]],
    doc_ids=["first", "second", "third"],
)
tied_ids = [doc_id for doc_id, _score in tied.get_top_ids(["same"], n=3)]

print("tie order:", tied_ids)
assert tied_ids == ["first", "second", "third"]

# %%
# The next examples compare scoring recipes, explain sparse versus dense paths,
# and persist an index.  The full contracts and failure modes live in
# :ref:`rank-bm25-index`.
#
# .. tags::
#
#    model-workflow: retrieval
#    plot-type: text
#    level: beginner
#    purpose: tutorial
