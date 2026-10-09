"""
Persist and Reload a Rank-BM25 Index
====================================

.. currentmodule:: scikitplot.rank_bm25

A BM25 index can be published to a directory and loaded in a later process.
This example uses pre-tokenized input, stable document identities, and a
temporary directory so the gallery leaves no artifact behind.

The saved state contains derived lexical data such as terms, frequencies, IDF
values, document lengths, parameters, and ``doc_ids``.  Treat the artifact as
potentially sensitive even though it does not serialize the original document
objects.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import tempfile
from pathlib import Path

from scikitplot.rank_bm25 import BM25Okapi

# %%
# 1. Build an index that can identify its results
# -----------------------------------------------

corpus = [
    ["python", "visualization"],
    ["lexical", "retrieval", "bm25"],
    ["vector", "retrieval", "annoy"],
]
doc_ids = ["visualization", "bm25", "annoy"]
index = BM25Okapi(corpus, doc_ids=doc_ids)

before = index.get_top_ids(["retrieval", "bm25"], n=3)
print("before save:", before)
print("build identity:", index.identity)

# %%
# 2. Publish into an isolated temporary directory
# -----------------------------------------------
# ``save`` returns the concrete generation it wrote.  Callers normally keep the
# publication root and let ``load`` follow its current-generation pointer.

with tempfile.TemporaryDirectory(prefix="scikitplot-rank-bm25-") as tmp:
    publication = Path(tmp) / "index"
    generation = index.save(publication)

    print("publication root exists:", publication.is_dir())
    print("generation directory created:", generation.is_dir())

    restored = BM25Okapi.load(publication)
    after = restored.get_top_ids(["retrieval", "bm25"], n=3)

    print("after load:", after)
    print("identity preserved:", restored.identity == index.identity)
    print("document IDs preserved:", restored.doc_ids == index.doc_ids)

    assert restored.identity == index.identity
    assert restored.doc_ids == index.doc_ids
    assert after == before

# %%
# 3. What is intentionally absent from this example
# --------------------------------------------------
# A bare callable passed as ``tokenizer=`` cannot be persisted because it does
# not carry a durable analyzer identity.  The implementation has a declaration
# type for that purpose, but it is not currently exported from the public
# ``scikitplot.rank_bm25`` namespace.  Until that public contract is resolved,
# the gallery keeps persistence examples on the supported pre-tokenized path
# rather than teaching a private import.
#
# .. tags::
#
#    model-workflow: retrieval
#    plot-type: text
#    level: intermediate
#    purpose: workflow
