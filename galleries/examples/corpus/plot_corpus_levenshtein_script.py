"""
Use Levenshtein distance with Corpus retrieval
==============================================

.. currentmodule:: scikitplot

:mod:`scikitplot.levenshtein` is an always-importable facade for edit distance.
It prefers the bundled Scikit-Plots implementation, can use the MIT-licensed
RapidFuzz package when available, and retains a dependency-free Python
fallback.  The separately licensed ``Levenshtein`` package is available only
when explicitly requested.

Edit distance is lexical, not semantic.  It is particularly useful for OCR
variants, spelling variants, identifiers, short labels, and deterministic
retrieval fallbacks. ``score_cutoff`` is backend-neutral normalized similarity,
so low-quality candidates can be rejected consistently across accelerators.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from scikitplot import levenshtein
from scikitplot.corpus import CorpusDocument, RetrievalConfig

# %%
# Basic distance
# --------------

print("kitten -> sitting:", levenshtein.distance("kitten", "sitting"))
print("active backend:", levenshtein.backend_info())
assert levenshtein.distance("kitten", "sitting") == 3

# %%
# Rank noisy lexical variants
# ---------------------------

choices = [
    "Scikit-Plots",
    "scikit plot",
    "science plots",
    "scikit-learn",
]

for match in levenshtein.rank(
    "scikit plots",
    choices,
    score_cutoff=0.55,
):
    print(match.choice, round(match.similarity, 3), match.distance)

# %%
# Adapt the scorer to Corpus
# --------------------------
# The adapter is lazy: importing :mod:`scikitplot.levenshtein` does not import
# Corpus.  When used with Corpus it returns normal ``RetrievalHit`` records
# with explicit lexical provenance.

documents = [
    CorpusDocument(
        doc_id="a",
        input_path="memory://a",
        chunk_index=0,
        text="hemoglobin concentration",
        normalized_text="hemoglobin concentration",
    ),
    CorpusDocument(
        doc_id="b",
        input_path="memory://b",
        chunk_index=0,
        text="haemoglobin concentration",
        normalized_text="haemoglobin concentration",
    ),
    CorpusDocument(
        doc_id="c",
        input_path="memory://c",
        chunk_index=0,
        text="platelet count",
        normalized_text="platelet count",
    ),
]

scorer = levenshtein.make_corpus_scorer(score_cutoff=0.60)
hits = scorer(
    "hemoglobin concentration",
    documents,
    RetrievalConfig(top_k=3),
)

for hit in hits:
    print(hit.rank, hit.doc.doc_id, round(hit.score, 3), hit.backend)

assert hits[0].doc.doc_id == "a"
assert hits[0].backend.startswith("levenshtein:")
