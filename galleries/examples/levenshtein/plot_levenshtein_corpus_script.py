"""
Use Levenshtein with Corpus
===========================

Rank Corpus documents by edit distance. Corpus owns the adapter,
:func:`scikitplot.corpus.make_levenshtein_ranker`, so the Levenshtein facade
never imports Corpus.
"""

from scikitplot.corpus import CorpusDocument, RetrievalConfig, make_levenshtein_ranker

documents = [
    CorpusDocument.create("a.txt", 0, "kitten"),
    CorpusDocument.create("b.txt", 0, "bitten"),
    CorpusDocument.create("c.txt", 0, "sitting"),
    CorpusDocument.create("d.txt", 0, "unrelated"),
]

ranker = make_levenshtein_ranker(
    backend="python",
    score_cutoff=0.5,
)
hits = ranker("kitten", documents, RetrievalConfig(top_k=3))

assert [hit.doc.text for hit in hits] == ["kitten", "bitten", "sitting"]
assert all(hit.match_mode == "levenshtein" for hit in hits)
assert all(hit.backend == "levenshtein:python" for hit in hits)

for hit in hits:
    print(
        f"{hit.doc.text:10s} "
        f"score={hit.score:.3f} "
        f"backend={hit.backend}"
    )
