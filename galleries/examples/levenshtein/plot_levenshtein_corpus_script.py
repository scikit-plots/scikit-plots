"""
Use Levenshtein with Corpus
===========================

Build a local deterministic retrieval scorer without making Corpus an import
requirement of the Levenshtein facade.
"""

from scikitplot.corpus import CorpusDocument, RetrievalConfig
from scikitplot.levenshtein import make_corpus_scorer

documents = [
    CorpusDocument.create("a.txt", 0, "kitten"),
    CorpusDocument.create("b.txt", 0, "bitten"),
    CorpusDocument.create("c.txt", 0, "sitting"),
    CorpusDocument.create("d.txt", 0, "unrelated"),
]

scorer = make_corpus_scorer(
    backend="python",
    score_cutoff=0.5,
)
hits = scorer("kitten", documents, RetrievalConfig(top_k=3))

assert [hit.doc.text for hit in hits] == ["kitten", "bitten", "sitting"]
assert all(hit.match_mode == "levenshtein" for hit in hits)
assert all(hit.backend == "levenshtein:python" for hit in hits)

for hit in hits:
    print(
        f"{hit.doc.text:10s} "
        f"score={hit.score:.3f} "
        f"backend={hit.backend}"
    )
