"""Probe for CX-04: the MCP Corpus+Annoy profile in each retrieval mode.

Run from the repository root with the Annoy extension loadable::

    python -B maintenances/mcp/_maintenance/evidence/probe_hybrid.py DOCS_DIR

For each mode it prints the top source per query and the legs that ran. Then it
breaks the query embedder, which shows how semantic and hybrid fail. HashEmbedder is near-lexical, so equal hit counts across modes are
not a quality result; the probe checks wiring and failure behaviour only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from scikitplot import corpus
from scikitplot.mcp._corpus_annoy import CorpusAnnoyRetriever

QUERIES = ("roc_auc_score", "ValueError", "train_test_split", "how are errors reported")


class _Broken:
    """Embedder that corpus accepts for building but that fails at query time."""

    def __init__(self, inner):
        self._inner, self.armed = inner, False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, texts, *args, **kwargs):
        if self.armed:
            raise RuntimeError("embedder offline (probe)")
        return self._inner(texts, *args, **kwargs)


def main(docs: str) -> int:
    for mode in ("semantic", "hybrid", "keyword"):
        with tempfile.TemporaryDirectory() as tmp:
            for p in Path(docs).glob("*.md"):
                (Path(tmp) / p.name).write_text(p.read_text())
            r = CorpusAnnoyRetriever.from_corpus_annoy(
                tmp, embedder=corpus.HashEmbedder(dimension=64), mode=mode, n_trees=5
            )
            print(f"== mode={mode} type={type(r).__name__}")
            for q in QUERIES:
                out = r.search(q, 3)
                top = Path(out[0].source_uri).name if out else "-"
                legs = sorted((leg.leg, leg.status) for leg in getattr(out, "legs", []))
                print(f"  {q!r:28} top={top:12} n={len(out)} legs={legs}")
    print("== embedder failure at query time (strict=False)")
    for mode in ("semantic", "hybrid"):
        with tempfile.TemporaryDirectory() as tmp:
            for p in Path(docs).glob("*.md"):
                (Path(tmp) / p.name).write_text(p.read_text())
            emb = _Broken(corpus.HashEmbedder(dimension=64))
            r = CorpusAnnoyRetriever.from_corpus_annoy(tmp, embedder=emb, mode=mode, n_trees=5)
            emb.armed = True
            out = r.search("roc_auc_score", 3)
            legs = sorted((leg.leg, leg.status) for leg in getattr(out, "legs", []))
            print(f"  mode={mode:8} status={out.status} n={len(out)} legs={legs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
