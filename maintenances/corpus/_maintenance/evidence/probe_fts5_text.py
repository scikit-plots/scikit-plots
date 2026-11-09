"""
Probe SQLite FTS5 free-text search with characters no FTS5 string can hold (CX-05).

Run from the repository root:
``python -B maintenances/corpus/_maintenance/evidence/probe_fts5_text.py [ROOT]``.
``ROOT`` defaults to this checkout; pass another checkout to probe it instead.
Writes one line per query and path to standard output, and exits non-zero if
any search raises or any MCP lexical leg fails. Every query gets a fresh store:
CPython 3.11 and newer report a failure to encode a lone surrogate as the
connection's previous error, which would misname it after a failed NUL query.
"""

import logging
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[4]
)
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from scikitplot.corpus import CorpusDocument, SQLiteStorage, StorageQuery  # noqa: E402
from scikitplot.mcp._hybrid import Bm25Retriever  # noqa: E402

QUERIES = [
    "\x00",
    "\x00null",
    "roc\x00auc",
    "roc_auc_score()\x00",
    "\ud800",
    "roc\udfffauc",
    "\udc80null",
    "\x01roc",
    "\x7froc",
    "\x1croc",
    "\ufeffroc",
    "\uffffroc",
    "roc auc",
]
DOCS = [
    "Use roc_auc_score() from sklearn.metrics for the ROC curve.",
    "A null value is not a number.",
    "roc auc appear together here.",
]


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _search_text(store, name, query):
    return [name[doc.doc_id] for doc, _ in store.search_text(query, 3)]


def _full_text(store, name, query):
    found = store.query(StorageQuery(full_text=query, limit=3)).documents
    return [name[doc.doc_id] for doc in found]


def _strict(leg, query):
    return len(leg.search(query, 3))


def _attempt(fn, *args):
    """Return ``(text, failed)``: the call's result, or the exception it raised."""
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001  # the probe records every failure
        return f"RAISE {type(exc).__name__}: {str(exc)[:60]}", True
    return f"ok {out}", False


def _store(path):
    store = SQLiteStorage(path)
    docs = [CorpusDocument.create("p.md", i, text) for i, text in enumerate(DOCS)]
    store.save_batch(docs)
    return store, {doc.doc_id: f"d{i}" for i, doc in enumerate(docs)}


def main() -> int:
    """Probe every query on both storage paths and on the MCP lexical leg."""
    bad = 0
    _say(f"root={ROOT}")
    for query in QUERIES:
        path = os.path.join(tempfile.mkdtemp(), "c.db")
        store, name = _store(path)
        try:
            line1, bad1 = _attempt(_search_text, store, name, query)
        finally:
            store.close()
        store, name = _store(os.path.join(tempfile.mkdtemp(), "c.db"))
        try:
            line2, bad2 = _attempt(_full_text, store, name, query)
        finally:
            store.close()
        leg = Bm25Retriever.from_corpus_sqlite(path)
        strict = Bm25Retriever.from_corpus_sqlite(path, strict=True)
        try:
            out = leg.search(query, 3)
            status = out.legs[0].status if out.legs else "?"
            line3, bad3 = _attempt(_strict, strict, query)
        finally:
            leg.close()
            strict.close()
        _say(f"{query!r:24} search_text {line1:42} full_text {line2:42}")
        _say(f"{'':24} mcp leg {status:8} strict {line3}")
        bad += bad1 + bad2 + (status == "failed") + bad3
    _say(f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
