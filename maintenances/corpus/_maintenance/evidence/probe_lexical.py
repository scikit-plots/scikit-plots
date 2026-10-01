"""
Probe the MCP lexical leg over 13 realistic technical queries (finding CX-01).

Run from the repository root: ``python -B maintenances/corpus/_maintenance/evidence/probe_lexical.py``.
Exits non-zero if any query makes the leg fail.
"""
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4]))
import tempfile, os, logging
logging.disable(logging.WARNING)
from scikitplot.corpus import SQLiteStorage, CorpusDocument
from scikitplot.mcp._hybrid import Bm25Retriever
path = os.path.join(tempfile.mkdtemp(), "c.db")
st = SQLiteStorage(path)
docs = [
 "Use roc_auc_score() from sklearn.metrics to compute the area under the ROC curve.",
 "The C++ extension annoylib is built with meson.",
 "Error: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').",
 "plot_confusion_matrix is deprecated; use ConfusionMatrixDisplay.from_estimator instead.",
 "Set random-state=42 for reproducible splits.",
 "The NOT operator and the AND keyword appear in this sentence about boolean logic.",
]
for i, t in enumerate(docs):
    st.save(CorpusDocument(doc_id=f"d{i}", input_path="doc.md", chunk_index=i, text=t)) if hasattr(st, "save") else st.add(CorpusDocument(doc_id=f"d{i}", input_path="doc.md", chunk_index=i, text=t))
r = Bm25Retriever.from_corpus_sqlite(path)
queries = ["roc_auc_score()", "sklearn.metrics:roc_auc_score", "C++ extension", 'dtype("float64")', "ValueError: Input contains NaN",
           "random-state", "plot_confusion_matrix deprecated", "AND", "NOT operator", "what is roc auc", "NEAR(roc curve)", "roc*", "area under the curve?"]
bad = 0
for q in queries:
    out = r.search(q, 3)
    status = out.legs[0].status if hasattr(out, "legs") and out.legs else "?"
    ids = [c.doc_id for c in out]
    err = (out.legs[0].error or "")[:60] if out.legs else ""
    bad += status == "failed"
    print(f"{q!r:36} {status:9} {ids} {err}")
print("FAILED legs:", bad, "/", len(queries))

_sys.exit(1 if bad else 0)
