"""R4-2 gate: span recall and context coverage per chunker on a fixed corpus.

Corpus: docs/source/user_guide/corpus/*.rst at a pinned commit (argv[2]).
Queries: every 4th sentence (regex ``(?<=[.!?])\\s+``) of at least 40 chars
that lies inside one paragraph.
Relevant: a retrieved unit whose context span covers the sentence span.
Ranking: BM25 (RetrievalIndex, match_mode="keyword") over the retrievable
units: every chunk, except for parent_child, where only children are indexed
and each hit returns its parent as context.
Metrics: Recall@1, Recall@5, MRR@10, mean returned-context length, and
paragraph coverage@1 (fraction of the sentence's paragraph -- blank-line
delimited -- inside the top-1 returned context).
"""
import json, logging, pathlib, re, sys
logging.disable(logging.WARNING)
from scikitplot.corpus import (
    BuilderConfig, CorpusBuilder, RetrievalConfig, RetrievalIndex,
)
import subprocess, tempfile
# The corpus is pinned to a commit: the guide this benchmark is published in
# is part of the corpus, so reading the working tree would change the result
# with every edit of the page.
ROOT, COMMIT = pathlib.Path(sys.argv[1]), sys.argv[2]
GUIDE = "docs/source/user_guide/corpus"
names = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", f"{COMMIT}:{GUIDE}"],
                       check=True, capture_output=True, text=True).stdout.split()
pinned = pathlib.Path(tempfile.mkdtemp())
for name in sorted(n for n in names if n.endswith(".rst")):
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{COMMIT}:{GUIDE}/{name}"],
                          check=True, capture_output=True).stdout
    (pinned / name).write_bytes(blob)
files = sorted(pinned.glob("*.rst"))
SENT = re.compile(r"(?<=[.!?])\s+")
PARA = re.compile(r"\n\s*\n")

def sentences(text):
    out, pos = [], 0
    for m in SENT.finditer(text):
        out.append((pos, m.start())); pos = m.end()
    out.append((pos, len(text)))
    # One paragraph only: a "sentence" across a blank line (a heading without
    # a full stop) would penalise every chunker that keeps paragraphs whole.
    return [(s, e) for s, e in out if e - s >= 40 and not PARA.search(text, s, e)]

def paragraph_of(text, s, e):
    start = 0
    for m in PARA.finditer(text):
        if m.start() >= e:
            return start, m.start()
        start = m.end()
    return start, len(text)

CHUNKERS = {
    "sentence": ("sentence", {"min_length": 1}),
    "paragraph": ("paragraph", {}),
    "fixed_window_500": ("fixed_window", {"window_size": 500, "step_size": 400}),
    "recursive_500": ("recursive", {"chunk_size": 500}),
    "recursive_200": ("recursive", {"chunk_size": 200}),
    "parent_child_1500_200": ("parent_child", {"parent_size": 1500, "child_size": 200}),
    "structure_rst_500": ("structure", {"markup": "rst", "chunk_size": 500}),
}
results = {}
for label, (name, kwargs) in CHUNKERS.items():
    docs = []
    for path in files:
        docs += CorpusBuilder(BuilderConfig(chunker=name, chunker_kwargs=kwargs, normalize=False,
            filter_kwargs={"min_words": 0, "min_chars": 0})).build(path).documents
    by_id = {d.doc_id: d for d in docs}
    units = [d for d in docs if d.metadata.get("chunk_role") != "parent"]
    index = RetrievalIndex(RetrievalConfig(match_mode="keyword", top_k=10))
    index.build(units)
    texts = {str(p): p.read_text(encoding="utf-8") for p in files}
    r1 = r5 = rr = cov = clean = n = 0
    for path in files:
        text = texts[str(path)]
        for k, (s, e) in enumerate(sentences(text)):
            if k % 4:
                continue
            n += 1
            hits = list(index.search(text[s:e]))
            ctxs = []
            for h in hits:
                doc = h.doc
                if doc.parent_doc_id and doc.parent_doc_id in by_id:
                    doc = by_id[doc.parent_doc_id]
                ctxs.append(doc)
            rel = [i for i, d in enumerate(ctxs)
                   if pathlib.Path(str(d.input_path)).name == path.name and d.char_start is not None
                   and d.char_start <= s and e <= d.char_end]
            first = rel[0] if rel else None
            r1 += first == 0
            r5 += first is not None and first < 5
            rr += 1 / (first + 1) if first is not None and first < 10 else 0
            if ctxs:
                top = ctxs[0]
                clean += len(top.text)
                ps, pe = paragraph_of(text, s, e)
                if pathlib.Path(str(top.input_path)).name == path.name and top.char_start is not None:
                    inter = max(0, min(pe, top.char_end) - max(ps, top.char_start))
                    cov += inter / (pe - ps)
    results[label] = {"units": len(units), "queries": n,
        "recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4), "mrr@10": round(rr / n, 4),
        "mean_context_chars@1": round(clean / n, 1), "paragraph_coverage@1": round(cov / n, 4)}
print(json.dumps(results, indent=1))
