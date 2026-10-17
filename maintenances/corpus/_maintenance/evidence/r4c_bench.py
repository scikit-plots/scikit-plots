"""R4-3 gate: the stages of RetrievalExecutionPlan against a bare search, on a fixed corpus.

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
    RetrievalExecutionPlan, CharBudgetCompressor, StageKind,
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


TOKEN = re.compile(r"\w+")

def tokens(text):
    return TOKEN.findall(text.lower())

def longest_run(query, hits):
    """Deterministic reranker: the longest run of query tokens a hit contains in order."""
    q = tokens(query)
    out = []
    for hit in hits:
        t = tokens(hit.doc.text)
        best = 0
        prev = [0] * (len(t) + 1)
        for a in q:
            cur = [0] * (len(t) + 1)
            for j, b in enumerate(t, 1):
                if a == b:
                    cur[j] = prev[j - 1] + 1
                    best = max(best, cur[j])
            prev = cur
        out.append(float(best))
    return out

def halves(query):
    """Multi-query transform: the two halves of the query."""
    words = query.split()
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])] if mid else []

CONFIG = RetrievalConfig(match_mode="keyword", top_k=10)
PLANS = {
    "search": None,
    "plan_empty": RetrievalExecutionPlan(retrieval=CONFIG),
    "plan_rerank": RetrievalExecutionPlan(retrieval=CONFIG, rerank=longest_run),
    "plan_multi_query": RetrievalExecutionPlan(retrieval=CONFIG, transform=halves, top_k=10),
    "plan_compress_1500": RetrievalExecutionPlan(retrieval=CONFIG, compress=CharBudgetCompressor(1500)),
}
CHUNKERS = {
    "recursive_200": ("recursive", {"chunk_size": 200}),
    "recursive_500": ("recursive", {"chunk_size": 500}),
}
texts = {str(p): p.read_text(encoding="utf-8") for p in files}
results, seconds = {}, {}
QUERY_SETS = {
    "sentence": lambda q: q,
    # The first half of the sentence's words: the query no longer pins one
    # chunk, so ranking matters.
    "half": lambda q: " ".join(q.split()[: max(1, len(q.split()) // 2)]),
}
for label, (name, kwargs) in CHUNKERS.items():
    docs = []
    for path in files:
        docs += CorpusBuilder(BuilderConfig(chunker=name, chunker_kwargs=kwargs, normalize=False,
            filter_kwargs={"min_words": 0, "min_chars": 0})).build(path).documents
    index = RetrievalIndex(CONFIG)
    index.build(docs)
    baseline_ids = {}
    for qset, shape in QUERY_SETS.items():
      for variant, plan in PLANS.items():
          r1 = r5 = r10 = rr = n = covered = kept = 0
          stage_time = {k.value: 0.0 for k in StageKind}
          statuses = {}
          for path in files:
              text = texts[str(path)]
              for k, (s, e) in enumerate(sentences(text)):
                  if k % 4:
                      continue
                  n += 1
                  query = shape(text[s:e])
                  result = index.search(query) if plan is None else plan.run(index, query)
                  statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
                  ids = [h.doc.doc_id for h in result]
                  if plan is None:
                      baseline_ids[(qset, path.name, s)] = ids
                  elif variant == "plan_empty":
                      assert ids == baseline_ids[(qset, path.name, s)], "empty plan differs from search"
                  if plan is not None:
                      for st in result.stages:
                          stage_time[st.stage.value] += st.seconds
                  rel = [i for i, h in enumerate(result)
                         if pathlib.Path(str(h.doc.input_path)).name == path.name and h.doc.char_start is not None
                         and h.doc.char_start <= s and e <= h.doc.char_end]
                  first = rel[0] if rel else None
                  r1 += first == 0
                  r5 += first is not None and first < 5
                  r10 += first is not None and first < 10
                  rr += 1 / (first + 1) if first is not None and first < 10 else 0
                  if plan is not None and result.passages is not None:
                      kept += sum(p.chars for p in result.passages)
                      for p in result.passages:
                          d = p.hit.doc
                          if pathlib.Path(str(d.input_path)).name != path.name or d.char_start is None:
                              continue
                          if any(d.char_start + a <= s and e <= d.char_start + b for a, b in p.spans):
                              covered += 1
                              break
          row = {"queries": n, "recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4), "recall@10": round(r10 / n, 4),
                 "mrr@10": round(rr / n, 4), "statuses": statuses}
          if variant == "plan_compress_1500":
              row["context_chars"] = round(kept / n, 1)
              row["sentence_in_context"] = round(covered / n, 4)
          results[f"{label}/{qset}/{variant}"] = row
          if plan is not None:
              seconds[f"{label}/{qset}/{variant}"] = {k: round(v / n * 1000, 3) for k, v in stage_time.items() if v}
print(json.dumps(results, indent=1))
print(json.dumps({"ms_per_query_by_stage": seconds}, indent=1), file=sys.stderr)
