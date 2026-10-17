"""
R5-2 gate: bounded corrective retrieval against a bare search.

Run: ``python bench_r5b.py <repository> faa7665 > bench_r5b.out``.

The corpus is that of the R5-1 benchmark (``bench_r5a.py``): the corpus user
guide at a pinned commit, ``recursive`` (500) chunks, BM25 (keyword). Three
task sets, every 4th sentence of at least 40 characters:

* ``answerable``: the first half of a corpus-guide sentence (82 tasks);
  relevant chunks cover the sentence;
* ``noisy``: the same, with the first three words of an off-corpus sentence
  appended (a query the user muddled);
* ``off_corpus``: the first half of a sentence of the Levenshtein user guide,
  which is not indexed (nothing is relevant: the right answer is none).

Variants: ``search`` (one search, always answers); ``grade_only`` (one
graded round; ``abstained`` returns nothing); ``correct`` (up to 3 rounds:
the query, its first half, its second half). Grader: ``TermCoverageGrader``
at the settings in ``GRADERS``.
Reported per task set: recall@1 (an abstention is a miss), the share
answered, precision@1 over the answered, and mean rounds; for
``off_corpus``, the share abstained.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

from scikitplot.corpus import CorrectiveRetrieval, TermCoverageGrader

EVIDENCE = (
    pathlib.Path(__file__).resolve().parents[3] / "agent" / "_maintenance" / "evidence"
)
sys.path.insert(0, str(EVIDENCE))  # the R5-1 corpus and tasks, reused as they are
from bench_r5a import EVERY, build, pinned_guide, sentences  # noqa: E402

OFF_CORPUS = "docs/source/user_guide/levenshtein"
NOISE_WORDS = 3
#: Grader settings (top_k, accept, reject): the default, then two stricter
#: ones that read only the first hit. A grid over this benchmark's own data,
#: not a tuned choice: there is no held-out set.
GRADERS = ((3, 0.8, 0.3), (1, 0.9, 0.6), (1, 1.0, 0.8))
MAX_ROUNDS = 3


def halves(query, rounds):
    """Return the next half of the query: the first, then the second, then none."""
    words = query.split()
    mid = len(words) // 2
    parts = [" ".join(words[:mid]), " ".join(words[mid:])] if mid else []
    n = len(rounds) - 1
    return parts[n] if n < len(parts) else None


def off_corpus_queries(root, commit):
    """Return the first half of every 4th sentence of the off-corpus guide."""
    out = []
    for path in pinned_guide(root, commit, OFF_CORPUS):
        text = path.read_text(encoding="utf-8")
        for k, (s, e) in enumerate(sentences(text)):
            if k % EVERY:
                continue
            words = text[s:e].split()
            out.append(" ".join(words[: max(1, len(words) // 2)]))
    return out


def score(retrieve, tasks):
    """Score one variant on one task set of ``(query, relevant)``."""
    hit1 = answered = rounds = 0
    for query, relevant in tasks:
        result = retrieve(query)
        hits = list(result)
        rounds += len(getattr(result, "rounds", (None,)))
        if hits:
            answered += 1
            hit1 += hits[0].doc.doc_id in relevant
    n = len(tasks)
    return {
        "recall@1": round(hit1 / n, 4),
        "answered": round(answered / n, 4),
        "precision@1_answered": round(hit1 / answered, 4) if answered else None,
        "rounds": round(rounds / n, 2),
    }


def main(root, commit):
    """Run every variant on every task set and return the report."""
    _, index, answerable = build(pinned_guide(root, commit))
    off = off_corpus_queries(root, commit)
    noisy = [
        (f"{query} {' '.join(off[i % len(off)].split()[:NOISE_WORDS])}", relevant)
        for i, (query, relevant) in enumerate(answerable)
    ]
    task_sets = {
        "answerable": answerable,
        "noisy": noisy,
        "off_corpus": [(query, set()) for query in off],
    }
    variants = {"search": index.search}
    for top_k, accept, reject in GRADERS:
        grader = TermCoverageGrader(top_k=top_k, accept=accept, reject=reject)
        name = f"top{top_k}/{accept}/{reject}"
        variants[f"grade_only {name}"] = CorrectiveRetrieval(
            grader=grader, max_rounds=1
        ).bind(index)
        variants[f"correct {name}"] = CorrectiveRetrieval(
            grader=grader, rewrite=halves, max_rounds=MAX_ROUNDS
        ).bind(index)
    report = {name: len(tasks) for name, tasks in task_sets.items()}
    for label, retrieve in variants.items():
        row = {name: score(retrieve, tasks) for name, tasks in task_sets.items()}
        row["off_corpus"]["abstained"] = round(1 - row["off_corpus"]["answered"], 4)
        report[label] = row
    return report


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    result = main(pathlib.Path(sys.argv[1]), sys.argv[2])
    sys.stdout.write(json.dumps(result, indent=1) + "\n")
