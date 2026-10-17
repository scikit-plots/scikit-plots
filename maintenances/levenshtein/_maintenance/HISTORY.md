# History

## 2026-10-10 — maintenance baseline

Established a dedicated maintenance/skill/docs/gallery surface around the new
`scikitplot.levenshtein` facade.

Baseline runtime tests: 23 passed in the focused lane.

Two non-correctness findings remain open:

- LV-001: runtime accelerator failure skips the remaining safe accelerator;
- LV-002: ranking resolves backend capability per candidate.

No runtime behavior was changed as part of creating this maintenance baseline.

## 2026-10-12 — the Corpus adapter moves to Corpus

`make_corpus_scorer` is removed from the public surface. It was a relevance
ranker for `CustomRetrievalIndex` that imported Corpus lazily, so the
low-level algorithm depended on the pipeline above it. Corpus now owns the
adapter as `scikitplot.corpus.make_levenshtein_ranker` (same ordering,
cutoff and per-hit backend provenance), plus `make_pairwise_ranker` for any
pairwise function. `test_levenshtein_never_imports_corpus` and the structural
checker hold the boundary (L8).

The user guide gained the missing `python_api` page (every public name,
`BackendInfo` included) in its index; both guide-sync checks had failed on the
uploaded tree, so the recorded `user_guide_sync` PASS was not true until now.
The recorded fingerprint was also stale on the uploaded tree.

Lanes re-run at the final fingerprint `5e4dbc96...`: focused runtime 34 passed,
maintenance 8 passed, user-guide sync 7 passed, gallery 5/5.

The same day, an independent review found that the Corpus ranker logged one
warning per document (2000 for 2000) when its backend was unavailable: the
root was LV-002. One per-call `_ExecutionPlan` now resolves the backend once,
warns once per call, and after a runtime failure tries the next available
backend of `internal -> rapidfuzz -> python` (LV-001). Both findings are
closed in `REVIEW.json`, each with the tests that hold its acceptance
criteria. An unknown backend name, or an unavailable one with `strict=True`,
now raises even when there are no choices.

Correction (round 33, increment 2b): the fingerprint `5e4dbc96...` recorded
above was computed before those review fixes changed `_core.py` and the
tests, so the delivered STATE, EVIDENCE and contract review were stale and
`test_structural_contract_is_current` failed. The fingerprint of the tree
after LV-001 and LV-002 is `54b674e0...`. Lanes re-run there: focused runtime
and user-guide sync 41 passed, maintenance 8 passed; the runtime is unchanged
since.
