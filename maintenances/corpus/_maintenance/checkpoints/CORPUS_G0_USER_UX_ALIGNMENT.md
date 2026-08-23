# Corpus G0 User / Documentation / Gallery Alignment Checkpoint

Date: 2026-08-19
Status: VERIFIED — FOCUSED G0 GATE GREEN

## Source authority

```text
archive: scikit-plots(20260818-204251).zip
sha256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Scope

G0 is deliberately documentation/maintenance-only. No gallery `.py` file and no
Corpus processing, retrieval, reader, storage, security, or runtime algorithm is
changed in this increment.

Changed/added paths:

```text
scikitplot/corpus/__init__.py
scikitplot/corpus/README.md
scikitplot/corpus/MAINTAINING.md
scikitplot/corpus/_maintenance/STATE.json
scikitplot/corpus/_maintenance/REGISTRY.md
scikitplot/corpus/_maintenance/VERIFICATION.md
scikitplot/corpus/_maintenance/HISTORY.md
scikitplot/corpus/_maintenance/checkpoints/CORPUS_G0_USER_UX_ALIGNMENT.md
```

## Requirements applied

1. Teach the current public API ladder: Pipeline -> Builder -> Fluent -> Runtime.
2. Preserve `FluentCorpus.build() -> CorpusPlan`.
3. Teach `.materialize()` as the explicit operational boundary.
4. State that materialization itself does not read the source.
5. Describe `RuntimePolicy(allow_network=False)` only as a network-source gate.
6. Prefer generic `RetrievalConfig.index_kwargs` in new documentation.
7. Add a user-facing README rather than expanding maintenance docs into a user guide.
8. Establish the subsequent gallery rule:

```text
missing optional dependency/resource/capability -> visible SKIP
real API/contract/security regression             -> visible FAIL
```

A skip must not fabricate data and must not be implemented as indiscriminate
`except Exception`.

## Verification gate

```text
[x] __init__.py compiles
[x] maintenance tracker gate passes
[x] focused plan/public API tests pass (39 passed; 1 known-environment test deselected)
[x] README portable Fluent/Runtime smoke path executes successfully
[x] no gallery .py file changed in G0
[x] changed-file audit contains only G0 user/maintenance paths
```

## Next exact action

After verification, mark this checkpoint VERIFIED, then start G1 on
`plot_corpus_fluent_corpus_script.py`. Preserve its working declarative sections,
remove stale pre-materialization wording, and add one small real runtime scenario.
Optional dependencies/resources in gallery work must skip seamlessly and visibly.

## Verification result

```text
python -m py_compile scikitplot/corpus/__init__.py
PASS

python scikitplot/corpus/_maintenance/check_trackers.py
PASS — 80 source / 78 test files, 57100 / 31914 LOC

focused plan + API manifest gate
39 passed, 1 deselected

portable README FluentCorpus -> materialize -> run -> memory storage smoke
PASS
```

The deselected test is the already-recorded harness condition where `requests`
is present in the subprocess environment before the configuration assertion. G0
does not weaken or modify that unrelated import-hygiene contract.
