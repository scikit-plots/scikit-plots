# History — `scikitplot.cython`

Read only when the *why* behind a current rule is unclear.

## The campaign that shaped the project

A 30-finding review and fix campaign ran on this submodule and grew the suite
from 966 to 1 255 passing tests. Its most instructive finding:

> A genuine interprocess exclusivity bug — a non-blocking lock probe with
> `timeout_s=0` **destroyed live locks** held by other processes.

Not a crash. A probe that answered its question by breaking the thing it was
asking about, and which no single-process test could see.

That campaign produced `MAINTAINING.md` with guard tests making every finding
permanently verifiable, and it established the methodology now applied to every
other submodule:

- zero hallucination; source-grounded claims
- minimal-impact root-cause fixes only
- an always-green gate — rewrite tests to correct contracts, never delete them
- per-turn evidence: a named test and reproducible output
- Conventional Commits, NumPyDoc, no `==` pins, Python 3.8 → 3.15+

**Corpus, MCP, the annoy family, the sphinx family and `_cli` all inherited that
discipline from here.** This submodule was first, and its maintenance pair is
written last — which is why this set adds so little to its verification and so
much to the *record* of why its tests exist.

## Its own documents

`ADR-0001-runtime-lifecycle.md` records the lifecycle decision.
`OPERATIONS.md` and `DEV_NOTES.md` are both **tested** —
`test__operations_docs.py` and `test__maintainer_docs.py` assert the docs match
the code. That is rare and should survive this maintenance set unchanged.

## Observation O-6

`__pycache__` shipping in the release archive was recorded during the Corpus
campaign. It is present here, in `_cli`, and in the archive generally. Still open.
