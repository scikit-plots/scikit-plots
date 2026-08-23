# History — `_sphinx_ai_backend`

Read only when the *why* behind a current rule is unclear.

## This submodule

A prior effort built the assistant: an `_ExtSettings` security registry with 79
tests, a discovery-contract CI gate, a Docker proxy deployment, a local dataset
sink, and a competitive benchmark against Biel.ai which concluded that
RAG-with-citations is a *composition* problem over owned `scikitplot.corpus` and
`scikitplot.annoy` components. "I'm Feeling Lucky" was replaced with a
deterministic "Explain this page" using confidence-ordered question selection.

`_static/_backup/css_js_v1` … `v6` are that effort's frontend iterations.

## The bootstrap package

`scikitplot_sphinx_llm_ai_assistant_maintenance_bootstrap_20260820.zip` proposed
the two-submodule split and a maintenance skeleton. **Adopted**, with two
corrections:

1. **Do not rename `sphinx_llm/` to `upstream/`.** Renaming the vendored
   directory is itself a modification of the vendored tree, and the stated goal
   is to leave it untouched.
2. **A third submodule is proposed**: `_sphinx_ai_backend`, holding the
   deployable services. They are not Sphinx code, they have a different release
   cadence, and they carry 9 896 LOC with zero tests — a gap invisible while
   they sit inside an extension whose suite covers the extension.

## What the earlier campaigns established

Corpus (R00–R16, IMPL-01–18), MCP and the annoy family each produced a
maintenance pair with the same discipline. The principle carried here:

> A document that describes what a script can check should be replaced by the
> script; one that records why cannot be.

## 2026-08-22 — family graph reduced to one live edge

- `_sphinx_llm` frozen. The three-member graph in `DEPENDENCY_MAP.md` and
  `MAINTAINING.md` described a build-time edge that no longer exists.
- `MAINTAINING.md` anchor refreshed and both LOC figures corrected: 15 files,
  9 942 LOC all files, 8 384 LOC of `.py`/`.js`. The single 9 896 figure was not
  reproducible under either counting rule.
- The vendoring correction ("decline the `sphinx_llm/` -> `upstream/` rename")
  is retired as moot.
- `SX-S00-B6` opened: this submodule's maintenance vocabulary and checkpoint
  namespace still diverge from the assistant's.
