# Corpus G11 — Hamlet gallery notebook-cell lifecycle

Date: `2026-08-19`

```text
source authority     scikit-plots(20260818-204251).zip
gallery              plot_corpus_fluent_hamlet_retrieval_script.py
scope                RuntimeCorpus lifecycle across Sphinx-Gallery notebook cells
status               VERIFIED / WHOLE-SCRIPT + CELL-BY-CELL GREEN
```

## Finding

The canonical Hamlet example opened `RuntimeCorpus` with a context manager and
then placed later runtime operations under additional `# %%` gallery cell
boundaries.  In a generated/interactive notebook, the materialization cell
finishes before the next cell executes, so `RuntimeCorpus.__exit__()` closes the
runtime.  A later `runtime.run()` then correctly raises:

```text
RuntimeError: RuntimeCorpus is closed.
```

This is a gallery lifecycle defect, not a RuntimeCorpus defect.

## Fix

The multi-cell example now uses:

```text
runtime = fluent.materialize(...)
run/search/add/export across independent # %% cells
runtime.close() in the final cleanup cell
```

A compact context-manager form remains documented for single-cell/script use.

## Verification

```text
py_compile                         PASS
whole-script execution             PASS
shared-kernel cell-by-cell         PASS
executable gallery chunks          18
runtime open through export        PASS
runtime closed in cleanup          PASS
run() documents                    12
add() total documents              14
search/export                      PASS
```

No Corpus runtime implementation changed.  The one-shot `run()` and closed
runtime guards remain intact.
