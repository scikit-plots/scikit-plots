# B14 — Export the deployable services as `_sphinx_ai_backend`

**Status:** PROPOSED. Does not block A00–A13 or B00–B13.

See `../DEPENDENCY_MAP.md` §1 and `_sphinx_ai_backend/_maintenance/SUBMODULE_STRUCTURE.md`.

## Why

`_sphinx_ai_assistant` contains four things that are not Sphinx code:

| Directory | Files | LOC |
|---|---:|---:|
| `_hf_spaces_proxy/` | 9 | ~5 077 |
| `_hf_spaces_model/` | 3 | ~2 446 |
| `_cf_worker/` | 2 | ~556 |
| `dev_proxy.py` | 1 | ~1 800 |

**9 896 LOC of internet-facing service code with zero tests.** The proxy accepts
public input, forwards it to a model, and collects a dataset.

That gap is invisible while the code sits inside an extension whose suite covers
the extension. Separating it makes the gap countable.

Second reason: **different lifecycle.** These are deployed, not installed. They
must keep serving readers of already-published docs, on a cadence unrelated to
`pip install scikit-plots`. Shipping them inside a pip-installed package
conflates two release cadences.

Third: different skills, different review. Docker, edge workers and dataset
policy are not Sphinx extension work, and reviewing them together means
reviewing neither well.

## How

**A move, not a rewrite.**

1. One commit that relocates the four paths and changes **no file contents**, so
   the diff is reviewable as a move.
2. A second commit that adds tests.

Mixing the two makes a move indistinguishable from an edit.

## Also move

`robots.txt` and `DATASET_COLLECTION_GUIDANCE.md` are policy, not code. They
belong in `_maintenance/`.

## Exit criteria

```text
[ ] the four paths relocated with no content change
[ ] the assistant still builds docs with no backend running
[ ] at least one test exercises the proxy's input handling
[ ] a secrets audit is recorded (currently UNCHECKED, not clean)
[ ] the deployment contract is written down: what version serves which docs
```
