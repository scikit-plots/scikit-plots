# Submodule Structure — `_sphinx_ai_backend`

> **PROPOSED submodule** — contents live in `_sphinx_ai_assistant/` today.

> Read `DEPENDENCY_MAP.md` first if the change crosses a family boundary.

## 1. Role

**PROPOSED: the deployable services the extension talks to — proxy, model space, edge worker.**

## 2. Antecessors / successors

| | |
|---|---|
| Antecessors | — |
| Successors | _sphinx_ai_assistant (is its client) |

### 3. What moves here, and why

| From `_sphinx_ai_assistant/` | Files | LOC (all files) | LOC (.py/.js) |
|---|---:|---:|---:|
| `_hf_spaces_proxy/` | 9 | 6 471 | 5 077 |
| `_hf_spaces_model/` | 3 | 2 553 | 2 446 |
| `_cf_worker/` | 2 | 556 | 499 |
| `dev_proxy.py` | 1 | 362 | 362 |
| **total** | **15** | **9 942** | **8 384** |

Both columns are recomputed by `check_trackers.py` and gated against these
values. The previously recorded single total of ~9 896 LOC was not reproducible
under either rule: it mixed a code-only count for the proxy and model spaces
with an all-files count for the worker, and recorded `dev_proxy.py` as ~1 800
lines when the file is 362. The argument for the export is unchanged — zero
tests on 8 384 lines of internet-facing service code is the same argument — but
a campaign resting on "a countable gap gets closed" has to be able to recount.

Three reasons, in order of weight:

1. **Zero tests on internet-facing code.** The proxy accepts user input, forwards
   it to a model, and collects a dataset. Separating it makes the gap countable.
2. **Different lifecycle.** These are *deployed*, not installed. They must keep
   serving readers of already-published docs, on a cadence unrelated to
   `pip install scikit-plots`.
3. **Different skills and different review.** Docker, edge workers and dataset
   policy are not Sphinx extension work, and reviewing them together means
   reviewing neither well.

### 4. Move the policy documents too

`robots.txt` and `DATASET_COLLECTION_GUIDANCE.md` currently sit beside the app.
They are policy, not code — they belong in `_maintenance/`.

### 5. Where a new thing goes

| Adding | Put it |
|---|---|
| A proxy endpoint | `_hf_spaces_proxy/` — **with a test** |
| Model-space behaviour | `_hf_spaces_model/` |
| Edge routing | `_cf_worker/` |
| A dataset field | `_dataset_schema.py`, and update the collection guidance |
| Sphinx extension behaviour | **not here** — `_sphinx_ai_assistant/` |

### 6. Before this submodule exists

The export is a **move, not a rewrite**. Do it in one commit that changes no
file contents, so the diff is reviewable — then add tests as a second commit.
Mixing the two makes it impossible to tell a move from an edit.

## Review checklist

```text
[ ] Does it edit vendored code?                       -> reject; add beside it
[ ] Does it cross a family boundary?                  -> DEPENDENCY_MAP.md §1
[ ] Does it touch the MCP edge?                       -> that edge is UNVERIFIED
[ ] Does it add a config authority?                   -> there are already four
[ ] Is it internet-facing?                            -> it needs a test
[ ] python _maintenance/check_trackers.py             -> exit 0
[ ] sphinx-build succeeds WITHOUT a live backend
```
