# Verification — proving the MCP tree is healthy

Folds in the former `MCP_VERIFICATION_MATRIX.md`.

---

## 1. The three commands

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
$ python -m pytest scikitplot/mcp -q -p no:cacheprovider
$ python -c "import sys, scikitplot.mcp; print([m for m in ('mcp','pydantic','scikitplot.corpus','annoy') if m in sys.modules])"
```

Expected:

```text
physical tracker matches the tree (17 source / 15 test files, ...)
<suite result>
[]
```

The third is the **optionality proof**: importing MCP must pull in neither the
SDK nor a retrieval backend. It prints `[]` today; making it a test is finding
`MCP-M00-03`.

---

## 2. Current known state

```text
$ python -m pytest scikitplot/mcp -q
2 skipped, 4 errors
E   ModuleNotFoundError: No module named 'pydantic'
```

**This is a recorded finding (`MCP-M00-01`), not a broken checkout.** Four of six
test files import `pydantic` transitively without `pytest.importorskip`, so the
suite fails at *collection* rather than skipping. Install `[mcp]` to run it, or
fix the guards — the fix is the better answer.

With `[mcp]` installed, the suite is expected green.

---

## 3. What the gate checks

| Check | Fails when |
|---|---|
| DRIFT | recorded inventory differs from the tree by more than 10% |
| TRIPWIRE | a module exceeds 1 200 LOC; test:source ratio below 0.50; a code subpackage has no tests |
| BOUNDARY | `mcp`/`pydantic` imported at module scope outside `_server.py` |
| BOUNDARY | `corpus` or `annoy` imported at module scope anywhere |
| LOGICAL | a contract in `TRACKER.json` names a module that does not exist |

Indentation is the boundary signal: an indented (call-time) import is legitimate
and is exactly how optional dependencies stay optional. Docstring prose is not a
violation — only a real statement is.

After a deliberate structural change:

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py --update
```

then regenerate `TRACKER_PHYSICAL.md` to match. A recorded exception must be
written into `TRACKER.json` → `physical.known_exceptions` to be honoured; an
undocumented exception is indistinguishable from an oversight.

---

## 4. Evidence standard

Inherited from the Corpus campaign, unchanged:

- **"I tested it" is insufficient.** Paste the command and its output.
- A claim about a capability requires a **probe**, not an assumption.
- A finding is marked resolved with **evidence**, never deleted.
- A test is never weakened to make a change pass. If a test fails, either the
  change is wrong or the test encoded a contract that is being deliberately
  changed — and the second case requires a written justification.

---

## 5. Per-run exit criteria

The M00–M12 runs and their exit gates live in the review kit
(`MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md`). This file covers the tree-level checks
that apply to **every** run, whichever one is active.
---

## 6. M14 Corpus + Annoy CLI/showcase gate

```console
$ SCIKITPLOT_MCP_RUN_LIVE=0 PYTHONPATH=<tree> \
    pytest -q -o addopts='' scikitplot/mcp/tests -p no:cacheprovider
164 passed, 2 skipped

$ python scikitplot/mcp/_maintenance/check_trackers.py
physical tracker matches the tree (18 source / 16 test files, 4686 / 2656 LOC)
```

Additional verified seams:

- `CorpusAnnoyRetriever(...).get(doc_id)` returns the same stable indexed
  document used by MCP resources.
- `from_corpus_annoy(..., embedder=HashEmbedder(...))` uses one local batch
  embedder for document and query vectors.
- `--docs-jsonl` and `--corpus-annoy` are mutually exclusive.
- `--corpus-annoy` remains SDK-free under `--self-test`.
- the Docker HTTP gallery path is opt-in and loopback-bound during automated
  execution.

Environment limitation for this checkpoint: the current harness has no usable
native Annoy extension and no `mcp` SDK, so the real Annoy + Streamable HTTP
round trip is **not** recorded as PASS here.
