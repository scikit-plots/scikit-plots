# Reconciliation — bootstrap plan vs actual source tree

A00 revalidated the bootstrap against:

```text
archive: scikit-plots(20260821-211419).zip
sha256:  4990b417e7d6309bc3ca2c4691ee735b1fcdf9e698c38a129908419ea80178d6
reviewed: 2026-08-22
```

Read this before any later checkpoint. It records **current source truth** and the corrections that led to it. Sections 7–10 preserve A01 evidence and closure; earlier sections preserve the A00 reconciliation.

## 1. The vendoring action already happened; A01 verifies rather than re-vendors

The bootstrap proposed vendoring NVIDIA during A01. The current tree already
contains:

```text
_sphinx_llm/sphinx_llm/
  LICENSE
  LICENSE_HEADER
  __init__.py
  docref.py
  markdown_builder.py
  summary.py
  txt.py
  version.py
  tests/
  vendor.lock.json
```

The lock declares:

```json
{
  "repository": "https://github.com/NVIDIA/sphinx-llm.git",
  "commit_hash": "2a971d7da6a5d7df81f7bff3612ee1822a060c17",
  "tree_mode": "bash-sha256sum",
  "tree_hash": "1fa9ef908e475aedee2eea1593dadaba59da9bef48f57f078c2cd998f7754a8a",
  "generated_utc": "2026-08-21T15:59:35Z"
}
```

A00 does **not** accept this as proof of parity. Therefore:

| Original plan | Current A01 responsibility |
|---|---|
| copy/vendor pinned NVIDIA source | verify the existing vendor tree and lock |
| establish provenance | compare with pinned upstream; classify every file |
| run upstream tests | still required |
| protect future sync | still required |

A01 must not re-copy the tree first; that would destroy the evidence it is meant
to verify.

## 2. Keep `sphinx_llm/`; do not rename it to `upstream/`

The vendored production modules use relative imports, while preserved NVIDIA
tests contain absolute `sphinx_llm` imports. Renaming would force edits to the
very upstream tests used to demonstrate parity and buys no semantic advantage.

Accepted structure:

```text
_sphinx_llm/
├── sphinx_llm/        vendored NVIDIA candidate; A01 owner
├── core/              downstream only
├── adapters/          downstream only
├── curation/          downstream only
├── compat/            downstream only
├── tests/             downstream fixtures
├── upstream/README.md retired bootstrap placeholder only
└── _maintenance/
```

## 3. Backend export is still proposed, but its maintenance shell now exists

The current source contains `_sphinx_ai_backend/MAINTAINING.md` and a maintenance
skeleton. That **does not mean the service code has moved**.

The deployable paths are still present under `_sphinx_ai_assistant`:

```text
_hf_spaces_proxy/
_hf_spaces_model/
_cf_worker/
dev_proxy.py
```

Therefore B14 remains a future **move, not rewrite** checkpoint. The shell is a
planning/ownership marker only.

A00 also observed that the backend shell's current checker is not green. That is
outside `_sphinx_llm` A00 and must be repaired in the backend's own bootstrap
before it is treated as a live maintenance control plane.

## 4. Consumer boundary clarified

Earlier text alternated between “assistant consumes only artifacts” and
“assistant uses a consumer facade.” These are compatible only if edge timing is
explicit:

```text
runtime browser/agent  -> static published artifacts ONLY
Sphinx build-time      -> optional public Python facade, if A11 proves needed
private sphinx_llm/*    -> never imported by downstream consumers
```

This keeps the runtime static-security boundary while allowing a narrow Sphinx
integration API later if required.

## 5. MCP edge remains claimed, not proven

Wiring is not sufficient evidence. Later checkpoints still need to establish
live reachability, source-verification semantics, unavailable/degraded behavior,
and preservation of `RetrievalResponse.DEGRADED`. The final point also depends
on MCP run M04.

## 6. A00 closure result (historical checkpoint state)

A00 reconciles maintenance truth only. It intentionally does **not**:

- recompute the NVIDIA vendor tree hash;
- compare local files with the pinned upstream revision;
- run NVIDIA tests;
- integrate `_sphinx_llm` as a live Sphinx extension;
- implement adapters/curation/artifacts;
- move backend code;
- change assistant runtime behavior.

Those boundaries made A01 reviewable. Sections 7–10 record source verification, the local environment blocker, and the explicit pinned-upstream-CI-equivalent closure.

## 7. A01 source parity is green; local exact-lock execution is environment-blocked

A01 verified the existing vendor tree rather than replacing it.

Results:

```text
pinned NVIDIA commit:       2a971d7da6a5d7df81f7bff3612ee1822a060c17
upstream-derived files:     13
byte parity:                GREEN / all UPSTREAM_PRESERVED
portable manifest:          GREEN
import isolation:           GREEN
license source provenance:  GREEN
current NVIDIA main delta:  zero at 2026-08-22 observation
preserved upstream tests:   ENVIRONMENT_BLOCKED
checkpoint A01:             COMPLETE / PINNED_UPSTREAM_CI_EQUIVALENT
```

The legacy `bash-sha256sum` tree hash is also reproducible, but only at the
original documented target path. Its Bash pipeline includes path-bearing
`sha256sum` output in the aggregate, so identical bytes moved to another
checkout path produce a different aggregate. A01 therefore adds
`VENDOR_BASELINE.json` with a path-independent relative-file digest manifest
without rewriting `vendor.lock.json` or any NVIDIA-derived file.

Source parity alone did not close A01. The exact upstream `pyproject.toml` and
`uv.lock` are preserved as maintenance fixtures. For Python 3.13, the current
interpreter is missing Sphinx, Docutils, sphinx-markdown-builder and a functional
OpenAI distribution, while pytest 9.0.2 differs from pinned 9.1.1;
network/DNS is unavailable. The local proof therefore remains
`ENVIRONMENT_BLOCKED`. Section 10 records the independent, machine-checked
`PINNED_UPSTREAM_CI_EQUIVALENT` proof that closes A01 without rewriting this
local result.

---

## 8. Preserved tests require their upstream path shape

A01 found that byte-identical upstream tests are not self-contained after a
plain directory move. NVIDIA places them under `src/sphinx_llm/tests/`, and
several tests walk four parents to reach repository `docs/source/`. From the
vendored `_sphinx_llm/sphinx_llm/tests/` location that calculation points at the
wrong directory.

Do not patch preserved tests. Maintenance now keeps an exact pinned
`docs/source` fixture and `run_upstream_tests.py` creates an ephemeral
`repo/src/sphinx_llm` + `repo/docs/source` staging tree. Layout verification is
GREEN. `prepare_upstream_test_environment.py` can stage and, when explicitly asked with `--sync`, materialize the exact lock in a disposable directory. Behavior execution remains blocked in this sandbox because package retrieval is unavailable.


---

## 9. A01 uses an exact pinned dependency environment

A01 is provenance/baseline verification, not the compatibility matrix. Maintenance therefore preserves NVIDIA's pinned `pyproject.toml` and `uv.lock` byte-for-byte and records the Python-3.13 lock anchors in `UPSTREAM_TEST_ENVIRONMENT.json`. `run_upstream_tests.py` rejects missing or mismatched anchors.

`prepare_upstream_test_environment.py` stages those files only into a disposable destination outside `_sphinx_llm`; network/package activity occurs only with explicit `--sync`. An offline sync in this sandbox resolved the preserved lock but failed on the first unavailable cached wheel (`pytest==9.1.1`), so the local exact-lock proof correctly remains `ENVIRONMENT_BLOCKED`. Broader Sphinx/Python/dependency combinations belong to A02.


## 10. A01 closure uses an explicit pinned-upstream-CI-equivalent proof

A01 originally treated local `GREEN_EXACT_LOCK` as its only behavior-proof path.
That was stronger than the checkpoint's provenance objective and allowed an
infrastructure-only DNS/package-access failure to freeze the campaign after all
locally vendored behavior inputs had been proved byte-identical to the pinned
upstream revision.

The durable rules now define a second fail-closed proof mode:
`PINNED_UPSTREAM_CI_EQUIVALENT`. It is accepted only when the local vendor source,
tests, licenses, `docs/source` fixture, `pyproject.toml`, `uv.lock`, and pinned
NVIDIA test workflow are byte-verified against one exact commit; the local staged
upstream layout is GREEN; and NVIDIA's official workflow/job for the selected
Python/Sphinx baseline succeeds on that same commit. `UPSTREAM_CI_BASELINE.json`
is the machine evidence and `check_trackers.py` enforces these bindings.

For pinned commit `2a971d7da6a5d7df81f7bff3612ee1822a060c17`, those gates are GREEN,
including Test #211 and its Python 3.13 / Sphinx `>=9,<10` job. A01 is therefore
COMPLETE. The local exact-lock attempt remains `ENVIRONMENT_BLOCKED` (16/50
exact, 15 mismatched, 19 missing) and is retained as a reproducibility gap, not
rewritten as local GREEN. A02 was eligible after A01; it is now BLOCKED on config-override semantic parity and owns broader compatibility.
