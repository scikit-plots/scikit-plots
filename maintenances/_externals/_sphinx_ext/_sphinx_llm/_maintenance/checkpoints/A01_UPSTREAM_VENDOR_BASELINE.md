# A01 — Upstream Vendor Baseline Verification

Status: **COMPLETE**
Closure proof: **PINNED_UPSTREAM_CI_EQUIVALENT**
Local exact-lock reproduction: **ENVIRONMENT_BLOCKED**
Type: **bounded upstream-verification checkpoint**
Subsystem: **_sphinx_llm**

## Objective

Prove what the existing `sphinx_llm/` vendor tree actually is, preserve a
reviewable NVIDIA baseline, and establish behavior evidence without mixing
upstream verification with downstream feature work.

The vendoring action happened before A00. A01 verifies it; it does not re-vendor
or rename the tree.

## Prerequisite

- A00 **COMPLETE**

## Fixed evidence entering A01

```text
working source:  scikit-plots_20260821-211419_A00-applied.zip
source sha256:   82afef355312dd068183dc2ef32184051e3b02121ed6de683c852058aac9a388
lineage source:  scikit-plots(20260821-211419).zip
lineage sha256:  4990b417e7d6309bc3ca2c4691ee735b1fcdf9e698c38a129908419ea80178d6
local vendor:    sphinx_llm/
lock:            sphinx_llm/vendor.lock.json
pinned NVIDIA:   2a971d7da6a5d7df81f7bff3612ee1822a060c17
```

The final A01 maintenance reconciliation was performed on the incremental input
`scikit-plots_20260821-211419_A01-ci-corroboration-applied.zip`
(`321d0334a7f12301d128c7dd92a68b57afdf33e08d8ffc4802a8d86599647458`).

## Findings and proof

### 1. Pinned source parity — GREEN

All 13 upstream-derived files (license/header, package modules, and preserved
upstream tests) are byte-identical to the pinned NVIDIA checkout and classified
`UPSTREAM_PRESERVED` in `../VENDOR_BASELINE.json`.

```text
portable algorithm: sha256-sorted-relative-file-digests-v1
portable aggregate: e19f91b85e2c6a130e772eaee6a4788cefd8391ba739c50d9471b57601229845
upstream-preserved files: 13
local vendoring metadata: README.md, vendor.lock.json
```

### 2. Legacy lock — reproducible but path-dependent

The legacy `bash-sha256sum` tree hash
`1fa9ef908e475aedee2eea1593dadaba59da9bef48f57f078c2cd998f7754a8a`
reproduces at the documented original `/work/.../sphinx_llm` target, but changes
when identical bytes are moved because path-bearing `sha256sum` records enter
the aggregate.

A01 does not rewrite the global vendoring tool or vendor lock. It adds the
portable relative-file manifest and leaves `SLLM-015` with the maint-tools
owner.

### 3. Import isolation — GREEN

- normal project-root import search does not expose an unintended top-level
  local `sphinx_llm` package;
- the preserved test harness intentionally places the vendor parent on the
  staged import path so upstream absolute imports resolve to the verified local
  vendor.

### 4. License provenance — source GREEN; distribution DEFERRED

`LICENSE` and `LICENSE_HEADER` match pinned Apache-2.0 upstream exactly. No
upstream-derived implementation/test file is modified. Distribution-level
notice inclusion remains deferred until `_sphinx_llm` is actually
packaged/installable.

### 5. Current upstream delta — GREEN at observation

At the 2026-08-22 observation, pinned revision `2a971d7d…` and NVIDIA `main`
were identical in the reviewed comparison. Future sync work must recheck this;
it is not a permanent assumption.

### 6. Preserved test layout — GREEN

NVIDIA tests expect their original repository relationship:

```text
repo/
├── docs/source/
└── src/sphinx_llm/tests/
```

Direct execution from the vendored package path is therefore not a valid
upstream-layout proof. A01 keeps the test bytes untouched, preserves the pinned
9-file `docs/source` fixture, and stages an ephemeral upstream-shaped tree with
`run_upstream_tests.py`. `--layout-only` is GREEN.

### 7. Local exact-lock proof path — ENVIRONMENT_BLOCKED

A01 preserves upstream `pyproject.toml` and `uv.lock` byte-for-byte and derives
the complete Python-3.13 selection into `UPSTREAM_TEST_LOCKSET.json`.

```text
selected distributions: 50
current exact:           16
current mismatched:      15
current missing:         19
local result:            ENVIRONMENT_BLOCKED
```

The runner checks all 50 selected distributions plus required imports before
pytest. A live `uv sync` resolves the lock but this execution environment cannot
retrieve packages because package-index/DNS access is unavailable.

This result remains **ENVIRONMENT_BLOCKED**. It is not changed to GREEN by A01
closure.

### 8. Pinned official-CI equivalent proof — GREEN / ACCEPTED

A01 has a second explicit proof mode governed by `RULESET.md` rule 39 and
machine-checked by `UPSTREAM_CI_BASELINE.json`.

The accepted proof requires all of the following to bind to the **same exact
NVIDIA commit**:

1. vendor source/tests/license byte parity;
2. pinned `docs/source` fixture byte parity;
3. pinned `pyproject.toml` byte parity;
4. pinned `uv.lock` byte parity;
5. pinned NVIDIA test-workflow byte parity;
6. GREEN local upstream-shaped staging layout;
7. official NVIDIA workflow SUCCESS at the pinned commit;
8. official selected Python/Sphinx job SUCCESS at that commit;
9. workflow execution of the preserved upstream test path using the preserved
   project/lock semantics.

For commit `2a971d7da6a5d7df81f7bff3612ee1822a060c17` these gates are GREEN:

```text
workflow: Test #211
run id:   31871592483
result:   SUCCESS

job:      test (3.13, >=9,<10)
job id:   94980976384
result:   SUCCESS

workflow fixture sha256:
2e89600dc67f68da5aa7a2d25fe0a298b2c0b039d8551d30eeeebc903a4342b5
```

`UPSTREAM_CI_BASELINE.json` is the machine contract; the human rationale is in
`../UPSTREAM_CI_CORROBORATION.md`.

This proof mode is intentionally narrower than A02: it establishes the pinned
baseline only. It does not claim compatibility across other supported Python,
Sphinx, or dependency versions.

## Why A01 can close while local execution remains blocked

The checkpoint has two explicit behavior-proof modes:

```text
LOCAL_GREEN_EXACT_LOCK
    exact 50-distribution local environment
    + required imports GREEN
    + preserved suite GREEN

PINNED_UPSTREAM_CI_EQUIVALENT
    exact local byte-equivalence of all behavior inputs
    + upstream-shaped layout GREEN
    + official exact-commit selected job SUCCESS
```

A01 closes with the second mode. This is not an exception inferred from a green
badge: the mode is defined in the durable rules, represented in a schema-backed
machine evidence file, and enforced by the maintenance checker. The blocked
local proof remains recorded separately.

## Scope completed

1. reproduced and characterized the legacy vendor-lock hash;
2. compared every upstream-derived vendor byte with the pinned NVIDIA revision;
3. classified every upstream-derived source/test/license file;
4. verified license/header source provenance;
5. verified import isolation;
6. recorded current-upstream delta at observation;
7. preserved and byte-verified NVIDIA's docs test fixture;
8. built a fail-closed upstream-shaped test harness without modifying upstream
   tests;
9. preserved `pyproject.toml` + `uv.lock` and froze the exact Python-3.13
   50-distribution lockset;
10. built a disposable exact-lock environment preparer and recorded the local
    infrastructure blocker;
11. preserved and byte-verified NVIDIA's pinned `test.yml` workflow;
12. bound the official exact-commit Python-3.13/Sphinx-9 successful job to the
    local equivalence evidence;
13. closed A01 under `PINNED_UPSTREAM_CI_EQUIVALENT`.

## Non-goals preserved

- no rename of `sphinx_llm/` to `upstream/`;
- no NVIDIA vendor-file edits;
- no directive adapters, curation, manifest, `llms.txt` enhancement, assistant
  integration, or backend moves;
- no claim that local exact-lock execution succeeded;
- no A02 compatibility work in this checkpoint.

## Execution record

```yaml
checkpoint: A01
status: COMPLETE
started_at: 2026-08-22
completed_at: 2026-08-22
closure_proof_mode: PINNED_UPSTREAM_CI_EQUIVALENT
source_anchor:
  archive: scikit-plots_20260821-211419_A00-applied.zip
  sha256: 82afef355312dd068183dc2ef32184051e3b02121ed6de683c852058aac9a388
closure_reconciliation_input:
  archive: scikit-plots_20260821-211419_A01-ci-corroboration-applied.zip
  sha256: 321d0334a7f12301d128c7dd92a68b57afdf33e08d8ffc4802a8d86599647458
upstream_anchor:
  repository: NVIDIA/sphinx-llm
  commit: 2a971d7da6a5d7df81f7bff3612ee1822a060c17
production_code_modified: false
contracts_touched:
  - SLLM-C01
  - SLLM-C12
findings_closed:
  - SLLM-001
  - SLLM-002 source-classification/delta portion
  - SLLM-017
  - SLLM-018
  - SLLM-019
open_external:
  - SLLM-015 path-dependent global vendor hash
risks:
  - local exact-lock reproduction remains ENVIRONMENT_BLOCKED
  - future direct execution from the vendored test path remains invalid; use the staged harness
  - legacy vendor hash is checkout-path dependent
  - official CI-equivalent proof must be invalidated if any pinned byte/workflow/run identity changes
rollback: remove A01 downstream maintenance evidence only; NVIDIA vendor/runtime bytes are unchanged
```

## Verification gates

- [x] legacy tree hash reproduced and portability limitation characterized
- [x] pinned-upstream file comparison recorded
- [x] every upstream-derived source/test/license file classified
- [x] pinned docs/source fixture byte parity recorded
- [x] upstream-shaped staging layout GREEN
- [x] exact upstream `pyproject.toml` + `uv.lock` byte parity recorded
- [x] full Python-3.13 50-distribution lockset derived and checked
- [x] environment preparer fail-closed and source-tree-safe
- [ ] local exact-lock environment materialized — **ENVIRONMENT_BLOCKED / NON-BLOCKING REPRODUCTION GAP**
- [ ] local preserved suite executed under exact lock — **ENVIRONMENT_BLOCKED / NOT RELABELED**
- [x] pinned NVIDIA test workflow byte parity recorded
- [x] official exact-commit Test #211 SUCCESS recorded
- [x] official Python 3.13 / Sphinx `>=9,<10` job SUCCESS recorded
- [x] `PINNED_UPSTREAM_CI_EQUIVALENT` prerequisites machine-checked
- [x] import isolation recorded
- [x] license source status recorded; distribution gate deferred explicitly
- [x] current-upstream delta anchored at observation
- [x] no downstream feature work mixed into A01
- [x] state, trackers, registry, verification, upstream docs reconciled
- [x] `HISTORY.md` updated only after closure

## Closure result

**A01 COMPLETE.** The pinned NVIDIA baseline is frozen under
`PINNED_UPSTREAM_CI_EQUIVALENT`. Local exact-lock execution remains explicitly
`ENVIRONMENT_BLOCKED` and must never be reported as local GREEN. **A02 is now
eligible but not started in A01.**
