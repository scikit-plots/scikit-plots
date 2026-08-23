# Maintaining `_sphinx_llm`

This is the **human and fresh-chat entry point** for
`scikitplot._externals._sphinx_ext._sphinx_llm`.

## Current state

```text
maintenance checkpoint A00        COMPLETE
maintenance checkpoint A01        COMPLETE
NVIDIA source snapshot            PRESENT at sphinx_llm/
pinned-source byte parity         GREEN / UPSTREAM_PRESERVED
portable vendor manifest          GREEN
legacy vendor tree hash           MATCHES at recorded /work path; PATH_DEPENDENT
upstream test harness layout      GREEN
pinned A01 test environment       ENVIRONMENT_BLOCKED / exact 50-distribution lockset recorded
A01 behavior proof mode            PINNED_UPSTREAM_CI_EQUIVALENT / GREEN
preserved NVIDIA tests            ENVIRONMENT_BLOCKED locally
current NVIDIA main delta          ZERO at 2026-08-22 observation
local downstream implementation   A02 COMPAT SHIM IMPLEMENTED
production behavior changed A02   YES / bounded to outer setup + compat/
A02 compatibility matrix          GREEN 10/10 upstream cells
A02 semantic config parity         IMPLEMENTED / DOWNSTREAM MATRIX 0/10 GREEN
A02 unit helper tests              GREEN / 10 passed
A02 dependency-light tests          GREEN / 48 passed (10 config + 14 matrix + 11 closure + 6 readiness + 7 CI rebase)
A02 CircleCI execution path        INTEGRATED / dedicated opt-in parameter, 10 evidence producers + aggregator
A02 closure evidence verifier      GREEN / recomputation + coherent CircleCI provenance gate
A02 reconciliation readiness       GREEN / read-only receipt pins evidence + 13 target digests
A02 current-CI rebase path           GREEN / separate candidate + semantic verifier; never in-place
next action                       SEMANTICALLY REBASE current CI if drifted; TRIGGER run_sphinx_llm_a02=true; A03 LOCKED
```

The name of A01 is historical. Vendoring happened before this campaign. A01 is
now **COMPLETE** using the explicit `PINNED_UPSTREAM_CI_EQUIVALENT` proof mode in
`RULESET.md` rule 39. The accepted proof binds NVIDIA's successful official
Python-3.13/Sphinx-9 test job to the exact pinned commit only after local
byte-verification of the vendor source/tests/license, `docs/source`,
`pyproject.toml`, `uv.lock`, and the upstream test workflow, plus a GREEN staged
upstream-layout check. The local exact Python-3.13 lock environment remains
`ENVIRONMENT_BLOCKED` (16/50 exact, 15 mismatched, 19 missing) and is **not**
relabeled GREEN; it remains a reproducibility gap. Broader compatibility belongs to A02. Its pinned NVIDIA matrix is GREEN and the
config-parity shim is implemented downstream without changing vendor bytes, but
the downstream Sphinx/ifconfig matrix remains 0/10 GREEN. The local runtime is
environment-blocked; the provider-neutral harness is now wired into a dedicated
opt-in CircleCI workflow whose parameter defaults false.

## Source anchors

```text
current incremental input:
  archive:  scikit-plots_20260821-211419_A02-current-ci-rebase-ready-applied.zip
  sha256:   ba83b4f25ef9c2549956e727f54250a81a26800030b5156a3110455194338f49

A01 original verification input:
  archive:  scikit-plots_20260821-211419_A00-applied.zip
  sha256:   82afef355312dd068183dc2ef32184051e3b02121ed6de683c852058aac9a388

lineage source supplied by user:
  archive:  scikit-plots(20260821-211419).zip
  sha256:   4990b417e7d6309bc3ca2c4691ee735b1fcdf9e698c38a129908419ea80178d6

NVIDIA/sphinx-llm:
  pinned commit:     2a971d7da6a5d7df81f7bff3612ee1822a060c17
  describe:          v1.0.0-1-g2a971d7
  upstream date:     2026-08-15
  reference archive sha256:
                     4b3fc9173a67a4a93292639886d52afacf8eabc9f955477e6c2a8ae4d227c3e5
  local vendor path: sphinx_llm/
  local lock path:   sphinx_llm/vendor.lock.json
  portable evidence: _maintenance/VENDOR_BASELINE.json
  portable aggregate:
                     e19f91b85e2c6a130e772eaee6a4788cefd8391ba739c50d9471b57601229845
  source parity:     GREEN / 13 UPSTREAM_PRESERVED files
  A01 environment:   exact upstream pyproject.toml + uv.lock preserved; current env BLOCKED
  upstream tests:    GREEN / PINNED_UPSTREAM_CI_EQUIVALENT (local exact-lock run remains BLOCKED)
  license:           Apache-2.0

jdillard/sphinx-llms-txt:
  reference commit:  9d0660ba71c3c5dfe3023ebc2d281ddcb3070241
  describe:          v0.7.1-5-g9d0660b
  upstream date:     2026-08-03
  role:              curation/design reference
  license:           MIT
```

If the source anchor changes, revalidate physical/behavioral claims. If the
NVIDIA anchor or vendor bytes change, re-run A01/A02 gates before claiming
parity.

## Fresh-chat read order

Read these first, in order:

1. `_maintenance/RECONCILIATION.md`
2. `MAINTAINING.md`
3. `_maintenance/MAINTENANCE_MODEL.md`
4. `_maintenance/RULESET.md`
5. `_maintenance/UPSTREAM.md`
6. `_maintenance/VENDOR_BASELINE.json`
7. `_maintenance/UPSTREAM_CI_BASELINE.json`
8. `_maintenance/UPSTREAM_CI_CORROBORATION.md`
9. `_maintenance/UPSTREAM_TEST_HARNESS.md`
10. `_maintenance/UPSTREAM_TEST_ENVIRONMENT.json`
11. `_maintenance/DEPENDENCY_MAP.md`
12. `_maintenance/TRACKER_LOGICAL.md`
13. `_maintenance/TRACKER_PHYSICAL.md`
14. `_maintenance/SUBMODULE_STRUCTURE.md`
15. `_maintenance/BUILD_FLOW.md`
16. `_maintenance/DIRECTIVE_COMPATIBILITY.md`
17. `_maintenance/ARTIFACT_CONTRACT.md`
18. `_maintenance/REGISTRY.md`
19. `_maintenance/VERIFICATION.md`
20. `_maintenance/checkpoints/A01_UPSTREAM_VENDOR_BASELINE.md`
21. `_maintenance/UPSTREAM_COMPATIBILITY_BASELINE.json`
22. `_maintenance/UPSTREAM_COMPATIBILITY.md`
23. `_maintenance/A02_MATRIX_PLAN.json`
24. `_maintenance/A02_MATRIX_EXECUTION.md`
25. `_maintenance/checkpoints/A02_UPSTREAM_COMPATIBILITY.md`

A01 is closed, so `_maintenance/HISTORY.md` now contains its compressed closure
record. Current machine truth still lives in `STATE.json`, `TRACKER.json`, and the
A01 baseline/evidence files.

Do not create new parallel files named `FINAL`, `REVISED`, `EXPANDED`,
date-suffixed, or chat-specific variants inside the source tree.

Run before any new work:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/check_trackers.py
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/run_upstream_tests.py --layout-only
```

The local exact-lock reproduction path remains available as a non-blocking A01
reproduction check:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/prepare_upstream_test_environment.py \
  --destination /tmp/sphinx-llm-a01-env --sync --run-tests
```

Network/package access is attempted only because `--sync` is explicit, and only
inside the disposable destination. A local GREEN would strengthen reproducibility,
but A01 closure already rests on the separate machine-checked pinned-upstream-CI
equivalence proof. Do not rewrite the current local `ENVIRONMENT_BLOCKED` result.

## Governing rule

> **Canonical LLM output is a deterministic semantic representation of the
> resolved Sphinx document, not a scrape of its presentation. Never report the
> representation as complete if meaningful nodes were dropped without an
> explicit policy, warning, or failure.**

## Ownership and dependency boundary

```text
vendored NVIDIA baseline: sphinx_llm/  (UPSTREAM_PRESERVED source)
                       |
                       v
        downstream core / adapters / curation / compat
                       |
          static artifacts + optional public build-time facade
                       |
                       v
              _sphinx_ai_assistant
```

Rules:

- `_sphinx_llm` must never import `_sphinx_ai_assistant`.
- The browser/runtime assistant consumes **published static artifacts**.
- A future Python facade may be used only for bounded Sphinx build-time
  integration; consumers may not import private vendored internals.
- Keep the vendored directory name `sphinx_llm/`; preserved upstream tests use
  absolute `sphinx_llm` imports.
- `upstream/` is a retired bootstrap placeholder, not an implementation owner.
- A02's downstream config-parity implementation is present, but the downstream
  Sphinx/ifconfig parity matrix is still 0/10 GREEN. The canonical execution
  plan is `_maintenance/A02_MATRIX_PLAN.json`; the dedicated CircleCI workflow is
  gated by `run_sphinx_llm_a02=false` by default. Cell jobs emit one evidence JSON
  each with CircleCI pipeline/workflow/job/project/revision provenance; the aggregator must reach 10/10, the post-CI verifier must confirm one coherent expected-project workflow, and `prepare_a02_reconciliation.py` must emit a read-only target-digest receipt before any human closure patch is prepared.

## Static vs dynamic rule

```text
canonical page Markdown         STATIC / BUILD-TIME
llms.txt                        STATIC / BUILD-TIME
llms-full.txt                   STATIC / BUILD-TIME / OPTIONAL
manifest/provenance             STATIC / BUILD-TIME
directive compatibility report STATIC / BUILD-TIME
version/locale routing          MAY BE DYNAMIC, but selects prebuilt artifacts
interactive chat                DYNAMIC, owned elsewhere
runtime DOM -> Markdown         FALLBACK ONLY, never canonical
```

## Upstream/downstream policy

- NVIDIA `sphinx-llm` is the primary architectural/behavioral baseline.
- A01 byte-compared all 13 upstream-derived source/test/license files to the
  pinned checkout; they are `UPSTREAM_PRESERVED`.
- `README.md` and `vendor.lock.json` inside the vendor directory are downstream
  vendoring metadata, not upstream source files.
- The legacy `bash-sha256sum` aggregate includes absolute paths and is therefore
  not portable across checkout locations. Preserve it as historical vendoring
  evidence; use `_maintenance/VENDOR_BASELINE.json` for portable file integrity.
- Prefer downstream features beside the vendor tree (`core/`, `adapters/`,
  `curation/`, `compat/`) rather than modifying preserved files.
- If an upstream-derived file requires a local patch, classify and record it as
  `UPSTREAM_PATCHED`; never hide divergence.
- Use `sphinx-llms-txt` primarily for curation ideas: page/block exclusions,
  source-code inclusion, URL templating, deterministic ordering, and bounded
  full-output policies.

## Fresh-chat continuation prompt

> Focus only on `_sphinx_llm` checkpoint A02. A01 is COMPLETE. A02's pinned
> NVIDIA matrix is GREEN 10/10 and the downstream config-parity shim is now
> implemented without changing `sphinx_llm/**`. Run `check_trackers.py`,
> `verify_upstream_compatibility.py`, the dependency-light helper + matrix
> harness tests, then read `A02_MATRIX_EXECUTION.md`, `CIRCLECI_INTEGRATION.md`, `A02_CLOSURE_EVIDENCE.md`, and `A02_RECONCILIATION_READINESS.md`.
> Trigger the dedicated CircleCI workflow with `run_sphinx_llm_a02=true`; preserve
> all ten cell JSON files plus `A02_MATRIX_RESULT.json`. Verify the downloaded artifact with `verify_a02_closure_evidence.py`; cell jobs are evidence
> producers and the aggregator is the semantic gate. Closure eligibility additionally requires one coherent CircleCI pipeline/workflow/revision for `scikit-plots/scikit-plots`. If eligible, run `prepare_a02_reconciliation.py` to emit a read-only receipt outside the source tree; do not edit state until that receipt is human-reviewed and its target digests still match. The local sandbox remains
> environment-blocked; do not relabel it GREEN. Close A02 only at real downstream
> matrix 10/10 GREEN. Do not start A03 in the same closure reconciliation.

## Updating maintenance state

For every material change:

1. update `REGISTRY.md`;
2. update `STATE.json`;
3. update `TRACKER.json` and human trackers when structure/contracts change;
4. update `VERIFICATION.md` when gates/evidence change;
5. update `UPSTREAM.md` for every upstream import/sync;
6. update exactly one bounded checkpoint;
7. append `HISTORY.md` only after a coherent checkpoint closes;
8. never create `*_FINAL.md`, `*_REVISED.md`, or date-suffixed parallel truth.
