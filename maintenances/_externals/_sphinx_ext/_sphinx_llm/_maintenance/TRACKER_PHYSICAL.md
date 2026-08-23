# `_sphinx_llm` Physical Tracker

## Current physical state — A01 complete; A02 CI + semantic-rebase + closure/reconciliation-readiness tooling ready, real parity matrix not run

```text
_sphinx_llm/
  __init__.py        downstream Sphinx extension setup; selects config-parity generator
  sphinx_llm/        verified pinned NVIDIA source baseline
  core/              downstream ownership README only
  adapters/          downstream ownership README only
  curation/          downstream ownership README only
  compat/            downstream config-parity implementation + future compatibility shims
  tests/             downstream unit/integration compatibility fixtures
  upstream/          retired bootstrap placeholder README only
  _maintenance/      durable maintenance state + portable vendor evidence
    upstream_test_fixture/      pinned NVIDIA docs/source fixture for staged tests
    upstream_test_environment/  pinned NVIDIA pyproject.toml + uv.lock
    UPSTREAM_TEST_ENVIRONMENT.json exact Python-3.13 A01 environment evidence
    UPSTREAM_TEST_LOCKSET.json   exact 50-distribution Python-3.13 selection
    prepare_upstream_test_environment.py disposable exact-lock environment preparer
    run_upstream_tests.py       ephemeral upstream-layout behavior harness
    upstream_ci_fixture/test.yml pinned NVIDIA behavior-workflow fixture
    UPSTREAM_CI_BASELINE.json    machine-checked accepted A01 CI-equivalence proof
    UPSTREAM_COMPATIBILITY_BASELINE.json A02 10-cell matrix + behavior/source probes
    UPSTREAM_COMPATIBILITY.md     human A02 compatibility result and blockers
    verify_upstream_compatibility.py dependency-free A02 evidence verifier
    verify_a02_closure_evidence.py read-only real-artifact closure verifier
    prepare_a02_reconciliation.py read-only human-reconciliation readiness receipt
    render_a02_circleci_rebase.py read-only semantic rebase renderer for drifted current CI
```

A01 byte-compared the upstream-derived source/test/license files to pinned
NVIDIA commit `2a971d7d…`. All 13 are `UPSTREAM_PRESERVED`. The only additional
files under the vendor boundary are downstream vendoring metadata:
`README.md` and `vendor.lock.json`.

A01 behavioral baseline is **closed** through `PINNED_UPSTREAM_CI_EQUIVALENT`.
The preserved tests retain NVIDIA repository-relative `docs/source` assumptions,
so maintenance recreates the upstream layout ephemerally and byte-verifies its
9-file docs fixture. The pinned `pyproject.toml`, `uv.lock`, and NVIDIA test
workflow are also byte-verified; the official exact-commit Python-3.13/Sphinx-9
job succeeded. Local exact-lock execution remains `ENVIRONMENT_BLOCKED`: this
interpreter is 16/50 exact, 15 distributions mismatch, and 19 are missing, and
live package retrieval is DNS-blocked. That local result remains visible as a
reproduction gap and is not the proof used to close A01.

## A02 compatibility state

The exact pinned NVIDIA `Test #211` matrix is GREEN in all ten recorded jobs and
the preserved suite exercises `html`, `dirhtml`, suffix modes, tag forwarding,
external `confdir`, lifecycle behavior, and summary/cache/security paths.

The preserved NVIDIA Markdown subprocess still does not natively propagate
primary Sphinx config overrides. A02 now contains a **downstream-only compatibility
implementation** that keeps all `sphinx_llm/**` bytes unchanged: the outer extension
selects `ConfigParityMarkdownGenerator`, captures the effective primary Sphinx
configuration, transfers it through a private SHA-256-integrity-checked snapshot,
injects a child bootstrap before user extensions, and reapplies the snapshot at
`config-inited` priority 1 for Sphinx 5+ lifecycle parity. Ten dependency-light
unit tests are GREEN. The real programmatic `ifconfig` parity gate is present but
is `ENVIRONMENT_BLOCKED` here because Sphinx/`sphinx-markdown-builder` are absent.
A02 therefore remains **BLOCKED**, now on executable regression proof rather than
missing implementation.


A02 post-CI closure tooling is also maintenance-only: `_maintenance/verify_a02_closure_evidence.py` accepts the downloaded directory/ZIP artifact, recomputes all ten cells, verifies the supplied aggregate, and emits a read-only closure decision. Its eleven dependency-light tests cover GREEN, blocked, tampered aggregate, duplicate basename, ZIP input, traversal rejection, all-GREEN local/manual evidence, mixed workflow identity, wrong repository identity, and duplicate CircleCI job IDs. `_maintenance/prepare_a02_reconciliation.py` then accepts only closure-eligible evidence, reruns the maintenance checker, and emits a read-only receipt that pins the evidence set plus 13 closure-target file digests; six tests prove non-mutation and fail-closed readiness behavior.

The project Python floor (`>=3.8`) versus upstream floor (`>=3.9`) is tracked as
a packaging/support-policy gap, not inferred as runtime compatibility from syntax.

## Production-like Python baseline

| Path | A01 origin state | LOC | Responsibility |
|---|---|---:|---|
| `__init__.py` | `DOWNSTREAM_ONLY` | 46 | outer extension setup; selects downstream config-parity generator |
| `compat/__init__.py` | `DOWNSTREAM_ONLY` | 2 | compatibility package |
| `compat/primary_build_context.py` | `DOWNSTREAM_ONLY` | 251 | effective-config capture, private snapshot, restore helpers |
| `compat/_child_config.py` | `DOWNSTREAM_ONLY` | 54 | child bootstrap + Sphinx-5-safe reapply |
| `compat/markdown_generator.py` | `DOWNSTREAM_ONLY` | 158 | Apache-2.0 NVIDIA-derived generator shim with local modification notice; vendor remains untouched |
| `sphinx_llm/__init__.py` | `UPSTREAM_PRESERVED` | 2 | upstream package init |
| `sphinx_llm/docref.py` | `UPSTREAM_PRESERVED` | 662 | document reference/routing |
| `sphinx_llm/markdown_builder.py` | `UPSTREAM_PRESERVED` | 79 | Markdown sub-build |
| `sphinx_llm/summary.py` | `UPSTREAM_PRESERVED` | 184 | optional summary generation |
| `sphinx_llm/txt.py` | `UPSTREAM_PRESERVED` | 1122 | llms/page-output generation |
| `sphinx_llm/version.py` | `UPSTREAM_PRESERVED` | 10 | upstream version |

Vendored tests under `sphinx_llm/tests/`: **5 Python files including
`__init__.py`, 2,809 LOC; 4 test modules / 95 test functions**.

Portable vendor evidence:

```text
_maintenance/VENDOR_BASELINE.json
algorithm: sha256-sorted-relative-file-digests-v1
aggregate: e19f91b85e2c6a130e772eaee6a4788cefd8391ba739c50d9471b57601229845
files:     13 UPSTREAM_PRESERVED
```

The legacy lock's `bash-sha256sum` tree aggregate is path-dependent; it matches
at the recorded original `/work/.../sphinx_llm` path but cannot be the sole
portable integrity gate.

## Physical ratchets

1. `sphinx_llm/` is the only vendored NVIDIA tree. Do not create a second
   implementation under `upstream/`.
2. Downstream-only features must not be added directly to an
   `UPSTREAM_PRESERVED` file; prefer `core/`, `adapters/`, `curation/`, or
   `compat/`.
3. Every file under the vendor boundary must be represented in
   `VENDOR_BASELINE.json` as upstream-derived or downstream vendoring metadata;
   ephemeral cache files are ignored.
4. `_sphinx_llm` must not import `_sphinx_ai_assistant` anywhere.
5. `_maintenance/` and downstream tests must never become runtime dependencies.
6. `upstream/` is a retired placeholder and may not receive production code.
7. New downstream modules over 1,500 logical lines require decomposition review
   before merge; this is a review trigger, not an automatic rejection.
8. Generated build artifacts (`llms.txt`, page Markdown, manifests) never live
   in the source package tree.
9. A new top-level source directory requires `SUBMODULE_STRUCTURE.md` and
   `TRACKER.json` updates.
10. A01 closure must satisfy one explicitly named proof mode from `RULESET.md`: local `GREEN_EXACT_LOCK`, or `PINNED_UPSTREAM_CI_EQUIVALENT` with all byte-equivalence prerequisites machine-checked.
11. A blocked local exact-lock run must remain recorded as `ENVIRONMENT_BLOCKED`; official CI proof may not relabel it GREEN. Broader version compatibility belongs to A02.
12. Environment synchronization may target only a disposable path outside the source/vendor tree.

## Inventory update rule

A future upstream sync must regenerate `VENDOR_BASELINE.json` from the reviewed
upstream delta, classify every changed file, execute preserved upstream tests,
and reconcile both machine and human trackers before the new baseline is called
clean.

13. Local full exact-lock proof is the complete `UPSTREAM_TEST_LOCKSET.json` selection; six anchor packages are only additional import-health probes.


## Repository CI boundary — A02 only

`.circleci/config.yml` is a repository-level owner, not `_sphinx_llm` production
code. A02 adds only the dedicated `run_sphinx_llm_a02` parameter, the two generic
A02 jobs, and one opt-in workflow. `CIRCLECI_INTEGRATION_BASELINE.json` records
the public-main source used because the supplied archive omitted hidden CI files.
During A02, the maintenance checker pins the reviewed integrated digest. For a drifted live CI file, `render_a02_circleci_rebase.py` produces a separate candidate/diff from semantic top-level anchors and `verify_a02_circleci_integration.py --candidate` checks the canonical A02 wiring without the historical digest pin. The renderer never overwrites the live input; future unrelated CI edits must be consciously reviewed rather than silently accepted.
