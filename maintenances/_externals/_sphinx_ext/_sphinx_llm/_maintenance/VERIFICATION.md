# `_sphinx_llm` Verification Contract

## Status vocabulary

```text
GREEN                executed and passed against the recorded source anchor
RED                  executed and failed for a product/contract reason
ENVIRONMENT_BLOCKED  could not execute because required environment/dependency is absent
NOT_RUN               not attempted against this anchor
DEFERRED              intentionally outside the active checkpoint
NOT_APPLICABLE        gate does not apply to the current implementation phase
```

Never convert `ENVIRONMENT_BLOCKED` into local `GREEN`. A checkpoint may use an
explicit alternate proof mode only when its rules define that mode and the drift
checker verifies every prerequisite. For A01, `PINNED_UPSTREAM_CI_EQUIVALENT` is
separate from local `GREEN_EXACT_LOCK`: it binds official NVIDIA CI on the exact
pinned commit to locally byte-verified source/tests/license, docs fixture,
project metadata, lockfile, workflow, and a GREEN staged layout. The current
local exact-lock attempt remains `ENVIRONMENT_BLOCKED` even though A01 is
COMPLETE by the alternate proof.

## Maintenance control-plane gates

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/check_trackers.py
python -m json.tool scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/STATE.json >/dev/null
python -m json.tool scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/TRACKER.json >/dev/null
python -m json.tool scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/VENDOR_BASELINE.json >/dev/null
python -m json.tool scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/UPSTREAM_CI_BASELINE.json >/dev/null
python -m json.tool scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/UPSTREAM_COMPATIBILITY_BASELINE.json >/dev/null
```

When `jsonschema` is available, validate maintenance JSON evidence against the checked-in schemas, including `STATE.json`, `TRACKER.json`, `VENDOR_BASELINE.json`, `UPSTREAM_TEST_ENVIRONMENT.json`, `UPSTREAM_TEST_LOCKSET.json`, `UPSTREAM_CI_BASELINE.json`, and `UPSTREAM_COMPATIBILITY_BASELINE.json`.

The drift checker must additionally prove:

- every current vendor file is classified in `VENDOR_BASELINE.json` or is an
  explicitly ignored ephemeral cache file;
- each recorded file SHA-256 matches current bytes;
- every pinned upstream test-fixture file matches its recorded digest;
- the test runner itself is required maintenance state and runs the drift checker before staging;
- the path-independent aggregate matches the recorded portable manifest;
- `UPSTREAM_PRESERVED` claims have pinned byte-parity evidence;
- tracker classifications agree with vendor evidence;
- A01 `COMPLETE` must identify one valid proof mode: local `GREEN_EXACT_LOCK` + preserved-suite GREEN, or fully verified `PINNED_UPSTREAM_CI_EQUIVALENT`; a blocked local run remains recorded as blocked;
- `_sphinx_llm -> _sphinx_ai_assistant` reverse dependency is absent;
- retired `upstream/` does not become a second implementation tree.

## A01 verification snapshot

| Gate | A01 status | Evidence |
|---|---|---|
| pinned checkout identity | **GREEN** | uploaded NVIDIA checkout HEAD = `2a971d7d…`; origin matches NVIDIA repository |
| pinned file byte parity | **GREEN** | 13 upstream-derived source/test/license files byte-identical |
| portable vendor manifest | **GREEN** | relative-file aggregate `e19f91b…29845` |
| legacy lock reproduction | **GREEN WITH PORTABILITY LIMIT** | lock hash reproduces at original `/work/...` path; changes after relocation because Bash hash input includes paths |
| origin classification | **GREEN** | all upstream-derived files `UPSTREAM_PRESERVED`; vendoring README/lock are downstream metadata |
| import isolation | **GREEN** | normal project root does not expose top-level `sphinx_llm`; intentional vendor test path resolves local vendor |
| license source provenance | **GREEN** | `LICENSE` and `LICENSE_HEADER` byte-identical; no modified upstream files |
| distribution notice inclusion | **DEFERRED** | verify when subsystem is actually packaged/installable |
| current NVIDIA `main` delta | **GREEN AT OBSERVATION** | pinned commit and `main` identical on 2026-08-22 comparison |
| pinned NVIDIA test workflow fixture | **GREEN / BYTE-VERIFIED** | preserved `test.yml` digest `2e89600d…`; matrix includes Python 3.13 / Sphinx `>=9,<10` and preserved test path |
| official Python 3.13 / Sphinx 9 job | **GREEN / A01 PROOF** | exact pinned commit job `94980976384` (`3.13, >=9,<10`) reports SUCCESS; bound by `UPSTREAM_CI_BASELINE.json` |
| official NVIDIA Sphinx docs build | **GREEN CORROBORATION** | `Build Sphinx Docs #275` for `2a971d7` reports `Status Success`, 5 jobs completed; not required for A01 closure |
| vendor Python syntax compilation | **GREEN** | all vendored Python source/tests compile |
| upstream test fixture parity | **GREEN** | 9 `docs/source` fixture files byte-identical to pinned NVIDIA revision |
| upstream test harness layout | **GREEN** | ephemeral `repo/src/sphinx_llm` + `repo/docs/source` reproduces upstream path assumptions |
| pinned A01 environment fixture | **GREEN** | upstream `pyproject.toml` + `uv.lock` byte-verified; full 50-distribution Python-3.13 lockset recorded |
| full selected lockset gate | **GREEN** | runner checks all 50 selected external distributions before required import probes and pytest |
| uv dry-run / lockset parity | **GREEN** | `uv sync --dry-run` proposes exactly the same 50 name/version pairs as `UPSTREAM_TEST_LOCKSET.json` |
| current A01 environment | **ENVIRONMENT_BLOCKED** | 16/50 exact, 15 mismatched, 19 missing; live sync resolves lock but package download fails on DNS |
| local preserved NVIDIA behavior tests | **ENVIRONMENT_BLOCKED** | staged harness is ready; exact local environment cannot be materialized here; this remains a reproduction gap |
| A01 behavior proof mode | **GREEN** | `PINNED_UPSTREAM_CI_EQUIVALENT`; all rule-39 local equivalence and official-run prerequisites machine-checked |
| A01 checkpoint | **COMPLETE** | pinned baseline closed without relabeling local exact-lock `ENVIRONMENT_BLOCKED`; A02 is eligible |

## A01 preserved-test command contract

Do **not** execute the preserved tests directly from
`_sphinx_llm/sphinx_llm/tests`. Their source files are unchanged from NVIDIA and
resolve `docs/source` using the upstream repository layout. Use the checked-in
maintenance harness, which recreates that layout ephemerally while copying the
verified vendor bytes unchanged:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/check_trackers.py
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/run_upstream_tests.py --layout-only
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/run_upstream_tests.py
```

Exact environment preparation:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/prepare_upstream_test_environment.py \
  --destination /tmp/sphinx-llm-a01-env --sync --run-tests
```

`--layout-only` proves only the staging relationship. For the **local** A01 proof
mode, all 50 selected distributions must match exactly, required imports must be
GREEN, and the preserved suite must execute successfully. The runner never
installs dependencies or uses the network; missing packages remain
`ENVIRONMENT_BLOCKED`.

A01 is currently closed by the separate `PINNED_UPSTREAM_CI_EQUIVALENT` mode.
That mode is valid only while `UPSTREAM_CI_BASELINE.json` and its local byte
bindings remain GREEN. If a future local exact-lock run succeeds, record it as
additional reproduction evidence; do not erase the historical blocked result.
If any behavior test fails for a product reason, record `RED`/`BLOCKED_PRODUCT`
and re-evaluate reliance on the pinned baseline rather than hiding the failure.

## Required gates after A01+

| Gate | Purpose |
|---|---|
| upstream compatibility matrix | prove pinned NVIDIA behavior survives supported Sphinx/Python versions |
| Sphinx `html` fixture build | canonical HTML + Markdown coexistence |
| Sphinx `dirhtml` fixture build | routing/suffix semantics |
| primary tags/config forwarding | Markdown build sees same semantic conditions |
| directive compatibility fixture | rich extension semantics preserved |
| unknown-node positive control | deliberately unsupported semantic node is observable |
| raw-content malicious fixture | script/iframe/dangerous URL policy enforced |
| llms.txt golden/parser test | structure/order/descriptions/links deterministic |
| llms-full size-policy test | skip/keep/note/error semantics explicit |
| manifest schema test | all generated page artifacts inventoried |
| compatibility schema test | node handling counts and unknowns valid |
| provenance hash test | source/fidelity/transforms recorded consistently |
| build determinism test | same source/config produces stable semantic output |
| optional-generation disabled test | representation works with no LLM/provider package |
| summary credential transport test | key not in config/artifacts/logs; insecure auth rejected |
| locale/version matrix | routes cannot cross build identity |
| assistant producer-consumer test | runtime assistant uses static artifact; any build-time Python integration uses only public facade |

## Semantic comparison philosophy

Do not overfit tests to byte-for-byte Markdown formatting when the contract is
semantic. Use targeted assertions for headings, links, every tab/dropdown body,
code/output roles, media metadata, generated API content, and expected
exclusions. Use byte/golden tests when deterministic format itself is the
contract (`llms.txt`, normalized schemas/manifests, vendor parity).

## Release condition

A strict release advertising canonical machine representation should require:

```text
unsupported_semantic_nodes == 0
unexplained_content_loss    == 0
manifest_schema             GREEN
routing_matrix              GREEN
llms_index                  GREEN
raw_content_safety          GREEN
```

## A02 verification snapshot

| Gate | A02 status | Evidence |
|---|---|---|
| upstream compatibility matrix | **GREEN 10/10** | exact pinned NVIDIA Test #211; matrix/workflow byte-bound in `UPSTREAM_COMPATIBILITY_BASELINE.json` |
| html / dirhtml routing | **GREEN upstream baseline** | preserved routing/link tests executed by every matrix job |
| suffix modes | **GREEN upstream baseline** | file/url/auto/both/replace tests present in preserved suite |
| primary tags | **GREEN upstream baseline** | `test_tags_forwarded_to_markdown_build` + source forwarding loop |
| external confdir | **GREEN upstream baseline** | `test_confdir_outside_srcdir` + `-c self.app.confdir` |
| preserved vendor config overrides | **NOT NATIVE / vendor unchanged** | pinned `MarkdownGenerator` still lacks native override propagation; this is retained as upstream-source truth |
| downstream config-parity shim | **GREEN SOURCE + UNIT** | outer setup + integrity-checked snapshot child handoff; `tests/test_primary_build_context.py`: 10 passed |
| programmatic `ifconfig` parity matrix | **NOT RUN / 0 of 10 GREEN** | canonical plan + isolated per-cell runner + fail-closed aggregator are ready; local Sphinx stack is unavailable; all 10 required cells must pass before A02 closes |
| summary/cache/security | **GREEN upstream baseline** | preserved summary/docref/page-summary tests run across official matrix |
| Python 3.8 project floor | **DEFERRED_POLICY** | project >=3.8 vs upstream >=3.9; syntax-only probe is not support proof |
| vendor bytes | **GREEN / unchanged** | A01 portable baseline remains authoritative |
| A02 checkpoint | **BLOCKED** | implementation exists; downstream semantic parity matrix is 0/10 GREEN and must reach 10/10 before A03 |
| canonical matrix plan | **GREEN** | `A02_MATRIX_PLAN.json` mirrors the same 10 ordered upstream cells and binds evidence filenames |
| dependency-light A02 helper + matrix + closure + readiness + CI-rebase tests | **GREEN 52/52** | 10 config-transfer + 14 matrix/orchestration + 11 closure/provenance + 6 read-only reconciliation-readiness + 11 structural-YAML CI-rebase tests |
| parity runner cell inventory | **GREEN** | canonical plan/baseline/checker agree on all 10 ordered cells |
| parity runner environment guard | **GREEN** | deliberately wrong expected Python returns RED before semantic execution |
| matrix drift negative controls | **GREEN** | missing cell, GREEN-without-evidence, and premature A02 completion all fail closed |
| CircleCI integration | **GREEN / NOT EXECUTED** | dedicated `run_sphinx_llm_a02=false` parameter; 10 small evidence producers + aggregate gate; aggregate artifacts retained with `when: always`; exact integration baseline recorded |
| current public CircleCI main drift | **REBASE REQUIRED / TOOLING GREEN** | recorded integration base: 1,265 lines; current public main observed 2026-08-22: 1,237 lines and no `run_sphinx_llm_a02`; semantic renderer creates a separate candidate/diff and candidate verifier checks A02 semantics without historical digest pin |
| post-CI closure evidence verifier | **GREEN / NO REAL ARTIFACT YET** | directory/ZIP input; recomputes all 10 cells, matches provided aggregate, rejects duplicates/traversal/stale evidence, never mutates state; requires coherent CircleCI pipeline/workflow/project/revision identity; 11 tests GREEN |
| A02 reconciliation readiness | **GREEN / NO REAL ARTIFACT YET** | read-only preparer requires closure-eligible CircleCI evidence, reruns maintenance checker, pins 13 target-file digests, refuses source-tree receipt output, and never mutates state; 6 tests GREEN |

Run the A02 evidence and regression gates:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/verify_upstream_compatibility.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=<repository-root> \
  python -m pytest -q -c /dev/null \
  scikitplot/_externals/_sphinx_ext/_sphinx_llm/tests/test_primary_build_context.py \
  scikitplot/_externals/_sphinx_ext/_sphinx_llm/tests/test_a02_matrix_orchestrator.py \
  scikitplot/_externals/_sphinx_ext/_sphinx_llm/tests/test_a02_closure_evidence.py

python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/run_a02_matrix.py --plan

# Run once inside each required environment; example cell:
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/run_a02_config_parity.py \
  --expect-python 3.13 --expect-sphinx '>=9,<10' \
  --evidence-out /tmp/a02-py313-sphinx9.json
```

A GREEN compatibility verifier means the **recorded evidence and downstream shim
are internally truthful**, not that semantic parity has passed. The helper suite
must remain separate from the final Sphinx integration gate. In this environment the runner must report `ENVIRONMENT_BLOCKED`. A real GREEN in
one environment proves only that cell; A02 closes only after all 10 required cells
are GREEN (or an explicitly reviewed equivalent matrix).


Verify the repository CI wiring before triggering A02:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/verify_a02_circleci_integration.py

# If live CI has drifted from the recorded base, render/review a separate candidate:
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/render_a02_circleci_rebase.py \
  --input .circleci/config.yml --output /tmp/config.a02.yml \
  --diff-out /tmp/A02_CURRENT_CIRCLECI_REBASE.patch
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/verify_a02_circleci_integration.py \
  --candidate /tmp/config.a02.yml
```


After downloading the real CircleCI artifact, independently verify it before any maintenance-state reconciliation:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_llm/_maintenance/verify_a02_closure_evidence.py \
  /path/to/sphinx-llm-a02 \
  --decision-out /tmp/A02_CLOSURE_DECISION.json
```

Only `ELIGIBLE_FOR_HUMAN_RECONCILIATION` with recomputed `GREEN_10_OF_10` **and** coherent CircleCI provenance for `scikit-plots/scikit-plots` is sufficient to begin the separate A02 closure reconciliation.
