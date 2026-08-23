# `_sphinx_llm` Maintenance Model

This file answers six questions once: **why, when, where, which, how many, and
how much**.

## WHY

This subsystem exists to create machine-consumable documentation without
silently changing or losing Sphinx semantics. Its dangerous failure mode is not
an exception; it is a plausible Markdown artifact that omits a dropdown, tab,
video, generated API section, conditional branch, link target, or custom node
while still claiming to be canonical.

The maintenance system therefore optimizes for **semantic fidelity,
provenance, bounded output, upstream recoverability, and explicit trust
boundaries**.

## WHEN

Trigger a bounded checkpoint when any of these changes:

- the vendored or target NVIDIA `sphinx-llm` revision/behavior;
- Sphinx or `sphinx-markdown-builder` compatibility floor;
- enabled Sphinx extension set or custom node/directive class;
- HTML/dirhtml routing or suffix policy;
- `llms.txt` proposal/spec behavior;
- curation/ignore/size/source-code rules;
- manifest/provenance/compatibility schema;
- build-time summary provider, prompt, cache, credential, or privacy policy;
- public producer/build-time integration surface;
- version/locale publication layout;
- static HTML compatibility conversion.

Elapsed time alone is not a trigger.

## WHERE

```text
MAINTAINING.md                 human entry point
sphinx_llm/                    vendored NVIDIA baseline (A01 verifies it)
core/                          downstream representation/artifact contracts
adapters/                      semantic directive/node handling
curation/                      selection/ignore/order/size/code policies
compat/                        upstream-version + Tier-2 HTML compatibility
tests/                         downstream tests/fixtures
upstream/                      retired bootstrap placeholder; NO production code
_maintenance/
  RECONCILIATION.md            source-vs-bootstrap corrections; read first
  MAINTENANCE_MODEL.md         this six-question model
  RULESET.md                   durable invariants
  UPSTREAM.md                  provenance + sync protocol
  DEPENDENCY_MAP.md            family boundary and edge types
  TRACKER_LOGICAL.md           contracts and owners
  TRACKER_PHYSICAL.md          files, ratchets, boundaries
  TRACKER.json                 machine-readable tracker
  STATE.json                   campaign state and executed proof snapshot
  VENDOR_BASELINE.json          portable A01 vendored-source evidence
  UPSTREAM_TEST_ENVIRONMENT.json exact pinned A01 test-environment evidence
  UPSTREAM_TEST_LOCKSET.json    full Python-3.13 selected distribution set
  upstream_test_environment/    byte-preserved upstream pyproject.toml + uv.lock
  prepare_upstream_test_environment.py disposable environment preparer
  SUBMODULE_STRUCTURE.md       placement/dependency rules
  BUILD_FLOW.md                target generation flow
  DIRECTIVE_COMPATIBILITY.md   extension/node compatibility matrix
  ARTIFACT_CONTRACT.md         static outputs and routing
  SECURITY_MODEL.md            trust/build-time LLM rules
  REGISTRY.md                  open findings and exact next actions
  VERIFICATION.md              gates and status vocabulary
  CHECKPOINT_TEMPLATE.md       bounded work schema
  checkpoints/                 one file per bounded campaign step
  schemas/                     machine contracts
  history/                     superseded/completed evidence only
```

## WHICH

Use the smallest owner matching the responsibility:

- exact/patched NVIDIA-derived baseline -> `sphinx_llm/`, with provenance;
- stable representation model/artifact orchestration -> `core/`;
- custom semantic nodes/directives/media -> `adapters/`;
- selection/ignore/order/size/source-code policy -> `curation/`;
- upstream-version shims or lower-fidelity offline HTML -> `compat/`;
- runtime chat/UI/model-service authority -> **not here**;
  `_sphinx_ai_assistant` / future backend ownership applies.

Do not put production code in the legacy `upstream/` placeholder.

## HOW MANY

Prefer a small number of durable maintenance documents. Add a new top-level
maintenance document only for a distinct long-lived contract. Campaign-specific
detail belongs in `checkpoints/`; completed detail moves to `history/` or is
compressed into `HISTORY.md`.

Do not create `*_FINAL.md`, `*_REVISED.md`, `*_V2.md`, or chat-specific copies.

## HOW MUCH

Use ratchets rather than arbitrary perfection:

- verified upstream-preserved files should not accumulate downstream features;
- new downstream modules should have one clear responsibility;
- a new module over the configured threshold requires decomposition review;
- `llms.txt` remains small/curated;
- `llms-full.txt` may be omitted when policy limits are exceeded;
- unknown semantic-node loss is never accepted merely to keep a build green;
- vendored source origin may be promoted to `UPSTREAM_PRESERVED` by exact pinned-source byte evidence, but checkpoint closure still requires every executable gate mandated by its rules;
- vendor integrity evidence must be portable across checkout paths; path-dependent legacy hashes remain historical evidence, not the sole current proof.

A narrow truthful artifact is better than an apparently complete artifact with
unreported omissions.
