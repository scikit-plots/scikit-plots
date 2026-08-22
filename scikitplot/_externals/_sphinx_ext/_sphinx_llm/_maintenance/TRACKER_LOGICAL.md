# `_sphinx_llm` Logical Contract Tracker

Statuses describe **current source truth**, not aspirations. A01 proved the pinned NVIDIA baseline through byte-level local equivalence plus the explicit `PINNED_UPSTREAM_CI_EQUIVALENT` behavior proof. The local exact Python-3.13 environment remains `ENVIRONMENT_BLOCKED` and is recorded as a reproduction gap, not local GREEN. `UpstreamCore` therefore HOLDS for the pinned baseline; broader compatibility and downstream contracts remain owned by A02+ checkpoints.

| ID | Contract | Owner | Status | Invariant | Proof |
|---|---|---|---|---|---|
| `SLLM-C01` | `UpstreamCore` | `sphinx_llm/` | **HOLDS** | NVIDIA-compatible pinned behavior remains isolated and synchronizable. | A01 byte parity + portable manifest + import isolation + staged-layout GREEN + pinned workflow/lock equivalence + official Python-3.13/Sphinx-9 job SUCCESS; local exact-lock reproduction separately BLOCKED |
| `SLLM-C02` | `SemanticBuild` | `core/` | **PLANNED** | Canonical Markdown is built from resolved Sphinx semantics, not final HTML. | Sphinx fixture differential tests |
| `SLLM-C03` | `MarkdownPageRouting` | `core/` | **PARTIAL** | html/dirhtml/suffix modes map each document deterministically to Markdown URLs/files. | A02 preserved upstream suite exercises html/dirhtml and file/url/auto/both/replace suffix modes across the successful pinned matrix; downstream canonical producer work remains later |
| `SLLM-C04` | `NodeAdapterRegistry` | `adapters/` | **PLANNED** | Known custom semantic nodes are rendered by explicit adapters; registration is inspectable. | registry unit tests |
| `SLLM-C05` | `DirectiveCompatibility` | `adapters/ + tests/roots/test-llm-directives` | **PLANNED** | Enabled extensions lose no unexplained semantic content. | compatibility report unsupported==0 |
| `SLLM-C06` | `CurationPolicy` | `curation/` | **PLANNED** | Inclusion/exclusion/order/code/size behavior is deterministic and separately configurable. | curation tests |
| `SLLM-C07` | `LlmsIndex` | `core/` | **PLANNED** | llms.txt is small, semantic, link-based, described, and deterministic. | golden/parser tests |
| `SLLM-C08` | `LlmsFull` | `curation/` | **PLANNED** | llms-full.txt is optional, bounded, never silently truncated, and reports skip/note policy. | size-policy tests |
| `SLLM-C09` | `ArtifactManifest` | `core/` | **PLANNED** | Every represented document has a machine-readable inventory record. | manifest schema validation |
| `SLLM-C10` | `CompatibilityReport` | `core/` | **PLANNED** | Node classes, handling mode, unknowns, and losses are counted and reviewable. | compatibility schema + zero-loss gate |
| `SLLM-C11` | `Provenance` | `core/` | **PLANNED** | Each artifact records source kind, fidelity, hashes, transforms, warnings, and generation origin. | provenance schema tests |
| `SLLM-C12` | `SummaryGeneration` | `sphinx_llm/ + core/` | **PARTIAL** | Optional LLM summaries are opt-in, credential-safe, cached, provenance-recorded, and fallback-safe. | A02 pinned upstream matrix exercises summary/cache/security paths; A10 still owns downstream provenance/provider hardening |
| `SLLM-C13` | `HtmlCompatibilityFallback` | `compat/` | **PLANNED** | Static generated HTML without canonical Sphinx representation can be converted offline and labeled compatibility fidelity. | fallback fixture tests |
| `SLLM-C14` | `ConsumerBoundary` | `core/ + published artifacts` | **PLANNED** | Runtime consumers use static artifacts; any Python build-time consumer uses only an explicit public facade, never private vendor internals. | API/import-surface + static-consumer tests |
| `SLLM-C15` | `DiscoveryLinks` | `core/` | **PLANNED** | Published HTML advertises canonical Markdown and applicable llms.txt without exposing secrets. | HTML metadata tests |
| `SLLM-C16` | `LocaleVersionRouting` | `core/` | **PLANNED** | Version/language builds cannot cross-link to the wrong artifact set. | matrix build tests |
| `SLLM-C17` | `RawContentSafety` | `adapters/` | **PLANNED** | Executable/raw browser markup cannot pass into agent Markdown without explicit sanitization policy. | malicious raw-node fixtures |
| `SLLM-C18` | `OptionalDependencyBoundary` | `core/` | **PLANNED** | Missing optional generation/provider dependencies degrade explicitly without breaking deterministic representation. | dependency-blocker tests |

## Contract status vocabulary

- `PLANNED` — required architecture, not yet implemented locally.
- `PARTIAL` — relevant implementation/source exists but one or more invariants
  are unproved or integration is incomplete.
- `HOLDS` — current source + regression gate prove the invariant.
- `VIOLATED` — current source contradicts the invariant.
- `DEFERRED` — intentionally postponed with a recorded owner/prerequisite.
- `SUPERSEDED` — replaced by another named contract; retained for traceability.

A contract may not move to `HOLDS` based only on code inspection when a practical
executable gate exists.

## `SLLM-C19` — PrimaryBuildSemanticContext

**Owner:** `compat/` plus the upstream integration boundary
**Status:** `PARTIAL`

**Invariant:** The Markdown sub-build receives every primary-build semantic input
required to resolve the same document content. Tags and `confdir` are necessary
but not sufficient; explicit config overrides must also survive the boundary.

**A02 evidence:** pinned upstream tests prove tag forwarding and external `confdir`.
The preserved NVIDIA generator still has no native config-override forwarding and
remains byte-identical. A downstream `compat/` shim now captures the effective
primary config, transfers it through an integrity-checked private snapshot, injects a
child bootstrap first, and reapplies the snapshot at `config-inited` priority 1
for Sphinx 5+ lifecycle parity. Ten dependency-light helper tests are GREEN. The
programmatic `ifconfig` regression is checked in. A canonical 10-cell execution
plan, minimal import-isolated CI cell runner, and fail-closed evidence aggregator
are also checked in with eleven dependency-light closure harness tests GREEN. The post-CI verifier also requires one coherent CircleCI pipeline/workflow/project/revision identity with unique job IDs for the expected repository. A six-test reconciliation-readiness layer accepts only closure-eligible evidence and emits a read-only receipt pinning the evidence plus 13 target-file digests. An eleven-test structural-YAML CircleCI-rebase layer renders the canonical A02 blocks into a separate current-CI candidate, refuses ambiguous/partial/in-place mutation, and validates the proposal through candidate YAML-structure verification. The downstream
parity matrix is still 0/10 GREEN; the local environment cannot materialize
Sphinx. This contract remains `PARTIAL` until all 10 required compatibility cells
are GREEN (or an explicitly reviewed equivalent matrix).


### A02 execution boundary

The semantic-context contract is tested through a dedicated opt-in repository CI
workflow (`run_sphinx_llm_a02=false` by default). A cell produces evidence; only
the 10/10 aggregator gates semantic execution, the read-only closure-evidence verifier independently recomputes the downloaded artifact and validates coherent CircleCI provenance, the reconciliation-readiness preparer pins the exact evidence/target state before human review, and the current-CI rebase renderer keeps drift handling separate from live-file mutation. This tooling does not change ownership
of representation semantics or unlock A03 by itself.
