
## Task: R173T4 — Conversation Context v2 + artifact identity

### Context
- Goal: make the request builder able to carry real conversation history, and stop
  generated artifacts from being truncated or anonymously named.
- Rationale: the panel had an artifact ledger and a transcript, and the request
  builder read neither. Every downstream symptom (lost follow-ups, `snippet-1.txt`,
  "Remember conversation" overstating itself) descends from that one fact.

### Implementation steps
- [x] T4a server chat contract v2 — history accepted, bounded, fenced as untrusted
- [x] T4a mirror contract to `_hf_spaces_model/` (caught by the parity gate)
- [x] `/health` advertises `contracts` + history bounds so the browser discovers v2
- [x] SERVER_SYSTEM_POLICY declares the `file=` capability — closes direct/proxy parity
- [x] CommonMark fence nesting — generated files containing fences no longer truncate
- [x] `_LANG_EXT` allowlist — rst/pyx/pxd/pyi/cfg/diff/... stop becoming `.txt`
- [x] One contextual filename resolver for both naming sites
- [x] Revision diff statistics (`+N` / `−M`) on file cards
- [x] Two new browser harnesses promoted from probes
- [ ] T4b browser v2 sender + context receipt
- [ ] T4c retry split (Retry answer / Re-ask with current context / Edit & branch)

### Results review — 2026-09-07
- Browser gate: 140/140 (138 baseline + 2 new harnesses)
- Architecture gates: 389/389
- Chat contract + model-service parity: 34/34
- `test_app.py`: 46/46
- 10 remaining failures in the broad proxy/integration sweep were reproduced on the
  **pristine** baseline zip in the same environment (missing PyYAML and friends) and
  are environmental, not regressions.

### Deviations from plan
- The claimed broken-pipe artifact defect did not exist. An executed probe disproved
  it and found CommonMark fence truncation instead — a silent truncation on the
  success path, which is worse. Scope changed accordingly.
- The naming resolver shipped with a suffix-truncation collision bug caught by its own
  probe before release: three blocks in one answer resolved to one identical filename.

---
# `_sphinx_ai_assistant` current todo

## Run 172 local full-suite validation — CLOSED

The exact Fix 7 workspace completed the user's Sphinx-enabled full suite with **2322 collected, 2318 passed, 4 skipped, 0 failed**. The four skips are the known Redis live/chaos gates when `redis-server` is unavailable.

### Next actions

1. Preserve Fix 7 SHA-256 `53381d32540402dac5c96a4b2eff75c02f94fc188a27f360120fbc6e36852b71` as the behavioral verification anchor.
2. Keep R172P1 Docker-context coverage retained.
3. Do not make further behavior changes under the closed Run 172 failure-repair phase.
4. Start future feature work as a new run/checkpoint.


## Parallel closed packaging failure — R172P1 proxy Docker build context

- external build failure: BuildKit could not calculate the `_providers` COPY checksum because `_providers` was absent from the sent context;
- classification: path/layout packaging defect; runtime Python was not reached;
- root cause: `.dockerignore` starts with `*` and re-included `_utils/` but omitted `_providers/`;
- fix: re-include `_providers/` and `_providers/**`, then re-exclude provider cache/bytecode;
- regression: derive local Dockerfile COPY sources and require each source to exist and be explicitly visible through the deny-by-default allowlist;
- verification: focused deployment/supply-chain 15/15; neighboring integration 51/51 warning-strict; independent ignore simulation flips `_providers` from ignored to included.

## Test-tree invariants

- `foo.py -> test_foo.py`;
- `__init__.py -> test___init__.py`;
- one collected Python owner per source module;
- large feature chunks only under non-collected `_cases/<source>/`;
- no historical `test_runNN_*` canonical filenames except `test_run_redis_chaos.py`, which mirrors the real source `run_redis_chaos.py`;
- `_integration/` and `_architecture/` are explicit cross-module planes;
- no parent-depth runtime discovery; use `tests/_paths.py`;
- no stale test-to-test filename references after a move.

## Closed local failure 1 — hermetic command adapter

- local node: `_architecture/test_release_gate_process_hermeticity.py::test_run170_command_adapter_cannot_read_parent_secret`;
- classification: test environment/isolation mismatch **plus** production pipe-resource cleanup gap;
- test fix: invoke the probe with `sys.executable` instead of a virtualenv-dependent `#!/usr/bin/env python3`;
- production fix: `publish_release.command_publisher()` now closes stdout on every exit path and reaps killed children before failing;
- regression: `test_publish_release.py` proves failed children leave both stdin and stdout closed;
- verification here: Run 170 hermeticity 10/10, canonical publisher owner 17/17, combined 27/27 with `ResourceWarning` treated as error.


## Closed local failure 2 — Run 162 witness fixture namespace drift

- local node: `_hf_spaces_proxy/security/test_anchor_archive_health.py::test_run163_anchor_bootstrap_and_offline_verify`;
- classification: canonical test-fixture namespace drift; production cryptographic logic was not implicated;
- root cause: `_threshold_root(recovery=False)` emitted `with-key-*`, while `_witnessed()` and all Run 162/163+ consumers use the intended `wit-key-*` namespace;
- fix: correct only the non-recovery fixture prefix from `with` to `wit`; recovery remains `rec`;
- verification: exact node 1/1, Run 162 32/32, Run 163 22/22, Run 164 21/21, Run 166 16/16, Run 167 20/20 in four 5-node batches.


## Closed local failure 3 — command-adapter pipe ownership

- local node: `_hf_spaces_proxy/security/test_anchor_archive_health.py::test_run163_command_adapter_bounds_output`;
- classification: production subprocess resource-ownership defect;
- root cause: output overflow killed/reaped the child but left parent stdout/stderr readers open;
- scope: warning-strict existing regressions demonstrated the copied defect in Run 151 and Runs 159, 160, 162, 163, 164; Run 161 was already safe;
- fix: reap on every exit, close owned pipes, and join readers before overflow classification;
- regression: existing bounded-output tests now assert the actual `Popen` object is reaped and its pipes are closed;
- verification: bounded-output family 7/7, Run151 14/14, Run159 33/33, Run160 34/34, Run162 32/32, Run163 22/22, Run164 21/21, Run166 16/16 with `ResourceWarning` promoted to error.


## Closed local failure 4 — hermetic Python fixture interpreter lookup

- local node: `_hf_spaces_proxy/security/test_archive_native_status_evidence.py::test_run160_command_adapter_bounds_output_while_produced`;
- classification: test environment/isolation mismatch; production hermeticity was correct;
- root cause: a temporary Python script relied on `#!/usr/bin/env python3` while the adapter intentionally supplied only `PATH=os.defpath`;
- fix: argv-capable fixtures name `sys.executable`; the path-only Run159 verifier fixture uses an absolute interpreter shebang;
- regression: focused probes replace `os.defpath` with an empty directory so interpreter discovery through child PATH cannot accidentally make the test pass;
- verification: exact node 1/1, portability probes 3/3, Run151 14/14, Run159 33/33, Run160 34/34, warning-strict.

## Closed local failure 5 — Run 161 retention fixture path ownership

- local node: `_hf_spaces_proxy/security/test_audit_archive_retention.py::test_run161_rejects_archive_auditor_operator_overlap`;
- classification: canonical test-fixture destructuring / ownership typo; production archive-health logic was not implicated;
- root cause: `_setup()` returns `mp` as tuple slot 10 and `targets` as slot 11, but the test discarded `mp` and attempted `targets.membership_path`;
- fix: bind the membership path to `mp` and use it for membership read/write and `membership_path=` while leaving `targets` as the adapter tuple list;
- verification: exact node 1/1, Run161 40/40, neighboring Run160 34/34 and Run162 32/32, warning-strict.



## Closed local failure 6 — Run 158 previous lifecycle directory ownership

- local node: `_hf_spaces_proxy/security/test_verify_attestation_lifecycle.py::test_run158_status_versions_cannot_skip`;
- classification: canonical test fixture/path-ownership typo; production validation order was correct;
- root cause: `_initialized()` returned the initialized lifecycle directory as `out`, but the test discarded it and passed nonexistent `tmp_path / "out"`;
- fix: bind the returned `out` and pass `previous_dir=out`;
- verification: exact node 1/1, Run158 32/32, Run157 22/22, Run159 33/33, warning-strict.


## Closed local failure 7 — canonical logger test-state isolation

- local node: `test___init__.py::TestGenerateMarkdownFiles::test_disabled_by_config_no_md`;
- classification: cross-test global-state leakage / test isolation defect; production logging was not implicated;
- root cause: the hostile-parent integration test imported the canonical package and assigned a warning-only `_Log` directly to `m._logger`, leaving it installed for later tests;
- fix: use `monkeypatch.setattr(m, "_logger", _Log())` so teardown restores the previous canonical logger;
- verification: untouched ordered pair reproduces the failure; repaired pair 2/2, hostile owner 11/11, Markdown class 10/10, hostile + `test___init__.py` 633 passed / 3 skipped, integration plane 182/182.

## Run 173 — improvement sequence

### R173T1 — local Save + Global Share parity — CLOSED

- canonical schema 2.1 across browser, Python proxy, and Worker;
- 2.0 accepted only as migration input;
- JSON/HTML/TXT/YAML/TOML share one canonical snapshot and Global transport;
- Save file promoted to a first-class destination;
- Global Share preserves validated resource/model/rating/feedback metadata while stripping unsafe source URLs.

### R173T2 — feedback export/review — CLOSED

1. local each-feedback export via panel;
2. cloud merged-feedback representation/review;
3. preserve explicit network-consent boundary: local export never grants cloud submission permission.

#### R173T2A — feedback artifact identity/privacy boundary — CLOSED

- local filenames identify JSON request vs JSONL cloud projection;
- feedback network model attribution is transport-minimal;
- page source evidence is sanitized on both client and server;
- next: deterministic merged cloud feedback view/export while keeping per-record lifecycle authority.

#### R173T2A1 — conversation artifact provenance naming — CLOSED

- local Save filenames now identify lifecycle role + format + timestamp for JSON/HTML/TXT/YAML/TOML;
- Global Share names identify lifecycle role + format without capability leakage;
- Global viewer can download the server-canonical artifact through fixed POST `/v1/share/download`;
- next remains deterministic cloud-merged feedback view/export, not monolithic cloud write authority.

## After R173T2B

- [x] Add deterministic cloud-merged feedback JSONL derived view with role-bearing filename.
- [x] Bind merged bytes with a non-authoritative SHA-256/count manifest.
- [x] Prevent prior merged exports from re-entering local snapshot authority.
- [x] Ingest the user's contribution-section local/cloud files and map request/projection/provider/merged artifact parity before coding the next slice.


## R173T3 — contribution provenance/cloud merge hardening

- [x] Map single-pair, rated-answer, and whole-conversation request/projection/provider parity.
- [x] Add scope + lifecycle-role filenames for contribution request JSON and cloud-projection JSONL.
- [x] Minimize contribution model evidence to attribution-only client-side and server-side.
- [x] Sanitize page provenance before contribution digest/idempotency/normalization.
- [x] Add deterministic derived cloud-merged contribution JSONL + integrity/authority manifest.
- [x] Prevent default and cryptographically manifest-bound custom merged exports from re-entering input authority.
- [x] Re-minimize historical contribution rows during derived export.
- [x] Make merged JSONL and manifest writes atomic per artifact.
- [x] Close test-owned SQLite connection warning without changing production ledger ownership.
- [x] Complete exact R173T3 candidate packaged-byte replay and mark checkpoint COMPLETE; final immutable rebuild/replay follows the evidence freeze.


## R173T3A — post-T3 maintenance consistency closure

- [x] Mark the completed feedback sequence closed.
- [x] Close the completed contribution-ingestion audit todo.
- [x] Record immutable final R173T3 packaged-byte replay separately from candidate evidence.
- [x] Refresh fresh-chat handoff to start after R173T3.
- [x] Preserve unrelated CSS dark-mode TODO as optional future UI cleanup.
- [x] Keep runtime/tests/skills byte-identical to R173T3.

## R173T85 — PDF thumb geometry/cascade parity — CLOSED

- [x] Diagnose computed CSS authority rather than adding another optical offset.
- [x] Replace competing checked transforms with one shared travel-token authority.
- [x] Keep Mic default travel at 12px and PDF/Copy/Panel large-toggle travel at 16px.
- [x] Preserve PDF/Panel 34×18 track and 14×14 thumb geometry.
- [x] Add dependency-free cascade/parity regression and rerun neighbor/maintenance gates.

## R173T86 — workspace tab visual parity — CLOSED

- [x] Trace font/color differences to computed-style context rather than markup.
- [x] Make the shared format-tab component own line-height.
- [x] Remove the workspace-only selected colour/underline authority.
- [x] Preserve T79 content-sized layout and T84 icon/ARIA/keyboard behavior.
- [x] Add visual-parity and mutation regression coverage.


## R173T87 — Share artifact responsive action layout — CLOSED

- [x] Trace narrow-row failure to duplicate flex shorthand resetting metadata basis to 0%.
- [x] Group managed lifecycle actions structurally instead of relying on direct-child wrapping.
- [x] Make stacking respond to actual Share container width rather than viewport width.
- [x] Keep three-action rows compact and four-or-more action rows balanced 2x2 when tight.
- [x] Update stale DOM/static tests and mutation controls to the stronger structure.


## R173T88 — mobile model action menu trigger anchor — CLOSED

- [x] Trace the large mobile menu gap to whole-row absolute positioning.
- [x] Give the ellipsis trigger and action popup one local positioned host.
- [x] Keep the mobile popup adjacent to the trigger regardless of model-card height.
- [x] Flip above near the visible model-sheet bottom boundary.
- [x] Preserve Edit/Delete/Edited/Reset behavior and desktop action rail.
- [x] Repair CSS mutation-target plumbing in the model-responsive harness.
- [x] Add positive-control mutants for row re-anchoring, lost positioning context, and lost edge flip.

## R173T92 — answer-section disclosure discoverability — CLOSED

- [x] Diagnose why open `<summary>` rows read as static headings, especially on touch.
- [x] Replace hover-dependent discoverability with a persistent theme-aware disclosure surface.
- [x] Normalize chevron semantics to right=closed / down=open.
- [x] Add visual `Show section` / `Hide section` guidance without duplicating accessibility state.
- [x] Preserve native `<details>` state, forced-colors and reduced-motion behavior.
- [x] Add focused regression and mutation controls; rerun the complete registered Node harness plane.

## R173T93 — inline snippet scroll handoff — CLOSED

- [x] Trace the rare wheel/touch dead zone through both inner-code and outer-sheet overflow ownership.
- [x] Remove the generic file-sheet overflow shorthand that contradicted axis-specific ownership.
- [x] Make inline answer snippets yield vertical scrolling to the conversation while preserving horizontal long-line pan.
- [x] Preserve rounded snippet chrome without using overflow clipping as scroll ownership.
- [x] Keep native wheel/touch chaining; add no JavaScript wheel interceptor.
- [x] Add focused regression plus four positive-control mutations and rerun canonical static UI harnesses.

## R173T94 — Presented-file segmented-control parity — CLOSED

- [x] Compare Presented-file composition against the known-good normal artifact card before editing.
- [x] Trace the visual drift to multiple historical grid/layout authorities rather than DOM order alone.
- [x] Reduce Presented-file primary layout to one `[artifact group] [overflow]` grid.
- [x] Make Presented-file Download reuse `ai-md-artifact-download-label` while retaining its behavior hook.
- [x] Move preview/download flex geometry into the shared artifact-group contract.
- [x] Remove dead save-as/patch/continue/secondary/more direct-control CSS.
- [x] Replace contradictory historical assertions with one structural-parity contract.
- [x] Add five positive-control mutants and rerun the complete Node/mutation planes.


## R173T95 — Presented-file responsive segment continuity — CLOSED

- [x] Reproduce the defect boundary at the legacy 560px responsive rule.
- [x] Remove full-width Download geometry from the in-group mobile state.
- [x] Make Presented list/card/primary ancestors explicitly shrinkable.
- [x] Move Presented heading stacking from viewport width to component width.
- [x] Preserve T90 per-file/bulk label compaction thresholds.
- [x] Add four positive-control mutations and rerun complete Node/mutation planes.


## R173T98 — toolbar dropdown surface ownership — CLOSED

- [x] Reproduce the transparent dropdown as a surface declaration problem, not stacking/opacity.
- [x] Trace the current-source regression to T96 borrowing the speak-toggle local surface token.
- [x] Give the toolbar dropdown its own light/dark semantic surface fallback.
- [x] Recognize PyData `data-theme` / `data-mode` dark attributes plus Bootstrap/.dark compatibility.
- [x] Add a focused regression rejecting a second color after `var()` and rejecting speak-token coupling.
- [x] Add three positive-control mutants and rerun complete Node/mutation/maintenance planes.
