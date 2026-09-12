# `_sphinx_ai_assistant` Maintenance History

## 2026-08-20 — maintenance normalization package prepared

- Introduced Corpus/MCP-style live trackers, state, registry, checkpoints,
  schemas, and verification semantics for the multi-runtime assistant.
- Established sibling `_sphinx_llm` as the future owner of canonical Sphinx
  Markdown/`llms.txt`/`llms-full.txt`/directive representation.
- Kept current source behavior unchanged; runtime HTML conversion remains legacy
  current behavior until staged migration gates close.
- Recorded conservative security contract states instead of treating existing
  desired-policy prose as proof.

## 2026-08-22 — capability pivot: the assistant owns its representation

- `_sphinx_llm` frozen. It is not required by any assistant surface: the
  assistant already writes `page.md` and `llms.txt` itself on `build-finished`.
- Governing rule rewritten. It previously ended *"canonical representation is
  consumed from `_sphinx_llm`"*, which pointed the one dispute-resolving
  sentence at frozen code.
- B01 inverted (define -> sever), B08 inverted (consume -> produce), A12
  withdrawn.
- Reverse-dependency rule corrected: it scanned raw text, so a module docstring
  naming the assistant in an architecture diagram was reported as a dependency.
  It now detects imports. The gate was red for prose.
- Representation contract recorded explicitly: canonical is the build-time
  artifact, convenience is the browser conversion, and the distinction is not
  about quality — an external agent cannot fetch a `blob:` URL.

## 2026-08-22 — maintenance reconciled to the pivot

- `DEPENDENCY_MAP.md` rewritten: two live members, one live edge, and the false
  "Identical in all three `_maintenance/` folders" claim removed — the copies
  were never identical and each is now marked with its owner.
- `INTEGRATION_CONTRACT.md` re-titled and rewritten: the contract is no longer
  with `_sphinx_llm` but between this extension's own build and browser layers,
  plus the HTTP edge to the backend.
- `RULESET.md` 16-20 replaced: `_sphinx_llm` integration -> representation
  ownership.
- `REGISTRY.md`: `AIA-002` (P0) and `AIA-013` (P1) **WITHDRAWN**. Both existed to
  migrate canonical ownership away from the assistant; the pivot makes the
  assistant's own generation the intended design rather than a debt.
- `TRACKER_LOGICAL.md`: `AIA-C04` `RepresentationConsumer` (PLANNED) ->
  `RepresentationProducer` (ACTIVE).
- `MAINTAINING.md`: anchor and physical scope remeasured; ownership section
  inverted; representation contract stated.

## 2026-08-22 — capability increments 1-5 landed

Five bundles applied and verified against a pristine re-extraction. No new
runtime dependency; `__init__.py` still has zero module-scope non-stdlib imports.

- **Copy mode toggle.** `browser` (convenience, default) or `static` (canonical).
  A failed static fetch names the alternative rather than silently substituting
  the browser conversion.
- **`llms.txt`** now follows the llmstxt.org layout. Titles come from each page's
  own heading and descriptions from its first prose paragraph, so the index
  improves as the docs do.
- **Directive fidelity.** 15 rules in one table, declared in Python and
  serialised to the browser, so a rule cannot exist on one side only.
- **Root-cause fix.** `strong_em_symbol` was `"**"`; markdownify doubles it for
  `<strong>`, so every bold run in the documentation converted to `****text****`
  — 209 occurrences across 18 published pages, in every `page.md` and everything
  sent to an AI provider. Now 0.
- **Video links.** Embed URLs mapped to watch URLs using the templates
  `_sphinxcontrib_youtube` already uses for its own epub and latex output. Vimeo
  deliberately unchanged: its `_platform_url` *is* the player URL, so no watch
  form is invented.

Not proven from a source snapshot, and recorded as such: that COPY(static) and
ASK AI resolve to byte-identical content, and that no surface degrades with
`_sphinx_llm` absent. Both need a live build.

## 2026-08-28 — maintenance/runtime separation and proxy v6.4 streaming recovery

- Moved maintenance-only material out of the runtime extension into the mirrored
  repository-level path
  `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/`.
- Retained historical backups, checkpoints, schemas, lessons, and active task
  material without making runtime code depend on them.
- Renamed the maintenance work queue from source-local `tasks/` to maintenance
  `todo/`; the runtime drop-in no longer carries `tasks/`.
- Updated `check_trackers.py` so it works in both a standalone maintenance
  archive and a full repository checkout, and fails if runtime-local maintenance
  directories reappear.
- Added `APP_STREAMING_RUNBOOK.md` and checkpoint B15 with deterministic stub and
  real-model curl probes plus the v6.4 response-mode state machine.
- Anchored the runbook to the v6.4 proxy behavior: `stub/*` fail-closed,
  no blank bearer header, upstream opened before downstream success, JSON kept as
  JSON, SSE kept as SSE, and terminal stream failures made explicit.


## 2026-08-29 — B16 unified Share conversation architecture

- Replaced three JSON/HTML/TXT Share sheet instances and hard-coded routing with
  one registry-driven Share shell and lazy immutable format panels.
- Moved live Share serializer/MIME/extension metadata behind `_EXPORT_FORMATS`
  so direct download and Share use the same format ownership boundary.
- Replaced duplicate-ID/first-element export-mode synchronization with observer
  controls and one true interactive button per rendered control.
- Added explicit conversation-scoped UI identity plus format/conversation guards
  for delayed IndexedDB/network completions. New chat/clear invalidates cached
  Share UI state without pretending that browser identity is authorization.
- Added one canonical sheet registry for open/close/Escape/toolbar/focus concerns.
- Reworded local link choices as Session-only and Portable, with explicit
  exposure semantics for self-contained URLs.
- Added a 97-assertion architecture contract, 35-assertion execution smoke, and
  Share-specific mutation cases. Runnable non-Sphinx regression state closed at
  409 passed / 3 skipped; the full Sphinx fixture plane remains environment-gated.
- Closed `AIA-018`; opened `AIA-019` for payload preflight + real-browser E2E.
  `AIA-006` remains P0 OPEN and is not affected by this browser UX checkpoint.


## 2026-08-29 — B18 Run 1 — secret lifecycle containment

Endpoint profile storage moved to schema v3. Legacy v1/v2/raw storage is rewritten after sanitization so stale bearer values are physically removed from the browser storage blob. New custom profile persistence/export omits token fields. Build-time profile and legacy flat Share/Feedback token config values are ignored and never serialized into static HTML. Runtime-entered tokens remain an explicit current-page-memory compatibility path only. B18 remains open for server/log/prompt/retention work.

### 2026-08-29 — B18 Run 1 packaged-copy verification

The Run 1 secret-lifecycle overlay was re-extracted into a clean directory and rerun against the packaged bytes: endpoint secret lifecycle 24/24, client secret boundary 3 passed, Node harnesses 28 passed, mutation gate 107 passed, standalone non-Sphinx suite 417 passed / 3 skipped, and maintenance drift checker GREEN. The Sphinx-dependent suite remains environment-blocked in this sandbox.


## 2026-08-29 — B18 Run 2 — export / Share active-content isolation

Run 2 introduced one privacy-sanitized canonical conversation snapshot for direct
JSON/HTML/TXT downloads and Share serialization. HTML embedded JSON now receives
raw-text-safe encoding before entering `<script type="application/json">`, and
the generated offline viewer ships a restrictive no-script/no-network CSP. The
self-contained hash transport moved from executable `c1.<fmt>.<content>` to a
validated structured `c2` envelope. Legacy c1/IndexedDB HTML is opened only as
`text/plain`. HTTP(S) share bases strip query/hash; file/custom schemes cannot
produce a path-leaking c2 URL. Global Share server representation/auth remains
open for Run 3. Detailed B17 subfindings were reconciled to canonical `SEC-P0-*`
IDs to remove collisions with durable AIA registry IDs.

## 2026-08-29 — B18 Run 3 / B19 Global Share server authority

Global Share stopped accepting caller-rendered content/MIME as authority. HF and
Cloudflare now store canonical structured snapshots, own JSON/HTML/Text
representation, split public read from private edit capability, enforce
PATCH/DELETE authorization, and return no-store/noindex/sandboxed responses.
Client edit capability is memory-only. Share application logs omit capabilities;
the bundled HF Uvicorn access log is disabled. Resource limits and trusted
forwarding/public-base rules are explicit. Cloudflare KV aggregate quota remains
conservative rather than atomic and is documented as such.

## 2026-08-29 — B20 Prompt Authority & Credential Destination Binding

- Added negotiated `scikitplot-chat-v1` browser/server contract.
- Moved authoritative system policy and provider-native reasoning mapping to bundled servers.
- Added server model allowlists and rejected client system/developer/messages/tools/destination authority.
- Split Path 1/2/3 credentials into `BACKEND_AUTH_TOKEN`, `HF_SPACES_AUTH_TOKEN`, and `HF_TOKEN`; disabled credential-bearing redirect following.
- Hardened the independently reachable `_hf_spaces_model`: it now advertises/parses the same typed contract, rejects arbitrary OpenAI messages, and reconstructs policy locally. Path 2 preserves the structured envelope across proxy → model-service.
- Added authority/cross-path tests and mutation positives.
- CORS, resource limits, centralized log redaction, contribution retention, and sensitive-input preflight remain later-run work rather than being mislabeled closed.



## 2026-08-29 — B21 / Run 5 logging, telemetry, and public diagnostics

- Added byte-identical proxy/model `_telemetry.py` privacy boundaries.
- Unified ordinary log-message and exception/traceback sanitization and length
  bounding; removed raw model traceback formatting and nested exception leakage.
- Prohibited partial-secret token fragments in diagnostics.
- Removed stable feedback/session/share identifiers and raw exception messages
  from Worker application logging; fixed reflected public network/contract errors.
- Kept bundled HF Uvicorn access logs disabled and made model-service access-log
  disablement explicit.
- Minimized proxy/model/dev public discovery to coarse client-required
  capability/readiness state. Detailed storage/repository/routing/token-class/CORS
  topology remains operator-only.
- Preserved browser discovery compatibility by modeling service-managed storage
  as coarse state instead of restoring detailed server diagnostics.
- Added adversarial logging/privacy tests and mutation positives.
- `AIA-C06 EndpointDiscovery` moved to HOLDS; `AIA-C20 PrivacyDataLifecycle`
  moved to PARTIAL. `AIA-022` / `SEC-P0-15` intentionally remain PARTIAL because
  provider/CDN/reverse-proxy access logs can still observe path-based Share read
  capabilities outside application source control.


## 2026-08-29 — B22 / Run 6 feedback-contribution lifecycle

- Split ordinary rating feedback from explicit content contribution. Browser network rating telemetry is now opt-in and content-minimal; server normalization discards Q&A/comment/model/page/session data even from direct callers.
- Advanced canonical collection schema to v3 with lifecycle state (`telemetry`, `quarantined`, `eligible`, `legacy_unreviewed`) and model evidence (`client_reported`, `legacy_unverified`).
- Enabled exact versioned contribution consent (`1.0.0`).
- Replaced direct raw contribution persistence with bounded process-local mutable quarantine. Pending intake receives a separate delete capability and can be physically removed before promotion.
- Added independent operator review/promotion using `CONTRIBUTION_REVIEW_TOKEN`; only promoted rows become `trainingStatus=eligible`.
- Changed the dataset cleaner to fail closed: ordinary training output accepts eligible contributions only; feedback telemetry and unreviewed/quarantined rows are excluded.
- Hardened Cloudflare feedback to the same minimal telemetry class; contribution review parity remains a documented residual.
- Reconciled B07 through B22 and marked `AIA-C13 FeedbackProvenance` HOLDS; `AIA-C20 PrivacyDataLifecycle` remains PARTIAL for Run 7 preflight and durable control-plane/erasure residuals.
- Working-tree gates: 21 browser assertions; 10 privacy lifecycle tests; 10 dedup tests; 32 Node harnesses; 137 mutations; 529 passed/3 skipped non-Sphinx. Full suite remains environment-blocked by missing Sphinx: 995 passed/3 skipped/5 failed/62 errors.


## 2026-08-29 — Run 7 sensitive-input/privacy preflight

- Added one local advisory privacy-preflight engine shared by inference, Share, and explicit contribution.
- Findings expose categories/counts and Unicode codepoints only; matching secret/personal values never enter dialog state/logging.
- Inference reviews the user message plus the prepared automatically attached page context before transcript/network mutation.
- Share reviews the canonical snapshot and uses the exact post-review snapshot for generated/opened results.
- Contribution reviews the exact consented outbound payload before quarantine egress.
- Added explicit Go back / Redact & continue / Continue unchanged behavior; redaction mutates an operation copy only.
- Added fail-closed behavior when flagged data cannot be reviewed and an async conversation-identity guard so delayed dialogs cannot publish into a new chat.
- Added B23, AIA-024, AIA-C21, SEC-P0-28, seven mutation positives, and dedicated source/DOM/race harnesses.
- Broad non-Sphinx suite: 545 passed, 3 skipped. Full suite remains environment-blocked by missing Sphinx: 1011 passed, 3 skipped, 5 failed, 62 errors.


## 2026-08-29 — B24 / Run 8 Share formats and artifact lifecycle

- Promoted YAML and TOML to first-class export/share formats with canonical MIME types and real parser round-trip gates.
- Centralized Share information architecture around Format → Destination → Content & privacy → Advanced → Result.
- Added Standard/Minimal/Complete/Customize privacy presets before serialization.
- Added conservative self-contained URL byte preflight and long/oversize fallbacks.
- Added one page-memory managed-artifact lifecycle registry: Local Blob removal really revokes; Global Revoke performs server DELETE with the memory-only edit capability; self-contained removal is explicitly local-only; Share-sheet and direct toolbar Download records are tracked but cannot claim device-file deletion.
- Preserved page-memory Global revoke capability across New chat while continuing to exclude edit tokens from sessionStorage.
- Added browser+server YAML/TOML real-parser round-trips and artifact-lifecycle mutation positives (including direct-download tracking); broad non-Sphinx suite reached 563 passed / 3 skipped before packaging. Full attempt reached 1029 passed / 3 skipped before the unchanged missing-Sphinx wall (5 failed / 62 errors).
- Candidate drop-in archive reproduced the full Run 8 packaged-copy plane (563 passed / 3 skipped non-Sphinx plus focused parser/lifecycle/capability/mutation gates); final metadata rebuild is rechecked before delivery.
- Real-browser accessibility/responsive/focus/clipboard acceptance remains an explicit AIA-019 residual.

## 2026-08-29 — Run 9 / B25 Global link artifact tracking

- Added bounded `sessionStorage` public Global artifact ledger (25 entries).
- Preserved all links handed to the user across new-chat/reload as read-only lifecycle records.
- Kept edit/revoke capability page-memory-only.
- Added explicit content-free HEAD lifecycle probe to HF + Cloudflare services.
- Successful revoke now becomes a lifecycle tombstone until Forget.
- Added Run 9 source/DOM/server/mutation gates.

## 2026-08-29 — Run 10 / B26 Global lifecycle fail-closed recovery

- Revalidated the exact Run 9 overlay and traced the Global URL preserved in the supplied MHTML without assuming remote status.
- Rebuilt `ai-assistant-global-share:v2` recovery from an allowlist and destructively scrubbed legacy/tampered forbidden fields, including any persisted edit capability/content fingerprint.
- Made reason-unknown 404 non-terminal/recheckable while detaching it from implicit current-conversation PATCH state.
- Added 410-to-fresh-POST client fallback and aligned explicit expiry behavior across HF/Cloudflare GET/PATCH/DELETE/HEAD lifecycle routes.
- Added legacy/tamper, 404→200 recovery, unavailable Revoke+Forget escape, expired-delete, and five Run 10 mutation regressions.



## 2026-08-29 — Run 11 / B27 CORS, identity, resource and deployment parity

Source: `scikitplot__sphinx_ai_assistant_b18_run10_global_link_lifecycle_hardening_overlay.zip` (`cd3897bc7078fd8a1df4cbb7c8e3282772868062b7cbe10c450f723c2ed673fc`).

Closed B05/B06 at the bundled application boundary. Wildcard CORS defaults were replaced with exact defaults; HF proxy/model and Worker now reject explicit disallowed browser Origin before work; HF proxy/model and Worker enforce request ceilings while streaming; dev proxy checks declared size before read; all HF proxy rate-limited routes share direct-peer/default-deny-XFF identity; HF abuse maps are cardinality-bounded; feedback retractions no longer bypass rate limits; Worker rate state uses unique expiring KV event keys instead of same-key counters; bundled Wrangler main now matches the shipped `index.js`. Strict distributed rate accounting is explicitly deferred as `SEC-P0-31`, not mislabeled green.


## 2026-08-29 — Run 12 / B28 contribution receipt lifecycle and withdrawal

- Replaced route-owned process-local contribution quarantine authority with `_contribution_ledger.py`: bounded memory compatibility plus optional local transactional SQLite.
- Added `CONTRIBUTION_REQUIRE_DURABLE=true` fail-closed intake for deployments that require restart-durable receipt state.
- Kept receipt mutation authority client-held and digest-only server-side; raw Q&A is cleared from receipt state after promotion.
- Added atomic promotion/withdrawal state claims, closing the double-promotion read/write race.
- Added authenticated content-free receipt status and preserved the receipt capability after promotion.
- Added privacy-minimal withdrawal tombstones keyed only by contribution dedup key; ordinary dataset LWW now suppresses withdrawn eligible rows and excludes tombstones from output.
- Added best-effort current-view removal across Hugging Face/GitHub/GitLab/Bitbucket plus stale in-process mirror-retry suppression.
- Corrected pending deletion semantics: active-ledger content removal is not a forensic/global physical-erasure promise. SQLite secure-delete/WAL truncation are defense in depth only.
- Added restart reclamation for stale `promoting`/`withdrawing` states and receipt-stable provider paths/tombstone timestamps so crash replays do not deadlock receipts or fan out duplicate logical artifacts.
- Added bounded terminal receipt tombstone retention and moved withdrawn-record suppression to the provider write-lock boundary, closing a retry-after-withdrawal resurrection race.
- At the Run 12 checkpoint, shared multi-replica receipt authority, provider-history/global erasure, strict distributed rate accounting, infrastructure Share-log redaction, real-browser E2E and canonical Sphinx release verification remained explicit residuals. Runs 13–16 subsequently close/ratchet several of those boundaries as recorded below.


## 2026-08-29 — Run 13 / B29 Share fixed-path fragment transport

- Replaced newly generated `/v1/share/<read-capability>` URLs with fixed `/v1/share#share=<locator>` viewer URLs.
- Added fixed body-locator read/status/update/revoke operations to HF and Cloudflare while retaining legacy path routes for old-link compatibility.
- Added exact dual locator validation for HF 32-hex IDs and Worker canonical UUIDs; a runtime parity test caught and closed the initial format mismatch.
- Migrated browser status/update/revoke and bounded session lifecycle recovery to fragment-aware fixed paths without persisting private edit authority.
- Added Worker fixed-update rate limiting and disabled automatic Worker invocation logs as URL-log defense in depth.
- Added B29/AIA-C27 and Run 13 transport/runtime/mutation gates. `SEC-P0-15` is now partial only for legacy capability-bearing paths and deployment-configured payload-level telemetry.


## 2026-08-29 — Run 14 / B30 legacy Share compatibility drain

Stamped all new/fixed-updated Share objects with transport generation 2; generation-2 objects are now ineligible for capability-bearing `/v1/share/{id}` routes. Legacy HEAD/GET/authenticated DELETE remain only for pre-generation objects and emit standards-shaped deprecation/sunset/successor metadata. Legacy PATCH is retired with 410 so path transport cannot refresh TTL; fixed `/update` is the one-way migration path. `SEC-P1-34` moves from OPEN/BOUNDED to PARTIAL/DRAINING rather than CLOSED because already-issued pre-generation URLs still necessarily expose their locator until expiry/revoke/migration.


## 2026-08-29 — Run 15 / B31 distributed rate-limit authority

- Added optional shared Redis fixed-window authority for HF replicas using one atomic Lua operation and HMAC-pseudonymized identity keys.
- Added `RATE_LIMIT_REQUIRE_SHARED=true` fail-closed semantics so a configured authoritative HF deployment cannot silently split quota into process-local counters.
- Added a sharded `RateLimitBucket` Durable Object authority to the Worker and made the bundled Wrangler deployment require it.
- Preserved unique-event KV and HF memory counters only as explicitly non-authoritative compatibility gates.
- Added truthful backend/shared/authoritative/readiness discovery without endpoints/secrets/raw identities.
- Added 11 Python authority/positive-control tests, 15 Worker runtime assertions, and retained all existing client mutation gates; broad non-Sphinx tree reached 645 passed / 3 skipped before packaging.
- `SEC-P0-31` moves from pure infrastructure deferral to partial/deployment-conditional: Worker bundled authority is active by config; horizontally scaled HF must still provision and require Redis.


## 2026-08-29 — Run 16 shared contribution receipt authority

- Added optional shared Redis contribution receipt authority with atomic bounded Lua lifecycle/capacity operations and same-slot `{contribution}` key ownership.
- Added dedicated HMAC-SHA256 receipt-key pseudonymization and fail-closed `CONTRIBUTION_REQUIRE_SHARED=true`.
- Added cross-replica operation claim digests/leases, then deliberately rejected automatic promotion lease takeover because Redis cannot fence already-issued external Git/provider writes.
- Added `promotion_uncertain` / `withdrawal_uncertain` recovery semantics; participant withdrawal remains the monotonic privacy-safe resolution.
- Normalized provider transport/timeout mutation ambiguity so possible-write/lost-response failures cannot reopen ordinary promotion.
- Preserved Run 12 memory/SQLite behavior and clarified that Redis shared coordination is distinct from external persistence/backup durability.
- Added B32/AIA-C30 and closed SEC-P0-34; SEC-P0-26 remains partial only for production activation/external Redis durability evidence and provider-complete erasure remains separate.


## 2026-08-29 — B18 Run 16.2 — portable data Share + Global link delivery

Current Self-contained generation moved from host-page `#ai-share-c2` links to a bounded inert `data:text/html;charset=utf-8;base64,...` artifact built from the reviewed normalized snapshot. c1/c2 remain compatibility-only. The UI states that base64 is not encryption and copied links are non-revocable. **Run 16.2.1 correction:** Open now attempts the exact generated data URL and never substitutes an origin-bound Blob; if browser policy blocks the navigation, the same copied data URL remains the explicit-open path. Global creation now always resolves a validated public read URL: compatible UUID-only responses are converted to the configured fixed `/v1/share#share=<id>` viewer URL, and managed Global artifacts expose Copy link.


## 2026-08-29 — Run 16.2.2 — visible Global creation feedback

Fixed a silent UI/network failure mode where `Create global link` could appear inert if a successful HTTP response had an empty or non-JSON body, or if request setup failed before callbacks. The result area now shows `Creating global link…` immediately, then either the copyable public URL or a bounded error + Retry. Fetch/CORS, HTTP, serialization, and invalid-success-response errors are surfaced without reflecting arbitrary response HTML.

## 2026-08-29 — Run 16.2.3 — Local preview copy/inspect action consistency

Added Copy link and Inspect to the Local preview result while preserving browser-local Blob semantics. The result now uses **Copy link | Open | Inspect | Remove from browser**. Created-artifact rows for local/self-contained artifacts use **Copy link | Open | Remove**; Global lifecycle controls remain unchanged. Copying a local Blob URL does not upload it or make it portable.

## 2026-08-29 — Run 16.2.4 — official docs CORS deployment repair

Live Global Share creation reached the HF endpoint but returned `403 Origin not allowed` from proxy v6.5.0. The documentation origin is `https://scikit-plots.github.io`; the code default matched it, revealing that an `ALLOWED_ORIGINS` deployment override could replace the built-in default. Run 16.2.4 makes the official docs origin immutable within the exact allowlist, treats `ALLOWED_ORIGINS` as additive extras, validates configured origins, mirrors the rule in the Worker, bumps proxy version to 6.5.1, and adds non-sensitive `/health` CORS diagnostics.


## 2026-08-29 — Run 16.2.5 — HF proxy `_utils` package cleanup

Moved all private/helper Python modules out of `_hf_spaces_proxy/` root into an import-light `_utils/` package. `app.py` and `deduplicate_dataset.py` are now the only supported root-level Python entrypoints. Updated both package and top-level HF Space import paths, changed Docker to copy `_utils/` atomically, moved the unused legacy `deduplicate_dataset_v1.py` under `_utils/`, synchronized direct-test imports and active maintenance paths, and added a layout regression guard. Runtime/security behavior is unchanged.

## 2026-08-29 — Run 16.2.6 — HF Share public-base + local-file CORS repair

Live Global Share diagnostics separated two deployment failures. The HF proxy now derives its external HTTPS Share base from Hugging Face's deployment-owned `SPACE_HOST` when `SHARE_PUBLIC_BASE_URL` is absent, and safely upgrades an accidental `http://<same SPACE_HOST>` configuration without trusting arbitrary forwarded headers. Local `file://` pages remain denied by default because browsers serialize them as `Origin: null`, which is also shared by sandboxed opaque documents. Operators that explicitly need local-file Global Share can enable `SHARE_ALLOW_OPAQUE_ORIGIN=true`; the allowance is restricted to `/v1/share` routes and mirrored in the Worker. Proxy version is 6.5.2 and `/health` exposes only the safe boolean compatibility state.

## 2026-08-29 — Run 16.2.7 Global revoke-to-Forget unlock

- Fixed a UI lifecycle lock leak after successful Global revocation: the server
  revoke completed and the artifact correctly transitioned to `revoked`, but
  `artifact.busy` remained true, so the newly rendered **Forget** action was
  disabled.
- Successful revoke now clears the operation lock before rendering the terminal
  lifecycle state.
- Forget remains strictly browser-local lifecycle cleanup; it does not issue a
  second server mutation and does not claim additional revocation.
- DOM regression now asserts **Create -> Revoke -> enabled Forget -> local record
  removal**.

## 2026-08-29 — Run 17 — Dataset contribution purpose separation and conversation records

Dataset contribution is removed from Share → More actions and promoted to a first-class **Contribute to dataset** sheet. Feedback telemetry is renamed truthfully and remains content-free; a separate **Contribute this Q&A…** shortcut routes into the canonical contribution workflow. The sheet supports This Q&A, Rated answers, and Whole conversation, where whole conversation is one schema-v4 ordered `messages[]` record. Inspect, privacy preflight, consent 2.0.0, quarantine, receipt deletion and post-promotion withdrawal are one lifecycle. Endpoint Configuration becomes **Runtime & Data** with separate Feedback telemetry and Dataset contributions sections. Proxy deployment version becomes 6.6.1. Telemetry is additionally ratcheted to a two-sided explicit-permission contract: structured versioned browser consent, self-gated rating/retraction helpers, server consent validation, and content-free public feedback events. The fresh-chat design source is `_maintenance/FRESH_CHAT_DATASET_CONTRIBUTION_UX_HANDOFF.md`.

## 2026-08-30 — Run 18 / B37 lifecycle and privacy closure

- Replaced HF process-memory-only Global Share authority with a truthfully
  classified memory/SQLite/Redis `ShareStore` abstraction and fail-closed
  durable/shared deployment requirements.
- Added pre-request recoverable CREATE envelopes for Global Share and dataset
  contribution. Current browser creates transmit only a digest of the locally
  held revoke/delete capability; exact replay returns the same object and
  divergent payload/capability replay conflicts.
- Added explicit `outcome_unknown` UX for ambiguous create results and exact
  operation retry instead of duplicate/orphan-prone fresh creates.
- Added explicit contribution management-receipt Save/Import without silent
  long-lived capability persistence.
- Split public feedback DOM integration from network telemetry permission;
  rating remains local-only by default.
- Enforced contribution review/storage limit parity and reject-not-truncate
  semantics for reviewed schema-v4 content.
- Made transcript recovery per-tab opt-in/bounded, moved microphone device ID to
  session scope, minimized Share Standard metadata, and sanitized/restricted
  runtime endpoint query handling.
- Added atomic Redis Share expiry cleanup and truthful Worker KV eventual-
  consistency semantics.
- Proxy deployment version: **6.7.0**.
- Initial working-tree gates: B37 Python **6 passed**, B37 browser **56/56**,
  focused compatibility/mutation **229 passed**, runnable non-Sphinx **711
  passed, 3 skipped**. Exact-package evidence follows the release cycle.


## 2026-08-30 — Run 19 / B38 supply-chain and deployment hardening

- Revalidated `SEC-P0-10` against exact Run-18 delivery anchor.
- Replaced floating/ranged release inputs with immutable Linux/amd64 base and
  exact hash-locked 30-package Python runtime closure.
- Removed broad production standard extras; direct runtime requirements are five
  exact packages.
- Added two-stage non-root strict runtime, deny-by-default Docker context and
  hardened read-only deployment reference.
- Added shared strict Redis TLS policy across rate-limit, Share and contribution
  authority.
- Added machine policy, CycloneDX Python-lock SBOM, offline verifier and explicit
  fresh dependency/image/SBOM/provenance release gates.
- Ratcheted known vulnerable historical Click/Starlette versions to reviewed
  current dependency closure instead of freezing the old environment.
- Proxy deployment version: **6.8.0**.
- Working-tree B38 gate: **8 passed**; complete runnable non-Sphinx tree:
  **719 passed, 3 skipped**. Exact-package evidence follows the release cycle.

## Run 20 / B39 — release evidence and production guardrails

- Added short-lived, content-addressed production release evidence with a
  deterministic runtime-source digest so unchanged dependencies cannot inherit
  a scan after application source changes.
- Bound dependency/image scans, full-image SBOM, SLSA provenance and signature
  verification to exact subjects; provenance must also include the resolved
  immutable base-image manifest.
- Added one fail-closed `verify_release_gate.py` combining B38 source policy and
  B39 evidence binding.
- Added privacy-minimal `release_subjects.py` and explicit Redis operational
  probe that accepts only an environment-variable name and emits no
  authority/topology values.
- Added production logging/WAF/APM/third-party telemetry Off requirements so
  browser consent cannot become an infrastructure telemetry bypass.
- Proxy deployment version: **6.9.0**.
- Final verifier self-review made schema v1 closed/bounded, requires exact runtime proxy version, CycloneDX >=1.6, bounded tool identity, distinct resolved base manifest, and deterministic Docker-context/runtime-source parity including non-Python `_utils/` inputs while excluding generated bytecode.
- `SEC-P1-39` closed at evidence-binding boundary; `SEC-P1-38` remains open for
  actual external scanner/signature/Redis/logging production facts.

## Run 21 / B40 — Runtime isolation and secret boundary

Internal assistant lifecycle coordination moved off the public document event
surface; optional host-page integration became a v2 consent-gated bounded
projection. Opaque-origin Share compatibility was split into read and write
authority, strict deployment refuses opaque writes, runtime bearer entry became
site-owner opt-in/default Off, and standalone Share viewers gained frame and
Permissions-Policy denial. Proxy deployment version ratcheted to **7.0.0**.


## Run 22 / B41 — Separate-origin isolation and capability messaging

- Added opt-in fail-closed separate-origin assistant runtime with a tiny docs-origin host bridge.
- Added exact source/origin/version/channel bootstrap followed by a transferred MessageChannel with bounded monotonic capability envelopes.
- Added bounded page-context/canonical-read/print/UI/public-integration capabilities; no arbitrary parent DOM or transcript/secret read capability exists.
- Added parent-origin namespace for all isolated local/session storage.
- Preserved B40 consent semantics by revalidating bounded public lifecycle projections at the host boundary.
- Added restrictive isolated bootstrap document, deployment header/CORS guidance and explicit compromised-parent/deployment residuals.
- Frozen working-tree acceptance: B41 **11/11 + 34/34**, Node **45/45**, mutation/privacy **212/212**, runnable **754 passed / 3 skipped**, two-root compile **68/68**, JS/TOML/release-subject/maintenance GREEN, Sphinx-inclusive **1220 / 3 / 5 / 62** missing-`sphinx`-only.

## Run 23 / B42 — Hostile-parent and egress boundary hardening

- Ratcheted isolation protocol to **2.0.0** and moved handshake-channel generation into the isolated frame using WebCrypto only.
- Added generated exact parent-origin policy with deny-all source default and closed schema.
- Removed popup sandbox escape and blocked frame-self HTTP(S) navigation that could otherwise collapse SOP by navigating the frame onto the docs origin.
- Centralized ambient credential policy: service fetches omit credentials by default; explicit compatibility reaches at most `same-origin`.
- Kept canonical docs reads as a distinct same-origin, redirect-blocked, streaming-bounded capability.
- Added independent default-Off cross-origin microphone delegation and parent-origin + docs-root storage partitioning.
- Corrected the Sphinx setup regression test to require the new isolation-policy build-finished hook rather than an obsolete two-hook count.
- Working-tree runnable boundary: **766 passed, 3 skipped**. Sphinx-inclusive: **1232 passed, 3 skipped, 5 failed, 62 errors**, with missing `sphinx` as the only non-green family.

## Run 24 / B43 — bounded remote response and context ingestion

- Added pre-buffer response ceilings to browser chat/control/canonical reads and removed unsafe whole-body compatibility fallbacks from security-sensitive readers.
- Added total-byte and unterminated-line ceilings for browser SSE consumption.
- Stream-bounded isolated parent-policy/canonical reads and standalone HF/Worker Global Share viewer JSON.
- Changed HF and dev proxy upstream chat handling to streamed collection under an 8 MiB default / 32 MiB hard ceiling; Worker gained equivalent declared/chunked enforcement without whole-body buffering.
- Added non-secret effective response-limit health diagnostics and operator documentation.
- Proxy deployment version ratcheted to **7.1.0**.
- Working-tree boundary: B43 **8/8 + 13/13**, mutation/privacy **236/236**, runnable **775 passed / 3 skipped**, Sphinx-inclusive **1241 / 3 / 5 / 62** missing-`sphinx`-only before packaging.


## Run 25 / B44 — Provider Response Boundaries & Semantic Context Integrity

- Proxy version ratcheted to **7.2.0**.
- Added bounded provider-control metadata reads and no-body mutation response handling.
- Added scoped bounded Hugging Face Hub client transport with factory restoration.
- Tightened GitLab custom API authority and rejected unsupported provider `api_base`.
- Moved same-origin and isolated context visibility authority to the live rendered DOM before serialization.

## Runs 26–34 — focused UI/configuration/diagnostic overlays

- B45/B46/B48 refined export-mode/contribution icons and removed the external sprite dependency.
- B47 refined the Contribute-to-dataset sheet for adaptive, human-readable inspection and input/JSON sizing.
- B49/B50 made request failures privacy-safe/actionable and aligned proxy/model failure attribution/configuration.
- B51/B52 made feedback and per-bubble More menus panel-body-aware with shared flip/clamp geometry.
- B53 aligned Hugging Face Space model allow-list defaults/diagnostics with the models exposed by the client.
- These overlays preserved the established privacy, contribution lifecycle, provider response-boundary and release-control invariants.

## Run 35 / B54 — provider-native contribution review

- Added optional `CONTRIBUTION_REVIEW_MODE=provider-pr`; compatibility default remains `ledger`.
- Added one provider-neutral review contract over Hugging Face PRs, GitHub PRs, GitLab MRs and Bitbucket PRs.
- Native review writes the future eligible bytes to an isolated provider review ref; the configured canonical branch is the only training-eligible authority.
- Opaque receipt-derived branch/title identifiers prevent contribution/user text from leaking into Git metadata.
- Manual provider-UI merge is observed and ratchets the local lifecycle to `eligible`; close/decline remains noneligible.
- Pending participant deletion closes native review before ledger clearing; post-merge withdrawal preserves the existing tombstone/current-view-removal contract.
- Only the Primary storage target owns approval; mirrors remain replication targets.
- Proxy deployment version ratcheted to **7.3.0**.
- Working-tree boundary: B54 **13/13**, focused **151 passed**, Node **50/50**, runnable non-Sphinx **824 passed, 3 skipped**, compile **72/72**, Sphinx-inclusive **1290 / 3 / 5 / 62** missing-`sphinx`-only.

## Run 40 / B59 — CORS default origins and Space configuration

- Added both current documentation origins as package defaults: Scikit-plots GitHub Pages and Scikit-plots Learn on Read the Docs.
- Added `ALLOWED_ORIGINS_MODE=additive|replace`; downstream/fork deployments can replace package defaults without editing proxy source.
- Kept wildcard CORS as an explicit insecure compatibility mode and preserved strict-mode rejection.
- Mirrored the origin-composition contract in the Cloudflare Worker.
- Preserved privacy-minimal health diagnostics while adding default-origin count/coverage facts.
- Expanded the HF Space README with a clean Variables-vs-Secrets guide, current Scikit-plots examples, custom-site recipes, and explicit classification of `TRAINING_DATASET_REPO` as non-secret configuration.

## Run 44 / B63 — contribution action-group UX

- Reused Endpoint Configuration I/O button primitives throughout the dataset contribution sheet.
- Grouped management controls into Payload, Private recovery, Maintainer support, Review lifecycle, and recovery-import surfaces instead of a flat undifferentiated button list.
- Added local Copy JSON and Download JSON actions alongside Inspect JSON; neither action submits content.
- Applied the existing danger-button treatment to delete/withdraw and added responsive stacking plus ARIA preview state.
- Retargeted the existing withdrawal mutation anchor to the shared action-button constructor without weakening the privacy/lifecycle mutation.
- Working-tree boundary: Node **51/51**, focused contribution/provider/privacy **54/54**, runnable non-Sphinx **849 passed, 3 skipped**.

## Run 45 / B64 — feedback review, training eligibility and quality signal

- Added the shared **Feedback | Dataset contribution | Activity** workspace while keeping local rating, anonymous telemetry, reviewed feedback, and dataset contribution as separate authorities.
- Added provider-native one-Q&A feedback review with stable PR/MR identity, unchanged no-op, revision updates, status, withdrawal, and direct provider-review lookup.
- Ratcheted feedback review consent to **2.0.0** and added an independently versioned training-consent marker; historical review-only browser consent fails closed.
- A maintainer merge now makes explicitly consented feedback Q&A records training-eligible; close/decline never does.
- Added server-derived `qualityScore` (`0..1`) and `qualityPercent` (`0..100`) while retaining the raw signed rating and scale bounds.
- Training builder now admits eligible `feedback` as well as eligible `contribution`, while privacy-minimal rating telemetry remains excluded.
- Proxy public API version ratcheted to **7.4.0**.
- Runnable non-Sphinx working-tree boundary: **864 passed, 3 skipped**.

## Run 46 / B65 — feedback payload inspection and model attribution

- Added Contribution-style Inspect/Copy/Download JSON controls to the Feedback tab; all are local-only and display the exact review payload.
- Made the assistant transcript turn the model-attribution authority so changing the currently selected model after generation cannot relabel an older answer.
- Feedback review now fails closed client-side and server-side if originating `provider` + concrete model name are unavailable.
- Included model identity in feedback-review no-op fingerprints.
- Replaced text-filled feedback popup icon slots with Octicon `comment-discussion` and `pulse` SVGs.
- Corrected `ratingTitle` validation to measure the title rather than the rating label.
- Runnable non-Sphinx boundary: **869 passed, 3 skipped**.

## 2026-09-06 — Run 172 maintenance + skill normalization

- Established Run 172 as the immutable release anchor and the user-supplied
  `run172(2)` archive as a separate local-debugging workspace anchor.
- Moved the maintenance corpus out of `skills/` into the repository-level
  `maintenances/.../_sphinx_ai_assistant` mirror described by the existing
  architecture documents.
- Added a focused `SKILL.md` that makes fresh-chat continuation independent of
  conversation history and defaults the next phase to local test-first repair.
- Moved maintenance-only `check_trackers.py`, `dev_proxy.py`, and the old proxy
  configuration example into `_maintenance/tools/` / `_maintenance/examples/`.
- Moved feedback/dataset operator guides beside `_hf_spaces_proxy` and reduced
  the runtime submodule root to three Markdown guides.
- Archived superseded fresh-chat handoffs under `_maintenance/history/fresh_chat/`.
- Removed generated cache/bytecode/backup debris.
- Relocation-focused regression set: 70/70 passed after correcting one stale
  test that still imported `dev_proxy` as a runtime package module.
- Maintenance drift checker: GREEN in repository mode.


## 2026-09-11 — R173T88 mobile model action menu trigger anchor

Closed a mobile placement defect where model actions were positioned from the
variable-height model row instead of the `⋮` disclosure trigger. Added a local
action host, below/above edge-aware placement, CSS-target-aware responsive
harness plumbing, and three mutation controls. Verification: model responsive
38/38, override 125/125, remove/revert 23/23, quick model 91/91, Node/UI
161/161, mutation 473/473, maintenance core 35/35, family 2/2 GREEN.

## 2026-09-11 — R173T89 mobile speak-toggle resting visibility

Fixed the speak-hint disclosure chevron becoming visually lost on touch/mobile.
The base rule had a self-cancelling `background-color` / `background: transparent`
sequence and depended too much on desktop hover for contrast. The toggle now
keeps its intended resting surface, resets native appearance, binds the SVG
stroke to `currentColor`, and owns a stronger touch resting color with dark and
forced-colors fallbacks. Verification: focused 14/14, speak neighbor 37/37,
Node/UI 162/162, mutation 479/479.


## 2026-09-11 — R173T90 artifact Download mobile compaction

Split the old one-size-fits-all `22rem` artifact icon-only threshold. Per-file
Download controls now compact at `26rem` because they directly compete with long
filenames; bulk footer labels retain text until `22rem`. The named container
query remains the authority, so resized/docked panels still respond to actual
surface width rather than viewport width. Verification: artifact contract
198/198, Node/UI 160/160, mutation anchors/catalogue 243/243 and 240/240 mutants
caught.

## 2026-09-11 — R173T91 speak-toggle sticky-hover contrast

Closed the rare post-tap case where the speak disclosure remained clickable
but its SVG appeared transparent. Touch browsers may retain `:hover`; the
higher-specificity interaction rule used `color: inherit`, overriding T89's
explicit coarse-pointer resting colour. Hover/focus/active now retain an
explicit base-text foreground, the controls use a real surface token, and the
speak icon no longer assumes a host accent always contrasts. Verification:
T91 19/19, T89 neighbor 14/14, speak lifecycle 37/37, Node/UI 161/161,
mutation structure 246/246 and 243/243 mutants caught.

## 2026-09-11 — R173T93 inline snippet scroll handoff

Closed the rare case where numbered code embedded in a normal answer could
capture vertical mouse/touch scrolling. R173T69 had fixed the inner `<pre>`, but
snippet wrappers still inherited file-preview vertical ownership and overscroll
containment. Real file sheets now keep explicit axis ownership, while inline
snippets are non-scrolling wrappers and only their code cell owns horizontal
pan. No JavaScript wheel interception was added. Verification: T93 20/20,
numbered-preview neighbor 200/200, canonical static Node 154/154, mutation
structure 254/254 and 251/251 mutants caught.

## 2026-09-11 — R173T94 Presented-file segmented-control parity

Closed a real Presented-files composition drift where runtime DOM already used
the shared artifact segment builder but old preview/download/save-as/patch grid
rules remained in the cascade. Presented-file Download now reuses the normal
`ai-md-artifact-download-label` visual class, the shared group owns segment
geometry, and the Presented row has one two-column authority: artifact group +
overflow. Dead pre-overflow CSS and contradictory regression expectations were
removed. Verification: T94 18/18, latest-preview 202/202, diff 35/35,
working-file 143/143, raw-body 21/21, Node/UI 164/164, mutation structure
259/259 and 256/256 mutants caught.


## 2026-09-11 — R173T95 Presented-file responsive segment continuity

Closed the device-mode continuation of T94 at 560px and small-phone widths. A
legacy `@media (max-width:560px)` declaration still set the in-group Download
segment to `width:100%`, collapsing the flexible Preview segment and making the
divider/content appear reordered. Removed that obsolete authority, made the
Presented list/card/primary grid explicitly shrinkable, and moved heading
stacking to the component's named container. Verification: T95 21/21, T94
neighbor 18/18, latest-preview 202/202, diff 35/35, working-file 143/143,
Node/UI 165/165, mutation anchors 260/260 and 260/260 mutants caught.

## 2026-09-11 — R173T97 snippet / Presented-file menu workflow parity

Unified the two file overflow workflows without lying about capabilities.
Anonymous snippets now follow inspect → save → track → continue, then graduate
to the canonical tracked-file action list after promotion; Presented files keep
inspect → save → patch → continue. Promotion now returns the registered ledger
entry so Continue editing can stage the exact tracked revision. Verification:
T97 16/16, latest-preview 203/203, working-file 144/144, Node/UI 169/169,
mutation anchors 272/272 and 269/269 mutants caught.
