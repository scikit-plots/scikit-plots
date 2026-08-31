# `_sphinx_ai_assistant` physical tracker

## Baseline inventory — 2026-08-28 v6.4 anchor

| Path | Lines | Role | Ratchet |
|---|---:|---|---|
| `__init__.py` | 7,383 | Sphinx config/build injection + Markdown/llms generation | no unrelated responsibility without review |
| `_static/ai-assistant.js` | 30,307 | browser runtime | no new major subsystem without contract/test review |
| `_static/ai-assistant.css` | 17,067 | UI styling | component families need responsive/accessibility gates |
| `_hf_spaces_model/app.py` | 2,446 | model service | security boundary |
| `_hf_spaces_proxy/app.py` | 2,444 | routing/API/persistence/stream bridge | security + reliability boundary |
| `_hf_spaces_proxy/_utils/_shared_logic.py` | 1,469 | shared proxy/service logic | security boundary |
| `_hf_spaces_proxy/_utils/_dataset_schema.py` | 1,206 | feedback/training schema | provenance boundary |
| `_hf_spaces_proxy/_utils/_storage.py` | 608 | provider-neutral record storage | credentials + persistence boundary |
| `_cf_worker/index.js` | 499 | edge relay | must match routing/security policy |
| `dev_proxy.py` | 511 | local dev relay | must not become production authority |
| `tests/test___init__.py` | 3,671 | Sphinx extension integration tests | preserve during extraction |

Current Sphinx extension registers **106** `add_config_value` calls at the anchor.

## Maintenance separation ratchet

The repository-level maintenance tree is:

```text
maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/
```

The runtime module must not regain source-local `tasks/`, `_maintenance/`,
`MAINTAINING.md`, or backup trees. Planning/history belongs under the mirrored
maintenance module.

## Forbidden physical dependencies

- production runtime must not import from `maintenances/`;
- production runtime must not import/reference maintenance `_backup/` as live source;
- `_sphinx_llm` must not import `_sphinx_ai_assistant`, and the assistant must not
  depend on `_sphinx_llm`;
- browser bundles must not contain configured production secret values;
- service credentials must remain server-side;
- maintenance tools may inspect runtime source but runtime source must never need
  maintenance files to execute.

## Streaming ratchet

The v6.4 proxy state machine is a physical reliability boundary:

- open upstream before downstream success is committed;
- preserve JSON as JSON;
- preserve true SSE as SSE;
- fail pre-header protocol errors with real HTTP errors;
- make post-start stream failure explicit;
- reserve `stub/*` locally.

Regression owner: `tests/test_proxy_streaming_state.py`. Operational owner:
`_maintenance/APP_STREAMING_RUNBOOK.md`.

## Monolith rule

Existing file size is a baseline, not a target. Growth that introduces a new
responsibility family triggers decomposition review. Refactoring must preserve
behavior/security tests first; line-count reduction alone is not success.


## B16 Share conversation physical ratchet

The historical baseline inventory above remains an anchor, not a current line-count
claim. At B16 closure the browser files are 30,573 JS lines and 17,130 CSS lines.
The Share refactor does not claim monolith decomposition; it removes one repeated
responsibility pattern inside the existing runtime.

Physical invariants added by B16:

- one outer `ai-assistant-panel-conv-share-sheet`;
- no format-specific JSON/HTML/TXT Share sheet siblings;
- one `_sheetRegistry` owns cross-sheet traversal concerns;
- no duplicate export-mode DOM id used as synchronization authority;
- format panels are lazy-created from `_EXPORT_FORMATS` and remain format-bound;
- runtime still has no dependency on `maintenances/`.

Regression owners: `tests/test_share_conversation.mjs`,
`tests/test_share_conversation_dom.mjs`, and the Share mutants in
`tests/_mutants.py`.

## Run 4 prompt-authority physical ratchet

Run 4 introduces one explicit public chat envelope across bundled inference
services rather than allowing each service to interpret arbitrary provider
request bodies.

Physical invariants:

- `_hf_spaces_proxy/_utils/_chat_contract.py` and `_hf_spaces_model/_chat_contract.py`
  are deployment copies of one contract and must remain byte-identical;
- the browser sends `scikitplot-chat-v1` only after `/health` advertises it;
- HF proxy and Cloudflare construct provider system authority server-side;
- Path 2 forwards the structured envelope, not proxy-authored `messages`, to the
  direct model service;
- `_hf_spaces_model/app.py` parses the typed contract and reconstructs policy
  before inference;
- Path 1/2/3 credential variables are physically separate; `HF_TOKEN` cannot be
  reused by Path 1/2 routing helpers;
- credential-bearing clients use no automatic redirect following.

Regression owners: `tests/test_chat_authority.py`,
`tests/test_chat_authority.mjs`, `tests/test_model_service_authority.py`, and
the Run 4 client authority mutants.



## Run 5 logging/privacy physical ratchet

Run 5 adds an explicit application telemetry boundary without turning logs into a
parallel user-data store.

Physical invariants:

- `_hf_spaces_proxy/_utils/_telemetry.py` and `_hf_spaces_model/_telemetry.py` are
  deployment copies of one privacy boundary and remain byte-identical;
- proxy/model ordinary messages and exception summaries pass through that
  boundary before structured emission;
- model service contains no raw `traceback.format_exc()` response/log path;
- bundled HF Uvicorn access logging is disabled;
- Worker `_log(...)` sanitizes/drops sensitive fields and no callsite re-adds
  session/feedback/share identifiers or raw exception messages;
- `dev_proxy.py` does not log partial tokens or exact upstream URLs;
- public proxy/model/dev discovery exposes coarse client-needed state, not
  backend/storage/credential topology;
- Docker deployment copies `_telemetry.py` so packaged behavior matches source.

Regression owners: `tests/test_logging_privacy.py`,
`tests/test_logging_privacy_mutations.py`, `tests/test_discovery_contract.py`,
and the broad non-Sphinx suite.


## Run 6 feedback/contribution physical ratchet

Physical invariants added by B22:

- browser feedback network payload is a dedicated telemetry builder and does not serialize transcript/note/model/page/session content;
- `_dataset_schema.py` owns lifecycle states including `telemetry`, `quarantined`, `eligible`, `withdrawn`, and `legacy_unreviewed`;
- `_contribution_ledger.py` owns bounded receipt state with memory compatibility and local transactional SQLite backends;
- `/v1/contribute` writes first to the mutable receipt ledger, not `_persist_record(...)`;
- user delete/withdraw authority and independent review promotion use separate capabilities; promotion/withdrawal lifecycle claims are atomic;
- only the review route may construct durable `trainingStatus=eligible` rows;
- `deduplicate_dataset.py` defaults to eligible contribution rows only;
- Cloudflare feedback stores the same minimal telemetry class but does not yet own contribution review.

Regression owners: `tests/test_feedback_contribution_privacy.py`, `tests/test_feedback_contribution_privacy.mjs`, `tests/test_deduplicate_multisource.py`, and the Run 6 mutants.


## B23 privacy preflight physical ratchet

Run 7 adds one browser-side privacy-preflight responsibility inside the existing runtime rather than format/path-specific scanners. Physical owners:

- `_static/ai-assistant.js`: `_privacyPreflightScan`, `_privacyRedactValue`, `_privacyPreparePageContext`, `_privacyPreflightReview` and outward-operation integration;
- `_static/ai-assistant.css`: accessible privacy warning/dialog presentation;
- `tests/test_privacy_preflight.mjs`: source/integration/value-non-retention contract;
- `tests/test_privacy_preflight_dom.mjs`: executed dialog/redaction contract;
- `tests/test_share_conversation_dom.mjs`: async conversation-identity race gate.

Ratchets:

- no separate inference/Share/contribution secret scanners with divergent semantics;
- finding state may not retain matched source bytes;
- preflight must run before transcript/network/link mutation for flagged egress;
- explicit redaction operates on an operation copy;
- direct-service policy remains server-owned and independent of this client warning layer.


## B24 Share formats/artifact lifecycle physical ratchet

Run 8 physical invariants:

- `_EXPORT_FORMATS` has exactly five live entries in canonical order: JSON, HTML, Text, YAML, TOML;
- `_EXPORT_STUB_FORMATS` is empty for these formats;
- one `ai-assistant-panel-conv-share-sheet` owns destination/content/result state;
- format descriptor switching replaces one host child and cannot duplicate destination sections;
- Local preview removal calls `URL.revokeObjectURL`;
- Global managed artifacts retain the private edit capability only in page memory and Revoke calls server DELETE;
- sessionStorage may persist public Global recovery state but not edit capability;
- New chat clears active Share result but retains page-memory managed artifacts for old-link revocation;
- self-contained artifacts use truthful local-only lifecycle copy; Share-sheet and direct toolbar downloads enter the shared page-memory registry and use truthful device-file lifecycle copy;
- HF + Worker Share format allowlists include YAML/TOML and remain server-owned.

Regression owners: `test_run8_serializers.mjs`, `test_run8_client_roundtrip.py`, `test_run8_share_formats.py`, `test_share_conversation.mjs`, `test_share_conversation_dom.mjs`, `test_global_share_capability.mjs`, and Run 8 mutation positives.

## Run 9 — Global artifact lifecycle tracking

Modified runtime owners:

- `_static/ai-assistant.js` — bounded session public Global ledger, restore/status/revoke lifecycle state;
- `_hf_spaces_proxy/app.py` — content-free `HEAD /v1/share/{id}` status route;
- `_cf_worker/index.js` — matching HEAD route + HEAD CORS allowance;
- `_hf_spaces_proxy/README.md` — endpoint contract documentation;
- tests — Global capability/DOM/server/mutation lifecycle gates.

No edit capability or conversation snapshot is added to persistent browser storage.

## Run 10 — Global lifecycle fail-closed recovery

Modified runtime owners:

- `_static/ai-assistant.js` — allowlisted/destructive session recovery migration; non-terminal/recheckable unavailable state; current PATCH-state detach; 410 fresh-create fallback;
- `_hf_spaces_proxy/app.py` — expired DELETE returns 410 before mutation authorization semantics can imply successful revoke;
- `_cf_worker/index.js` — explicit expiry checks for GET/PATCH/DELETE, matching existing HEAD semantics;
- `_hf_spaces_proxy/README.md` — documents five Share formats and explicit expired-route 410 parity;
- `tests/test_global_share_capability.mjs` — storage trust/410/server-parity source contract;
- `tests/test_share_conversation_dom.mjs` — 404 remains recheckable/revocable/locally-forgettable in live page memory and can recover to 200; legacy/tampered storage cannot restore Revoke/PATCH authority;
- `tests/test_share_server_authority.py` — expired DELETE clears entry and returns 410;
- `tests/_mutants.py` — Run 10 positive controls.

No new credential persistence is introduced. `404` does not erase its bounded public locator solely from reason-unknown absence; `410`/successful DELETE remain terminal.



## Run 11 — B05/B06 physical ratchet

Modified runtime/config owners:

- `_hf_spaces_proxy/app.py` — exact-origin default + early Origin guard; shared streamed body gate; common trusted client identity; chat/share/feedback/contribution rate gates; hard-bounded identity maps.
- `_hf_spaces_proxy/_utils/_shared_logic.py` — discovery default matches runtime origin policy.
- `_hf_spaces_proxy/README.md` — exact CORS, hard body ceiling, rate/identity operator contract.
- `_hf_spaces_model/app.py` — hard-clamped streamed body gate and early browser-Origin denial.
- `_hf_spaces_model/README.md` — streamed/hard body ceiling documentation.
- `dev_proxy.py` — loopback exact-origin default and declared-size pre-read enforcement.
- `_cf_worker/index.js` — exact-origin guard, streaming body reader, chat rate gate, aligned share/feedback defaults, unique TTL KV event limiter.
- `_cf_worker/wrangler.toml` — bundled `main=index.js`, current compatibility date, security vars and soft-limiter semantics.
- `tests/test_b05_b06_release_security.py` — B05/B06 cross-runtime regression owner.
- `tests/test_chat_authority.mjs` — Worker fake KV now supports event-list limiter without masking prompt-authority defects.

Physical ratchets: no `await request.body()` in bundled HF body gates; no Worker `request.text()` body buffering; no wildcard bundled production default; no unbounded HF rate identity table; no same-key Worker KV counter loop; bundled Wrangler entrypoint must exist.


## B28 contribution lifecycle physical ratchet

Run 12 adds one explicit receipt control-plane boundary. `_contribution_ledger.py` owns lifecycle mutation; `app.py` owns authenticated routes and compensating workflow; `_storage.py` owns provider current-view writes/removals and stale retry suppression; `_dataset_schema.py` owns privacy-minimal withdrawal tombstones; `deduplicate_dataset.py` owns last-write-wins training suppression. SQLite is a local transactional backend only and must not be presented as replica-safe shared authority.


## B29 Share fixed-path/fragment physical ratchet

Run 13 physical owners:

- `_static/ai-assistant.js` — exact fragment/legacy locator parser, fixed `/status` `/update` `/revoke` transport, fragment-aware bounded lifecycle ledger;
- `_hf_spaces_proxy/_utils/_share_contract.py` — static fixed viewer shell and exact dual backend locator validation;
- `_hf_spaces_proxy/app.py` — fixed viewer/read/status/update/revoke routes plus fragment URLs while retaining explicit legacy route compatibility;
- `_cf_worker/index.js` — matching fixed transport, dual locator validation, fixed-update abuse gate, fragment URLs;
- `_cf_worker/wrangler.toml` — automatic invocation logs disabled as URL-log defense in depth;
- `_hf_spaces_proxy/README.md` — current-vs-legacy endpoint contract;
- `tests/test_run13_share_capability_transport.py`, `test_share_fixed_transport.mjs`, existing Share harnesses, and Run 13 mutants — regression owners.

Physical ratchet: current generated public URLs contain no locator in the path; current browser operation paths are constant; edit capabilities remain memory-only; legacy `/v1/share/{id}` routes may not silently become the generator path again.


## Run 14 physical ratchet — legacy Share drain

- `_hf_spaces_proxy/app.py` — current create/fixed update stamp `transport_version=2`; legacy HEAD/GET/DELETE require pre-generation state; legacy PATCH returns 410; fixed update is migration authority.
- `_cf_worker/index.js` — matching `transportVersion=2`, legacy generation gate, standards-based deprecation/sunset signaling, retired PATCH and fixed-update migration.
- `_hf_spaces_proxy/README.md` / `_cf_worker/wrangler.toml` — deployment contract distinguishes current fixed paths from bounded pre-generation compatibility.
- `tests/test_run14_legacy_share_retirement.py` + `test_share_legacy_retirement.mjs` — runtime regression owners.

Physical ratchet: the set of objects eligible for a capability-bearing request path can only shrink. New/fixed-updated objects may not re-enter it.


## Run 15 — distributed rate-limit authority physical ratchet

Physical owners added/changed by B31:

- `_hf_spaces_proxy/_utils/_rate_limit.py`: Redis atomic fixed-window backend, HMAC identity keys, bounded manifest/error surface, lazy async client initialization.
- `_hf_spaces_proxy/app.py`: backend selection, fail-closed shared-required policy, explicit route-family scopes, safe health/discovery state, lifecycle initialization/close.
- `_hf_spaces_proxy/requirements.txt` + README: fresh-deploy Redis dependency and explicit horizontal-deployment activation contract.
- `_cf_worker/index.js`: HMAC-derived per-scope identity shard, `RateLimitBucket` Durable Object, fail-closed required-authority branch, truthful KV fallback metadata.
- `_cf_worker/wrangler.toml`: `RATE_LIMIT_DO` binding, SQLite-backed exported class, `RATE_LIMIT_REQUIRE_AUTHORITATIVE=true`, dedicated HMAC secret contract.
- `tests/_import_cf_worker_for_node.mjs`: Node-only shim for Cloudflare's `cloudflare:workers` import so runtime harnesses can execute the shipped Worker logic.
- `tests/test_run15_distributed_rate_authority.py` + `test_run15_worker_rate_authority.mjs`: shared-domain, privacy, cross-PoP, fail-closed and positive-control gates.

Ratchets:

- required authoritative mode never falls back silently to local/KV;
- Worker authority is sharded by route family + HMAC identity, not one global singleton;
- raw IP-like identity never becomes a Redis key or Durable Object name;
- local/KV fallback remains compatibility behavior and advertises `authoritative:false`;
- no built-in local/permissive edge binding is relabeled globally atomic;
- health/discovery cannot reveal Redis URL, HMAC secret, raw identity, or internal shard name.


## Run 16 — shared contribution receipt authority physical ratchet

Physical owners added/changed by B32:

- `_hf_spaces_proxy/_utils/_contribution_ledger.py`: shared Redis receipt backend, HMAC receipt keyspace, atomic create/capacity/lifecycle scripts, claim digests/leases, reconciliation-required promotion uncertainty and monotonic withdrawal.
- `_hf_spaces_proxy/app.py`: shared-required configuration/readiness gate, claim-aware promotion/withdrawal, uncertainty status and fail-safe transport ambiguity handling.
- `_hf_spaces_proxy/_utils/_storage.py`: provider transport/timeout normalization into ambiguous transient mutation outcomes.
- `_hf_spaces_proxy/README.md`, `DATASET_COLLECTION_GUIDANCE.md`, requirements/Docker comments: shared authority activation, durability split and uncertainty semantics.
- `tests/test_run16_shared_contribution_authority.py`: two-replica authority, privacy, uncertainty, direct-ledger parity and positive-control gates.

Ratchets:

- shared-required contribution intake never silently falls back to process/local authority;
- raw receipt capabilities never become Redis key/index identifiers;
- an expired promotion lease cannot be reissued after possible external side effect;
- provider mutation ambiguity never reopens ordinary promotion eligibility;
- participant withdrawal remains available from promotion uncertainty and recovery is monotonic;
- shared Redis coordination is never mislabeled external persistence durability.

## Run 17 — B36 dataset contribution UX / conversation-record physical ratchet

Physical owners:

- `_static/ai-assistant.js` — canonical contribution payload builders; dedicated **Contribute to dataset** sheet; main/menu/per-answer entry points; exact JSON inspection; privacy/consent submission; receipt management; Share contains no contribution controller.
- `_static/ai-assistant.css` — responsive contribution scope cards, inspect/consent/result surfaces.
- `_hf_spaces_proxy/_utils/_dataset_schema.py` — schema v4, `recordType`, ordered bounded `messages[]`, per-message model/feedback normalization, receipt-scoped Q&A/conversation dedup keys.
- `_hf_spaces_proxy/app.py` — v4 consent 2.0.0 enforcement with v2/v3 consent-1.0 compatibility and fail-closed rejection of empty conversation records.
- `_hf_spaces_proxy/README.md` + `DATASET_COLLECTION_GUIDANCE.md` — current schema, purpose separation, record families and lifecycle operator contract.
- `tests/test_dataset_contribution_ux.mjs` + `test_dataset_contribution_dom.mjs` — source and real mini-DOM workflow ownership.
- `tests/test_run17_dataset_contribution_ux.py` — server/schema compatibility and one-record quarantine/delete contract.
- `tests/_mutants.py` — positive controls for consent downgrade, identity re-linkage, privacy bypass, Share recoupling, error-row inclusion, conversation splitting, and lifecycle overclaim.
- `tests/test_feedback_telemetry_consent.mjs` + `test_feedback_contribution_privacy.mjs` — executable browser consent migration/default-off/network-stop/public-event privacy contract.
- `_cf_worker/index.js` + `_hf_spaces_proxy/app.py` — reject feedback telemetry lacking the current schema/consent marker/version/timestamp; operator persistence never substitutes for reader permission.

Physical ratchets:

- no contribution controller/action under `_buildConversationShareSheet`;
- no Q&A/content fields in ordinary feedback telemetry;
- one whole-conversation selection produces one conversation record;
- error rows remain excluded while preserving answer-index alignment for feedback;
- exact reviewed JSON is the submitted JSON;
- no content-bearing quick action bypasses consent/quarantine lifecycle.
- no legacy boolean telemetry preference re-enables network collection;
- no feedback rating/retraction helper transmits after telemetry opt-out;
- no public feedback DOM event carries Q&A/note/model/page/conversation content.


## Run 19 — supply-chain/deployment physical ratchet

Physical owners added/changed by B38:

- `_hf_spaces_proxy/requirements.txt` — five exact minimal direct runtime dependencies;
- `_hf_spaces_proxy/requirements.lock` — 30 exact hash-locked Linux/amd64 wheel components;
- `_hf_spaces_proxy/Dockerfile` — immutable digest, explicit Linux/amd64 stages, builder-only install path, non-root runtime and strict deployment profile;
- `_hf_spaces_proxy/.dockerignore` — deny-by-default build context;
- `_hf_spaces_proxy/docker-compose.hardened.reference.yml` — read-only/rootless/no-new-privileges/capability-dropped reference;
- `_hf_spaces_proxy/_utils/_redis_security.py` — one TLS transport policy for all Redis authorities;
- `_hf_spaces_proxy/security/supply_chain_policy.toml` — machine-readable release ratchets;
- `_hf_spaces_proxy/security/python-runtime.cdx.json` — scoped CycloneDX Python lock SBOM;
- `_hf_spaces_proxy/security/verify_supply_chain.py` — offline lock/Docker/SBOM policy verifier;
- `_hf_spaces_proxy/security/SECURITY_RELEASE_GATES.md` — networked dependency/image/SBOM/provenance release evidence contract;
- `tests/test_run19_supply_chain_deployment_hardening.py` — executable B38 regression owner.

Physical ratchets: image digest and target architecture move together; runtime
root is incompatible with strict mode; Redis TLS cannot be independently
downgraded per subsystem; checked-in Python SBOM cannot masquerade as a full
image SBOM; and a stale vulnerability scan cannot satisfy a later release.

## Run 20 release-evidence physical ratchet

The B39 release-control plane stays under `_hf_spaces_proxy/security/` and is
excluded from the deny-by-default Docker runtime context. Runtime application
code does not import maintenance files or production evidence artifacts.

Physical invariants:

- `release_evidence_policy.toml` owns promotion-evidence policy;
- `verify_release_evidence.py` is stdlib-only and performs no network access;
- `verify_release_gate.py` combines B38 source and B39 external-evidence binding;
- `release_subjects.py` emits only canonical non-secret source subjects;
- `probe_redis_authority.py` accepts an environment-variable **name**, never a
  Redis URL CLI argument, and emits no authority/topology values;
- `release-evidence.example.json` is intentionally non-authoritative and cannot
  pass unchanged;
- evidence files are not copied into the application image;
- `tests/test_run20_release_evidence_production_guardrails.py` owns the B39
  substitution/staleness/privacy negative gates.

## Run 21 / B40 runtime-isolation ownership

- `_static/ai-assistant.js`: private lifecycle bus, consent-gated public
  projections, page-integration v2, attachment gate, runtime-token enforcement.
- `__init__.py` + `_example_conf.py`: `ai_assistant_allow_runtime_tokens=False`
  site-owner policy and privacy-safe integration guidance.
- `_hf_spaces_proxy/app.py`: opaque-origin read/write classifier, strict write
  refusal, viewer frame/permission policy.
- `_cf_worker/index.js` + `wrangler.toml`: equivalent opaque-origin split and
  viewer hardening, both flags default Off.
- `tests/test_run21_runtime_isolation_secret_boundary.*`: executable B40
  negative/positive controls.


## Run 22 / B41 separate-origin isolation ownership

- `_static/ai-assistant-isolation-host.js` — documentation-origin capability adapter; exact-origin handshake, MessageChannel transfer, bounded page/canonical adapters, UI resize and revalidated public-integration projection.
- `_static/ai-assistant-isolated.html` + `_static/ai-assistant-isolated.css` — isolated-origin bootstrap document with no inline script/style and restrictive baseline meta CSP.
- `_static/ai-assistant-isolated-frame.js` — frame-side exact-parent handshake, replay-checked request/response channel, config secret scrub, parent-origin storage namespace and delayed load of the full assistant runtime.
- `_static/ai-assistant.js` — fail-closed parent self-suppression, scoped local/session storage, isolated page-environment abstraction, context/canonical/print capability use, and B40 public projection forwarding.
- `__init__.py` + `_example_conf.py` + `ISOLATION_DEPLOYMENT.md` — exact isolation-origin/path validation, configuration, asset registration and deployment guidance.
- `tests/test_run22_separate_origin_isolation.py` + `.mjs` — B41 negative/positive controls.

Physical ratchets: no same-origin fallback after isolation request; no arbitrary parent-DOM RPC; no runtime window-message bus after port transfer; no unscoped isolated storage; no parent query/fragment crossing; no page-integration projection without its independent consent and host-side schema validation.

## Run 35 / B54 provider-native review physical ownership

- `_hf_spaces_proxy/_utils/_storage.py` — provider-neutral `ReviewReceipt` and native review adapters for Hugging Face, GitHub, GitLab and Bitbucket; bounded provider metadata/no-body mutations remain under B44.
- `_hf_spaces_proxy/app.py` — `CONTRIBUTION_REVIEW_MODE`, native-review intake/recovery, manual-merge synchronization, participant close/delete and API-promotion compatibility.
- `_static/ai-assistant.js` — human-facing **IN REVIEW / APPROVED / NOT ACCEPTED** lifecycle copy without exposing maintainer review URLs.
- `_hf_spaces_proxy/README.md` + `DATASET_COLLECTION_GUIDANCE.md` — operator configuration, provider mapping, permissions and canonical-branch eligibility contract.
- `tests/test_run35_provider_review_workflow.py` — provider adapter + lifecycle semantic owner.
- `tests/test_dataset_contribution_ux.mjs` — browser source-contract owner for provider-native review states.

Physical ratchets: one Primary review authority; opaque review refs/titles; no content in review metadata; merge is the only provider-native path to canonical eligibility; pending participant delete closes native review before ledger deletion; provider bodies/tokens remain non-public; mirrors cannot create competing review decisions.
