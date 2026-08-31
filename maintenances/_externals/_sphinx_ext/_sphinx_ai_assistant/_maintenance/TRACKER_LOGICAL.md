# `_sphinx_ai_assistant` Logical Contract Tracker

The initial statuses are deliberately conservative. `VIOLATED_OR_UNPROVED` means existing review evidence suggests a gap or the desired invariant cannot yet be proved from the supplied source without a dedicated checkpoint. It prevents aspirational documentation from becoming false source truth.

| ID | Contract | Owner | Status | Invariant | Proof |
|---|---|---|---|---|---|
| `AIA-C01` | `BuildConfiguration` | `__init__.py` | **PARTIAL** | Sphinx config is validated, serializes only intended client-safe fields, and has explicit ownership. | config inventory + serialization tests |
| `AIA-C02` | `ClientSafeSerialization` | `__init__.py + browser export/share serializers` | **PARTIAL** | Generated client/export serialization excludes production secrets and never interprets untrusted conversation bytes as executable content; persistent browser storage excludes credentials. | Runs 1–2: secret stripping/storage migration + canonical sanitized snapshot + HTML raw-text isolation; Run 16.2 current self-contained output is reviewed inert base64 data HTML while c2 remains compatibility-only; server/log/prompt gates remain |
| `AIA-C03` | `BrowserRuntime` | `_static/ai-assistant.js` | **PARTIAL** | Browser owns presentation/state only; security decisions remain server-enforced. | browser/service bypass tests |
| `AIA-C04` | `RepresentationProducer` | assistant build layer (`generate_markdown_files`, `generate_llms_txt`) | **ACTIVE** | The assistant emits the canonical static representation at `build-finished`; the browser consumes it for VIEW and ASK AI. | producer contract exit criteria; COPY(static) and ASK AI resolve identically |
| `AIA-C05` | `PromptAuthority` | `model/proxy service` | **HOLDS** | Server owns system policy; client/page content cannot set authoritative system role. | Run 4 typed-contract tests across HF proxy, Worker, dev proxy, and direct model service |
| `AIA-C06` | `EndpointDiscovery` | `proxy GET / + browser client` | **HOLDS** | Public discovery is schema-versioned, client/server drift-tested, and privacy-minimized to coarse capability/readiness signals rather than backend/storage/credential topology. | Run 5 discovery contract + browser/storage compatibility tests |
| `AIA-C07` | `ProxyRouting` | `_hf_spaces_proxy` | **HOLDS** | Server credentials are destination-bound to approved upstreams. | Run 4 Path 1/2/3 credential matrix + destination/redirect tests |
| `AIA-C08` | `CorsOriginPolicy` | `proxy + model + worker + dev` | **HOLDS** | Production browser origins are exact/least-privilege by default and explicit disallowed Origin is rejected before expensive/write handlers; Origin is never authentication. | B27 origin matrix |
| `AIA-C09` | `ShareAuthorization` | `proxy persistence/routes` | **HOLDS** | Public read capability does not imply edit/delete authority; HF + Cloudflare require a distinct edit token and store only its digest. | B19 server + client capability tests |
| `AIA-C10` | `ClientIdentity` | `proxy/worker` | **HOLDS** | HF rate-limited routes use direct peer by default and XFF only after explicit trusted-ingress declaration; Worker direct-edge abuse identity comes from `CF-Connecting-IP`. IP remains non-authenticated identity. | B27 cross-route spoof/identity test |
| `AIA-C11` | `RequestLimits` | `proxy/model/worker/dev` | **HOLDS** | Bundled request paths apply hard byte ceilings before or while application buffering; configurable chat ceilings cannot exceed 16 MiB. | B27 declared/chunked oversize gates |
| `AIA-C12` | `ModelServicePolicy` | `_hf_spaces_model` | **HOLDS** | Direct model endpoint enforces role/policy invariants independently of UI. | `test_model_service_authority.py` + contract-copy parity |
| `AIA-C13` | `FeedbackProvenance` | `proxy dataset logic` | **HOLDS** | Rating telemetry is content-minimal/training-ineligible; explicit contributions are version-consented, quarantined, evidence-labeled and require independent review before eligibility. | B22 Run 6 privacy/provenance/quarantine/dataset tests |
| `AIA-C14` | `SettingsRegistry` | `browser settings` | **PARTIAL** | Client settings have one validated schema/write path with explicit persistence and sensitivity. | Node registry tests + integration guard |
| `AIA-C15` | `CorpusBoundary` | `cross-submodule` | **PLANNED** | Corpus owns retrieval/evidence semantics; assistant consumes published retrieval contracts. | import/boundary tests |
| `AIA-C16` | `MCPBoundary` | `cross-submodule` | **PLANNED** | MCP owns protocol transport; assistant does not invent parallel MCP semantics. | integration boundary tests |
| `AIA-C17` | `RuntimeHtmlFallback` | `browser/representation compatibility` | **CURRENT_LEGACY** | DOM conversion remains observable fallback during staged static migration, not canonical target. | fallback-selection tests |
| `AIA-C18` | `ServiceParity` | `proxy/model/worker/dev` | **PARTIAL** | Bundled paths now match prompt/model, logging/privacy, CORS and application buffering boundaries. Strict distributed rate accounting/private upstream deployment remains infrastructure-owned (`SEC-P0-31`). | Runs 4–5 + B27; deployment quota evidence still separate |
| `AIA-C19` | `ShareConversationUX` | `browser export/share state` | **HOLDS** | Live export formats share one shell and canonical serializer metadata; repeated delivery controls observe one state; async Share results remain bound to their initiating format and conversation UI identity. | B16 source contract + execution smoke + mutation gate |

## Status vocabulary

- `HOLDS` — current source and a regression gate prove it.
- `PARTIAL` — useful implementation exists but the entire invariant is not proved.
- `VIOLATED_OR_UNPROVED` — evidence indicates risk or source truth has not yet been adjudicated; do not claim safe.
- `CURRENT_LEGACY` — intentionally still live during migration, not desired end-state.
- `PLANNED` — new architecture not yet live.
- `DEFERRED` / `SUPERSEDED` — explicit lifecycle states.


### AIA-C20 — PrivacyDataLifecycle

**Status: PARTIAL.** Runs 5-7 plus Runs 12/16 now prove application-log minimization, minimal opt-in rating telemetry, explicit versioned contribution, atomic review provenance, optional local transactional/restart-durable receipt state, optional shared atomic Redis receipt authority when explicitly required, ambiguity-safe promotion state, active-ledger content removal, enforceable post-promotion training withdrawal, current-view cleanup, fail-closed training eligibility, and advisory local sensitive-egress preflight. External Redis persistence and provider-history/global erasure remain explicit residuals.


**Owner:** browser + proxy + persistence + contribution pipeline
**Status:** `PARTIAL`

**Invariant:** User conversation/personal data is minimized, purpose-bound, retention-bounded, non-identifying by default, and deletion/withdrawal semantics are truthful.

**Verification:** Run 5 proves application-log/diagnostic minimization; Run 6 proves feedback/contribution minimization/provenance; Run 7 proves local category/count-only preflight; Run 12 proves atomic receipt lifecycle, SQLite reconstruction, withdrawal tombstones, training exclusion and provider current-view cleanup; Run 16 proves shared Redis receipt coordination, fail-closed shared-required behavior and uncertainty-safe promotion/withdrawal. External Redis persistence and provider-history/global-erasure residuals remain.


### AIA-C21 — SensitiveEgressPreflight

**Owner:** browser privacy preflight in `_static/ai-assistant.js`
**Status:** `HOLDS`

**Invariant:** Before browser-controlled inference, Share, or contribution egress, the actual outbound data is locally reviewed for high-confidence secret/sensitive/invisible-control signals. Findings retain category/count/codepoint metadata only; redaction changes an operation copy; flagged data cannot silently fail open when review UI is unavailable.

**Verification:** B23 `test_privacy_preflight.mjs` + `test_privacy_preflight_dom.mjs` + Share delayed-dialog/new-conversation race + Run 7 mutation positives. This is advisory user protection and is not server authorization or proof that data is non-sensitive.


### AIA-C22 — ShareArtifactLifecycle

**Owner:** browser Share sheet + Global Share DELETE capability
**Status:** `HOLDS`

**Invariant:** Every assistant-managed Share artifact exposes a lifecycle action whose wording matches the underlying storage semantics: local Blob previews are actually revoked, Global links are remotely revoked only with the private edit capability, self-contained copies are never called remotely revocable, and Share-sheet/direct-toolbar downloaded device files are lifecycle-tracked but never claimed deleted by the page. New-chat state does not discard still-usable page-memory Global revoke capabilities.

**Verification:** B24 Share source/DOM/capability gates + browser/server YAML/TOML parser gates + Run 8 artifact-lifecycle positive-control mutations.

### AIA-C23 — GlobalShareLifecycleTracking

**Owner:** browser Share lifecycle + HF/Cloudflare Global Share status route
**Status:** `HOLDS`

**Invariant:** Every Global Share URL handed to the user enters a bounded session-scoped public lifecycle ledger. New chat/reload does not make previously provided links invisible, but private edit/revoke authority remains page-memory-only. Status checks are explicit and content-free; revoked/expired/unavailable states are labeled truthfully.

**Verification:** B25 `test_global_share_capability.mjs` + `test_share_conversation_dom.mjs` + `test_share_server_authority.py` + Run 9 mutation positives.

### AIA-C24 — GlobalShareRecoveryFailClosed

**Owner:** browser Global recovery + HF/Cloudflare lifecycle routes
**Status:** `HOLDS`

**Invariant:** Reload recovery never restores private mutation authority or conversation-derived fields from Web Storage; a reason-unknown 404 remains bounded/recheckable rather than falsely terminal; unavailable/expired objects are detached from implicit PATCH state; unavailable same-page objects retain separate Revoke/Forget semantics; confirmed expiry is represented consistently across status/read/update/delete routes.

**Verification:** B26 `test_global_share_capability.mjs` + `test_share_conversation_dom.mjs` legacy/tamper and 404→200 recovery execution + `test_share_server_authority.py` expired DELETE 410 + Run 10 positive-control mutants.


### AIA-C25 — DistributedAbuseLimitSemantics

**Owner:** deployment ingress + HF/Worker abuse gates
**Status:** `HOLDS`

**Invariant:** Local/process/KV counters are truthfully labeled soft abuse controls, while a limiter is labeled shared/authoritative only when one configured atomic decision plane owns the claimed consistency domain. No limiter is treated as authenticated human identity or billing accounting.

**Verification:** B27 bounds local/KV fallback semantics; B31 adds Redis atomic shared mode for HF and sharded Durable Object authority for the bundled Worker, with truthful health/discovery and fail-closed required-mode gates.


### AIA-C26 — ContributionReceiptLifecycle

**Owner:** proxy contribution ledger + provider storage + dataset builder + browser receipt UX
**Status:** `HOLDS` across the bundled local lifecycle boundary; shared authority is covered by `AIA-C30` when required

**Invariant:** Receipt mutation authority is stored only as a digest; raw review content is cleared from ledger state after promotion; promotion/withdrawal claims are atomic; optional SQLite state survives process reconstruction; authenticated post-promotion withdrawal creates privacy-minimal dedup tombstones that suppress ordinary training output; current-view removal and active-ledger deletion are never described as provider-history/forensic erasure.

**Verification:** B28 `test_contribution_lifecycle_control_plane.py`, `test_feedback_contribution_privacy.py/.mjs`, `test_storage_multisource.py`, `test_deduplicate_multisource.py`, and Run 12 mutation positives; B32 extends cross-replica receipt coordination under `AIA-C30`. Provider-complete physical erasure remains outside this HOLDS scope.


### AIA-C27 — ShareCapabilityTransport

**Owner:** browser Global Share transport + HF/Cloudflare fixed viewer/routes
**Status:** `HOLDS` for current generated links

**Invariant:** Newly generated Global Share URLs keep the public read locator in an exact browser fragment on the fixed viewer path. Current read/status/update/revoke requests use fixed HTTP paths and bounded body locators, while mutation still requires the separate memory-only edit capability. Session recovery accepts the fragment form fail-closed. Legacy capability-bearing path routes are compatibility-only; Run 14 additionally rejects current-generation objects on those paths and prevents legacy PATCH from extending their lifetime.

**Verification:** B29 `test_run13_share_capability_transport.py`, `test_share_fixed_transport.mjs`, Global Share source/DOM harnesses, and Run 13 positive-control mutations. Full-body/WAF telemetry and retirement of old path links remain outside this HOLDS scope.


### AIA-C28 — LegacyShareCompatibilityDrain

**Owner:** HF/Cloudflare Share storage generation + legacy compatibility routes
**Status:** `HOLDS` at bundled application boundary

**Invariant:** Current Share objects carry server-owned transport generation 2 and are ineligible for `/v1/share/{id}` compatibility. Only pre-generation objects may use bounded legacy HEAD/GET/authenticated DELETE; legacy PATCH is retired and cannot extend TTL. Fixed `/update` migrates old objects to generation 2, so the legacy population can only shrink through migration, revoke, or expiry.

**Verification:** B30 `test_run14_legacy_share_retirement.py`, `test_share_legacy_retirement.mjs`, fixed-path server authority and Global capability harnesses. Actual deletion of legacy route code remains a post-drain release action.


### AIA-C29 — DistributedRateLimitAuthority

**Owner:** HF proxy Redis authority + Cloudflare Worker Durable Object authority
**Status:** `HOLDS_WHEN_AUTHORITATIVE_MODE_REQUIRED`

**Invariant:** When a deployment requires distributed quota authority, every request in the claimed replica/PoP consistency domain consults one atomic per-identity/per-route-family decision plane, raw identity is pseudonymized before shared storage/routing, and backend failure cannot silently degrade to independent counters.

**Verification:** B31 `test_run15_distributed_rate_authority.py` + `test_run15_worker_rate_authority.mjs`; bundled Worker config requires Durable Object authority, while horizontally scaled HF deployments must activate Redis and `RATE_LIMIT_REQUIRE_SHARED=true`.


## AIA-C30 — SharedContributionReceiptAuthority

**Status:** `HOLDS_WHEN_SHARED_MODE_REQUIRED`

**Invariant:** when horizontal contribution authority is explicitly required, all participating replicas consult one atomic Redis receipt lifecycle domain; receipt IDs are HMAC-pseudonymized before shared storage; authority failure is fail-closed; and expired/ambiguous promotion ownership becomes reconciliation-required uncertainty rather than a reassignable promotion lease because external provider mutations are not fenced by Redis.

**Verification:** B32 `test_run16_shared_contribution_authority.py` cross-replica semantic client, Lua/source contract assertions, real FastAPI shared-required/ambiguity behavior, storage transport classification, SQLite compatibility/restart regression, and positive-control mutants. External Redis persistence policy and production activation remain deployment evidence rather than this contract's claim.

### AIA-C31 — DatasetContributionPurposeSeparation

**Owner:** browser feedback/contribution surfaces + HF contribution schema/lifecycle
**Status:** `HOLDS`

**Invariant:** Rating telemetry, Share/export, and content-bearing dataset contribution are distinct purposes and control planes. Every content contribution entry point converges on one exact-payload inspection/privacy/consent controller; schema-v4 whole-conversation intake remains one ordered user/assistant record and enters the existing quarantine/review/delete/withdraw lifecycle.

**Verification:** B36 `test_dataset_contribution_ux.mjs`, `test_dataset_contribution_dom.mjs`, `test_run17_dataset_contribution_ux.py`, existing contribution lifecycle/privacy suites, and Run-17 positive-control mutants.


### AIA-C32 — FeedbackTelemetryExplicitPermission

**Owner:** browser feedback telemetry controller + HF/Worker `/v1/feedback` gates
**Status:** `HOLDS`

**Invariant:** local rating interaction does not authorize network telemetry. Only a current explicit structured browser permission may enable the official telemetry helper, every official feedback request carries the current consent contract, HF/Worker reject missing/stale/malformed consent before persistence work, and turning telemetry Off stops all future feedback network requests including retractions. The public feedback DOM event remains content-free.

**Verification:** B36 `test_feedback_telemetry_consent.mjs`, `test_feedback_contribution_privacy.mjs`, `test_feedback_contribution_privacy.py`, Worker source contract assertions, and positive-control mutants for fail-open consent, helper-gate removal, public-event content exposure, and post-opt-out retraction.

## AIA-C33 — RecoverableCreateManagementCapability

**Status:** HOLDS

Share and contribution CREATE operations establish operation identity and
management authority before transmission. Current browser clients send only the
management-capability digest, never the raw revoke/delete token, and the server
binds resource ID + operation ID + payload digest + capability digest. Exact
replay resolves the same object; a changed payload or digest conflicts; an
ambiguous transport/server outcome remains explicitly `outcome_unknown` and is
retried with the same envelope. Contribution management authority can be
explicitly exported/imported without silent localStorage persistence.

Verification owner: B37 Run 18 Python service tests + browser operation-envelope
source/mini-DOM tests.

## AIA-C34 — BrowserOriginPrivacyResidue

**Status:** HOLDS_WITH_SAME_ORIGIN_TRUST_BOUNDARY

Local feedback does not imply network telemetry or public DOM integration.
`ai-assistant-feedback` requires its own current structured permission and
remains content-free. Transcript recovery is explicit per-tab opt-in with
bounded untrusted-state restoration, microphone device identity is
session-scoped, endpoint diagnostics omit query/fragment/userinfo, and
credential-like endpoint query names are rejected. This application contract
does not isolate secrets from arbitrary compromised same-origin JavaScript.

Verification owner: B37 Run 18 executable browser harness + endpoint/security
source tests.

## AIA-C35 — GlobalShareStorageAuthority

**Status:** HOLDS_WITH_TRUTHFUL_BACKEND_LIMITS

HF Global Share storage reports and enforces the semantics of the configured
backend: memory is compatibility/process-local, SQLite is restart-durable and
single-instance, Redis is shared and atomically lifecycle-managed but is called
durable only with separate operator evidence. Required durability/shared modes
fail closed. Redis expiry cleanup is atomic. Cloudflare KV remains explicitly
eventually consistent; deterministic retry recovery is not promoted into an
atomic global create-once claim.

Verification owner: B37 Run 18 ShareStore service tests, Redis atomic-source
contract, Worker/browser Share tests and health/discovery assertions.


## AIA-C36 — ReproducibleDependencyClosure

**Status:** `HOLDS_WITH_FRESH_RELEASE_SCAN_REQUIRED`

**Invariant:** production Python dependencies are exact and hash-locked for the
declared runtime platform; direct requirements cannot silently expand through
broad extras; the Docker build consumes only the lock; and the checked-in
CycloneDX Python SBOM exactly matches package/version/hash closure. Immutable
base identity and dependency hashes provide reproducibility, while fresh
advisory/image scanning remains separate time-sensitive release evidence.

**Verification:** B38 `test_run19_supply_chain_deployment_hardening.py` plus
`_hf_spaces_proxy/security/verify_supply_chain.py`; external scanner/install
proof remains `SEC-P1-38`.

## AIA-C37 — LeastPrivilegeDeploymentProfile

**Status:** `HOLDS_AT_APPLICATION_AND_REFERENCE_DEPLOYMENT_BOUNDARY`

**Invariant:** strict deployment refuses root execution and unsafe browser-origin
policy; release container construction separates builder/runtime, binds the
platform to the wheel lock, runs UID/GID 1000, denies build-context files by
default, and provides a read-only/no-new-privileges/capability-dropped operator
reference. The application does not claim kernel/orchestrator enforcement it
cannot observe.

**Verification:** B38 Docker/policy/source assertions plus strict-startup tests.

## AIA-C38 — SharedRedisTransportAuthority

**Status:** `HOLDS_WITH_PRODUCTION_REDIS_OPERATIONS_EVIDENCE_REQUIRED`

**Invariant:** rate-limit, Share, and contribution Redis backends use one shared
validator. Strict mode requires `rediss://`, forbids query parameters that could
weaken redis-py TLS configuration, and forces peer certificate and hostname
verification. TLS transport does not prove ACL/authentication, persistence,
replication, backup, or failover configuration.

**Verification:** B38 three-control-plane transport tests and shared-helper
source/runtime contract; production operational evidence remains external.

## AIA-C39 — ReleaseEvidenceSubjectBinding

**Owner:** `_hf_spaces_proxy/security` release evidence policy/verifiers
**Status:** `HOLDS_WITH_EXTERNAL_EVIDENCE_REQUIRED`

**Invariant:** production promotion evidence is short-lived, content-addressed,
and bound to the exact dependency lock, Python SBOM, deterministic runtime-source
digest, immutable base manifest, final OCI image and matching SLSA provenance
subject. Path traversal/symlink substitution, stale evidence, artifact/source
drift and risk-waiver injection fail closed.

**Verification:** B39 Run 20 dedicated evidence tests + offline
`verify_supply_chain.py` + `release_subjects.py`; actual external scanner/signing
facts remain `SEC-P1-38`.

## AIA-C40 — InfrastructureTelemetryNonBypass

**Owner:** release evidence logging posture + existing application logging/telemetry boundary
**Status:** `HOLDS_AT_RELEASE_POLICY_BOUNDARY`

**Invariant:** browser telemetry consent cannot authorize infrastructure request
body/header/query/WAF/APM capture. Hardened promotion requires those channels and
third-party telemetry export to be disabled, while application/runtime logs keep
the existing content/capability minimization contract.

**Verification:** B39 release-evidence negative tests for body/telemetry posture +
existing logging/privacy mutation plane.

## AIA-C41 — RedisOperationalEvidenceBoundary

**Owner:** B38 Redis transport policy + B39 sanitized operational probe/release evidence
**Status:** `HOLDS_WITH_PRODUCTION_PROVIDER_EVIDENCE_REQUIRED`

**Invariant:** Redis URLs/identity/topology never become evidence output. The probe
may observe TLS/reachability/non-default identity/AOF/replication only through a
secret environment variable and emits coarse facts; Share/Contribution
production promotion separately requires least-privilege, persistence,
replication and recent backup/restore evidence.

**Verification:** B39 fake-client sanitization tests + B38 three-plane Redis TLS
gates. Provider durability/failover remains external `SEC-P1-38` evidence.

## AIA-C42 — PrivateLifecycleEventBoundary

**Status:** HOLDS_WITH_SAME_ORIGIN_RESIDUAL

Internal assistant coordination uses a private event bus. `document` is an
optional integration surface, not an internal transport. Public lifecycle events
require current page-integration consent and expose only bounded per-event
projections; raw model/profile/endpoint/token/Q&A/conversation detail is not
projected. Network telemetry permission does not grant page-event authority.

Verification owner: B40 Run 21 executable browser isolation harness + source
regression assertions.

## AIA-C43 — OpaqueOriginReadWriteSeparation

**Status:** HOLDS

`Origin: null` compatibility is route/method scoped. Read-only Share viewing may
be enabled independently; mutation/capability-management requires a second
high-risk opt-in, and strict HF deployments reject opaque-origin writes. Browser
preflights are classified by intended method so OPTIONS cannot bypass the split.

Verification owner: B40 Run 21 HF TestClient/subprocess tests + Worker source and
Wrangler parity assertions.

## AIA-C44 — RuntimeBearerCompatibilityDefaultOff

**Status:** HOLDS_WITH_SAME_ORIGIN_RESIDUAL

Static Sphinx output never serializes bearer credentials. Browser-entered
Share/Feedback bearer fields are additionally disabled by default and require an
explicit site-owner compatibility opt-in. The endpoint registry enforces this at
resolution/ingest, so programmatic profile injection cannot create token
authority while the policy is Off. Opt-in tokens remain page-memory-only.

Verification owner: B40 Run 21 executable endpoint-registry browser harness +
Python Sphinx-config source assertions.


## AIA-C45 — SeparateOriginRuntimeIsolation

**Status:** HOLDS_WHEN_EXPLICITLY_ENABLED

A configured isolation origin suppresses the full assistant runtime on the documentation page and runs UI/transcript/runtime state in a distinct-origin sandboxed frame. The main bundle independently self-suppresses when isolation was requested, so loss of the host bridge cannot create a silent same-origin downgrade.

Verification owner: B41 Python source/config tests + executable host/frame harness.

## AIA-C46 — CapabilityMessageChannelBoundary

**Status:** HOLDS

Window messaging is limited to one exact-origin/source/version/channel handshake that transfers one MessagePort. Runtime envelopes are bounded and monotonic, reject replay/out-of-order input, and dispatch only a fixed capability allowlist (`page.context.read`, `page.canonical.read`, `page.print`, `ui.resize`, `page.integration.emit`).

Verification owner: B41 executable browser harness + source assertions.

## AIA-C47 — IsolatedStorageTenantPartition

**Status:** HOLDS_AT_BROWSER_STORAGE_KEY_BOUNDARY

Every localStorage/sessionStorage key used by the isolated assistant runtime is transparently prefixed by the validated parent origin and normalized documentation root. Direct transcript sessionStorage helpers use the scoped store too; unrelated parent origins and separate projects under one origin cannot intentionally share fixed assistant storage keys merely because they use one isolated service origin.

Verification owner: B41 source assertions + storage-scope executable initialization test.

## AIA-C48 — HostPageContextCapabilityBoundary

**Status:** HOLDS_WITH_COMPROMISED_PARENT_RESIDUAL

The frame receives bounded page snapshots/canonical Markdown rather than ambient parent DOM access. Query/fragment identity is stripped; active/form/assistant/hidden content and dangerous attributes are removed before transfer; the isolated runtime still performs secret redaction, privacy review, invisible-control removal and injection fencing. A compromised parent can still alter source content before extraction (`SEC-P1-42`).

Verification owner: B41 host adapter/source tests + existing privacy/injection mutation plane.

## AIA-C49 — IsolatedPublicIntegrationProjection

**Status:** HOLDS

Isolated mode does not bypass B40 page-integration consent. Only the bounded B40 projection may traverse `page.integration.emit`, and the host independently validates type, key shape, depth and total size before dispatching on the parent document.

Verification owner: B41 source/executable capability tests + B40 projection harness.
## AIA-C50 — BootstrapSnapshotIntegrity

**Status:** HOLDS_AFTER_HOST_BRIDGE_STARTUP

The documentation-origin bridge snapshots and sanitizes assistant configuration, endpoint descriptors and the endpoint default before any asynchronous frame handshake. Secret-shaped or prototype-pollution keys are excluded, snapshot maps have null prototypes, and later mutations of the page globals cannot alter the INIT payload. A parent compromised before bridge startup remains the explicit `SEC-P1-42` residual.

Verification owner: B41 Python source assertions + executable late-global-mutation/prototype-pollution browser harness.


## AIA-C51 — FrameGeneratedBootstrapAuthority

**Status:** HOLDS

Protocol v2 generates the bootstrap channel capability inside the isolated frame with WebCrypto after the generated parent-origin policy is accepted. The capability is absent from the iframe URL, `Math.random()` is never a security fallback, and missing secure randomness fails closed.

Verification owner: Run 23 B42 Python source assertions + executable browser handshake harness.

## AIA-C52 — IsolationNavigationContainment

**Status:** HOLDS_AT_REPOSITORY_SANDBOX_BOUNDARY

The isolated frame cannot preserve script execution while self-navigating onto the documentation origin. HTTP(S) anchor navigation is intercepted and opened outside the isolated browsing context with `noopener,noreferrer`; popup escape/top-navigation sandbox authority is not granted.

Verification owner: Run 23 B42 sandbox/navigation source + executable click tests.

## AIA-C53 — GeneratedParentOriginPolicy

**Status:** HOLDS_WITH_DEPLOYMENT_HEADER_RESIDUAL

The build emits a closed-schema protocol-v2 isolation policy containing only the configured/derived exact parent origins and isolation origin. The source/default policy is deny-all; invalid or missing policy authority fails before HELLO. Real CDN/CSP/frame-ancestors truth remains `SEC-P1-43`.

Verification owner: Run 23 B42 policy-generator unit tests + isolated-frame policy harness.

## AIA-C54 — AmbientCredentialEgressBoundary

**Status:** HOLDS

Assistant-service requests centrally force browser credentials to `omit` by default. A deliberate site-owner compatibility opt-in can raise this only to `same-origin`; caller-requested `include` cannot escape the wrapper. Canonical documentation reads are a separately bounded same-origin content capability, not a general service-credential exception.

Verification owner: Run 23 B42 executable fetch-wrapper tests + source assertions.

## AIA-C55 — CrossOriginDeviceDelegationBoundary

**Status:** HOLDS

Cross-origin microphone delegation is independent of speech/UI availability and is default Off. When delegation is not explicitly enabled, isolated INIT suppresses voice controls that could imply an unavailable permission path.

Verification owner: Run 23 B42 Python/host configuration assertions.

## AIA-C56 — BoundedCanonicalReadBoundary

**Status:** HOLDS_AT_BROWSER_ADAPTER_BOUNDARY

Canonical Markdown/page reads performed by the docs-origin bridge are same-origin only, strip query/fragment from docs-root authority, reject redirects, bypass caches, and enforce streaming byte/character ceilings before complete buffering.

Verification owner: Run 23 B42 host source assertions + existing canonical representation tests.

## AIA-C57 — PreBufferRemoteResponseCeiling

**Status:** HOLDS_AT_COVERED_BOUNDARIES

Browser chat/control/canonical reads, isolated policy/canonical reads, Global
Share viewer reads, HF/dev-proxy chat responses and Worker chat forwarding apply
byte ceilings before whole-body buffering and recheck streamed bytes. Malformed
or declared-oversize lengths fail closed.

Verification owner: Run 24 B43 Python + executable browser/Worker tests.

## AIA-C58 — StreamCapabilityFailClosed

**Status:** HOLDS

Security-sensitive browser response readers do not reinterpret missing
`ReadableStream.getReader()` as permission to use unbounded `text()`/`json()`.
A transport incapable of enforcing the pre-buffer ceiling fails closed.

Verification owner: Run 24 B43 browser source/executable tests + Run 22/23
compatibility harnesses updated to provide real bounded streams.

## AIA-C59 — UpstreamResponseParity

**Status:** HOLDS_FOR_CHAT_RELAY_PLANE

HF/FastAPI, local dev relay, and Cloudflare Worker enforce compatible 8 MiB
default / 32 MiB hard upstream chat-response ceilings. HF/dev buffer only after
bounded streamed collection; Worker forwards unknown-length responses through a
byte-counting stream and rejects malformed/declared oversize before forwarding.

Verification owner: Run 24 B43 Python, Worker executable harness and deployment
configuration assertions.

## AIA-C37 — ProviderNativeContributionReview

**Owner:** HF proxy contribution lifecycle + storage-provider adapters
**Status:** `HOLDS_WHEN_PROVIDER_PR_ENABLED`

**Invariant:** With `CONTRIBUTION_REVIEW_MODE=provider-pr`, explicit dataset contributions are persisted for human review using the Primary provider's native PR/MR mechanism while the canonical branch remains the only training-eligible authority. Review metadata is opaque and content-free; native merge monotonically ratchets to eligible, close/decline never does, and participant management remains capability-separated from reviewer authority. `ledger` remains the backwards-compatible default.

**Verification:** B54 `test_run35_provider_review_workflow.py`, existing storage/provider bounded-response suites, dataset contribution lifecycle suites, and registered browser harnesses.
