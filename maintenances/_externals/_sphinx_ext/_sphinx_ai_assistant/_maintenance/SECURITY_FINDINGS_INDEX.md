# `_sphinx_ai_assistant` Security Finding Index

Status: **REVALIDATION REQUIRED at B02**. This file is a durable index, not proof
that each finding remains at the same line number after the source anchor
changes.

The previous review identified the following high-priority families. B02 must
reproduce or close each against the exact current source before implementation
work claims success.

| ID | Priority | Finding family | Target invariant |
|---|---|---|---|
| `SEC-P0-01` | P0 | server credential attached to configurable/unbound backend destination | credentials are destination-bound |
| `SEC-P0-02` | P0 | wildcard/permissive CORS in proxy/edge paths | explicit least-privilege origins with path parity |
| `SEC-P0-03` | P0 | share locator/UUID doubles as edit authority | read and write capabilities are distinct |
| `SEC-P0-04` | P0 | alternate/direct inference relay bypasses primary controls | all service paths enforce equivalent policy |
| `SEC-P0-05` | P0 | caller/browser can supply authoritative `system` content | server exclusively owns system policy |
| `SEC-P0-06` | P0 | feedback/training contribution poisoning path | contributions remain untrusted with provenance/review |
| `SEC-P0-07` | P0 | consent/version provenance insufficient or disabled | stored contribution carries consent version/state |
| `SEC-P0-08` | P1/P0 by deployment | untrusted forwarded client identity | only trusted proxy boundary can assert forwarding identity |
| `SEC-P0-09` | P1/P0 by resource | body limit applied after excessive buffering/allocation | enforce protective limits early |
| `SEC-P0-10` | **CLOSED — Run 19 source-controlled boundary** | floating/root/container/supply-chain hardening gaps | immutable/platform-bound base, exact hash lock, minimal dependency set, builder/runtime split, non-root strict profile, deny build context, scoped SBOM/verifier and mandatory fresh scan/provenance gates |
| `SEC-P0-11` | P0 target | credential/token values can reach generated/browser config escape hatches | production secrets never client-visible/persisted |
| `SEC-P0-12` | P0 | HTML export embeds raw JSON containing literal `</script>` | untrusted snapshot data cannot terminate raw-text container / execute |
| `SEC-P0-13` | P0 | legacy `#ai-share-c1.html.<payload>` can decode arbitrary HTML into a Blob | current self-contained generation is trusted-renderer output from the reviewed normalized snapshot, made inert and base64 data-URL encoded; legacy c1 HTML remains inert and c2 is compatibility-only |
| `SEC-P0-14` | P0 | Global Share accepts caller content + MIME and serves it inline | server owns structured storage and representation/MIME |
| `SEC-P0-15` | P0/P1 | Share/read capability IDs and other sensitive identifiers are logged | capabilities/secrets never appear in logs |
| `SEC-P0-16` | P0/P1 | runtime endpoint-profile tokens persisted to localStorage | browser-entered tokens are page-memory-only |
| `SEC-P0-17` | P1 | source page URL can leak query/hash/local filesystem path into exports | Share metadata uses sanitized source descriptor |
| `SEC-P0-18` | P1 | feedback default/persistence can retain full conversation metadata without a distinct contribution action | rating telemetry is minimal; content contribution explicit |
| `SEC-P0-19` | P1 | traceback/log redaction paths can diverge | all formatted log text passes one redaction/bounding path |
| `SEC-P0-20` | P1 | Global Share capability URL scheme can be wrong behind a proxy | public URL uses externally trusted HTTPS origin / explicit configured base |
| `SEC-P0-21` | P1 | Share TTL and cache semantics can conflict with expiry/revocation | response caching never outlives privacy/expiry guarantees |
| `SEC-P0-22` | P1 | Share entry/body/aggregate storage quotas insufficient | per-item, entry-count, aggregate-byte, and cleanup limits are enforced |
| `SEC-P0-23` | P1/P2 | rate-limit identity trusts untrusted forwarded headers | only trusted proxy boundary can assert client identity |
| `SEC-P0-24` | P2 | malformed/non-object JSON can escape validation into 500 paths | endpoint validates top-level type/schema and returns bounded 4xx |
| `SEC-P0-25` | P2 | invisible/bidi controls can visually spoof shared content | raw data preserved but suspicious formatting is inspectable/warned |
| `SEC-P0-26` | P1 | process-local contribution quarantine is not durable/shared across replicas | production review/delete control plane is mutable, transactional, bounded, and replica-safe |
| `SEC-P0-27` | P1 | promoted append-only/mirrored contribution data lacks guaranteed physical erasure | UI/policy never promises global erasure without provider-complete deletion evidence |
| `SEC-P0-28` | P1 | likely secrets/personal data/invisible controls can leave browser egress without local user review | actual outbound inference/Share/contribution payload receives category/count-only advisory preflight before egress |
| `SEC-P0-31` | P2 | application abuse counters may be mistaken for globally authoritative distributed quotas | Run 15 adds Worker Durable Object + HF Redis authority with fail-closed required modes; HF horizontal deployment activation remains operator evidence |
| `SEC-P0-32` | P1 | concurrent contribution promotion can race after a shared read and persist the same receipt more than once | atomic lifecycle claim makes exactly one reviewer own promotion/withdrawal transition |
| `SEC-P0-33` | P1 | pending receipt deletion wording can overclaim forensic/global physical erasure beyond active-ledger evidence | scope deletion claims to active ledger/training/current view; keep forensic/history erasure explicitly unproved |
| `SEC-P0-34` | **CLOSED — Run 16** | ambiguous/timeout provider mutation outcomes enter `promotion_uncertain`, block re-promotion, and can resolve toward withdrawal; expired promotion leases are never automatically reassigned across unfenced external side effects |

| `SEC-P1-35` | P1 | contribution purpose/consent UX was conflated with Share and ambiguous feedback “dataset” wording | feedback telemetry, human Share/export, and content-bearing contribution remain distinct control planes; every content submission uses explicit inspect/privacy/versioned-consent/quarantine lifecycle |
| `SEC-P1-36` | P1 | feedback service consent evidence and content-rich public DOM event did not match the local-only telemetry promise | current structured telemetry consent is enforced client/server and the public event is content-free |
| `SEC-P1-37` | P1 | public DOM integration remained unconditional when network telemetry was Off; lifecycle CREATEs could orphan/duplicate objects; management recovery, Share durability, and reviewed-content fidelity had gaps | separate page-integration permission; recoverable capability-digest CREATE envelope; explicit receipt recovery; truthful ShareStore semantics; reject-not-truncate contribution review |
| `SEC-P1-38` | P1 / deployment | checked-in supply-chain policy cannot prove current final-image vulnerability state, locked artifact installation, signed provenance, or production Redis ACL/persistence/replication | fresh per-release dependency + built-image scan/SBOM/provenance evidence and operator Redis evidence; never treat digest/SBOM source files as deployment proof |
| `SEC-P1-39` | **CLOSED — Run 20 evidence-binding boundary** | fresh artifacts could still be stale/substituted or bound to a different runtime source/image; infrastructure telemetry and Redis operational facts lacked one fail-closed promotion envelope | short-lived content-addressed release evidence bound to lock + runtime source + image + base manifest + provenance, with telemetry/logging and Redis operational guardrails |

## Closure format

For each finding, B02/B03+ must record:

```text
current source anchor
exact owner/path
reproduction/exploit precondition
current status: CONFIRMED | DISPROVED | PARTIAL | DEFERRED
fix checkpoint
regression test that bypasses the browser when service-level
rollback/compatibility impact
```

Do not close `SEC-P0-05` with client prompt sanitization alone; direct service
callers must be unable to set authoritative system policy.

## B18 implementation dispositions

| Finding | Current disposition | Evidence / next owner |
|---|---|---|
| `SEC-P0-01` | **CLOSED — Run 4** | Path 1/2/3 credentials are independent and destination-bound; redirect auto-follow disabled |
| `SEC-P0-02` | **CLOSED — Run 11; Run 16.2.6 local-file compatibility bounded** | exact default origin policies + early explicit-Origin denial across bundled proxy/model/Worker/dev paths; Run 16.2.6 permits `Origin:null` only on Share routes under explicit `SHARE_ALLOW_OPAQUE_ORIGIN=true`; Origin remains non-authentication |
| `SEC-P0-08` | **CLOSED — Run 11 app identity boundary** | XFF default-deny/explicit trusted-ingress rule is shared by all HF rate-limited routes; Worker direct-edge identity is explicit |
| `SEC-P0-09` | **CLOSED — Run 11 app buffering boundary** | Content-Length precheck + incremental stream limits on HF proxy/model/Worker; dev declared-length guard before read |
| `SEC-P0-04` | **CLOSED — Run 4 bundled inference paths** | HF proxy, Worker, dev proxy, and direct model service enforce server-owned prompt/model policy |
| `SEC-P0-05` | **CLOSED — Run 4** | `scikitplot-chat-v1`; direct model and relay tests reject caller role authority |
| `SEC-P0-06` | **CLOSED — Run 6 core boundary** | contribution intake is quarantined/untrusted; separate review capability is required before `trainingStatus=eligible`; cleaner fails closed |
| `SEC-P0-07` | **CLOSED — Run 6** | exact `consentVersion` is enforced and retained; client consent remains explicitly a client assertion, not verified identity |
| `SEC-P0-11` | **CLOSED — Run 1 client/build boundary** | build/static and endpoint-profile serialization contain no credential values |
| `SEC-P0-12` | **CLOSED — Run 2 client export** | `test_active_content_isolation.mjs` + `export-html-raw-json-breakout` mutant |
| `SEC-P0-13` | **CLOSED — Run 2; current transport superseded safely by Run 16.2** | c1 HTML inert; c2 compatibility validated; current generation uses reviewed inert base64 data HTML + active-content gates |
| `SEC-P0-14` | **CLOSED — Run 3** | HF + Cloudflare accept structured snapshots + format enum only and render server-side |
| `SEC-P0-15` | **PARTIAL — Run 14 current + generation-2 path transport closed** | app events omit capabilities/content; current generated Share URLs use a fragment/fixed operations, and generation-2 objects are rejected on legacy capability-bearing paths. Historical pre-generation URLs drain by migration/revoke/expiry; full-body/WAF tracing is deployment-owned. |
| `SEC-P0-16` | **CLOSED — Run 1 browser persistence** | endpoint profile v3 + migration gate |
| `SEC-P0-17` | **PARTIAL — Run 2 export/share metadata closed** | canonical snapshot sanitizes source; broader logs/persistence privacy still B18 |
| `SEC-P0-18` | **CLOSED — Run 6** | rating telemetry is opt-in/minimal; content contribution is separate, version-consented and quarantine-first |
| `SEC-P0-19` | **CLOSED — Run 5 application boundary** | proxy/model share one bounded sanitizer/exception-summary path; raw traceback split is regression-gated |
| `SEC-P0-20` | **CLOSED — Run 16.2.6 hardened HF deployment derivation** | explicit `SHARE_PUBLIC_BASE_URL` remains authoritative; validated HF `SPACE_HOST` supplies the public HTTPS origin when ASGI sees internal HTTP; arbitrary remote HTTP still fails closed |
| `SEC-P0-21` | **CLOSED — Run 3 Share responses** | Global Share uses `private, no-store`, noindex/noarchive, no-referrer |
| `SEC-P0-22` | **PARTIAL — Run 3** | HF hard body/count/aggregate quotas; CF hard per-entry + conservative eventually-consistent count/aggregate gate; strict distributed quota requires Durable Object |
| `SEC-P0-23` | **CLOSED — Run 11 app identity boundary** | Every HF rate-limited public route uses the same direct-peer/explicit-XFF helper; Worker uses edge `CF-Connecting-IP`; topology changes require revalidation |
| `SEC-P0-24` | **CLOSED — Run 3 Share path** | POST/PATCH validate top-level object, schema, format, and bounded primitives with 4xx |
| `SEC-P0-25` | **CLOSED — Run 7 client inspection/redaction boundary** | invisible/bidi controls are surfaced as codepoint/count metadata; normal data is preserved and explicit Redact removes matched controls from an outbound copy |
| `SEC-P0-26` | **PARTIAL — Run 16 shared authority landed; external Redis durability/activation conditional** | memory remains process-local; SQLite remains local restart-durable; Redis now provides shared atomic receipt/review/withdraw authority with HMAC identifiers and shared-required fail-closed semantics, but repository code cannot prove external Redis persistence/production activation |
| `SEC-P0-27` | **PARTIAL — Run 12 training/current-view withdrawal landed** | withdrawal tombstones suppress ordinary training output and current provider views are removed best-effort; version history/backups/caches/provider infrastructure are not proved erased |
| `SEC-P0-28` | **CLOSED — Run 7 client boundary** | one local advisory preflight covers user+prepared page context, canonical Share snapshots, and explicit contribution payloads; matching values never enter findings/logs; direct API bypass remains governed by server controls |

| `SEC-P0-29` | P1 | provided Global links can become lifecycle-unmanaged across new-chat/reload, encouraging stale/public links the user can no longer see or reason about | bounded session public ledger tracks every provided Global link; edit capability stays page-memory-only; explicit status/revoke/forget semantics |
| `SEC-P0-30` | P1 | Global lifecycle recovery trusted legacy/tampered session state too broadly and reason-unknown/expired server states could diverge from current client mutation state | fail-closed allowlisted recovery; unavailable is recheckable; terminal/current-update state is detached; expired GET/PATCH/DELETE/HEAD semantics are explicit and regression-gated |

### Run 9 disposition

`SEC-P0-29` is **CLOSED — Run 9 browser/server lifecycle boundary** via B25. Public read capabilities are intentionally session-scoped and bounded; terminal states erase URL/UUID. Infrastructure request-path logging remains the independent `SEC-P0-15` residual.

### Run 10 disposition

`SEC-P0-30` is **CLOSED — Run 10 fail-closed lifecycle recovery boundary** via B26. Parsed Web Storage is allowlisted/scrubbed before use, reason-unknown 404 remains recheckable, stale current PATCH state is detached, and explicit expiry semantics are aligned across HF/Cloudflare lifecycle routes. The inability to revoke after page reload remains intentional because edit capability is not persisted.


### Run 11 disposition

`SEC-P0-02`, `SEC-P0-08`, `SEC-P0-09`, and `SEC-P0-23` are **CLOSED at the bundled application boundary** via B05/B06/B27. Run 15/B31 supersedes the old pure-infrastructure disposition for `SEC-P0-31`: the bundled Worker now requires sharded Durable Object authority and HF ships atomic Redis authority with fail-closed shared-required mode. The finding remains **PARTIAL / DEPLOYMENT-CONDITIONAL** because HF defaults to process-local compatibility until a horizontal deployment explicitly activates/proves the Redis consistency domain.


### Run 12 disposition

`SEC-P0-26` is **PARTIAL**: B28 closes bounded single-instance restart durability with SQLite; B32 adds an optional shared atomic Redis receipt authority and fail-closed shared-required mode. The remaining gap is external Redis production activation plus persistence/replication/backup/recovery evidence, which repository code cannot self-certify. `SEC-P0-27` is **PARTIAL**: B28 makes withdrawal enforceable in training output and attempts provider current-view deletion, but version history/backups/caches/infrastructure remain outside the erasure guarantee. `SEC-P0-32` and `SEC-P0-33` are **CLOSED — Run 12** by atomic lifecycle claims and truthful scoped deletion semantics.

`SEC-P1-34` is **PARTIAL / DRAINING LEGACY — Run 14**: new and fixed-updated objects carry transport generation 2 and are rejected on `/v1/share/{id}`. Only pre-generation objects remain eligible for deprecated HEAD/GET/authenticated DELETE; legacy PATCH is retired and cannot extend TTL. Fixed update permanently migrates an old object. Remove the route code after the bounded pre-generation population drains.


### Run 15 disposition

`SEC-P0-31` is **PARTIAL / DEPLOYMENT-CONDITIONAL — Run 15 B31**. The repository now contains an authoritative shared decision plane on both bundled deployment families: sharded SQLite-backed Durable Objects for Worker and atomic Redis Lua counters for HF. Required-authority modes fail closed and shared identities are HMAC-pseudonymized. The bundled Worker enables/requires its authority in `wrangler.toml`; horizontally scaled HF still requires operator provisioning and `RATE_LIMIT_BACKEND=redis` + `RATE_LIMIT_REQUIRE_SHARED=true`. Local/KV compatibility paths remain non-authoritative and are never billing/accounting evidence.

`SEC-P0-34` is **CLOSED — Run 16**: a provider timeout/transport failure may mean the mutation succeeded but its response was lost, so the receipt is moved to non-promotable reconciliation state rather than returned to ordinary quarantine. Participant withdrawal remains available as the privacy-safe monotonic resolution.

### Run 17 disposition

`SEC-P1-35` is **CLOSED — Run 17 B36**: content-bearing dataset contribution was hidden under Share and the feedback popup used ambiguous dataset language, making purpose/consent boundaries hard to discover. The browser now has separate Feedback telemetry, Share, and Dataset contribution control planes. All content contribution shortcuts converge on one exact-payload inspect/privacy/consent/quarantine controller; telemetry cannot imply content consent and Share owns no contribution authority.


### Run 17 telemetry-permission residual

`SEC-P1-36` is **CLOSED — Run 17 B36**: local rating telemetry was already Off
by default, but the bundled feedback service accepted requests without explicit
consent evidence and the public feedback DOM event exposed the full local Q&A
tuple to page listeners. Current client telemetry uses a structured versioned
permission; old boolean keys fail closed; rating and retraction helpers self-gate;
HF/Worker require schema 4 + current consent marker/version/timestamp; and the
public DOM event is rating-only. Operator persistence remains independent and
never implies reader permission.

### Run 18 disposition

`SEC-P1-37` is **CLOSED — Run 18 B37 at the application boundary**: public DOM rating events require a separate explicit current permission; Share/contribution CREATE is recoverable and binds resource, operation, reviewed payload, and capability digest without transmitting the raw management capability; contribution management authority has explicit receipt export/import; schema-v4 reviewed content rejects over-limit input rather than silently truncating it; and HF Share storage reports backend-truthful durability/sharing semantics with atomic Redis expiry cleanup. Same-origin script compromise, Worker KV eventual consistency, infrastructure full-body logging, provider history/backup erasure, and production durability configuration remain explicit residuals rather than being folded into this closure.


## Run 19 / B38 disposition — supply chain and deployment

### `SEC-P0-10` — CLOSED at source-controlled release-contract boundary

B38 replaces floating/ranged/root-oriented release behavior with an immutable
Linux/amd64 base identity, exact hash-locked Python closure, minimal direct
requirements, builder/runtime separation, non-root strict runtime,
deny-by-default Docker context, scoped CycloneDX Python SBOM, executable offline
verifier and documented fail-closed scanner/provenance release gates. One Redis
transport-security helper also prevents the three shared authority planes from
drifting into plaintext or query-parameter TLS downgrade behavior in strict
mode.

This closure is intentionally scoped to repository-owned controls. An immutable
digest is not evidence that the selected bytes are vulnerability-free.

### `SEC-P1-38` — OPEN deployment/release evidence

The final release environment must still prove a fresh exact-lock installation,
current dependency advisory result, built-image SBOM/CVE policy result,
provenance/registry evidence, and production Redis ACL/persistence/replication
properties. Those facts are external and time-sensitive. The current offline
execution environment cannot contact PyPI, so B38 does not fabricate a locked
install result. A newly disclosed advisory reopens the affected release gate
even if a previous build was GREEN. Run 20/B39 now verifies freshness and
subject/source binding for the evidence envelope, but it deliberately does not
convert those external facts into source-controlled truth.

### `SEC-P1-39` — CLOSED at evidence-binding boundary in Run 20

B39 adds a short-lived content-addressed release-evidence envelope and one
fail-closed verifier that binds external scan/SBOM/signature artifacts to the
exact checked-in lock/Python SBOM, deterministic Docker-owned runtime source,
resolved immutable base manifest and final OCI image. Provenance must name the
final image subject and base manifest; symlink/path/hash substitution, stale
evidence, hidden infrastructure body/credential/query logging, third-party
telemetry export, Redis default identity, unproved Share/Contribution
persistence/replication, stale restore tests and risk-waiver injection fail
closed.

This does **not** close `SEC-P1-38`: the repository can verify that evidence is
bound and structurally compliant, not that an external scanner/provider/operator
told the truth. Fresh networked scans, trusted signature verification, actual
image promotion and production Redis/logging evidence remain external release
facts.

### Run 21 disposition

`SEC-P1-40` is **CLOSED — Run 21 B40 at the repository-controlled event/CORS/token
boundary**: internal assistant lifecycle events no longer publish directly on
`document`; page integration uses an independent v2 permission plus bounded
projections; `Origin: null` read and write authority are separate with strict
write refusal; browser bearer-token entry is centrally default Off; and Share
viewer pages deny framing/sensitive permissions.

`SEC-P1-41` remains **OPEN / ARCHITECTURAL**: arbitrary compromised JavaScript on
the same documentation origin still shares the DOM/memory trust boundary. Full
isolation requires a separate-origin assistant execution surface or equivalent
browser-enforced compartment; B40 deliberately does not paper-close this.


### Run 22 disposition

`SEC-P1-41` is **CONDITIONALLY CLOSED — Run 22 B41 when separate-origin isolation is explicitly enabled and correctly deployed**: ordinary documentation-origin JavaScript no longer has ambient access to assistant DOM, transcript, runtime model/profile state, management receipts or isolated-origin Web Storage. The parent executes only a narrow page capability adapter, while the assistant runtime lives behind the browser Same-Origin Policy in a distinct-origin sandboxed frame. Same-origin compatibility mode intentionally retains the original residual and must not inherit this closure claim.

`SEC-P1-42` is **OPEN / ARCHITECTURAL**: a fully compromised documentation parent can still modify the page before context extraction, cover/remove/clickjack the isolated iframe, monkeypatch browser primitives before the host bridge initializes, or deny service. B41 limits what the parent can *read from* the assistant compartment; it cannot make a hostile parent an honest source of page context or presentation.

`SEC-P1-43` is **OPEN / DEPLOYMENT EVIDENCE**: production isolation depends on the isolated origin actually serving restrictive response headers (`frame-ancestors`, exact `connect-src`, no-referrer, nosniff, permissions policy) and proxy CORS allowing the isolated origin. The repository ships a restrictive baseline document and fail-closed protocol, but cannot prove CDN/reverse-proxy header truth from source bytes alone.

### Run 23 disposition

`SEC-P1-42` remains **OPEN / ARCHITECTURAL, NARROWED BY B42**. B42 removes several post-start and frame-escape opportunities: bootstrap entropy is frame-owned/WebCrypto-only and absent from the iframe URL; the host installs a capture listener before attachment; generated parent policy is deny-default; frame-self HTTP(S) navigation/popup escape is blocked; service traffic omits ambient credentials; microphone delegation is separately Off by default. None of those controls make a documentation parent already compromised before host startup an honest page/context/presentation authority.

`SEC-P1-43` remains **OPEN / DEPLOYMENT EVIDENCE**. Generated policy and source baselines do not prove real `frame-ancestors`, CSP, CORS, nosniff, referrer, CDN or reverse-proxy behavior.

`SEC-P1-44` is **CLOSED — Run 23 B42 at the repository-controlled browser boundary**: the B41 isolation compartment could be weakened by URL-carried/weak bootstrap authority, frame-self navigation onto the parent origin, popup sandbox escape, ambient credential attachment to service fetches, unscoped microphone delegation, and same-parent multi-project storage collision. Protocol v2 plus the B42 policy/navigation/egress/storage gates close those paths in shipped code.

### Run 24 disposition

`SEC-P1-45` is **CLOSED — Run 24 B43 at the covered response-ingestion boundaries**: browser assistant/control/canonical reads, isolated policy/canonical reads, standalone Global Share viewer reads, HF/dev-proxy chat responses and Worker chat forwarding now enforce response ceilings before and during stream consumption. Malformed declared lengths fail closed, unknown-length responses remain byte-counted, and security-sensitive browser code no longer falls back to whole-body `text()`/`json()` when a bounded stream reader is unavailable.

This is intentionally not a universal remote-I/O claim. Optional provider-storage clients and opaque third-party SDK internals remain a separate audit surface; a future run must review them before claiming every external response in the subsystem is pre-buffer bounded. `SEC-P1-38`, `SEC-P1-42`, and `SEC-P1-43` remain open in their existing scopes.


### Run 25 disposition

`SEC-P1-46` is **CLOSED — B44 provider-control response boundary**: repository-controlled record-storage mutations avoid body consumption when response content is unnecessary; metadata responses are bounded before JSON parsing; Hugging Face Hub control calls use a scoped bounded HTTP transport; GitLab custom API authority is HTTPS/host/path validated and other providers reject `api_base`.

`SEC-P1-47` is **CLOSED — B44 deterministic semantic-context boundary**: live rendered style and geometry are consulted before clone serialization in same-origin and isolated modes. Deterministic model-only surfaces are removed while ordinary below-the-fold documentation remains eligible.

`SEC-P2-48` remains **OPEN** for occlusion/z-index, animation/timing and reader-attention semantics. `SEC-P2-49` remains **OPEN / SEPARATELY SCOPED** for intentional large bulk downloads.
