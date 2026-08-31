# `_sphinx_ai_assistant` Durable Ruleset

## Authority and prompt trust

1. Browser code is never a security authority.
2. Documentation/page/retrieval content is untrusted reference data, not system
   instructions.
3. Authoritative model system policy is server-owned and immutable from normal
   client requests.
4. Client-side prompt-injection filters are defense in depth only; direct API
   callers must still be safe.

## Secrets and configuration

5. Production secrets must not be serialized into generated HTML, browser
   globals, URLs, logs, exports, or persistent browser storage.
6. Client config may advertise **capability/presence**, not secret value.
7. Escape-hatch self-hosting behavior must be explicitly labeled insecure/non-
   recommended and may not silently become the default.
8. Every configurable setting has one canonical schema/validation/ownership
   record before settings consolidation is declared complete.

## Service security

9. Credentials are bound to approved destinations; user-controlled routing may
   not receive server credentials.
10. CORS origins are explicit and least-privilege by default.
11. Share read capability and edit/delete capability are distinct.
12. Forwarded client identity is trusted only from an explicitly configured
    proxy boundary.
13. Body/resource limits apply before or while buffering enough to protect
    memory, not only after full allocation.
14. Feedback/training persistence records consent/provenance/authenticity and
    treats user-submitted data as untrusted.
15. Proxy, edge-worker, and direct-model paths must not create an easier bypass
    around each other's policy.

## Representation ownership

16. The assistant owns canonical page Markdown and `llms.txt`, generated at
    `build-finished` over the final HTML.
17. Canonical means the build-time artifact and nothing else. A browser
    conversion is convenience, however faithful, because no external agent can
    fetch it.
18. `VIEW` and `ASK AI` use the canonical static `.md`. `COPY` is convenience by
    default and canonical when its mode toggle selects `static`.
19. The selected representation is observable at the control that produced it;
    convenience output is never labelled canonical.
20. `_sphinx_llm` is frozen and imported by nothing. No assistant surface may
    degrade when it is absent from `extensions`.

## Multi-runtime maintenance

21. Browser JS, worker JS, Python services, Sphinx integration, and schemas all
    receive explicit gates.
22. Known large files are baseline debt; new monolith growth/new responsibility
    triggers review.
23. `_static/_backup` is historical only and may never become a runtime import/
    dependency source.
24. `_maintenance` prototypes are not production behavior until explicitly
    promoted with tests.

## Maintenance truth

25. Desired invariants are not marked `HOLDS` without current-source proof.
26. `ENVIRONMENT_BLOCKED` is not `GREEN`.
27. A security finding is not closed by client-only mitigation if the direct
    service endpoint still violates the invariant.
28. Every closed finding names a regression gate and exact source owner.


## User data, identity, logging, and abuse

29. Open-source client behavior is assumed discoverable and bypassable; client
    obscurity is never a security control.
30. Conversation/user/model content is transient by default. Persistence or
    contribution requires a distinct purpose and explicit action.
31. Do not create a stable user identity when a short-lived correlation ID is
    sufficient. IP address, conversation ID, session ID, model label, and share
    locator are not authenticated human identity.
32. Query strings, URL fragments, embedded URL credentials, local filesystem
    paths, and bearer capabilities are removed from Share/contribution metadata
    unless an explicit non-Share forensic workflow requires them.
33. PII/sensitive-content detection is warning/triage only; absence of a match
    is never evidence that data is non-sensitive.
34. Logs contain bounded operational metadata, not conversation content,
    credentials, complete share/edit locators, feedback comments, private URLs,
    or stable personal identifiers.
35. Exception/traceback text is passed through the same centralized redaction
    and length-bounding path as ordinary log messages.
36. Never log partial secrets as a debugging aid. Secret-shaped values are
    omitted/redacted as a whole.
37. Public health/discovery endpoints expose only the deployment information
    readers need; detailed topology, secret-presence diagnostics, and internal
    storage configuration are operator diagnostics.
37a. A public client dependency is satisfied with the least identifying form of
     the value: prefer coarse `configured`/`ready` capability state over repo IDs,
     backend URLs, storage targets, exact token classes, or credential presence.

## Export, Share, and active-content isolation

38. Every export/share format consumes one canonical privacy-filtered snapshot.
39. Untrusted conversation/model text may not become executable structure.
    HTML uses trusted rendering, context-safe embedded-data encoding, and a
    restrictive CSP; detection of `<script>` is not the primary control.
40. Self-contained Share links carry validated structured data, never arbitrary
    caller-supplied rendered HTML or MIME authority.
41. The server owns Global Share MIME/content rendering and security headers.
42. “Self-contained” means not encrypted and not revocable after distribution;
    UI copy may not imply stronger privacy/deletion semantics.
43. A public/read share locator is a capability and is never written to logs in
    full; mutation uses a separate private capability.

## Feedback, contribution, retention, and training safety

44. Simple rating telemetry and content contribution are different pipelines.
    A thumbs-up/down must not silently persist full query/answer content.
45. Contribution consent is versioned, purpose-bound, and server-recorded;
    `consentFlag=true` from an unauthenticated caller is a client assertion, not
    verified identity or provenance.
46. Raw contributions are UNTRUSTED/QUARANTINED until validation, sensitive-data
    checks, provenance checks, and review/promotion policy complete.
47. Withdrawal from training and physical data deletion are different actions
    and must use different labels/semantics.
48. Raw sensitive contribution intake does not go directly into append-only
    Git/repository history when later physical deletion is a user promise.
49. Retention is bounded by data class and purpose; “keep forever” is never a
    default for user conversation content.

## Detection and secret lifecycle

50. Browser-entered bearer tokens, when supported for self-hosted compatibility,
    are page-memory-only: never localStorage, profile export, Share export, URL,
    prompt, or log.
51. Build-time Sphinx configuration may contain endpoint addresses but non-empty
    bearer/API credentials are ignored and never serialized into HTML, even if
    sourced from environment variables.
52. High-confidence secret detection on user-entered queries warns before
    network transmission; warning telemetry records only detector type/count,
    never the matched secret.
53. Structural isolation/authorization controls must remain safe when every
    detection regex is bypassed.
54. Every bundled inference hop independently revalidates prompt authority.
    A relay-generated `system` message is not trusted merely because it came
    from another project service; preserve the typed contract across bundled
    hops and reconstruct authority at the final model boundary.


55. Rating telemetry is opt-in at the browser boundary and contains only bounded rating/event mechanics; question, answer, note, model, page, and conversation/session identity are excluded before network transmission.
56. Server feedback normalization re-enforces the minimal telemetry schema even for direct/legacy callers. Client preferences cannot enable server persistence; durable telemetry is an independent operator opt-in and remains training-ineligible.
57. Explicit contribution requires the exact current consent version. `consentFlag=true` is evidence of a client assertion, not verified identity.
58. Raw contribution content lands in a bounded mutable receipt ledger before any versioned/mirrored training storage. Receipt identity, delete/withdraw authority, and independent review authority are separate capabilities.
59. Only independent review authority may set `trainingStatus=eligible`; callers and browsers cannot self-promote lifecycle state.
60. Client-supplied model/provider identity is labeled `client_reported` (historical unknown evidence is `legacy_unverified`) unless a separate server-observed provenance mechanism proves stronger evidence.
61. Normal training dataset construction fails closed to `_source=contribution` + `trainingStatus=eligible`. Telemetry/quarantined/unreviewed rows require explicit audit/recovery handling and never enter training implicitly.
62. Active-ledger content removal, post-promotion training withdrawal, provider current-view deletion, and forensic/global erasure are different guarantees. Never collapse them into one "delete everywhere" claim without provider-complete evidence.
63. The bundled memory ledger is bounded/process-local. The SQLite ledger is local transactional/restart-durable only; horizontally scaled production collection requires a shared transactional authority with deletion/retention evidence.


## Sensitive-egress preflight

64. Browser sensitive-egress preflight is advisory user protection, never authentication, authorization, provenance proof, or evidence that unmatched content is safe.
65. Preflight findings contain only safe category/count/codepoint metadata. Matching values and surrounding source text never enter finding state, logs, telemetry, or warning copy.
66. Scan the actual outbound object, including automatically attached page context and structured Share/contribution fields, rather than only the visible composer string.
67. Explicit Redact transforms an outbound operation copy. It does not silently rewrite the user's composer, transcript, canonical source page, or previously stored history.
68. If flagged data requires review and the warning UI cannot be presented, browser egress fails closed rather than silently sending.
69. Async privacy decisions are bound to the initiating conversation/operation identity so a stale dialog cannot publish into a later conversation.

## Global artifact tracking

70. Every Global Share link handed to the user enters one bounded lifecycle registry; creation-path or conversation changes may not make it silently unmanaged.
71. Cross-reload Global tracking may persist the public read capability in `sessionStorage`, but never the private edit/revoke capability, conversation snapshot/content, prompt, credential, or content hash.
72. Global artifact history is capped and session-scoped. Terminal server-confirmed states erase stored URL/UUID and retain only non-content lifecycle metadata.
73. New chat clears current update state, not previously provided Global artifact history or still-live page-memory revoke capability.
74. Global status checks are explicit user actions, never automatic polling. Current transport uses a fixed `/v1/share/status` POST with a bounded locator body, `no-store`, and no redirect following; legacy HEAD exists only for compatibility.
75. `404` means unavailable, not necessarily revoked. Label `revoked` only after this client receives a successful authenticated revoke operation (current fixed `/revoke` or legacy DELETE); label `expired` only from explicit expiry evidence.
76. Successful revoke may retain a local lifecycle tombstone until explicit Forget; Forget is local record deletion and is never remote revocation.

## Fail-closed Global lifecycle recovery

77. Treat Web Storage recovery as untrusted input. Rebuild Global recovery state from an explicit allowlist and destructively scrub unknown/legacy fields; never return a parsed recovery object wholesale.
78. `404`/`unavailable` is non-terminal unless independent saved expiry evidence has already matured. Keep the bounded public locator recheckable and do not erase a live page-memory edit capability merely because the server returned reason-unknown not-found.
79. Reason-unknown `404`, confirmed `410`, and successful revoke all detach the matching object from implicit current-conversation update state. Only confirmed terminal `expired`/`revoked` destroys tracked public capability material and the live edit capability.
80. Global Share expiry semantics must agree across status, read, update, and revoke routes. An expired update target must allow Create Global link to fall back to a fresh create; expired operations must not be reported as active/successful merely because stale storage remains readable.
81. A reason-unknown unavailable Global artifact that still has a page-memory edit capability must expose both remote Revoke and explicit local Forget semantics; repeated remote 404 must never trap the user in an unremovable local lifecycle row.



## Browser-origin, request-limit, and abuse-state parity

82. CORS is not authentication. Reject an explicit unapproved browser `Origin` before expensive/write handlers, but do not invent an Origin requirement for legitimate server-to-server clients; those callers still require the endpoint's actual authorization/capability.
83. Production bundled services default to exact origin allowlists. `ALLOWED_ORIGINS=*` may exist only as an explicit insecure compatibility escape hatch and must never silently become the default.
84. Request-size enforcement validates declared length when present and also counts bytes while streaming. Platform ingress limits and post-parse length checks do not replace the application streaming ceiling.
85. Attacker-controlled rate-limit identity tables are hard-bounded. A full table with no expired entries fails closed for a new identity rather than allocating unbounded state; cleanup/retraction paths do not receive unlimited write bypasses.
86. Distributed limiter semantics match the primitive: process-local maps and eventually-consistent KV/edge counters are abuse gates, not globally atomic quotas or accounting. Avoid repeated writes to one Workers KV key; strict deployment quotas require a shared/transactional control plane.

## Contribution receipt lifecycle and withdrawal

87. Receipt management authority remains meaningful after promotion: store only its server-side digest, clear raw review content from ledger state after promotion, and preserve authenticated lifecycle status/delete semantics until expiry/withdrawal.
88. Promotion and withdrawal transitions use atomic state claims. Read-then-write review logic that permits two reviewers to persist the same receipt is prohibited.
89. A withdrawal tombstone is privacy-minimal and keyed only by the server-owned contribution dedup key. Dataset last-write-wins must suppress the earlier eligible row and must never emit the tombstone itself as training content.
90. Provider deletion APIs prove only the scope actually observed. Removing a file from a current branch/view does not prove removal from version history, mirrors, caches, backups, or provider infrastructure.
91. SQLite `secure_delete`/WAL truncation are defense in depth, not a forensic-erasure certificate. User-visible deletion language must remain scoped to the active ledger/training/current-view guarantee that the implementation can prove.
92. `CONTRIBUTION_REQUIRE_DURABLE=true` is fail-closed. A memory compatibility backend may not silently satisfy a deployment that explicitly requires restart-durable receipt state.

93. Transient contribution lifecycle states owned by a dead process must be recoverable without granting duplicate ownership. Restart recovery may reclaim `promoting`/`withdrawing` only when replay semantics are idempotent/stable.
94. Promotion/withdrawal replay identifiers and provider paths derive from receipt-stable lifecycle data, not retry wall-clock time; a crash retry must not silently create a second logical artifact.
95. Terminal receipt tombstones are bounded by retention. Status history may not permanently consume the finite receipt ledger and become an intake denial-of-service vector.
96. Withdrawal suppression is checked at the provider write commit boundary (inside the per-target lock), not only before waiting for that lock; delayed mirror retries must not resurrect withdrawn eligible content.


## Share request-path capability isolation

97. Current generated Global Share URLs place the public read locator only in an exact `#share=<id>` URI fragment on the fixed `/v1/share` viewer path; never construct a current public URL as `/v1/share/<capability>`.
98. Current Share read/status/update/revoke traffic uses fixed endpoint paths and carries the bounded public locator in the request body. A public capability may not be reintroduced into request targets merely for convenience.
99. The fixed viewer treats fragment state as untrusted input: accept only the exact supported locator forms, keep response data non-executable, and insert untrusted conversation values with DOM text APIs rather than HTML interpretation.
100. Legacy capability-bearing path routes are compatibility debt, not the current transport contract. Track and retire them explicitly; do not use their existence to generate new path-bearing links.
101. Disabling platform invocation/request-URL logs is defense in depth, not a substitute for capability-safe transport and not proof that full request-body/WAF/packet telemetry is absent.


## Legacy Share compatibility drain

102. Every newly created Share and every successful current fixed-path update receives the server-owned current transport generation marker. Client input never selects or downgrades that generation.
103. Capability-bearing `/v1/share/{id}` compatibility may operate only on explicitly pre-generation objects. Missing generation metadata is legacy evidence; unknown/tampered values fail closed, and current-generation objects return not-found on legacy paths.
104. Deprecated path PATCH may not extend object TTL or refresh path-capability lifetime. Current mutation uses fixed `/v1/share/update`; old-object fixed update is the one-way migration to current generation.
105. Legacy compatibility is a draining set, never a growing set: current create/update cannot add an eligible legacy object, and revoke/expiry/migration only remove from that set.
106. Deprecation signaling follows protocol syntax and remains non-authoritative: RFC 9745 `Deprecation` is a Structured Field Date, `Sunset` is the object HTTP-date expiry, and successor links do not replace capability/auth checks.
107. Historical compatibility must not be used as evidence that current request-path capability exposure is acceptable. Remove the legacy route code after the pre-generation population has drained and preserve fixed-path transport as the only current contract.


## Distributed rate-limit authority

108. A rate limiter may be labeled distributed/authoritative only when all replicas in the claimed consistency domain consult one atomic decision plane; process-local maps, eventually-consistent KV observations, and local/permissive edge bindings are soft abuse controls only.
109. When deployment policy requires shared authority, backend absence/misconfiguration/runtime failure fails closed. Never silently degrade from Redis/Durable Object authority to independent local/KV counters while retaining an authoritative claim.
110. Shared rate-limit identity keys use a dedicated keyed pseudonym (HMAC-SHA256 or equivalent), not raw IP-like identity and not a reusable API/provider credential.
111. Rate-limit scope is explicit and bounded by route family. A shared identity budget for chat does not implicitly consume Share/feedback/contribution budgets unless policy deliberately defines one combined scope.
112. Public health/discovery may expose backend class, shared/authoritative/readiness state, and consistency scope, but never connection URLs, credentials, HMAC secrets, raw client identities, or internal shard names.
113. Distributed abuse limiting is not authenticated identity or billing-grade accounting. Claims are scoped to the configured Redis consistency domain or Durable Object shard semantics and must not be generalized to unrelated/Active-Active stores.


## Shared contribution receipt authority

114. Shared contribution receipt authority may be claimed only when every participating replica consults one atomic transactional receipt domain. Process memory and local SQLite remain process-local/single-instance boundaries respectively.
115. When deployment policy requires shared contribution authority, backend absence, configuration failure, initialization failure, or runtime authority failure fails closed for contribution intake/management; never silently degrade to memory/SQLite while retaining the shared claim.
116. Receipt identifiers written to shared infrastructure use a dedicated keyed pseudonym (HMAC-SHA256 or equivalent). Raw receipt capabilities, delete tokens, review tokens, Redis URLs and contribution HMAC secrets never become shared key names, public discovery data or logs.
117. A database/Redis lease cannot fence external Git/provider mutations. An expired or transport-ambiguous promotion claim becomes reconciliation-required uncertainty and must not be automatically reassigned for re-promotion.
118. Provider mutation transport/timeouts with unknown commit outcome are security-relevant ambiguity, not ordinary retry failure. Preserve non-promotable uncertainty until explicit reconciliation or privacy-safe withdrawal.
119. Participant withdrawal is monotonic across uncertain promotion/withdrawal states: safe recovery may continue toward `withdrawn`, but an uncertain origin must never be restored to `trainingStatus=eligible` merely because an operation lease expired.
120. Shared transactional coordination and persistence durability are separate claims. Redis shared/authoritative state does not prove AOF/RDB/replica/backup/crash durability; require independent deployment evidence before asserting that property.
121. Direct ledger methods, HTTP routes, background/recovery helpers and tests must enforce the same uncertainty invariant; no lower-level shortcut may reinterpret an expired promotion claim as ordinary quarantine/pending state.


## Portable self-contained Share and public-link delivery

122. Current self-contained Share generation uses a bounded exact `data:text/html;charset=utf-8;base64,...` artifact built only from the reviewed normalized snapshot. Base64 is transport encoding, never encryption or secrecy.
123. Current portable self-contained HTML is static/inert: no executable script, forms, frames, workers, external resource fetches, or clickable external navigation. Legacy c1/c2 readers remain compatibility-only and may not become the current generation path again without a new security review.
124. A copied self-contained data URL is non-revocable and may persist in clipboard/history/messages. User-visible copy must say so; local Forget/revoke controls cannot imply remote deletion.
125. Self-contained **Open** must attempt the exact canonical `data:text/html;charset=utf-8;base64,...` artifact and must not silently replace it with an origin-bound `blob:` URL. Browser policy may still block page-initiated top-level `data:` navigation; when blocked, tell the user to open the same copied data URL explicitly rather than falling back to Blob transport.
126. A successful Global Share create/update must surface one validated copyable public read URL and register it in artifact lifecycle management. If a compatible backend returns a valid UUID but omits the URL, synthesize only the configured fixed `/v1/share#share=<id>` viewer URL; never fabricate a link after an invalid/missing locator.
127. Package-owned browser origins required by the shipped UI must not be accidentally removable by an additive deployment override; validate additional origins as exact HTTP(S) origins, retain explicit `*` only as an insecure compatibility escape hatch, and expose only privacy-safe CORS health diagnostics.

## Reverse-proxy public Share origin and opaque browser origins

128. Public Share URLs must derive from deployment-owned public-origin evidence. On Hugging Face Spaces prefer validated `SPACE_HOST` over an internal HTTP ASGI base; never fix scheme mismatch by blindly trusting caller-controlled forwarding headers.
129. `Origin: null` is not proof of a trusted local file. If local-file Global Share is enabled, make it an explicit deployment opt-in and scope it to Share routes only; unrelated API routes remain denied.
130. Public health diagnostics may reveal whether opaque-origin Share compatibility is enabled, but never custom allowlist contents, tokens, or deployment secrets.

## Dataset contribution purpose separation and schema v4

131. Rating telemetry and dataset contribution are separate purposes. Enabling or sending privacy-minimal feedback telemetry never implies consent to transmit question/answer/conversation content.
132. Share/export and dataset contribution are separate control planes. Share must not own contribution submission, consent, receipt-delete/withdraw capability, or training/dataset endpoint actions; every contribution shortcut converges on the canonical contribution controller.
133. Current schema-v4 contribution requires consent version `2.0.0`. Legacy v2/v3 consent may remain compatibility evidence only for those legacy schemas and may not authorize the broader v4 conversation contract.
134. Whole-conversation contribution is exactly one ordered `recordType="conversation"` record. Only user/assistant dialogue is eligible; runtime/error/system/tool rows are not silently converted into training dialogue, and assistant model evidence is preserved per message.
135. Inspect JSON and sensitive-egress preflight consume the same canonical preflight object. Submission uses exactly the preflight result (`review.value`): unchanged on Continue or the explicit redacted copy on Redact. Never rebuild a second independent payload after review.
136. Content-bearing contribution envelopes exclude stable browser session/conversation identifiers, feedback event chains, endpoint credentials, Share capabilities, contribution management capabilities, browser storage keys, and raw URL query/hash material unless a future explicit schema/security review proves a need.
137. Quick-access contribution UI may reduce navigation friction but must not bypass explicit review, versioned consent, quarantine, independent promotion, or truthful delete/withdraw lifecycle semantics.


## Feedback telemetry explicit permission

138. Local rating UI is never network consent. Missing, malformed, stale-version, storage-inaccessible, or legacy boolean telemetry preference state fails closed to local-only behavior.
139. Every `/v1/feedback` network request requires both a browser-side explicit current permission and a server-side current consent marker/version/timestamp check. Operator `FEEDBACK_PERSIST_ENABLED` may permit storage but may not create user consent.
140. Turning telemetry Off stops all future feedback network traffic, including retraction housekeeping. Do not send a hidden final request after opt-out.
141. The public `ai-assistant-feedback` DOM event is a privacy-minimal rating event. It may not rebroadcast question, answer, written note, model, page, stable conversation identifier, endpoint credential, or contribution capability to arbitrary page listeners.
142. Feedback telemetry consent evidence is an application contract, not cryptographic proof of human identity. Store/emit only the minimum versioned evidence needed to enforce the official client/server purpose boundary; never reinterpret it as dataset-contribution consent.

## Recoverable create authority and browser privacy — Run 18 / B37

143. Network feedback telemetry permission and public page-integration event permission are separate grants. Neither may imply the other; both fail closed on absent, malformed, stale-version, or inaccessible permission state.
144. Recoverable Share/contribution CREATE identity is established before network transmission. A timeout, transport failure, or server 5xx is `outcome_unknown`; retry reuses the exact operation/resource/payload rather than silently starting a fresh CREATE.
145. Current browser CREATE flows keep the raw revoke/delete management capability client-side and send only a one-way digest. The server binds resource ID, operation ID, reviewed payload digest, and management-capability digest; any replay mismatch conflicts.
146. Long-lived management capability persistence is explicit user action. Do not silently put Share edit tokens or contribution delete/withdraw tokens in localStorage; use page memory and explicit export/import receipt mechanics.
147. Content displayed by contribution **Inspect JSON** is the semantic content considered for storage. Enforce identical limits before review or reject over-limit submissions; never silently truncate reviewed training content server-side.
148. Browser transcript restoration requires explicit per-tab permission and bounded/untrusted Web Storage parsing. Malformed, oversized, or excessive restore state fails closed. Device identifiers that need no cross-session persistence remain session-scoped.
149. Global Share storage claims are backend-specific and truthful: memory is process-local/non-durable; SQLite is restart-durable single-instance; Redis is shared but durable only with separate persistence evidence. Required durable/shared modes fail closed rather than degrading while retaining the claim.
150. Distributed expiry cleanup must not perform a stale read followed by an unconditional delete that can erase a recreated object. Use one atomic compare/expiry/delete/accounting operation or an equivalent version-fenced transaction.
151. Application logs never emit runtime endpoint userinfo, query strings, or fragments. Runtime endpoint URLs reject credential-like query parameter names and direct users to dedicated memory-only token fields; benign routing query parameters may remain supported.
152. Eventually-consistent storage is never described as atomic global create-once authority. Deterministic operation/resource identity can provide retry recovery without proving linearizable distributed mutation semantics.


## Run 19 / B38 — supply-chain and deployment hardening

153. An immutable container image digest proves reproducible base identity, not absence of vulnerabilities. Current advisory/image-scan evidence is a separate per-release gate and may never be inherited indefinitely.
154. The production Python dependency plane uses exact direct versions plus a complete hash-locked transitive runtime lock. Broad convenience extras and unconstrained resolver ranges are forbidden in the release image path.
155. Platform-specific wheel locks and container architecture are one contract. A Linux/amd64 lock must build from an explicitly Linux/amd64 base selection rather than a silently architecture-dependent multi-platform index.
156. Strict production runtime is non-root and fail-closed when root execution is detected. Installer/build tooling belongs to the builder plane; the runtime image contains only what is required to execute the service.
157. Docker build context is deny-by-default and a read-only/rootless/capability-dropped deployment reference is maintained. Adding source files to the image requires an explicit allowlist change and review.
158. Shared Redis authority has one transport-security policy across rate limiting, Share, and contribution receipts. Strict mode requires `rediss://`, rejects query-string downgrade knobs, and forces certificate plus hostname verification.
159. A checked-in SBOM must state its scope truthfully. A Python-lock SBOM may not be represented as a complete container/OS SBOM; full-image SBOM generation remains a release-artifact gate.
160. Advisory floors are monotonic security ratchets, not a substitute for a fresh vulnerability database scan. A newly disclosed vulnerability reopens the affected release gate regardless of previous GREEN evidence.
161. High/Critical dependency or image findings fail the release gate unless a specific, time-bounded, reviewed exception records reachability/risk/owner/expiry; suppressing scanner output is not remediation.
162. Health/discovery surfaces expose only coarse deployment-policy/readiness facts. Redis URLs, credentials, TLS internals, image registry credentials, package provenance tokens, scanner secrets, and raw advisory artifacts never become public diagnostics.

## Release evidence and production guardrails — Run 20 / B39

163. A production scan/SBOM/attestation is not release evidence until it is
     cryptographically content-addressed and bound to the exact source inputs and
     final immutable artifact subject being promoted.
164. Dependency evidence must bind both the exact lock and the exact application
     runtime source. An unchanged lock cannot make changed `app.py`/runtime helper
     bytes inherit an older GREEN result.
165. Release evidence is short-lived. Standard hardened promotion rejects stale,
     expired, future-dated, path-traversing, symlinked, hash-mismatched, or
     risk-exception-bearing evidence rather than treating it as advisory.
166. Provenance JSON alone is not signer verification. Bind SLSA/in-toto
     provenance to the final image subject and resolved base-image manifest, and
     retain a separate trusted signature-verification result.
167. Release-evidence manifests contain no Redis URL/host/user/credential, registry
     credential, request sample, user content, raw capability, token, cookie, or
     Authorization material. Operational artifacts are referenced by relative
     path + digest rather than copied into public/runtime diagnostics.
168. Browser feedback consent never authorizes infrastructure telemetry. Request
     body, credential/capability header, query-string, WAF/APM body capture, and
     third-party telemetry export are separately disabled/verified for production
     promotion.
169. Redis operational probes are explicit operator actions and emit only coarse
     non-secret facts. Do not broaden Redis ACLs merely so a probe can inspect
     privileged configuration.
170. TLS/shared Redis is not automatically durable. Share/Contribution production
     authority requires separate persistence, replication, backup/restore, and
     least-privilege evidence; rate limiting is not mislabeled durable user data.
171. Repository verification may prove evidence **binding**, never external truth.
     Fresh scanner, registry/signature trust-root, Redis provider and logging/WAF
     facts remain release/deployment evidence and reopen when stale or changed.
172. Release-evidence schema versions are closed protocols, not extensible metadata bags. Unknown fields, oversized manifests, invalid tool identity, and unsupported/old evidence formats fail closed.
173. Runtime-source subjects and Docker build context are one contract. Every Docker-eligible `_utils/` file is hashed; generated cache/bytecode is explicitly excluded from the build context and may never create nondeterministic release subjects.
174. A resolved platform base-image manifest is distinct evidence from a multi-platform image index. Reusing the index digest as the resolved manifest, or presenting a release manifest for a different runtime `PROXY_VERSION`, fails promotion.


## Separate-origin assistant isolation — Run 22 / B41

175. Configuring a non-empty assistant isolation origin is a fail-closed security mode, not a preference. The parent page must suppress the full same-origin runtime even if the host-bridge asset fails; handshake failure never silently falls back to same-origin execution.
176. The isolation origin is an exact browser origin: HTTPS in production (HTTP only for localhost development), no credentials/path/query/fragment, and runtime-distinct from the documentation origin. A same-origin iframe is not isolation.
177. Window `postMessage` is bootstrap-only. HELLO/INIT require exact protocol, channel, `event.origin` and `event.source`; after one successful INIT a transferred `MessagePort` becomes the sole runtime channel and the window listener is removed.
178. Every capability-port envelope is bounded, versioned, channel-bound and monotonically sequenced. Replayed/out-of-order/oversized/unknown-capability messages fail closed; capability names are an allowlist, never arbitrary RPC method names.
179. The documentation host exposes no arbitrary DOM capability. Page context is a bounded snapshot with active/form/assistant/hidden content removed, query/fragment-stripped page identity, and oversize HTML downgraded to bounded text rather than token-cut HTML. Canonical Markdown is a same-origin bounded fetch performed by the host.
180. Isolated-origin browser storage is namespaced by the validated parent origin for every assistant local/session-storage key. Do not depend on third-party storage partitioning to keep unrelated documentation origins from sharing transcript/preferences/profile state.
181. B40 page-integration permission still governs public lifecycle events in isolated mode. The frame may request only the already-bounded projection and the host independently revalidates event type/detail before re-emitting it on the parent document. Network telemetry permission never grants this capability.
182. The isolated frame is sandboxed and referrer-suppressed; production response headers must additionally constrain `frame-ancestors`, CSP `connect-src`, MIME sniffing and permissions. Meta CSP is a baseline, not proof of deployment headers.
183. Separate-origin mode protects assistant DOM/storage/transcript from ordinary documentation-origin scripts under SOP, but it does not make a fully compromised parent trustworthy. Parent-side page tampering, overlay/clickjacking/removal, pre-bootstrap API monkeypatching and deployment-origin compromise remain separate threats and must not be described as closed.
184. The host bridge snapshots and sanitizes configuration/endpoint descriptors at startup before asynchronous cross-origin handshake. Later mutation of page globals cannot alter INIT authority; secret-shaped and prototype-pollution keys (`__proto__`, `prototype`, `constructor`) are rejected and bootstrap objects use null-prototype maps. This narrows post-start mutation races but does not close `SEC-P1-42` for compromise before bridge initialization.

## Hostile-parent and egress hardening — Run 23 / B42

185. Bootstrap capability entropy belongs to the isolated compartment. Protocol v2 generates the channel with WebCrypto inside the frame after parent-policy validation; no capability appears in iframe URL/query/fragment and missing secure randomness fails closed rather than falling back to `Math.random()`.
186. The isolation parent allowlist is a build artifact with a closed schema and deny-all source default. Only exact validated origins are emitted; copying isolated assets without an appropriate generated policy must not create an open embedding surface.
187. A sandboxed isolated frame must never preserve script authority while self-navigating onto the documentation origin. Do not grant popup escape/top-navigation; intercept HTTP(S) frame navigation and use only external `_blank` navigation with `noopener,noreferrer`.
188. Assistant-service fetches omit ambient browser credentials by default. An explicit site-owner compatibility switch may permit at most `same-origin`; `credentials="include"` is not a supported escalation and caller options cannot bypass the central wrapper.
189. Canonical documentation fetch is a distinct page-content capability: same-origin only, redirect-error, cache-no-store and streaming-bounded. Do not generalize that credential allowance to assistant/model/share/feedback/contribution service calls.
190. Cross-origin microphone delegation is a separate default-Off site-owner authority. Do not infer it from speech features, browser permission state, telemetry consent or page-integration consent; suppress unavailable voice UI when delegation is absent.
191. Isolated browser storage is partitioned by both validated parent origin and normalized documentation root. One isolation service origin must not merge transcripts/preferences for multiple docs projects hosted beneath the same parent origin.
192. The host installs its native capture-phase handshake listener before attaching the isolated frame and consumes a valid HELLO before later page listeners. This narrows post-start observation races; compromise before host startup remains `SEC-P1-42`.

## Bounded remote response and context ingestion — Run 24 / B43

193. Response-body safety limits are enforced before and while bytes are consumed. A size check performed only after `response.text()`, `response.json()`, `Response.read()`, or equivalent whole-body buffering is not a memory-safety control.
194. Security-sensitive browser remote reads require a stream-capable transport. If `ReadableStream.getReader()` or equivalent pre-buffer accounting is unavailable, fail closed instead of silently selecting an unbounded compatibility fallback.
195. Declared `Content-Length` is advisory input that must be syntactically valid, bounded, and independently rechecked against actual streamed bytes. Missing length never disables the streamed byte ceiling.
196. Chat/SSE response ceilings apply to the total decoded byte stream; SSE additionally bounds an unterminated logical line so an attacker cannot keep one event line growing indefinitely below unrelated message-count limits.
197. Standalone Global Share viewers are independent clients and enforce their own response ceiling; they do not rely solely on server-side Share payload limits or same-origin trust.
198. Public health may expose the effective non-secret response byte ceiling for deployment diagnosis, but never upstream URLs, payload content, provider credentials, response excerpts, or security-sensitive headers.
