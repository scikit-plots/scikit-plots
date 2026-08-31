# B37 — Lifecycle and privacy closure

Status: **IMPLEMENTED — Run 18 working tree; exact-package acceptance pending**

## Trigger

Run 17 separated Share, rating telemetry, and dataset contribution, but a focused
post-release audit found lifecycle and browser-origin residuals that could still
violate user expectations without being classic code-execution vulnerabilities:

1. HF Global Share used process-local memory while the UI could describe the
   result as a server-backed Global link without distinguishing restart/replica
   durability;
2. lost Share/contribution CREATE responses could create orphan/duplicate
   server objects because the browser learned management authority only after a
   successful response;
3. contribution delete/withdraw authority was practically page-lifetime unless
   the user manually copied hidden values;
4. the public `ai-assistant-feedback` DOM event remained an egress-capable
   same-origin integration surface even when network telemetry was Off;
5. contribution Inspect JSON could diverge from retained server content through
   server-side truncation/record limits;
6. browser transcript recovery was available by default and trusted restored
   storage too broadly;
7. runtime endpoint query strings could carry secret-like material into browser
   state/logging surfaces;
8. Share privacy presets did not make Standard materially safer than Complete.

B37 closes those application-layer residuals without claiming that same-origin
JavaScript, provider history/backups, infrastructure full-body tracing, or an
unconfigured deployment becomes cryptographically private.

## Recoverable CREATE operation envelope

Share and contribution CREATEs now establish recovery identity **before** the
network request:

```text
browser
  operationId       := random
  resourceId        := random
  managementToken   := random (private)
  tokenDigest       := SHA-256(managementToken)
  payloadDigest     := canonical reviewed payload
        |
        v
CREATE sends operationId + resourceId + tokenDigest + payload
(raw managementToken does not cross CREATE)
        |
        v
server create-once binding
  resourceId
  operationId
  payloadDigest
  tokenDigest
        |
        +-- exact replay ------------------> same object/receipt
        +-- payload/token mismatch --------> 409 conflict
        `-- response lost/5xx -------------> outcome unknown
                                              retry SAME envelope
```

The raw management capability is deliberately absent from current browser
CREATE requests and responses. It remains in page memory or in an explicitly
saved management receipt. Legacy API clients that do not supply the envelope
remain compatible with the historical server-generated-token response.

`outcome_unknown` is a first-class UI state. A timeout/fetch failure/server 5xx
never means "definitely not created" and Retry never silently creates a fresh
operation while the previous result may exist.

The replay binding includes the management-token digest. Reusing the same
resource/operation/payload with a different locally held token fails rather than
returning an object that the caller cannot later manage.

## Management receipt recovery

Contribution success exposes explicit **Save management receipt** and **Import
management receipt** actions. The receipt is bounded structured JSON containing
only the management fields needed for later delete/withdraw operations. The
browser does **not** silently persist the raw management token in localStorage.

This is capability recovery, not account identity. Losing the only receipt can
still mean losing management authority; the UI must not imply password-reset or
server-side identity recovery that does not exist.

## Global Share storage truthfulness

HF Share state now uses a `ShareStore` abstraction:

| Backend | Restart-durable | Shared across replicas | Authority claim |
|---|---:|---:|---|
| memory | no | no | compatibility/process-local |
| SQLite | yes for configured local file | no | single-instance durable |
| Redis | shared | yes | shared; durable only when operator separately confirms persistence |

`SHARE_REQUIRE_DURABLE=true` and `SHARE_REQUIRE_SHARED=true` are fail-closed
release controls. Health/discovery reveals only coarse backend/readiness/
durable/shared/consistency state; connection URLs and credentials remain secret.

Redis expiry cleanup is atomic: the read/expiry/delete/accounting decision is
one Lua operation so an expired read cannot race with recreation and delete a
new object under the same public resource identity.

Cloudflare Workers KV remains explicitly **eventually consistent**. Deterministic
resource/operation identity makes ordinary lost-response retries recoverable,
but it is not documented as transactional global create-once authority. A
Durable Object or transactional service is required before making that stronger
claim.

## Browser-origin telemetry separation

Network rating telemetry permission and public page-integration events are now
independent permissions.

- local rating remains functional with both permissions Off;
- telemetry consent does not imply DOM-integration consent;
- `ai-assistant-feedback` is not dispatched unless the current structured page
  integration permission is explicitly enabled;
- malformed/stale/inaccessible permission state fails closed;
- when enabled, the public event remains rating-only and excludes Q&A, written
  note, model, page, conversation/session ID, endpoint/capability data and
  telemetry-consent evidence.

This closes the application-owned public event channel. It does **not** claim
isolation from arbitrary compromised same-origin JavaScript, which remains part
of the current browser trust boundary.

## Exact contribution review

Schema-v4 review and server intake use the same semantic bounds. User-reviewed
contribution content is rejected when it exceeds the record/message/text limits
instead of being silently truncated after Inspect JSON. Non-dialogue roles in
API conversation input retain the Run-17 compatibility behavior: they may be
accepted as input but are filtered from training dialogue rather than converted
into user/assistant messages.

## Browser residue and endpoint hygiene

- **Remember conversation in this tab** is an explicit per-tab opt-in.
- transcript restoration is bounded by storage bytes, entry count, and text
  length; malformed/oversized state is destructively ignored/cleared.
- microphone device identity is sessionStorage-scoped rather than persistent
  localStorage state.
- endpoint logging uses one safe URL representation: scheme + host + pathname,
  never userinfo/query/fragment.
- runtime endpoint URLs reject credential-like query parameter names such as
  API-key/token/password/authorization/client-secret/signature forms; ordinary
  non-secret routing queries remain compatible.
- Minimal / Standard / Complete Share presets now represent materially different
  metadata exposure levels.

## Verification — working tree

- B37 Python lifecycle/privacy contract: **6 passed**;
- B37 executable browser/source contract: **56/56**;
- focused compatibility + mutation plane after contract changes: **229 passed**;
- complete runnable non-Sphinx tree: **711 passed, 3 skipped**;
- browser + Worker JavaScript syntax: **GREEN**;
- proxy deployment version: **6.7.0**.

Additional working-tree release gates:

- JavaScript harness registry: **43 passed**;
- mutation + logging/privacy positive controls: **212 passed**;
- complete runnable non-Sphinx tree: **711 passed, 3 skipped**;
- Python compile across all packaged `.py`: **58 files GREEN**;
- Wrangler TOML: **GREEN**, `invocation_logs=false`;
- maintenance drift checker: **GREEN**;
- Sphinx-inclusive boundary: **1177 passed, 3 skipped, 5 failed, 62 errors**,
  all confined to the unavailable `sphinx` dependency;
- controlled diff from exact Run 17: **5 added, 31 modified, 0 removed = 36
  paths**;
- pre-package tree membership: **254 files**.

Clean-extraction acceptance and final delivery SHA-256 are recorded only after
the package cycle.

## Deliberate residuals

B37 does not paper-close:

- `SEC-P0-10` reproducible/rootless/container/SBOM/CVE supply-chain hardening;
- provider history/backups/cache/global physical-erasure evidence;
- infrastructure WAF/full-body tracing and access-log ownership;
- production Redis TLS/ACL/persistence/replication evidence;
- Cloudflare KV eventual-consistency limitations;
- same-origin JavaScript as a browser privacy trust boundary;
- representative maintained Playwright/WebDriver cross-origin/storage/WebCrypto
  E2E;
- the local verification environment's missing `sphinx` dependency.

## Candidate package acceptance

**GREEN — clean independent extraction.** Candidate bytes reproduce 6 B37
Python tests, 56/56 B37 browser assertions, 43 Node harnesses, 212 mutation/
privacy controls, 711 passed + 3 skipped runnable non-Sphinx tree, 58-file
compile, syntax/TOML/maintenance GREEN, and Sphinx boundary 1177/3/5/62 with all
failures/errors confined to missing `sphinx`. Archive hygiene is 254 files,
exact two-root layout, and zero cache/bytecode contamination.

## Metadata-bearing prefinal acceptance

**GREEN.** Independent extraction reproduced the combined focused plane (261
passed) plus 56/56 B37 browser assertions, 711 passed + 3 skipped runnable
non-Sphinx tree, 58-file compile, syntax/TOML/maintenance GREEN, 254-file
cache-clean two-root archive hygiene, and the same 1177/3/5/62 missing-Sphinx
boundary.

## Final delivery acceptance contract

B37 is marked complete only for a delivery ZIP whose independent final-byte
extraction reproduces 6 B37 Python, 56/56 B37 browser, 43 Node, 212 mutation/
privacy, 711+3 runnable suite, 58-file compile, syntax/config/maintenance GREEN,
1177/3/5/62 missing-Sphinx-only boundary, and 254-file exact two-root/cache-clean
ZIP hygiene. Final SHA-256 is external; a failed final-byte gate invalidates the
release claim.
