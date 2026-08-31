# Separate-Origin Assistant Isolation (B42)

## Goal

Run assistant UI, transcript, management receipts, runtime model state, and
network capabilities on a browser origin that is different from the
Documentation origin. The documentation page retains only a narrow bounded
page adapter. This is an optional confidentiality boundary; same-origin mode
remains available for compatibility.

## Enable

```python
ai_assistant_isolation_origin = "https://assistant.example.com"
ai_assistant_isolation_frame_path = "/ai-assistant-isolated.html"
ai_assistant_isolation_context_max_chars = 200_000

# Exact documentation origins allowed to embed/handshake with the frame.
ai_assistant_isolation_parent_origins = ["https://docs.example.com"]

# High-trust cross-origin device permission; OFF by default.
ai_assistant_isolation_allow_microphone = False

# Assistant API traffic omits ambient browser cookies by default.
ai_assistant_allow_credentialed_fetch = False
```

If `ai_assistant_isolation_parent_origins` is empty, the build may derive one
exact parent origin from `html_baseurl` / `ai_assistant_base_url`. If it cannot,
the generated parent-origin policy denies all parents. Do not rely on implicit
runtime discovery for production.

Publish the extension `_static/` assets on the isolated origin, including
`ai-assistant-isolation-policy.json`. The source copy of that policy is
intentionally deny-all. A successful Sphinx build writes the deployment policy
into the built `_static/` directory.

## Protocol 2.0.0

1. The parent snapshots/sanitizes bootstrap config and endpoint descriptors.
2. The parent registers its capture-phase HELLO listener **before** attaching the
   iframe. The iframe URL contains protocol + parent origin, but no capability
   nonce.
3. The isolated frame fetches its same-origin parent-origin policy with
   `credentials="omit"`, validates a closed schema, verifies its own isolation
   origin, and verifies the exact parent origin is allow-listed.
4. The frame generates a 128-bit channel nonce using WebCrypto. If secure random
   generation is unavailable, isolation fails closed; there is no `Math.random`
   fallback.
5. The frame sends HELLO. The parent verifies exact `event.source`, exact origin,
   protocol, and nonce shape, then stops propagation so later parent listeners
   cannot observe the nonce.
6. The parent transfers one `MessagePort`; window-level messaging ends.
7. Port sequence numbers must be exactly monotonic (`previous + 1`), and every
   bounded message must match the allowed capability/request contract.

A parent script compromised **before** the host bridge installs its listener is
still inside the explicit hostile-parent residual. B42 prevents later parent
scripts from simply reading the nonce out of `iframe.src` or observing a normal
HELLO after the bridge owns the event.

## Capability boundary

The host accepts only:

- `page.context.read`
- `page.canonical.read`
- `page.print`
- `ui.resize`
- `page.integration.emit`

No host capability returns transcript, model responses, Web Storage, bearer tokens,
Share/contribution management capabilities, or arbitrary DOM access.

Canonical Markdown is a deliberate documentation-origin read and therefore may
use same-origin documentation credentials. It is stream-bounded before it is
returned to the isolated frame. This is distinct from assistant-service traffic.

## Ambient network authority

Assistant-service requests use `credentials="omit"` by default, including when
an endpoint happens to be same-origin with the assistant frame. Ambient cookies
or browser session state are not silently treated as API authorization.

`ai_assistant_allow_credentialed_fetch=True` is a site-owner compatibility
opt-in and permits only `same-origin`; the central service-fetch wrapper never
permits `credentials="include"`. Explicit caller `omit` remains `omit`.
Production deployments should prefer server-side authorization and keep this
flag False.

## Navigation boundary

The iframe needs both `allow-scripts` and `allow-same-origin` for its isolated
origin storage/CORS model. That combination means **frame-self navigation must
not be allowed to move the browsing context onto the documentation origin**.
B42 therefore:

- does not grant `allow-top-navigation`;
- removes `allow-popups-to-escape-sandbox`;
- intercepts live HTTP(S) anchor navigation in the isolated frame;
- resolves relative links against the documentation page context;
- opens user-selected HTTP(S) destinations in `_blank` with
  `noopener,noreferrer`, leaving the isolated frame on its own origin.

Do not add frame-self `location.assign`, `location.replace`, `window.location =`,
or `_self` HTTP(S) navigation without redesigning this boundary.

## Device permissions

Rendering speech UI does not automatically delegate microphone permission to a
cross-origin frame. `ai_assistant_isolation_allow_microphone=False` is the
default. When False, the iframe `allow` attribute omits microphone and the
isolated runtime hides speech UI. Enabling it still requires normal browser/user
permission and should be done only when voice input is intentionally offered.

## Context and storage partitioning

The host strips active/form/assistant/hidden nodes and dangerous attributes
before serializing visible page context. Page identity and docs-root metadata
cross without URL query or fragment material. Oversized HTML degrades to bounded
plain text; canonical Markdown is stream-bounded.

Isolated Web Storage is namespaced by both the exact parent origin and normalized
documentation-root path. This prevents accidental state mixing between multiple
documentation projects hosted under different roots on the same origin. It is
not a defense against a parent already controlled by an attacker.

## Required production headers

The shipped HTML contains a restrictive baseline meta CSP, but production must
also send real response headers from the isolated origin:

- `Content-Security-Policy`: preserve `default-src 'none'`, restrict
  `script-src`/`style-src` to self, restrict `connect-src` to the exact deployed
  proxy origins, and set `frame-ancestors` to exact documentation origins.
- `Referrer-Policy: no-referrer`.
- `X-Content-Type-Options: nosniff`.
- `Permissions-Policy`: deny capabilities not intentionally delegated.

Do not set `frame-ancestors *`. Browser CSP does not provide a generally usable
`navigate-to` allowlist in the current CSP directive set, so B42 keeps the
application-level frame-self navigation guard in addition to sandbox/header
requirements.

## Residual boundary

Separate-origin isolation materially protects assistant-origin transcript,
storage, and capabilities from ordinary or **later** compromised documentation
scripts. It cannot make a fully compromised parent page trustworthy: code that
runs before the host bridge can modify page content, observe/interfere with the
bootstrap, clickjack/remove the frame, monkeypatch browser primitives before
they are snapshotted, or deny service. Real `frame-ancestors`, CORS, CSP,
Permissions-Policy, and deployed-origin truth also remain production evidence,
not claims that static source can prove.
