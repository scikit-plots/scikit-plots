# `_sphinx_feedback`

Independent, theme-tolerant page feedback for Sphinx and reusable static HTML.

The core invariant is: **a feedback event represents a page reaction, never a person**.
The Sphinx adapter, browser controller, request/event contracts, storage coordinator,
and standalone ASGI transport are owned by this package. They do not import AI Learn
or AI Assistant. Applications may host this current page-feedback service at
`/v1/feedback`; that route accepts only `page.feedback-request.v1` and has no Assistant
rating-telemetry or legacy-contract fallback.

## Privacy and authority defaults

- No network request occurs on page view.
- The browser creates no cookies, visitor IDs, fingerprints, or `localStorage` IDs.
- Browser URL/query/fragment, referrer, user agent, device, account, locale, timezone,
  screen dimensions, and network identity are not request or durable-event fields.
- `page_id` comes from the canonical Sphinx `pagename`, not `window.location`.
- Feedback IDs are independent 192-bit CSPRNG event nonces (`feedback-` + 48 lowercase
  hex characters), never participant identifiers.
- Ambiguous retries reuse the exact request; pending state exists only in
  `sessionStorage` and is bound to site, page, revision, and configured endpoint
  authority. A changed endpoint cannot silently receive an old pending event.
- Unknown fields, duplicate JSON object keys, non-finite JSON numbers, malformed page
  IDs, control-character credit, and oversized requests fail closed.
- Request receipts are accepted by the browser only when the feedback ID, status, and
  SHA-256 commitment match the locally canonicalized request. Storage/provider adapters
  independently invert the durable event back to its unique normalized request and verify
  the same commitment before accepting authority.
- Provider credentials are server-only. `conf.py` contains public transport/config
  data, never GitHub/Hugging Face/GitLab/Bitbucket tokens.
- Abuse control identity is separate from feedback identity. The bundled ASGI adapter
  stores only a process-secret HMAC pseudonym in its local rate limiter, not a raw IP.
  Hosting providers and reverse proxies may independently keep access logs; this
  extension cannot govern infrastructure logs outside its process.

## Sphinx configuration

```python
extensions += ["_sphinx_ext._sphinx_feedback"]

feedback_page_enabled = True
feedback_site_id = "my-docs"
feedback_position = "sidebar"  # auto | sidebar | main-bottom | floating | none
feedback_page_main = True  # synchronized second view, not a second controller
feedback_position_fallback = "main-bottom"
feedback_detailed_enabled = True
feedback_buttons_ratings = {
    "left_button_rating": "left",
    "right_button_rating": "right",
}
feedback_counter_enabled = True
feedback_counter_source = "embedded"
feedback_aggregate_file = "_feedback/aggregate.json"  # beside conf.py; "" = no counters
feedback_endpoint = "https://feedback.example.org/v1/feedback"
```

`_example_conf.py` in this directory assigns every `feedback_*` value with its
default, allowed values and purpose, and shows the AI-assistant, two-site and
service variants; the test suite builds a site from it.

`feedback_aggregate_file` is a build-time snapshot selector, not a browser fetch
URL. It has two forms, and neither may leave its directory:

- `/name.json` (leading `/`) resolves beneath this extension's `_static/` asset
  root. That file is shipped with the extension and therefore shared by every
  site that installs it; the shipped `page-feedback-aggregate.json` is the
  `scikit-plots-learn` snapshot.
- `dir/name.json` (no leading `/`) resolves beneath the site's own
  documentation source directory (Sphinx `confdir`; the source directory when
  the build has no `conf.py`). Use this for any other site.

Blank means no snapshot: counters stay hidden. The snapshot's `site_id` must
equal `feedback_site_id`, or the build stops with a configuration error.

Automatic placement uses semantic/theme-compatible candidates and falls back to the
main content/body rather than disappearing on an unknown theme. `.. feedback::` is an
explicit mount escape hatch. If both sidebar and main-bottom mounts exist, they observe
one controller: one busy state, one nonce, one request, one accepted/retry state.

`feedback_endpoint` is explicit and independent from AI Assistant endpoint profiles.
This prevents a chat/share-only profile from silently becoming reviewed-feedback authority.
Deployments that colocate services may point both systems at the same public base, while
others can use a dedicated feedback service. AI Learn pages can be excluded to avoid
duplicating their section/generation feedback.

Embedded counters are offline build data. Sparse V3 aggregates keep a missing
page as unknown, so no counter is shown. A producer that knows it has a complete
reviewed snapshot may write `"complete": true`; only then may Sphinx safely render a
missing page as `0` negative, `0` positive, and `0 ratings`, matching AI Learn's known-zero
experience without fabricating zeros from incomplete data. The default design intentionally
has no live page-view counter fetch. The only supported aggregate contract is
`page.feedback-aggregate.v3`, which stores `count`, `score`, `positive_count`,
`negative_count`, and `neutral_count` directly from reviewed events. This is required
for trustworthy per-button counts because a `-5..+5` score plus total count cannot be
reverse-engineered into positive/negative event counts. `write_aggregate(..., complete_snapshot=True)` emits
`"complete": true` only when the caller explicitly certifies full site/revision coverage.
V3 may optionally carry `page_revision`; when
`feedback_page_revision` is configured it must match exactly, so older-revision feedback
is not silently presented as current feedback.


### Quick-button counter placement

The compact thumbs controls expose the reviewed per-sign count as an independently
configurable presentation detail. The public Sphinx setting is
`feedback_buttons_ratings`; use the distinct `left_button_rating` key for the
thumbs-down button and `right_button_rating` for the thumbs-up button. Each value is
`"left"` or `"right"`. A partial dictionary inherits the default for the omitted
button, while unknown keys or values fail the build instead of silently drifting.

The balanced default keeps the counts on the outside edges:

```python
feedback_buttons_ratings = {
    "left_button_rating": "left",
    "right_button_rating": "right",
}
```

```text
[0 | 👎]   [👍 | 0]   [⌄]
[0 | 👎 Not helpful]   [👍 | 0]   [⌄]
[0 | 👎]   [👍 Helpful | 0]   [⌄]
```

Both counts can instead follow their icons:

```python
feedback_buttons_ratings = {
    "left_button_rating": "right",
    "right_button_rating": "right",
}
```

```text
[👎 | 0]   [👍 | 0]   [⌄]
```

Or both can precede their icons:

```python
feedback_buttons_ratings = {
    "left_button_rating": "left",
    "right_button_rating": "left",
}
```

```text
[0 | 👎]   [0 | 👍]   [⌄]
```

The placement changes only visual ordering. Accessible button names, rating values,
selected state, transport payloads, counters, retry/idempotency semantics, and storage
contracts are unchanged. Logical CSS borders (`inline-start`/`inline-end`) preserve the
divider correctly in both LTR and RTL layouts. Because Python dictionaries cannot hold
two copies of the same key, do not repeat `left_button_rating`; use the distinct
left/right keys shown above.

## Standalone service

The package includes a dependency-free ASGI adapter:

```bash
uvicorn _sphinx_ext._sphinx_feedback._service.app:app
```

The reusable service defaults to **no storage targets**. A deployment must explicitly
choose a review/storage mode. SQLite is the deterministic local/private backend:

```bash
FEEDBACK_REVIEW_MODE=sqlite
FEEDBACK_SQLITE_PATH=/srv/feedback/feedback.sqlite3
FEEDBACK_ALLOWED_SITE_IDS=my-docs
```

`FEEDBACK_ALLOWED_SITE_IDS` is an optional exact server-side allowlist. It prevents a
direct client from changing the browser-supplied `site_id` to pollute another logical
site; any other `site_id` is answered with `422 site_not_allowed`. The Scikit-Plots proxy
binds its generic feedback route to `scikit-plots-learn,scikit-plots` by default (the
`scikit-plots-learn.readthedocs.io` and `scikit-plots.github.io` sites). Blank preserves
generic multi-site compatibility.

## Supporting any site

A site works when three values agree, with or without the AI assistant (this package
imports neither the AI assistant nor AI Learn, and a build may list both extensions):

| Site (`conf.py`) | Service (environment) | Failure when they disagree |
| --- | --- | --- |
| `feedback_site_id` | listed in `FEEDBACK_ALLOWED_SITE_IDS` (when set) | `422 site_not_allowed` on submit |
| site origin, e.g. `https://user.github.io` (no path) | listed in the CORS allowlist | browser blocks the response |
| `feedback_site_id` | `site_id` in the `feedback_aggregate_file` snapshot | build stops at `config-inited` |

The Scikit-Plots sites are configured this way:

| Site | `feedback_site_id` | `feedback_aggregate_file` |
| --- | --- | --- |
| `https://scikit-plots-learn.readthedocs.io/en/latest/` | `scikit-plots-learn` | `/page-feedback-aggregate.json` (packaged) |
| `https://scikit-plots.github.io/dev/` | `scikit-plots` | `_page_feedback/aggregate.json` (in `docs/source`) |

For high-assurance deployments, an optional local authority manifest can also bind
accepted pages (and optionally exact revisions) without adding browser secrets or page-view
network calls:

```bash
FEEDBACK_PAGE_AUTHORITY_FILE=/srv/feedback/page-authority.json
```

```json
{
  "contract": "page.feedback-authority.v1",
  "sites": {
    "my-docs": {
      "index": "",
      "guide/install": "rev-42"
    }
  }
}
```

An empty revision means “page existence is authoritative, revision is not pinned.” A
non-empty revision must match exactly. An explicitly configured empty manifest authorizes
zero pages; it never falls back to unrestricted mode. The manifest is server-side policy,
not participant identity.

SQLite mode is local-only: every configured target must use SQLite, so selecting the
local/private mode cannot silently acquire a network mirror. SQLite uses WAL, full
synchronization, a bounded busy timeout, lock-aware first-open initialization, and an
idempotent primary key on the event nonce. Readback cross-checks the stored feedback ID,
site/page index metadata, request commitment, and canonical event JSON before aggregation.
No participant, network-identity, or timestamp column is stored by this provider.

GitHub pull-request review is implemented as the external reviewed provider. Its
credential fallback is target-scoped and server-only:

```text
FEEDBACK_GITHUB_TOKEN
→ GITHUB_TOKEN
→ AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR
```

Hugging Face, GitLab, and Bitbucket are understood by the forward-compatible target
schema, but service configuration rejects them in this release because their write
adapters are not yet implemented and adversarially tested. The service never pretends
that an unsupported provider is safe or available.

Provider topology is deliberately simple: exactly one primary is authoritative; zero
or more mirrors are durability-only. Mirrors fan out concurrently and are bounded by
`FEEDBACK_MIRROR_TIMEOUT_SECONDS` (8 seconds by default); a primary success plus mirror
failure/timeout is accepted with a degraded mirror receipt. If the parent request is
cancelled after primary acceptance, unfinished mirror tasks are cancelled and joined rather
than left detached. A primary failure is never converted into success by a mirror. Provider
paths derive from a digest of
`site_id + page_id`, not a browser-controlled repository path. The convenience GitHub
environment shorthand keeps review links private by default; deployments that intentionally
want public provider links can opt in with an explicit storage-target registry.

For cross-origin static sites, configure an exact `FEEDBACK_ALLOWED_ORIGINS` allowlist.
Only HTTPS origins (plus localhost HTTP for development) are accepted; credentialed
CORS is not enabled. Blank means no CORS headers.

The standalone ASGI adapter uses the direct peer address for abuse control by default.
Behind a reverse proxy, opt in to forwarded-address processing only by declaring the
proxy networks you actually control:

```bash
FEEDBACK_TRUSTED_PROXY_CIDRS=10.0.0.0/8,2001:db8:1234::/48
```

`X-Forwarded-For` is ignored unless the immediate ASGI peer belongs to one of those
networks. The chain is then walked right-to-left until the first untrusted hop. Malformed
forwarding metadata falls back to the direct peer, so it may over-limit but cannot create
attacker-controlled rate identities. Literal or collectively equivalent whole-address-family
trust (for example two complementary `/1` networks) is rejected. Only declare proxies that
you control and configure them to overwrite or safely append forwarding metadata; trusting a
proxy that simply relays attacker-supplied `X-Forwarded-For` defeats any downstream parser.
This setting does not make the process-local limiter global across workers.

## Idempotency and broken-pipe recovery

The canonical request commitment is SHA-256 over normalized, key-sorted UTF-8 JSON.
Semantics are:

```text
same feedback_id + identical request  -> replay success
same feedback_id + different request  -> conflict
new feedback_id                        -> independent event
```

The GitHub adapter uses deterministic event paths/branches and reconciles ambiguous
failures after branch creation, event commit, and pull-request creation. Repository event
files are deterministic, key-sorted, two-space-indented UTF-8 JSON with a final newline so
maintainers can review them comfortably. Formatting is deliberately not event identity:
the adapter strictly decodes existing repository JSON (rejecting duplicate keys and
non-finite numbers), validates the event contract, compares canonical event semantics, and
then verifies the exact one-file review-branch diff before treating a retry/race as success.
This preserves idempotent replay for older compact one-line event files while different
durable content still conflicts. Redirects, over-fragmented responses, and oversized
provider responses are rejected. The branch scope is rechecked after pull-request
creation/reconciliation so a mid-flight branch mutation cannot produce a success receipt.
The deterministic review branch is also forbidden from colliding with the configured base
branch, preventing a maliciously chosen event nonce from turning reviewed publication into
a direct base write. The GitHub base branch name `feedback` is rejected at startup as well,
because Git ref namespace rules make it incompatible with every deterministic
`feedback/<nonce>` branch.

## Current boundary

The browser core is intentionally usable without Sphinx-specific markup, but non-Sphinx
hosts must provide their own explicit `site_id`, `page_id`, endpoint, and mount config.
The Sphinx package remains the adapter that supplies those values automatically.

Without `FEEDBACK_PAGE_AUTHORITY_FILE`, the server validates canonical page identifiers
but does not prove that every submitted page currently exists. `FEEDBACK_ALLOWED_SITE_IDS`
closes cross-site authority, while the optional manifest closes page/revision authority.
Neither mechanism authenticates a person; both constrain what anonymous page reaction the
service is willing to accept.


### Abuse-control deployment note

The bundled standalone limiter and the proxy retry accelerator are intentionally in-memory,
process-local controls. They never become feedback identity. Multi-worker/public deployments
that need a globally authoritative new-event quota should enforce that quota at a shared
edge/Redis layer; the Scikit-Plots proxy already keeps Redis authoritative when configured.

## Quick-action icon rendering

The compact quick controls use namespaced inline SVG geometry for thumbs-down, thumbs-up,
and the details chevron so their primary appearance is stable across operating systems and
emoji fonts. The SVGs are created with the DOM namespace API (not `innerHTML`), are marked
`aria-hidden`, and never replace the buttons' accessible names.

Unicode `👎`, `👍`, and `⌄` remain a fail-safe fallback if SVG DOM creation is unavailable or
throws. The detailed `-5…+5` scale intentionally keeps its expressive emoji faces. No icon
font, CDN, image request, AI Learn stylesheet, or other cross-extension asset dependency is
introduced.
