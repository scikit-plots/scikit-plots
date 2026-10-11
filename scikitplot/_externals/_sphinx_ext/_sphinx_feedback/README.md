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

Whether the deployed proxy accepts each site is visible without a submission:
`GET https://scikit-plots-ai.hf.space/health` returns `page_feedback.default_site_ids_allowed`
(`{"scikit-plots-learn": true, "scikit-plots": true}` once the Space runs this release or
sets `FEEDBACK_ALLOWED_SITE_IDS=scikit-plots-learn,scikit-plots`). Other site IDs a
deployment adds are counted (`allowed_site_id_count`), never listed.

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

GitLab (gitlab.com and self-managed) and Bitbucket Cloud run the same reviewed-change
engine as GitHub (`_service/_forge.py`): one branch, one immutable event file, one
review, the branch-scope proof, and reconciliation after a broken connection at every
step. Hugging Face is understood by the target schema but refused in this release; reach
it, and Bitbucket Data Center, through the `webhook` or `custom` provider. The service
never pretends that an unsupported provider is safe or available.

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
`FEEDBACK_ORIGIN_POLICY` (same values as `feedback_endpoint_policy`, default `strict`)
decides which origins may be listed: `strict` accepts HTTPS on port 443 and loopback
HTTP(S); `private-network` adds intranet HTTP and HTTPS on any port. Credentialed CORS is
not enabled. Blank means no CORS headers; a same-origin endpoint needs none.

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

## Endpoints and network policy

`feedback_endpoint` takes three forms, checked at build time by `feedback_endpoint_policy`
(the service applies the same rule to `FEEDBACK_ALLOWED_ORIGINS` through
`FEEDBACK_ORIGIN_POLICY`):

| Form | Example | `strict` (default) | `private-network` | `any` |
| --- | --- | --- | --- | --- |
| same-origin path | `/v1/feedback` | yes | yes | yes |
| HTTPS, port 443 | `https://feedback.example.org/v1/feedback` | yes | yes | yes |
| loopback HTTP(S), any port | `http://localhost:8000/v1/feedback` | yes | yes | yes |
| HTTPS, other port | `https://feedback.corp:8443/v1/feedback` | no | yes | yes |
| HTTP, private host | `http://feedback.internal:8080/v1/feedback` | no | yes | yes |
| HTTP, public host | `http://feedback.example.org/v1/feedback` | no | no | yes |

A private host is decided lexically, with no DNS lookup: RFC 1918, CGNAT (`100.64/10`),
link-local and unique-local IP literals; single-label names (`feedback`); names under
`.internal`, `.local`, `.localhost`, `.home.arpa`; and names under the suffixes listed in
`feedback_private_host_suffixes` (service: `FEEDBACK_PRIVATE_HOST_SUFFIXES`), for example
`[".corp.example"]`. Credentials, queries and fragments are refused under every policy.

Pick by deployment:

- **Public site, hosted service**: `strict`, HTTPS endpoint, origin in the allowlist.
- **Docs and service behind one server** (reverse proxy, intranet portal, offline
  laptop): `strict` with the same-origin path `/v1/feedback`; no CORS at all.
- **Intranet, internal CA or plain HTTP**: `private-network`.
- **Anything else**: `any`, knowing comments then cross networks unencrypted.

Browsers block an `http://` endpoint on an `https://` page (mixed content) whatever the
policy says.

## Storage providers

Exactly one primary is authoritative; up to four mirrors are durability-only. Configure
with `FEEDBACK_STORAGE_TARGETS` (a JSON list) or a one-target shorthand.

| Provider | Where events go | Network | Shorthand (`FEEDBACK_REVIEW_MODE` + env) |
| --- | --- | --- | --- |
| `sqlite` | one SQLite file | none | `sqlite` + `FEEDBACK_SQLITE_PATH` |
| `git` | one commit per event in a local working tree | none | `local` + `FEEDBACK_GIT_REPOSITORY_PATH` (+ `FEEDBACK_GIT_PATH`) |
| `github` | branch + reviewed pull request on github.com or GitHub Enterprise | HTTPS | `provider-pr` + `FEEDBACK_GITHUB_REPOSITORY` (+ `FEEDBACK_GITHUB_API_URL`) |
| `gitlab` | branch + merge request on gitlab.com or self-managed GitLab | HTTPS | `provider-pr` + `FEEDBACK_STORAGE_TARGETS` |
| `bitbucket` | branch + pull request on Bitbucket Cloud | HTTPS | `provider-pr` + `FEEDBACK_STORAGE_TARGETS` |
| `webhook` | HMAC-signed POST to any receiver | HTTP(S) | `custom` + `FEEDBACK_WEBHOOK_URL` (+ `FEEDBACK_WEBHOOK_ALLOW_HTTP`) |
| `custom` | your own Python adapter (`factory: "pkg.mod:make"`) | yours | `custom` + `FEEDBACK_STORAGE_TARGETS` |
| `huggingface`; Bitbucket Data Center | not implemented | | use `webhook` or `custom` |

Review modes bind provider families: `sqlite` (all SQLite), `local` (all SQLite or git:
the service makes no network request), `provider-pr` (primary opens a reviewed
pull/merge request), `custom` (any implemented provider). Each target accepts only its own
keys, so a key meant for another provider fails at start instead of being ignored:

```json
[
  {"id": "review", "provider": "github", "role": "primary", "repo": "org/docs-feedback",
   "api_url": "https://ghe.example/api/v3", "paths": {"feedback": "feedback"}},
  {"id": "archive", "provider": "webhook", "role": "mirror",
   "url": "https://feedback-ingest.example.workers.dev/v1/events"},
  {"id": "warehouse", "provider": "custom", "role": "mirror",
   "factory": "acme_feedback.store:make", "options": {"table": "page_feedback"},
   "token_env": "FEEDBACK_WAREHOUSE_TOKEN"}
]
```

Defaults: `api_url` `https://api.github.com` (github), `https://gitlab.com/api/v4`
(gitlab), `https://api.bitbucket.org/2.0` (bitbucket, the only accepted value); token
aliases `FEEDBACK_GITHUB_TOKEN` → `GITHUB_TOKEN` → `AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR`,
`FEEDBACK_GITLAB_TOKEN` → `GITLAB_TOKEN` (sent as `PRIVATE-TOKEN`),
`FEEDBACK_BITBUCKET_TOKEN` → `BITBUCKET_TOKEN` (an access token, sent as
`Authorization: Bearer`); `webhook` secret `FEEDBACK_WEBHOOK_TOKEN`; `paths.feedback`
`feedback`; base `branch` `main` (never `feedback`). GitLab `repo` may name nested groups
(`group/subgroup/project`). Review URLs must be on the forge's own host and under the
configured repository. `git` needs an absolute `repository_path` that is the top of a
working tree; commits use the identity `sphinx-feedback <sphinx-feedback@localhost>` and
leave pushing to you. `custom` options are short non-secret scalars; secrets come from
`token_env`. The adapter contract is in `_service/_custom.py`.

## Webhook delivery and Cloudflare

The `webhook` provider POSTs `page.feedback-delivery.v1`, canonical JSON
`{"contract", "event", "feedback_id", "request_hash"}`, with
`X-Feedback-Signature-256: sha256=<HMAC-SHA256(secret, body)>` and
`X-Feedback-Delivery: <feedback_id>`. The receiver answers 2xx
`{"status": "accepted"|"replay", "feedback_id", "request_hash"}`, or 409 for a feedback ID
stored with different content. It must be idempotent: the browser retries with the same
feedback ID. Redirects are not followed. Python receivers can use `_service._webhook.verify`.

Cloudflare, by role:

- **Receiver/storage**: `_service/receivers/cloudflare_worker.js` is a reference Worker
  that verifies the signature and stores each event once in D1 (primary key on
  `feedback_id`; KV is eventually consistent and is not used for authority). Deployment
  steps are in its header. Browsers never call it; only the service does.
- **In front of the service** (proxy or Cloudflare Tunnel for an intranet service): set
  `FEEDBACK_TRUSTED_PROXY_CIDRS` to the networks that actually connect to the service, so
  rate limiting sees the client address and not the edge's.
- **Static hosting** (Pages): any of the endpoint forms above; a Pages site with a
  separate service needs its origin in `FEEDBACK_ALLOWED_ORIGINS`.

The Scikit-Plots chat Worker has no `/v1/feedback` route on purpose: a chat/share edge
must not become feedback authority.

## Logging

Service records go to the `sphinx_feedback.service` logger. Every record carries a
structured `record.feedback` dict for JSON formatters, with an `event` field:

| `event` | Level | Fields |
| --- | --- | --- |
| `service_ready` | INFO | `mode`, `targets` (id, provider, role) |
| `stored` | INFO | `site_id`, `page_id`, `feedback_id`, `status`, `provider`, `degraded_mirrors` |
| `rejected` | WARNING | `site_id`, `page_id`, `feedback_id`, `code` |
| `conflict` | WARNING | `site_id`, `page_id`, `feedback_id` |
| `unavailable` | ERROR | `site_id`, `page_id`, `feedback_id`, `code` |

The provider exception behind `unavailable` is logged only at DEBUG. Never logged, at any
level: comment text, contributor credit, client addresses, request headers, credentials.
`feedback_id` is the per-event nonce, not a person.

```python
import logging

logging.getLogger("sphinx_feedback.service").setLevel(logging.INFO)
```

```bash
uvicorn scikitplot._externals._sphinx_ext._sphinx_feedback._service.app:app --log-config logging.yaml
```

The Sphinx adapter prints one line per build with `sphinx-build -v`: site ID, endpoint,
counter source, snapshot selector and whether the snapshot is complete.

## Generators (CLI)

```bash
python -m scikitplot._externals._sphinx_ext._sphinx_feedback init \
    --site-id my-docs --endpoint /v1/feedback --mode sqlite
python -m scikitplot._externals._sphinx_ext._sphinx_feedback aggregate \
    --site-id my-docs --events review-repo/feedback --sqlite feedback.sqlite3 \
    --out docs/source/_feedback/aggregate.json [--complete] [--revision REV]
```

```bash
python -m scikitplot._externals._sphinx_ext._sphinx_feedback check-provider
```

`check-provider` reads the same environment as the service and, for every forge target,
resolves the credential and the base branch and looks up a branch that cannot exist,
which exercises the calls a submission makes without writing anything; it also checks
`git` working trees and `sqlite` directories, and skips `webhook`/`custom` (any call would
deliver). Run it once against a real repository before enabling a forge.

`init` prints a `conf.py` block and the service environment, validated with the same
rules as the build and the service. `aggregate` reads the provider event layout
(`pages/<digest>/feedback-<nonce>.json`) and SQLite stores, keeps one site, skips other
sites' events, and writes a snapshot that is complete only with `--complete`.

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
