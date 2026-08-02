# Security findings — maintenance index

This is a **short, actionable index** into the full review. It does not replace
it. The complete evidence, exploit chains, P0–P3 roadmap, architecture decision
records, and remediation order live in:

- `SECURITY_REVIEW_AND_REMEDIATION_PLAN_FULL_ARCHIVE_UPDATED.md`
  (86 findings across all services; SHA-256
  `92ae5f2a1750b138f2b1d743cc2e50aa4f4636fbc31a8bebcf9dbc5d774bddb1`).

**Release verdict from the review: NO-GO until the P0s are fixed.** That verdict
is server-side and is *not* changed by any client-side work in this submodule.

## Scope reality (read first)

The P0s are overwhelmingly **server-side**. A browser widget cannot remediate
them, and a client-side "prompt-injection guard" is defense-in-depth only,
because the completion endpoint can be called directly, bypassing the browser.
The real fixes belong in `_hf_spaces_proxy/`, `_cf_worker/`, and
`_hf_spaces_model/`.

## P0 index (verified against current source)

| # | Finding | Where (verified) | Status |
|---|---|---|---|
| 1 | `HF_TOKEN` attached as Bearer to a configurable `BACKEND_URL` with no origin allow-list → token can be redirected to an attacker origin | `_hf_spaces_proxy/app.py:275,282` (+ header 14–16) | Open |
| 2 | CORS defaults to wildcard `*` | `_hf_spaces_proxy/app.py:414`; `_cf_worker/index.js:56` | Open |
| 3 | Share edit authorized by UUID possession (no owner/edit capability split) | `_hf_spaces_proxy/app.py:1546` (`PATCH /v1/share/{uuid}`), store keyed by UUID `:500` | Open |
| 4 | Public inference relay — completion endpoint callable directly, bypassing browser limits | `app.py:1083`; `_cf_worker/index.js` (second relay) | Open |
| 5 | Model Space accepts caller-supplied system messages (policy bypass) | `_hf_spaces_model/app.py` | Open |
| 6 | Training-data poisoning directly reachable (no proof the proxy generated the answer) | `app.py:1142` (`/v1/contribute`) | Open |
| 7 | Consent versioning disabled (`consentVersion: null`) | `_hf_spaces_proxy/_dataset_schema.py` | Open |
| 8 | `X-Forwarded-For` trusted without a trusted-proxy set → IP controls bypassable | `_hf_spaces_proxy/app.py` | Open |
| 9 | Body limit applied after full buffering (`await request.body()` first) | `_hf_spaces_proxy/app.py` | Open |
| 10 | Docker runs as root, floating base/dependency versions | `_hf_spaces_proxy/Dockerfile` | Open |

See the full archive for P1–P3, the exploit chains, and the secure target
architecture (server-owned system prompts, per-request capability tokens,
destination-bound credentials, read/edit capability split, queue-based dataset
writes, durable distributed quotas).

## Client-side items tracked here

| Item | Where | Status |
|---|---|---|
| Extended Settings had no single validated write path | `_static/ai-assistant.js` | **Addressed** — `_ExtSettings` registry built + 79/79 tests (see `EXT_SETTINGS.md`); wiring pending |
| `_persistCustom` comment claims tokens omitted but code persists them | `_static/ai-assistant.js` (`_EP._persistCustom`) | Open — see `CONFIG_ARCHITECTURE.md` token policy |
| Server↔client discovery drift undetected | `app.py` `GET /` ↔ `_fetchProxyDatasetInfo` | **Addressed** — contract + guard test (see `CONFIG_ARCHITECTURE.md`) |

## Method

Findings are recorded with a per-claim source location and re-verified against
the current tree before being marked. Client work follows the submodule's
always-green-before-packaging gate; server work is tracked for a separate,
grounded, run-by-run pass on the proxy / worker / model services.
