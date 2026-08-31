# B27 — CORS, Identity, Resource & Deployment Parity

Status: **COMPLETE — Run 11**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Finish B05/B06 release residuals across every bundled public service path without weakening the Run 1–10 authority/privacy/lifecycle contracts.

## Source and external evidence

- Input archive: `scikitplot__sphinx_ai_assistant_b18_run10_global_link_lifecycle_hardening_overlay.zip`
- SHA-256: `cd3897bc7078fd8a1df4cbb7c8e3282772868062b7cbe10c450f723c2ed673fc`
- Review date: 2026-08-29
- Cloudflare Workers request-body limits: platform ceilings are much larger than this application's 10 MiB default, so application streaming limits remain required.
- Workers KV: same-key writes are limited to one per second and KV visibility is eventually consistent.
- Workers Rate Limiting API: designed as a fast, permissive abuse gate rather than accurate accounting.
- `CF-Connecting-IP`: treated as an edge-provided identity only for direct Cloudflare edge traffic; revalidate if another Worker becomes the ingress.

## Landed controls

1. **Least-privilege browser origins** — bundled production defaults are exact, not wildcard. Explicit disallowed `Origin` is rejected before expensive/write handlers; missing `Origin` remains valid for server callers and is never authentication.
2. **Streaming byte ceilings** — HF proxy, HF model, and Worker reject declared oversize before reading and stop incremental reads at the ceiling; configurable chat body size is hard-clamped to 16 MiB. The loopback dev proxy requires and checks `Content-Length` before allocation.
3. **Identity trust** — HF uses the direct peer unless `TRUST_X_FORWARDED_FOR=true` explicitly declares a trusted overwrite ingress. Chat/Share/feedback/contribution share that helper. Worker uses edge `CF-Connecting-IP` for its abuse key.
4. **Bounded abuse state** — each HF in-memory identity map has a hard cardinality ceiling and fails closed for a new live identity when full. Feedback retractions use the same write gate rather than an unlimited bypass.
5. **Worker KV limiter repair** — unique expiring event keys replace same-key counter rewrites. This avoids the KV same-key write ceiling while remaining truthfully documented as eventually-consistent/soft.
6. **Deployment parity** — bundled `wrangler.toml` now points to its actual bundled `index.js`; compatibility date and recommended variables are current; discovery/README defaults match runtime policy.
7. **Direct model browser guard** — explicit disallowed browser origins are rejected before Gradio/REST work; server-to-server proxy calls without Origin remain usable.

## Verification

- `tests/test_b05_b06_release_security.py` — runtime/source contract for origin, streaming, identity, limiter bounds, Worker event-key limiter and Wrangler package path.
- existing Share/chat/model/feedback suites — no authority/privacy regression.
- `tests/test_chat_authority.mjs` fake edge KV updated to model `list` + `put`, so prompt-authority mutants remain meaningful rather than crashing.
- `tests/test_mutation.py` — existing 183 client security mutants remain green.
- broad non-Sphinx, syntax, maintenance, package-copy gates recorded in `VERIFICATION.md`.

## Contracts

- `AIA-C08 CorsOriginPolicy = HOLDS`
- `AIA-C10 ClientIdentity = HOLDS` at the bundled application trust boundary
- `AIA-C11 RequestLimits = HOLDS` at the bundled application buffering boundary
- `AIA-C18 ServiceParity = PARTIAL` only because strict distributed rate accounting/private upstream deployment remains infrastructure-owned, not because a bundled route still has wildcard CORS or post-buffer size enforcement

## Residual — SEC-P0-31

Process-local HF counters and eventually-consistent Worker KV event visibility are **abuse gates**, not globally authoritative quotas. Multi-replica/strict deployments must enforce a shared ingress/Durable Object/gateway policy. Cloudflare's Rate Limiting binding is also permissive/local by design and must not be documented as billing/accounting truth.

## Rollback

Revert the B27 production/test/config set together. Do not selectively restore wildcard CORS, full-body buffering, unbounded identity dictionaries, same-key KV counters, or a non-existent Worker `main` path.
