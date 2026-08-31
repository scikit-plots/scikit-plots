# B43 — Bounded Remote Response & Context Ingestion

Status: **PREFINAL EXACT-BYTE GREEN — immutable final build pending**

## Scope

Close repository-controlled response-memory exhaustion paths that remained after
request-body limits, streaming transport negotiation, and B42 isolation. This
checkpoint covers browser assistant/control/canonical reads, isolated policy and
canonical reads, standalone Global Share viewer reads, HF/FastAPI upstream chat
responses, the local dev proxy, and Cloudflare Worker forwarding.

It does **not** claim that third-party provider SDKs or every optional storage
provider API response is bounded; those remain a separate audit surface.

## Implemented

- browser chat response ceiling: **8 MiB** decoded upstream bytes;
- browser control/discovery ceiling: **512 KiB**;
- canonical static Markdown ceiling: **1 MiB**;
- SSE total response ceiling plus **256 KiB** unterminated-line ceiling;
- bounded readers reject malformed/oversized `Content-Length` before body read,
  then count bytes while streaming;
- security-sensitive browser readers fail closed when `ReadableStream.getReader`
  is unavailable instead of falling back to whole-body `text()`/`json()`;
- isolated parent-policy and canonical-document readers use the same pre-buffer
  streaming principle;
- standalone HF and Worker Global Share viewers stream-parse JSON under a **4
  MiB** hard viewer ceiling;
- HF proxy uses `httpx.send(..., stream=True)` for buffered and streaming chat
  paths and caps decoded upstream response bytes at
  `MAX_UPSTREAM_RESPONSE_BYTES` (default **8 MiB**, hard maximum **32 MiB**);
- dev proxy uses `httpx.stream()` and the same 8/32 MiB response policy;
- Worker rejects malformed/declared-oversize upstream lengths before forwarding
  and wraps unknown-length bodies in a byte-counting `ReadableStream` governed by
  `MAX_RESPONSE_BYTES` (default **8 MiB**, hard maximum **32 MiB**);
- HF/Worker health surfaces expose only the effective non-secret response ceiling;
- proxy deployment version ratcheted to **7.1.0**.

## Working-tree verification

- B43 Python: **8/8**.
- B43 executable browser/Worker: **16/16**.
- registered Node harness registry: **47/47**.
- mutation/logging/privacy: **244/244**.
- runnable non-Sphinx: **783 passed, 3 skipped**.
- Sphinx-inclusive: **1249 passed, 3 skipped, 5 failed, 62 errors**; all non-green
  cases remain in `test___init__.py` and terminate on unavailable `sphinx`.
- offline supply-chain verifier: **GREEN**.
- canonical release subject: proxy **7.1.0**, runtime-source SHA-256
  `0f64a53f86809ba6b5451342528185cd0c708453384adea7d0353c588f1f23f1`.
- complete two-root Python compile: **70/70**.
- clean source membership: **297 files**, exact `scikitplot/` + `maintenances/` roots, zero cache/bytecode.
- controlled Run 23 → Run 24 source freeze: **4 added · 30 modified · 0 removed = 34 paths**.
- candidate exact-byte extraction: **GREEN** — reproduced 8/8 B43 Python, 16/16 browser/Worker, 47/47 Node, 244/244 mutation/privacy, 783 passed + 3 skipped runnable, 70-file compile, syntax/TOML/release/maintenance GREEN, and 1249/3/5/62 missing-`sphinx`-only boundary.
- metadata-bearing prefinal exact-byte extraction: **GREEN** — reproduced 8/8 B43 Python, 16/16 browser/Worker, 47/47 Node, 244/244 mutation/privacy, 783 passed + 3 skipped runnable, 70-file compile, syntax/TOML/release/maintenance GREEN, and 1249/3/5/62 missing-`sphinx`-only boundary; archive remained 297 files, exact two roots, zero cache/bytecode.
- immutable final delivery build / exact-final-byte acceptance remains pending.

## Security disposition

`SEC-P1-45` is **CLOSED — Run 24 B43 at the covered response-ingestion
boundaries**. Application code no longer treats a post-`text()`/`json()` length
check as memory protection on those paths; limits are checked before and during
stream consumption.

This closure is deliberately scoped. Optional provider-storage clients and
third-party SDK internals must be audited separately before making a universal
"all remote responses bounded" claim. `SEC-P1-38`, `SEC-P1-42`, and
`SEC-P1-43` remain open in their existing external/architectural scopes.
