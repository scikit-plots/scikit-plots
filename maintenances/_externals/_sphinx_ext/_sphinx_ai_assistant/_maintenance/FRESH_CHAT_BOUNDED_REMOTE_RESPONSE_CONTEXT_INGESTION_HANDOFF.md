# Fresh-chat handoff — Run 24 / B43 Bounded Remote Response & Context Ingestion

Input release: Run 23 / B42 exact final SHA-256
`0700fe3cdf84b8494ad86f045904514ec75e7ee8550f7b004fe6c15a93215df6`.

B43 closes response-side memory exhaustion paths complementary to existing
request-body limits. Browser chat responses are bounded at 8 MiB, control and
service-discovery responses at 512 KiB, canonical static Markdown at 1 MiB, and
standalone Global Share viewer JSON at 4 MiB. Security-sensitive browser readers
require a real `ReadableStream`; they do not fall back to whole-body
`response.text()`/`response.json()` when bounded streaming is unavailable.

HF/FastAPI and the dev relay stream-read upstream chat bodies under
`MAX_UPSTREAM_RESPONSE_BYTES` (8 MiB default, 32 MiB hard maximum). The Worker
uses `MAX_RESPONSE_BYTES` with the same default/hard maximum, rejecting malformed
or declared-oversize lengths before forwarding and counting unknown-length
chunks in a wrapped stream. SSE also has a total-byte and unterminated-line
ceiling.

Working-tree gates before packaging: B43 Python 8/8, B43 browser/Worker 16/16, Node registry 47/47,
mutation/privacy 244/244, runnable non-Sphinx 783 passed / 3 skipped, and
Sphinx-inclusive 1249 passed / 3 skipped / 5 failed / 62 errors with unavailable
`sphinx` as the only failure/error family. Proxy version is 7.1.0 and the current
runtime-source subject is
`0f64a53f86809ba6b5451342528185cd0c708453384adea7d0353c588f1f23f1`.

Do not generalize this closure to optional provider-storage client responses or
third-party SDK internals without a separate bounded-response audit. Continue to
keep `SEC-P1-38`, `SEC-P1-42`, and `SEC-P1-43` open.

Source freeze: **70-file Python compile**, **297 clean files**, and exact Run 23 → Run 24 diff **4 added · 30 modified · 0 removed = 34 paths**. Candidate/prefinal/final byte acceptance is still pending at this handoff stage.

Candidate exact-byte acceptance: **GREEN**. A clean candidate extraction reproduced 8/8 B43 Python, 16/16 browser/Worker, 47/47 Node, 244/244 mutation/privacy, 783 passed + 3 skipped runnable, 70-file compile, release/config/maintenance GREEN, and 1249/3/5/62 missing-`sphinx`-only. Metadata-bearing prefinal and immutable final acceptance remain pending.
