# App streaming and real-model maintenance runbook

Status: **ACTIVE — proxy v6.4.0 contract**
Owner: `_hf_spaces_proxy/app.py`
Primary regression gate: `tests/test_proxy_streaming_state.py`

This runbook exists so a future maintainer can distinguish a browser/UI problem
from a proxy transport problem and an upstream model problem without reading an
old chat transcript.

## 1. Big picture

The browser can request OpenAI-compatible streaming with `"stream": true`, but
that is only a **preference**. It does not prove that the selected upstream
returns Server-Sent Events (SSE).

The proxy therefore opens the upstream response **before committing downstream
success**, checks the upstream status and content type, and then chooses the
correct downstream representation.

```text
Browser
  POST /v1/chat/completions
  stream:true
        |
        v
Proxy validates + resolves route
        |
        +-- reserved stub/* ----------------------> local deterministic responder
        |                                            (never forwarded upstream)
        |
        `-- real model
              |
              v
        open upstream first
              |
              +-- connect/protocol failure before headers
              |        -> real HTTP 502/504 JSON error
              |
              +-- non-2xx upstream
              |        -> structured HTTP error
              |
              +-- application/json
              |        -> preserve JSON as JSON
              |
              `-- text/event-stream
                       -> SSE passthrough
                              |
                              `-- body fails after stream starts
                                      -> terminal `event: error` frame
```

### Non-negotiable invariant

`HTTP 200` must mean the proxy has a usable upstream response mode. A pre-header
transport/protocol failure must never become a silent empty `200` response.

## 2. Why v6.4.0 exists

Three conditions could previously combine into an empty assistant answer:

1. a disabled `stub/*` model could fall through into a real inference path;
2. an empty HF token could produce an invalid blank `Authorization: Bearer `
   header and trigger `httpx.LocalProtocolError`;
3. a JSON-only model backend could be relabelled as `text/event-stream`, causing
   the browser SSE parser to ignore the JSON bytes because there were no `data:`
   frames.

v6.4.0 fixes all three and adds explicit empty-body / empty-stream safeguards.

## 3. Variables used by this runbook

For shell examples:

```bash
BASE="https://scikit-plots-ai.hf.space"
```

Relevant non-secret Space Variables:

```text
STUB_ENABLED=true|false
PROXY_PROTOCOL_RETRIES=0|1|2
HF_SPACES_MODEL_URL=https://scikit-plots-ai-model.hf.space/v1/chat/completions
HF_SPACES_MODEL_NAMESPACES=scikit-plots
```

`STUB_ENABLED=true` is for diagnostics and testing. It is not a Secret.

Provider credentials such as `HF_TOKEN` remain server-side Secrets.

## 4. Step 1 — liveness and version

```bash
# curl -s https://scikit-plots-ai.hf.space/health | jq
curl -s "$BASE/health" | python3 -m json.tool
```

Expected minimum:

```json
{
  "status": "ok",
  "version": "6.4.0"
}
```

If production does not report v6.4.0 or later, do not use the v6.4 streaming
assumptions below.

## 5. Step 2 — deterministic wire test

Enable the Space Variable:

```text
STUB_ENABLED=true
```

Then run:

```bash
# curl -N https://scikit-plots-ai.hf.space/v1/chat/completions \
curl -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stub/qa",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "ping"
      }
    ]
  }'
```

Healthy output must contain an SSE answer with `pong` and finish with:

```text
data: [DONE]
```

This test proves the public route, FastAPI handler, request validation, stub
intercept, SSE framing, and downstream transport without contacting a model
provider.

### Disabled-stub guard

With:

```text
STUB_ENABLED=false
```

the same `stub/qa` request must return:

```text
HTTP 503
code = stub_disabled
```

and must perform **zero upstream model calls**. `stub/*` is a reserved fail-closed
namespace.

## 6. Step 3 — test the real custom model Space

Use the Path-2 model namespace configured for the custom Space:

```bash
curl "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "scikit-plots/Qwen2.5-Coder-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Reply with exactly: pong"}
    ]
  }'
```

A healthy JSON backend returns an OpenAI-compatible completion document.

Now exercise the same path with browser-like streaming intent:

```bash
curl -i "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "scikit-plots/Qwen2.5-Coder-7B-Instruct",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Reply with exactly: pong"}
    ]
  }'
```

Two responses are valid:

### A. Upstream is JSON-only

```text
HTTP/1.1 200
content-type: application/json
```

The proxy must preserve JSON. It must **not** relabel those bytes as SSE.

### B. Upstream really streams SSE

```text
HTTP/1.1 200
content-type: text/event-stream
```

The proxy passes SSE through incrementally.

Both modes are valid because `stream:true` is intent, not a transport guarantee.

## 7. Step 4 — optional HF serverless route

When Path 3 is configured and the selected model is served by the HF router:

```bash
curl "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "messages": [
      {"role": "user", "content": "Reply with exactly: pong"}
    ]
  }'
```

If no HF token is configured, `_resolve_upstream_url()` must omit the
`Authorization` header entirely. It must never construct an empty bearer value.

## 8. Response-mode state machine

```text
request stream:false
    -> buffered POST
       -> success: preserve upstream status/body/content-type
       -> timeout/connect/protocol error: structured 502/504

request stream:true
    -> build + send upstream with stream=True
    -> wait for upstream headers
       |
       +-- local protocol error
       |     -> 502 upstream_local_protocol_error
       |     -> never retry blindly
       |
       +-- remote protocol/read failure before output
       |     -> bounded retry, PROXY_PROTOCOL_RETRIES (default 1, max 2)
       |     -> if exhausted: 502
       |
       +-- non-200 status
       |     -> close upstream + structured error
       |
       +-- content-type != text/event-stream
       |     -> buffer body
       |     -> empty body: 502 upstream_empty_response
       |     -> JSON/other usable body: preserve content type
       |     -> valid buffered SSE prefix: recover as SSE
       |
       `-- content-type == text/event-stream
             -> passthrough chunks
             -> require at least one `data:` field
             -> no data field: terminal event:error
             -> mid-stream timeout/protocol/read failure: terminal event:error
```

## 9. Failure interpretation

| Symptom | Meaning | First action |
|---|---|---|
| `503 stub_disabled` | Stub requested while diagnostic rig disabled | Set `STUB_ENABLED=true` only when intentionally testing |
| `502 upstream_local_protocol_error` | Proxy constructed/validated an invalid local HTTP exchange | Check header/request construction; do not retry blindly |
| `502 upstream_remote_protocol_error` before output | Upstream connection/protocol ended incorrectly | Check upstream health; bounded retry may already have run |
| `504 upstream_timeout` | Connect/read exceeded configured timeout | Check model startup/load latency and timeout budget |
| HTTP 200 JSON while request says `stream:true` | Upstream is JSON-only | Healthy; browser fallback parser should render it |
| HTTP 200 SSE + terminal `event: error` | Stream began, then failed | Treat answer as incomplete; retry from UI/operator |
| HTTP 200 but empty body | Invalid upstream success | v6.4 converts it to `502 upstream_empty_response` |
| SSE ends with no `data:` | Invalid/empty stream | v6.4 emits `upstream_empty_stream` terminal error |

## 10. `LocalProtocolError` privacy rule

Never log `str(exc)` for `httpx.LocalProtocolError`. Protocol-library messages
can include rejected header values. The proxy maps the exception to fixed labels
only:

```text
illegal-header
content-length-overrun
content-length-underrun
missing-host
request-line
unspecified
```

Logs must not include prompt text, model response body, token values,
Authorization values, or private endpoint payloads.

## 11. Operator decision tree for an empty panel answer

```text
Panel answer empty
   |
   +-- run stub/qa + ping
   |      |
   |      +-- fails -> proxy/browser transport problem
   |      |
   |      `-- passes
   |            |
   |            `-- run real model curl
   |                   |
   |                   +-- 4xx/5xx -> routing/auth/upstream problem
   |                   |
   |                   +-- 200 JSON with content -> UI JSON fallback problem
   |                   |
   |                   +-- 200 SSE with data -> UI SSE parser/rendering problem
   |                   |
   |                   `-- terminal SSE error -> upstream stream instability
```

This order keeps diagnosis bounded. Do not start by changing browser code when
the deterministic wire test has not passed.

## 12. Regression commands

From the runtime module root:

```bash
python -m pytest -q tests/test_proxy_streaming_state.py tests/test_stub_model.py
node --check _static/ai-assistant.js
python -m py_compile _hf_spaces_proxy/app.py _hf_spaces_proxy/_utils/_shared_logic.py
```

For the available standalone non-Sphinx suite:

```bash
python -m pytest -q tests --ignore=tests/test___init__.py
```

Record exact pass/skip/failure counts in `VERIFICATION.md`; do not copy old
counts forward without rerunning.

## 13. Rollback / compatibility

If a v6.4 deployment must be rolled back, remember that older versions may:

- allow disabled `stub/*` requests to enter real routing;
- construct a blank bearer header when no token is present;
- label a JSON model response as SSE merely because the browser requested
  streaming;
- commit downstream HTTP 200 before the upstream stream has proven viable.

Those are behavior regressions, not cosmetic differences. Prefer correcting the
upstream deployment while keeping the v6.4 bridge unless a tested compatibility
constraint requires otherwise.

## B43 response-memory boundary

Run 24 adds a response-side safety contract complementary to `MAX_BODY_BYTES`:

- browser chat responses: 8 MiB;
- browser control/discovery responses: 512 KiB;
- canonical static Markdown: 1 MiB;
- standalone Global Share viewer JSON: 4 MiB;
- HF/dev-proxy upstream responses: `MAX_UPSTREAM_RESPONSE_BYTES`, default 8 MiB, hard maximum 32 MiB;
- Worker upstream responses: `MAX_RESPONSE_BYTES`, default 8 MiB, hard maximum 32 MiB.

Limits are enforced from declared `Content-Length` when present and again while
streaming decoded bytes. Malformed lengths fail closed. A browser transport that
cannot expose `ReadableStream.getReader()` is not allowed to fall back to
`response.text()`/`response.json()` at these trust boundaries, because that
would apply the limit only after memory was already allocated.

The public HF/Worker health documents expose only the effective non-secret
response ceiling so operators can diagnose deployment drift without revealing
provider URLs, credentials, payloads, or response content.

## 14. Future-proofing rules

1. Treat streaming as negotiated capability, never as a request-side fact.
2. Keep `stub/*` local and fail-closed.
3. Never send empty credential headers.
4. Do not expose downstream success before upstream headers are valid.
5. Once visible output begins, never retry in a way that can duplicate text.
6. Convert terminal stream failure into an explicit machine-readable event.
7. Preserve JSON as JSON; preserve SSE as SSE.
8. Add a regression test before adding another upstream response mode.
9. Enforce response limits before whole-body buffering; never treat post-`text()`/`json()` size checks as memory safety.
