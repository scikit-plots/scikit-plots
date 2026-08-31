# B15 — App streaming protocol recovery and maintenance runbook

Status: **COMPLETE**
Type: **bounded proxy reliability checkpoint**
Subsystem: **_sphinx_ai_assistant / _hf_spaces_proxy**

## Objective

Eliminate silent empty assistant responses caused by request/stream protocol
mismatches and preserve a reproducible operator runbook outside the runtime
package.

## Scope

- `_hf_spaces_proxy/app.py` response-mode negotiation.
- reserved `stub/*` fail-closed behavior.
- prevention of blank Authorization headers.
- pre-header versus mid-stream error semantics.
- maintenance relocation and reproducible curl diagnostics.

## Non-goals

- No model-quality changes.
- No browser-controlled server credentials.
- No promise that every upstream provider supports SSE.

## Execution record

```yaml
checkpoint: B15
status: COMPLETE
started_at: 2026-08-28
completed_at: 2026-08-28
source_anchor: scikitplot__externals__sphinx_ext__sphinx_ai_assistant_app_streaming_v640_clean_runtime.zip @ 5c931a23733699da505fdcc0a001ecb4ae4e100e9fb3f978697f945895855aa8
production_code_modified: true
contracts_touched:
  - proxy chat routing
  - streaming transport negotiation
  - deterministic stub safety
  - maintenance/runtime separation
files_changed:
  - _hf_spaces_proxy/app.py
  - _hf_spaces_proxy/README.md
  - tests/test_proxy_streaming_state.py
  - maintenance APP_STREAMING_RUNBOOK.md
risks:
  - older deployed proxies do not provide the v6.4 guarantees
rollback: retain v6.4 unless a separately-tested compatibility requirement forces rollback
```

## Verification gates

- [x] `stub/qa` enabled returns `pong` and `[DONE]`.
- [x] disabled `stub/*` returns local 503 and never forwards upstream.
- [x] blank HF token does not create `Authorization: Bearer `.
- [x] JSON returned to a `stream:true` request remains JSON.
- [x] true SSE remains SSE.
- [x] pre-header protocol failure returns real 502/504.
- [x] mid-stream failure becomes an explicit terminal SSE error.
- [x] empty 200 body / empty SSE stream cannot silently finalize success.
- [x] curl flow and operator decision tree recorded in `APP_STREAMING_RUNBOOK.md`.

## Closure

Complete. Future changes to proxy streaming must update the runbook and the
streaming-state regression tests together.
