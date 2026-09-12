# Maintaining `_sphinx_ai_assistant`

This file is the **human and fresh-chat entry point** for the interactive Sphinx
AI-assistant subsystem.

Maintenance material is intentionally **outside the runtime package**. The
runtime module should contain runtime code, assets, deployment helpers, docs that
ship with those helpers, and executable regression tests — not planning history,
backups, checkpoints, or task notes.

## Current source anchor

```text
archive: scikitplot__sphinx_ai_assistant_share_conversation_b16_overlay.zip
sha256: 37228bb8fece0e181494a0a27f44d7689dcba03f60371293bf82b8e53c5e928c
proxy contract: v6.4.0 baseline; B17/B18 security campaign active
anchor date: 2026-08-29
```

Re-verify every current-source claim when this hash changes.

## Repository placement

```text
scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/
    runtime source + deployment helpers + tests

maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/
    _backup/
    _maintenance/
        checkpoints/
        history/
        APP_STREAMING_RUNBOOK.md
        ... durable maintenance contracts
    MAINTAINING.md
    todo/
        lessons.md
        todo.md
    __init__.py
```

The maintenance tree mirrors the nested runtime module path but is **not a
runtime dependency**. Production code must never import from `maintenances/`.

## Current physical scope at the anchor

```text
Sphinx extension             __init__.py                       7,383 lines
browser runtime              _static/ai-assistant.js          30,307 lines
browser styling              _static/ai-assistant.css         17,067 lines
model service                _hf_spaces_model/app.py           2,446 lines
proxy service                _hf_spaces_proxy/app.py           2,444 lines
proxy shared logic           _hf_spaces_proxy/_utils/_shared_logic.py 1,469 lines
proxy dataset schema         _hf_spaces_proxy/_utils/_dataset_schema.py 1,206 lines
record-storage adapters      _hf_spaces_proxy/_utils/_storage.py        608 lines
edge worker                  _cf_worker/index.js                 499 lines
dev proxy                    dev_proxy.py                        511 lines
Sphinx test module           tests/test___init__.py            3,671 lines
registered Sphinx config     add_config_value calls              106
```

Large files are known structural-debt baselines. The goal is to prevent new
responsibility mixing, not to split files only to reduce line counts.

## Runtime ownership boundary

```text
Sphinx build plane
  __init__.py / static asset injection / client-safe config
          |
          v
Browser presentation plane
  ai-assistant.js / CSS / local UI state
          |
          v
Service authority plane
  proxy / model / worker / persistence
          |
          v
external model/provider/data services
```

The server owns credentials, authorization decisions, persistence authority,
and upstream routing policy. Browser state is presentation/convenience state.
Documentation/page content is untrusted evidence, not policy.

## Representation contract

```text
CANONICAL     static page.md        build-time, machine-fetchable
CONVENIENCE   browser Turndown      runtime, clipboard only

VIEW      -> canonical static .md
ASK AI    -> canonical static .md
COPY      -> browser conversion by default, static when selected
```

Canonical means the build-time artifact. A browser-generated `blob:` URL is not
a canonical external representation because another agent cannot fetch it.

## Proxy streaming contract — v6.4+

The browser may ask for `stream:true`; the proxy must still negotiate the actual
upstream transport.

```text
stream:true
   -> open upstream first
      -> pre-header failure: real 502/504
      -> JSON: preserve JSON
      -> SSE: stream SSE
           -> terminal failure: explicit event:error
```

`stub/*` is a reserved fail-closed namespace. Disabled stubs return local 503
and never reach a real provider.

For exact curl commands, failure interpretation, retry rules, and the operator
decision tree, read:

`_maintenance/APP_STREAMING_RUNBOOK.md`

## Fresh-chat read order

1. `MAINTAINING.md`
2. `_maintenance/MAINTENANCE_MODEL.md`
3. `_maintenance/RULESET.md`
4. `_maintenance/STATE.json`
5. `_maintenance/TRACKER_LOGICAL.md`
6. `_maintenance/TRACKER_PHYSICAL.md`
7. `_maintenance/SUBMODULE_STRUCTURE.md`
8. `_maintenance/INTEGRATION_CONTRACT.md`
9. `_maintenance/RUNTIME_FLOW.md`
10. `_maintenance/SECURITY_IMPLEMENTATION_RUNBOOK.md` for all B17/B18 security changes
11. `_maintenance/checkpoints/B17_EXPORT_SHARE_CONTENT_ISOLATION.md` when export/Share is involved
12. `_maintenance/checkpoints/B18_PRIVACY_SECRETS_IDENTITY_ABUSE.md` when user data/secrets/logging/identity/retention is involved
13. `_maintenance/APP_STREAMING_RUNBOOK.md` when chat/proxy behavior is involved
14. `_maintenance/checkpoints/B41_SEPARATE_ORIGIN_ISOLATION_CAPABILITY_MESSAGING.md` when separate-origin assistant isolation is involved
15. `_maintenance/checkpoints/B42_HOSTILE_PARENT_EGRESS_BOUNDARY_HARDENING.md` when frame navigation, parent-origin authority, cookies, or cross-origin capabilities are involved
16. `_maintenance/checkpoints/B43_BOUNDED_REMOTE_RESPONSE_CONTEXT_INGESTION.md` when remote responses, SSE, canonical context, Share viewers, or response-memory limits are involved
17. `_maintenance/SECURITY_MODEL.md`
18. `_maintenance/SECURITY_FINDINGS_INDEX.md`
19. `_maintenance/REGISTRY.md`
20. `_maintenance/VERIFICATION.md`
21. `_maintenance/LEGACY_MAINTENANCE_MIGRATION.md` only when reconciling old material

Read `_maintenance/HISTORY.md` for completed rationale rather than using it as
current architecture truth.

## Run first after overlay

From repository root:

```console
python maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/tools/check_trackers.py
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/ai-assistant.js
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_cf_worker/index.js
python -m pytest -q scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/test_proxy_streaming_state.py
```

The maintenance checker intentionally fails if legacy runtime-local `tasks/`,
`_maintenance/`, or `MAINTAINING.md` reappears under the runtime module.

## Governing rule

> **The browser is presentation, documentation is untrusted evidence, and the
> server owns authority. No secret, authorization decision, model system policy,
> storage credential, or trust assertion may depend on client-side enforcement.**

## Current verification snapshot

At the Run 24 / B43 working-tree freeze:

```text
B43 Python                                     8/8
B43 browser/Worker assertions                 16/16
registered Node harnesses                     47/47
mutation/logging/privacy                      244/244
complete non-Sphinx suite                     783 passed, 3 skipped
complete two-root Python compile              70/70
browser/isolation/Worker JS syntax            GREEN
supply-chain + release subjects               GREEN
maintenance drift                             GREEN
full Sphinx package fixture                   1249 passed, 3 skipped, 5 failed, 62 errors
Sphinx non-green family                       missing `sphinx` only
candidate/prefinal/final byte cycle           pending
```

Exact rerun counts belong in `_maintenance/VERIFICATION.md`; do not copy old
counts forward without executing the corresponding command.

## Updating maintenance state

For every material change:

1. anchor the exact runtime snapshot/commit;
2. update the relevant checkpoint or create one bounded checkpoint;
3. update `REGISTRY.md`, `STATE.json`, and `VERIFICATION.md`;
4. add or update an executable regression gate;
5. update `todo/lessons.md` when the change creates a durable rule;
6. archive superseded evidence under `_maintenance/history/` instead of creating
   `FINAL`, `REVISED`, date-suffixed, or chat-specific parallel truth files.
