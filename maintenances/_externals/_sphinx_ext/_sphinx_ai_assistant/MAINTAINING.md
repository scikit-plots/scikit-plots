# Maintaining `_sphinx_ai_assistant`

This file is the **human and fresh-chat entry point** for the existing
interactive Sphinx AI-assistant subsystem.

The maintenance model is being normalized to the same durable control-plane
logic used by `scikitplot.corpus` and `scikitplot.mcp`, while respecting this
subsystem's larger multi-runtime trust boundary.

## Current source anchor

```text
archive: scikit-plots.zip
sha256: 3f0d6862c27c41811cc19a91e34bd9789a653328a9247dbf34492a1cccbd726d
```

Re-verify every current-source claim when this hash changes.

## Current physical scope at the anchor

```text
Sphinx extension             __init__.py                     5,592 lines
browser runtime              _static/ai-assistant.js         22,772 lines
browser styling              _static/ai-assistant.css        14,300 lines
model service                _hf_spaces_model/app.py          2,446 lines
proxy service                _hf_spaces_proxy/app.py          1,929 lines
proxy shared logic           _hf_spaces_proxy/_shared_logic.py 1,454 lines
proxy dataset schema         _hf_spaces_proxy/_dataset_schema.py 1,206 lines
edge worker                  _cf_worker/index.js                499 lines
dev proxy                    dev_proxy.py                       362 lines
Sphinx test module           tests/test___init__.py            3,507 lines
registered Sphinx config     add_config_value calls               90
```

Large files are **known structural debt baselines**. The maintenance goal is to
prevent unbounded growth and new responsibility mixing, not to declare them
incorrect merely because they are large.

## Ownership boundary

`_sphinx_llm` is **frozen**. It is not required by any assistant surface, and
nothing imports it: measured, 0 references in `__init__.py`,
`_static/ai-assistant.js`, `_static/__init__.py` or any test.

```text
_sphinx_ai_assistant
  build layer   canonical representation: page.md + llms.txt, at build-finished
  browser layer UI/state, endpoint discovery, interactive requests,
                convenience Markdown via vendored Turndown

_sphinx_ai_backend
  proxy, model space, edge worker; reached over HTTP, imported by nothing
```

So the assistant owns its canonical representation and emits it at build time.

## The representation contract

```text
CANONICAL     static page.md        build-time, machine-fetchable
CONVENIENCE   browser Turndown      runtime, clipboard only

VIEW      -> canonical      opens the static .md URL
ASK AI    -> canonical      sends the static .md URL
COPY      -> convenience by default, canonical when the toggle selects `static`
```

Canonical means the build-time artifact and nothing else. A browser conversion
cannot be canonical however faithful it is, because no external agent can fetch
a `blob:` URL — which is also why `View as Markdown` opens the static file.

## Fresh-chat read order

1. `_maintenance/MAINTENANCE_MODEL.md`
2. `_maintenance/RULESET.md`
3. `_maintenance/TRACKER_LOGICAL.md`
4. `_maintenance/TRACKER_PHYSICAL.md`
5. `_maintenance/SUBMODULE_STRUCTURE.md`
6. `_maintenance/CONFIG_ARCHITECTURE.md`
7. `_maintenance/INTEGRATION_CONTRACT.md`
8. `_maintenance/RUNTIME_FLOW.md`
9. `_maintenance/SECURITY_MODEL.md`
10. `_maintenance/SECURITY_FINDINGS_INDEX.md`
11. `_maintenance/REGISTRY.md`
12. `_maintenance/VERIFICATION.md`
13. `_maintenance/LEGACY_MAINTENANCE_MIGRATION.md` when reconciling the old docs

Do not create new parallel files named `FINAL`, `REVISED`, `EXPANDED`,
date-suffixed, or chat-specific variants inside the source tree.

Read `_maintenance/HISTORY.md` only for completed rationale.

Run first after overlay:

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/check_trackers.py
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/ai-assistant.js
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_cf_worker/index.js
```

## Governing rule

> **The browser is presentation, documentation is untrusted evidence, and the
> server owns authority. No secret, authorization decision, model system policy,
> or trust assertion may depend on client-side enforcement. The assistant
> consumes canonical static documentation representation; it does not define
> that representation's authority.**

## Desired vs current truth

Existing maintenance prose contains desirable target statements such as
“secrets live only on the server.” Current source must be checked before treating
those statements as facts. The current architecture has configuration/token
escape hatches and browser serialization/persistence surfaces that remain
security-review items. `TRACKER_LOGICAL.md` therefore distinguishes `HOLDS`,
`VIOLATED`, `PARTIAL`, and `PLANNED` explicitly.

## Current verification snapshot in the extracted source tree

```text
node --check _static/ai-assistant.js        GREEN
node --check _cf_worker/index.js            GREEN
node --check _maintenance/_ext_settings.js  GREEN
node _maintenance/test_ext_settings.js      GREEN (79 passed, 0 failed)
python compileall core/service modules       GREEN
pytest tests/test_discovery_contract.py      GREEN (4 passed)
full pytest suite                            ENVIRONMENT_BLOCKED/PARTIAL
  observed: 453 passed, 5 failed, 59 errors
  dominant blocker: Sphinx unavailable in the review environment
```

Do not interpret the full-suite line as a clean product failure or as green; rerun
in the canonical docs/test environment.

## Fresh-chat continuation prompt

> Review the current `_sphinx_ai_assistant` source from the supplied source
> snapshot. Verify the hash first. Read `MAINTAINING.md` and the maintenance
> rules/trackers/registry/security model/verification contract. `_sphinx_llm` is
> **frozen** — do not read it, do not depend on it, and do not restore any edge
> to it. Do not edit production code until the active checkpoint is explicit.
> Preserve server-owned authority, treat page/retrieved content as untrusted
> reference data, and prevent client-visible secrets. The assistant owns its
> canonical representation: `page.md` and `llms.txt` are written at
> `build-finished`, `VIEW` and `ASK AI` use the static `.md`, and the browser
> Turndown path is convenience that must never be labelled canonical. Record
> every new finding with evidence, severity, owner, regression gate, and exact
> next action.

## Updating maintenance state

For every material change update `REGISTRY.md`, `STATE.json`, the relevant
tracker, and `VERIFICATION.md`. Complete a checkpoint before starting a new
parallel architecture document. Archive superseded research rather than letting
multiple “final” documents compete as source truth.
