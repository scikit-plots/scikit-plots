# `_sphinx_ai_assistant` Verification Contract

## Status vocabulary

`GREEN`, `RED`, `ENVIRONMENT_BLOCKED`, `NOT_RUN`, `DEFERRED`, and
`NOT_APPLICABLE` have the same meanings as the sibling `_sphinx_llm` maintenance
model. Never hide a missing dependency by calling the suite green.

## Anchored review-environment snapshot

```text
node --check _static/ai-assistant.js        GREEN
node --check _cf_worker/index.js            GREEN
node --check _maintenance/_ext_settings.js  GREEN
node _maintenance/test_ext_settings.js      GREEN: 79 passed, 0 failed
python -m compileall selected modules        GREEN
pytest tests/test_discovery_contract.py      GREEN: 4 passed
full tests                                   ENVIRONMENT_BLOCKED/PARTIAL
  453 passed, 5 failed, 59 errors
  dominant blocker: ModuleNotFoundError: sphinx in Sphinx-dependent fixtures
```

Re-run in the canonical repository environment; these are snapshot observations,
not release certification.

## Maintenance gates

```console
python scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/check_trackers.py
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/ai-assistant.js
node --check scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_cf_worker/index.js
```

## Required multi-runtime gates

| Gate | Contract |
|---|---|
| Sphinx config inventory/serialization | AIA-C01/C02 |
| generated HTML secret positive-control | AIA-C02 |
| browser settings Node tests | AIA-C14 |
| discovery schema/client-server parity | AIA-C06 |
| static representation selection | AIA-C04/C17 |
| prompt-role direct API tests | AIA-C05/C12 |
| malicious page prompt positive-control | AIA-C05 |
| backend destination allow/bind tests | AIA-C07 |
| proxy + worker CORS matrix | AIA-C08/C18 |
| share read/edit capability tests | AIA-C09 |
| forwarded identity spoof tests | AIA-C10 |
| oversize body/resource tests | AIA-C11 |
| feedback consent/provenance tests | AIA-C13 |
| proxy/model/worker policy parity | AIA-C18 |
| `_sphinx_llm` facade import boundary | AIA-C04 |
| Corpus/MCP import/ownership boundary | AIA-C15/C16 |
| full Sphinx fixture suite | build/runtime regression |

## Security closure rule

A browser-only change cannot close a service-level P0. A service finding closes
only when the direct endpoint path itself satisfies the invariant and a
regression test bypasses the browser to prove it.

## Migration closure rule

Runtime DOM extraction may be removed/disabled only after a test inventory shows
canonical/static compatibility coverage for the supported page classes and the
fallback reason is no longer needed for ordinary scikit-plots docs.
