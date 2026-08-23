# `_sphinx_llm` Structure and Placement Rules

## Dependency graph

```text
sphinx_llm/  (vendored NVIDIA baseline)
     |
     +---------------------> compat/   (version/fallback shims)
     |
     v
   core/
     |
     +------------+-------------+
     |            |             |
     v            v             v
 adapters/     curation/   public build-time facade (A11, only if needed)
     |            |             |
     +------------+-------------+
                  |
                  v
        static published artifacts
                  |
                  v
        _sphinx_ai_assistant runtime
```

No arrow may point from `_sphinx_llm` to `_sphinx_ai_assistant` as a runtime or
private-code dependency.

## Directory ownership

| Directory | Owns | Must not own |
|---|---|---|
| `sphinx_llm/` | pinned NVIDIA-derived baseline and its preserved upstream tests | scikit-plots product/UI features |
| `core/` | representation lifecycle, routing, artifacts, manifest/provenance, discovery, bounded public build-time facade | browser UI, service auth |
| `adapters/` | custom node/directive semantic representation | general curation |
| `curation/` | inclusion/exclusion/order/size/code-file policies | Sphinx builder lifecycle |
| `compat/` | upstream-version shims, primary-build semantic-context parity, and lower-fidelity offline HTML fallback | canonical semantic meaning |
| `tests/` | downstream semantic/artifact fixtures | runtime service behavior |
| `upstream/` | retired bootstrap README only | **all production code** |
| `_maintenance/` | maintenance evidence/checks plus pinned upstream test fixture/harness | importable production behavior |

## Placement decision tree

```text
Is the change an upstream NVIDIA change/fix?
  yes -> sync/verify sphinx_llm/ in an upstream-only checkpoint;
         prefer a downstream compat shim if local behavior can stay outside.
  no  -> Does it define canonical representation/artifact lifecycle?
           yes -> core/
           no  -> Does it handle a custom semantic node/directive/media type?
                    yes -> adapters/
                    no  -> Is it selection/ignore/order/size/code policy?
                             yes -> curation/
                             no  -> Is it an upstream-version shim or offline HTML fallback?
                                      yes -> compat/
                                      no  -> It probably belongs outside _sphinx_llm.
```

## Consumer boundary

There are two distinct consumer paths:

1. **runtime path** — browser/agent consumers fetch static published artifacts;
   they never import Python or private upstream internals;
2. **optional build-time path** — another Sphinx extension may use a stable
   public facade introduced by A11 if integration requires it.

Neither path may import private helpers from `sphinx_llm/` directly.

## Test placement

- vendored NVIDIA tests remain under `sphinx_llm/tests/` and stay byte-identical
  where classified `UPSTREAM_PRESERVED`; do not patch their repository-relative
  fixture assumptions;
- `_maintenance/run_upstream_tests.py` recreates the upstream repository shape
  ephemerally using `_maintenance/upstream_test_fixture/docs/source`;
- downstream semantic fixture roots belong under `tests/roots/`;
- `tests/test_primary_build_context.py` verifies dependency-light config handoff, command-line allow-listing, integrity-checked snapshot, and fail-closed behavior;
- `tests/test_primary_config_override_integration.py` plus `tests/roots/test-primary-config-overrides/` prove executable `ifconfig` parity when Sphinx dependencies are available;
- adapter tests verify adapter contracts;
- artifact/schema tests verify downstream producer contracts;
- assistant/browser/service integration belongs to its owning subsystem.
