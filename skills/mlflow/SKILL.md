---
name: mlflow-maintainer
description: Maintain scikitplot.mlflow: project/session configuration, managed MLflow server lifecycle, readiness, provider compatibility, security-policy enforcement, workflow/CLI behavior, typing, tests and release evidence without taking ownership of upstream MLflow itself.
---

# `scikitplot.mlflow` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns **scikitplot's MLflow integration contract**, not the upstream `mlflow` implementation.

## Read first

1. `maintenances/mlflow/MAINTAINING.md`
2. `maintenances/mlflow/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/mlflow/_maintenance/STATE.json`
4. `maintenances/mlflow/_maintenance/FAMILY.md`
5. `maintenances/mlflow/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/mlflow/_maintenance/check_trackers.py --json
python -B maintenances/mlflow/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/mlflow/_maintenance/tests -q -p no:cacheprovider
```

## Preserve the ownership boundary

`scikitplot.mlflow` may import/use the external `mlflow` package lazily, launch `python -m mlflow server`, and wrap public APIs. It must not copy upstream server/client implementation or treat mocked providers as proof of real-version compatibility. `MlflowProvider` is the explicit test/customization seam.

## Security policy must be connected, not decorative

`SecurityPolicy`, `set_security_policy()` and `security_policy()` promise that active policies guard session/server operations. A policy object whose validation methods are only unit-tested in isolation is not enforcement. Any repair must route the relevant runtime entrypoints through the active policy before environment mutation, network access or process launch. Keep `SecurityPolicyViolationError` precise; do not silently downgrade it to a generic integration error.

Current finding `MLF-SEC-001`: an active `DEFAULT_SECURITY_POLICY` does not stop `ServerConfig(disable_security_middleware=True)` from reaching `build_server_args()`. Do not mark the security lane green until negative end-to-end policy probes cover actual runtime entrypoints.

## Keep optional dependencies genuinely optional

Importing `scikitplot.mlflow` must not require upstream MLflow or PyYAML. MLflow should remain lazy through `_compat.import_mlflow()` / provider injection. YAML support may import PyYAML at call time, but catch `ImportError`/`ModuleNotFoundError` for absence; do not catch arbitrary exceptions from a broken installed dependency and relabel them as "install PyYAML".

## Preserve subprocess and readiness semantics

Managed server launch must remain argv-based (`shell=False` by construction), use `sys.executable -m mlflow`, create a process group/session suitable for deterministic teardown, fast-fail an immediately exiting process, and bound readiness requests/timeouts. Keep port preflight as a hint rather than a reservation guarantee. Cross-platform process cleanup needs separate POSIX and Windows evidence.

## Treat environment/global MLflow state transactionally

`session()` mutates process environment and upstream MLflow globals. Every mutation must be restored in `finally`, including exceptions during readiness, experiment setup, client creation, run use, or server termination. New global state requires snapshot/restore tests. Provider state and security-policy state must not leak across tests or nested contexts.

## Keep config/project parsing fail-closed

TOML/YAML profiles, project-root markers, relative store paths, server/session fields and bundled configs are user-controlled input. Validate structure before side effects. Keep bundled TOML and YAML semantically aligned where intended, and never infer release compatibility from parsing alone.

## Typing is a shipped contract

`py.typed` is present, so `.pyi` mistakes are user-visible. Stub imports must be package-relative where required and stub/runtime APIs must not silently diverge. Current finding `MLF-TYP-001` tracks the absolute `_facade` import in `_session.pyi`. Use a real type checker in release evidence when available.

## Tests: distinguish logic from integration

The suite is intentionally heavy on stubs/mocks. That is good for deterministic unit coverage, but it does not prove a real MLflow distribution, CLI flag set, server startup, HTTP readiness, artifact APIs, registry behavior, or platform teardown. Keep those as distinct lanes.

Current finding `MLF-TEST-001`: several `TestRunDemo` cases leave the 10-second default sleep active even though the class claims all side effects are patched. Unit tests should replace sleep or pass `open_ui_seconds=0`; do not weaken the runtime workflow merely to speed tests.

## Evidence ladder

Keep these separate:

1. static contract + ownership review;
2. SDK-free/import/unit tests with stubs;
3. bundled TOML/YAML parsing and negative security probes;
4. real installed-MLflow import/version/CLI-capability checks;
5. managed server launch + readiness + teardown;
6. artifact/model/session integration against supported MLflow versions;
7. packaging/type-checking and supported-platform matrix.

Release remains blocked if levels 4-7 are unavailable, even when unit tests are green.


## Change-review sequence

For any change, first classify it as configuration, compatibility, process lifecycle, security, session/global-state, typing, or workflow behavior. Run the smallest focused runtime tests, then the full SDK-free batch, then maintenance mutation tests. If a change touches server arguments or security, add at least one negative test proving a forbidden value is rejected at the real entrypoint rather than only by a helper method.

When changing compatibility code, record the upstream MLflow versions actually exercised. Never update capability assumptions solely from a mocked `--help` string. When changing process teardown, capture both normal exit and forced-kill paths and keep platform-specific expectations explicit.

For public API changes, compare `__all__`, runtime annotations, `.pyi` files, bundled examples and CLI behavior together. A passing runtime import does not prove the typed or documented API is coherent.
