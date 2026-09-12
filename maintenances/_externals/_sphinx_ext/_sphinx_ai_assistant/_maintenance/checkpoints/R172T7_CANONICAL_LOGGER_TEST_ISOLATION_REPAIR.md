# R172T7 — canonical logger test-isolation repair

Date: 2026-09-07

## Trigger

The user-local full suite from Run 172 Fix 6 + retained proxy Docker-context repair stopped at:

```text
tests/test___init__.py::TestGenerateMarkdownFiles::test_disabled_by_config_no_md
```

Observed result: **1 failed, 1935 passed, 4 skipped**. `generate_markdown_files()` called
`log.info(...)`, but the canonical module logger was still a minimal `_Log` object created by
`test_b42_parent_origin_validator_and_generated_policy`.

## Classification

Cross-test global-state leakage / test isolation defect. Production logging behavior is correct.

## Root cause

The hostile-parent integration test imports the canonical package object and assigned
`m._logger = _Log()` directly. Because that is the same module singleton later used by
`test___init__.py`, the temporary warning-only logger escaped the test and survived until the
Markdown-disabled branch required `.info()`. Other direct `_logger` assignments found in the
integration tree load private module copies with `spec_from_file_location` and therefore do not
mutate the canonical singleton.

## Smallest fix

Give the hostile-parent test pytest's `monkeypatch` fixture and replace the direct assignment with
`monkeypatch.setattr(m, "_logger", _Log())`. Pytest now restores the prior logger automatically at
test teardown. Do not weaken `_get_logger()` or guard legitimate `.info()` calls in production.

## Verification

- untouched Fix 6 ordered reproducer (hostile-parent node -> Markdown-disabled node): **1 passed, 1 failed** with the same `_Log.info` error;
- repaired ordered reproducer: **2/2 passed**;
- hostile-parent integration owner: **11/11 passed**;
- `TestGenerateMarkdownFiles`: **10/10 passed**;
- hostile-parent owner + canonical `test___init__.py`: **633 passed, 3 skipped**;
- complete `_integration/` plane: **182/182 passed** (one unrelated unraisable sqlite warning in this local Python 3.13 environment);
- test-layout architecture: **9/9 passed**;
- full collection: **2322 tests, zero collection errors**;
- extracted candidate bytes: **ordered pair 2/2, retained Docker-context regression 1/1, layout 9/9, 2322/0 collection, maintenance GREEN, ZIP integrity GREEN**.

Input workspace:

```text
scikitplot__sphinx_ai_assistant_run172_fix6_proxy_docker_context_fix1.zip
SHA-256 f08dab2368162ff06f8377971dbce17fee4002da39d2e31df0e5a58bee9280b1
```

User log SHA-256:

```text
839f0ac2874f88c6c21ba0d94f7e53daa4897c632ddca09503cb162f34281180
```

## Next authority

The user's next `pytest -x -vv` rerun from the packaged Fix 7 workspace.
