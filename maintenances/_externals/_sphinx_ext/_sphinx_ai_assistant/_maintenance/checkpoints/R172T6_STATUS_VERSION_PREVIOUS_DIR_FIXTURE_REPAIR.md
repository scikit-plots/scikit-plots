# R172T6 — Run 158 previous-directory fixture ownership repair

Date: 2026-09-07

## Trigger

The user-local full suite from the combined Run 172 Fix 5 + proxy Docker-context
repair stopped at:

```text
tests/_hf_spaces_proxy/security/test_verify_attestation_lifecycle.py::
test_run158_status_versions_cannot_skip
```

Observed result: **1 failed, 1291 passed, 4 skipped**. The test expected
`VERSION_NOT_CONSECUTIVE` but `advance_lifecycle()` correctly failed first with
`ATTESTATION_PREVIOUS_DIR_INVALID`.

## Classification

Canonical test fixture/path-ownership typo. Production validation order is correct.

## Root cause

`_initialized(tmp_path)` creates and returns the valid initialized lifecycle directory
as tuple slot 2 (`tmp_path / "lifecycle"`). This test discarded that `out` value and
then passed a nonexistent hard-coded sibling, `tmp_path / "out"`, as
`previous_dir`. The version-continuity code was therefore unreachable.

## Smallest fix

Bind the returned `out` directory and pass `previous_dir=out`. Do not weaken
`advance_lifecycle()` to inspect a new status snapshot before validating the
previous lifecycle directory.

## Verification

- exact reported node: **1/1 passed**;
- Run 158 canonical owner: **32/32 passed** with `ResourceWarning` as error;
- Run 157 predecessor owner: **22/22 passed** with `ResourceWarning` as error;
- Run 159 successor owner: **33/33 passed** with `ResourceWarning` as error.

Input workspace:

```text
scikitplot__sphinx_ai_assistant_run172_fix5_proxy_docker_context_fix1.zip
SHA-256 6163f3fae50aeb85c50f97843f2eb3d863cc23f9043ebedc6dee29e8a5b277ee
```

User log SHA-256:

```text
afe832212ef85cffba8328a6a1dbfa4cf55ec252d1330f7830154c553204bc76
```

## Next authority

The user's next `pytest -x -vv` rerun from the packaged Fix 6 workspace.
