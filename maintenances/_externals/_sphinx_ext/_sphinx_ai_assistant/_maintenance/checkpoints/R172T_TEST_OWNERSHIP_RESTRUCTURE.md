# R172T — canonical test ownership restructure

Status: **READY FOR USER LOCAL FULL-SUITE RUN**

## Contract

Python test ownership is exact and mechanical:

```text
source/path/foo.py      -> tests/path/test_foo.py
source/path/__init__.py -> tests/path/test___init__.py
```

Large source modules may keep feature-sized helpers only under hidden
`_cases/<source-module>/` packages. Those fragments never start with `test_`
and are not collected directly; `_case_loader.py` exports their test objects and
fixtures through the single canonical owner. `_integration/`, `_architecture/`,
and `_static/ai_assistant/` are explicit exceptions because they intentionally
span modules or test non-Python assets.

## Restructure findings fixed

- six Run 171/172 config defaults added to the shared Sphinx fixture;
- stale parent-depth path derivation replaced by `tests/_paths.py`;
- old Run-number filenames replaced with semantic module ownership;
- stale direct test-to-test paths rebound after renames;
- pytest 9 fixture objects supported by `_case_loader.py`;
- case-local autouse share fixtures converted to explicit case-local marks;
- imported `TestClient` is not mistaken for a test class;
- discovery-contract paths use stable runtime/test roots;
- ZIP NFC/NFD alias test typo corrected (`café` vs decomposed `café`);
- model/proxy chat-contract copies restored to byte identity;
- Node/mutation harnesses use explicit runtime target arguments rather than old test-parent geometry.

## Verification before user local run

- collection: **2320 tests, zero collection errors**
- layout architecture: **9/9**
- canonicalized Python-owner slice: **361 passed, 2 skipped**
- broad non-security mirrored suite: **1166 passed, 4 skipped**
- path-sensitive release/attestation slice: **34/34**
- browser wrapper: **140/140**
- mutation + privacy/logging mutation: **224/224**
- exact user-reported Sphinx fixture failure cannot execute in this sandbox because `sphinx` is absent; static inventory confirms all six formerly missing defaults are present.

The user's local Sphinx-enabled full-suite run is the next authority.
