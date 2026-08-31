# B62 — Run 42 Test Resource Hygiene

Status: **GREEN**

## Scope

Fix a test-only resource leak exposed by strict pytest warning handling in
`test_run42_review_continuity.py`.

The production CORS behavior was already correct. The failing test opened the
Cloudflare Worker and Hugging Face Space proxy source files with bare
`open(...).read()` calls and left both handles to garbage collection. Under a
configuration that promotes `ResourceWarning`/unraisable warnings to failures,
this produced an `ExceptionGroup` even though both CORS assertions were valid.

## Change

The test now uses `pathlib.Path.read_text(encoding="utf-8")` for both source
reads. This makes file lifetime deterministic and preserves the existing
semantic assertions unchanged.

A scan of the Python test tree found no other bare `open(...).read()` source
reads; existing file opens use context managers or equivalent managed APIs.

## Runtime impact

None.

- no proxy code changed;
- no Worker code changed;
- no CORS policy changed;
- no contribution/review lifecycle behavior changed.

## Verification

- `test_run42_review_continuity.py`: 12/12 passed;
- replayed with `ResourceWarning` promoted to an error: GREEN;
- Python test-tree bare unmanaged source reads: none remaining;
- package cache/bytecode contamination removed before release.
