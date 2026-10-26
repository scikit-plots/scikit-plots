<!-- Archived verbatim from maintenances/_repo/todo/todo.md on 2026-10-26 (CI-3). -->

## Task: CI-2 - the six remaining failures and CodeQL 233/234 (2026-10-26)

### Root causes (each reproduced here)
- `test_public_api`: `annoy/conftest.py` (Rule 139) is not API; the test
  listed conftests by hand. Now every `*.conftest` is non-public.
- `misc` `TestLogger` (2): the module moved to `getLogger(__name__)` (Rule
  138), the tests did not; the parent check also depended on test order.
- `TestLogReport`: hand-attached handler, no level; failed once an earlier
  test raised the logger to WARNING (Rule 139). Now `captured_logger`.
- `test_explicit_unavailable_raises`: assumed faiss absent; CI installs
  `faiss-cpu`. Now `sys.modules["faiss"] = None` (Rule 146) plus a twin.
- `test_nesting_deeper...`: `analyse_pattern` caught `RecursionError` from
  a full stack; a GC finalizer there became unraisable. Now a counted bound
  (`_MAX_GROUP_DEPTH = 100`) before recursion (Rule 162).
- CodeQL 233/234: URL checked by substring in `_env` tests (Rule 163).

### Results Review - CI-2 - 2026-10-26
- Each failure reproduced before the fix and passes after (logger raised to
  WARNING; faiss installed; forced GC: unraisables at 154/215 thresholds
  before, 0 after; CodeQL query: 2 results before, 0 after).
- Not verified: a GitHub run.
