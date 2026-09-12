# R172T3 — command-adapter pipe ownership local repair

## Trigger

The user's Python 3.11.14 / pytest 9.0.2 rerun from Fix 2 stopped after
901 passes and 4 skips at
`test_anchor_archive_health.py::test_run163_command_adapter_bounds_output`.
Pytest raised an exception group containing two unraisable `ResourceWarning`s
for unclosed `BufferedReader` objects owned by the Run 163 command adapter.

## Classification

**Production subprocess resource-ownership defect.** The output-bound rejection
itself was correct; the adapter killed/reaped the child but did not close the
parent-owned stdout/stderr pipe objects before raising.

A warning-strict diagnostic sweep of already-existing bounded-output regressions
showed the same copied lifecycle defect in Run 151, Run 159, Run 160, Run 162,
and Run 164 adapters. Run 161 already closed reader pipes and was left unchanged.
Only adapters with a demonstrated existing regression were repaired.

## Fix

For the affected adapters:

- reap killed/timed-out children before propagating failure;
- close stdin on every exit path;
- join reader threads before interpreting overflow state;
- close stdout/stderr (or stdout-only adapters) deterministically;
- preserve the strict child environment and existing output/error semantics.

The existing bounded-output tests now capture the actual `Popen` object and
assert both process reaping and closed pipe ownership. Collection count is
unchanged.

## Verification

```text
bounded-output family, ResourceWarning=error   7/7 passed
Run151 finalize_publication owner             14/14 passed
Run159 native-status owner                    33/33 passed
Run160 native archive owner                   34/34 passed
Run162 witness owner                          32/32 passed (exact-node batches)
Run163 anchor owner                           22/22 passed (exact-node batches)
Run164 Merkle transparency owner              21/21 passed (exact-node batches)
Run166 continuation owner                     16/16 passed
```

During verification a Run 160 ordering race was caught: checking the overflow
flag before reader-thread completion could parse an empty response as JSON.
The final repair keeps drain/join authoritative before overflow classification.

## Next authority

The user's next `pytest -x -vv` run against the packaged R172T3 / Fix 3
workspace.
