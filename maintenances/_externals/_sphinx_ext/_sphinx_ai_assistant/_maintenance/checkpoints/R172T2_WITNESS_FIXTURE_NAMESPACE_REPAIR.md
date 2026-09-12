# R172T2 — witness fixture namespace local repair

## Trigger

User local Python 3.11.14 / pytest 9.0.2 rerun from Fix 1 stopped after
882 passes and 4 skips at
`test_anchor_archive_health.py::test_run163_anchor_bootstrap_and_offline_verify`.
The shared Run 162 helper raised `KeyError: 'wit-key-1'` before Run 163 anchor
verification began.

## Classification

**Canonical test-fixture namespace drift.** Production witness/anchor verification
was not reached by the failing path.

`_threshold_root(recovery=False)` created `with-key-*`, `with-identity-*`, and
`with-op-*`, while `_witnessed()` and the Run 162/163+ test family consistently
consume `wit-key-*`. Recovery roots correctly use the independent `rec-*`
namespace.

## Fix

Change the non-recovery fixture prefix only:

```text
with -> wit
```

Do not add aliases, fallback lookup, or relaxed key matching to production code.

## Verification

```text
exact failing Run163 node             1/1 passed
Run162 witness canonical owner       32/32 passed
Run163 anchor canonical owner        22/22 passed
Run164 Merkle transparency owner     21/21 passed
Run166 continuation owner            16/16 passed
Run167 rebridge owner                20/20 passed (4 x 5 exact-node batches)
```

The Run 167 file is intentionally verified in exact-node batches because this
maintenance line already records that Runs 163–168 can exceed a single process
window. Only completed pytest summaries count as evidence.

## Input identity note

The handoff text reported Fix 1 SHA-256
`1a80bec0850fd6d2699aeceedadea144f70427db03b72ef9c3a375c9bc5989cf`,
but the actually uploaded ZIP bytes hash to
`cbe13233bb8e3aa69715c11f611fc3e8cf8a67b910098f5b9a06a5a0b3ce8afd`.
This repair is based on the uploaded bytes; the mismatch is recorded rather than
silently treating the reported digest as verified.

## Next authority

The user's next `pytest -x -vv` run against the packaged R172T2 / Fix 2
workspace.
