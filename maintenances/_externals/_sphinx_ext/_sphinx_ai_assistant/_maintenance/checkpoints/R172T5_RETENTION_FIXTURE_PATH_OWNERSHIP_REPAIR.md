# R172T5 — retention fixture path-ownership local repair

## Trigger

The user's Python 3.11.14 / pytest 9.0.2 rerun from Fix 4 stopped after
952 passes and 4 skips at
`test_audit_archive_retention.py::test_run161_rejects_archive_auditor_operator_overlap`.
The test destructured `_setup()` with the membership-path slot discarded and then
called `targets.membership_path`, but `targets` is intentionally a list of
`(archive_id, provider, auditor)` tuples.

## Classification

**Canonical test-fixture destructuring / ownership typo.** Production retention,
auditor-independence, membership verification, and archive-health logic were not
implicated. `_setup()` already returns the membership `Path` as its tenth value
and the adapter target list as its eleventh.

## Fix

The failing canonical owner now binds the tenth return value to `mp` and uses it
consistently for membership read, rewrite, and the `membership_path=` argument.
`targets` remains the adapter tuple list passed to production. No runtime or
security implementation was relaxed or changed.

## Verification

```text
exact reported Run161 node                         1/1 passed
Run161 audit-retention canonical owner            40/40 passed
Run160 native-archive neighboring owner           34/34 passed
Run162 witness neighboring owner                  32/32 passed
all canonical owners above warning-strict where replayed
test-layout architecture                             9/9 passed
full pytest collection                              2321 / 0 errors
maintenance drift                                    GREEN
```

## Next authority

The user's next `pytest -x -vv` run against the packaged R172T5 / Fix 5
workspace.

## Packaging rule

Fix 5 remains on the Run 172 local-debugging line. Final archive SHA-256 is kept
external to this checkpoint so maintained evidence does not self-reference the
archive whose bytes it describes.
## Packaged-byte candidate replay

An independently extracted deterministic candidate reproduced the Run 161
canonical owner **40/40** with `ResourceWarning` promoted to error, test-layout
architecture **9/9**, full collection **2321 / 0 errors**, and maintenance drift
**GREEN**. Candidate ZIP membership contained **0** cache/bytecode entries. The
final archive is rebuilt after this metadata freeze; its digest remains external.
