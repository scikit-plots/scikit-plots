# A00 — Maintenance Bootstrap

Status: **COMPLETE**
Type: **bounded maintenance/control-plane checkpoint**
Subsystem: **_sphinx_llm**

## Objective

Freeze truthful source/upstream anchors, maintenance rules, trackers, schemas,
and fresh-chat workflow before any production changes.

## Scope actually executed

- maintenance/control-plane files only;
- reconcile bootstrap assumptions with `scikit-plots(20260821-211419).zip`;
- record actual vendored-path presence without claiming A01 parity;
- reconcile runtime-artifact vs optional build-time-facade consumer boundary;
- harden the maintenance checker so the old “not vendored” contradiction cannot
  remain silently green.

## Explicit non-goals

- no NVIDIA vendor-file edits;
- no vendor tree hash/parity verification (A01);
- no upstream tests (A01);
- no downstream producer implementation;
- no assistant/backend runtime changes;
- no B14 backend move.

## Execution record

```yaml
checkpoint: A00
status: COMPLETE
started_at: 2026-08-22T00:15:00+03:00
completed_at: 2026-08-22T00:22:00+03:00
source_anchor:
  archive: scikit-plots(20260821-211419).zip
  sha256: 4990b417e7d6309bc3ca2c4691ee735b1fcdf9e698c38a129908419ea80178d6
upstream_anchor:
  repository: NVIDIA/sphinx-llm
  pinned_commit: 2a971d7da6a5d7df81f7bff3612ee1822a060c17
  local_path: sphinx_llm/
  verification: VENDORED_UNVERIFIED_A01
production_code_modified: false
contracts_touched:
  - SLLM-C01 UpstreamCore (owner/status truth only)
  - SLLM-C12 SummaryGeneration (owner/status truth only)
  - SLLM-C14 ConsumerBoundary (clarified runtime vs build-time edge)
files_read:
  - MAINTAINING.md
  - README.md
  - _maintenance/RECONCILIATION.md
  - _maintenance/STATE.json
  - _maintenance/TRACKER.json
  - _maintenance/TRACKER_LOGICAL.md
  - _maintenance/TRACKER_PHYSICAL.md
  - _maintenance/REGISTRY.md
  - _maintenance/UPSTREAM.md
  - _maintenance/MAINTENANCE_MODEL.md
  - _maintenance/SUBMODULE_STRUCTURE.md
  - _maintenance/DEPENDENCY_MAP.md
  - _maintenance/VERIFICATION.md
  - _maintenance/check_trackers.py
  - sphinx_llm/vendor.lock.json
files_changed:
  - MAINTAINING.md
  - README.md
  - upstream/README.md
  - _maintenance/MAINTENANCE_MODEL.md
  - _maintenance/SUBMODULE_STRUCTURE.md
  - _maintenance/UPSTREAM.md
  - _maintenance/DEPENDENCY_MAP.md
  - _maintenance/RECONCILIATION.md
  - _maintenance/TRACKER_LOGICAL.md
  - _maintenance/TRACKER_PHYSICAL.md
  - _maintenance/TRACKER.json
  - _maintenance/STATE.json
  - _maintenance/REGISTRY.md
  - _maintenance/VERIFICATION.md
  - _maintenance/check_trackers.py
  - _maintenance/checkpoints/A00_MAINTENANCE_BOOTSTRAP.md
  - _maintenance/checkpoints/A01_UPSTREAM_VENDOR_BASELINE.md
  - _maintenance/HISTORY.md
findings_opened:
  - A01 vendor parity remains unverified despite vendor.lock.json presence
  - backend maintenance shell exists but its code export/checker are not complete (outside A00)
findings_closed:
  - stale pre-vendoring source anchor/status in _sphinx_llm maintenance control plane
  - conflicting upstream/ vs sphinx_llm/ ownership statements
  - ambiguous artifact-only vs facade consumer wording
risks:
  - the recorded vendor tree may differ from pinned NVIDIA until A01 proves otherwise
  - source archive has no guarantee of current upstream delta; A01 must use explicit external/upstream evidence
  - backend shell maintenance defects are visible but deliberately untouched by this checkpoint
rollback: restore only the A00 maintenance/documentation files; no production rollback is required
```

## Verification gates

- [x] current source archive and SHA-256 recorded
- [x] tracker checker passes after reconciliation
- [x] `STATE.json` and `TRACKER.json` parse
- [x] state/tracker schemas validate in the available A00 environment
- [x] read order is self-contained and begins with reconciliation
- [x] current vendor presence is represented as unverified rather than absent
- [x] reverse dependency rule remains enforced
- [x] no production file changed

## Closure

A00 is complete. The next bounded checkpoint is **A01**, now defined strictly as
verification of the existing vendored NVIDIA baseline. Do not start A02 or any
downstream feature until A01 is closed or explicitly blocked/deferred with
recorded evidence.
