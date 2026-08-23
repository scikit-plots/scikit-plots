# `_sphinx_llm` Maintenance History

## 2026-08-20 — maintenance bootstrap prepared

- Established `_sphinx_llm` as a planned independent producer subsystem.
- Selected NVIDIA `sphinx-llm` commit `2a971d7da6a5d7df81f7bff3612ee1822a060c17` as the pinned baseline
  for the first vendoring review.
- Selected `sphinx-llms-txt` commit `9d0660ba71c3c5dfe3023ebc2d281ddcb3070241` as a secondary curation
  design reference.
- Added planned contracts for semantic node adapters, static artifacts,
  curation, manifest/provenance, optional full output, HTML fallback, and a
  stable assistant consumer facade.
- No production implementation was vendored or modified by this bootstrap.

## 2026-08-22 — A00 reconciled against the live source snapshot

- Re-anchored maintenance truth to `scikit-plots(20260821-211419).zip` (`4990b417e7d6309bc3ca2c4691ee735b1fcdf9e698c38a129908419ea80178d6`).
- Recorded the already-present `sphinx_llm/` vendor tree as
  `VENDORED_UNVERIFIED` rather than falsely `NOT_YET_VENDORED`.
- Kept `sphinx_llm/` as the vendor path; demoted `upstream/` to a retired
  placeholder so preserved upstream tests need no rename edits.
- Changed A01 from a vendoring action to a bounded vendor-integrity/parity
  verification checkpoint.
- Clarified the consumer boundary: runtime uses static artifacts; a future
  Python facade, if needed, is build-time-only and cannot expose private vendor
  internals.
- Noted that `_sphinx_ai_backend` now has a maintenance shell but B14's service
  move has not occurred; its checker is outside A00 and is not silently treated
  as green.
- Hardened `check_trackers.py` to detect vendor-presence/status contradictions,
  checkpoint-state inconsistencies, retired-placeholder misuse, and reverse
  assistant dependencies.
- No production or vendored NVIDIA file was modified.

## 2026-08-22 — A01 pinned NVIDIA baseline closed

- Verified 13 upstream-derived source/test/license files byte-for-byte against
  NVIDIA commit `2a971d7da6a5d7df81f7bff3612ee1822a060c17` and classified them
  `UPSTREAM_PRESERVED`.
- Added a relocation-stable vendor manifest after proving the legacy
  `bash-sha256sum` aggregate is checkout-path dependent.
- Preserved NVIDIA's 9-file `docs/source` fixture and built an ephemeral
  upstream-shaped test harness so preserved tests need no local edits.
- Preserved upstream `pyproject.toml` and `uv.lock`, derived the complete
  Python-3.13 50-distribution lockset, and made the local runner fail closed on
  any distribution/version mismatch.
- Recorded that local exact-lock execution is still `ENVIRONMENT_BLOCKED`
  (16 exact, 15 mismatched, 19 missing) because package retrieval is unavailable;
  this result was not relabeled GREEN.
- Preserved and byte-verified NVIDIA's pinned test workflow and accepted the
  explicit `PINNED_UPSTREAM_CI_EQUIVALENT` proof mode only after binding local
  source/tests/docs/project/lock/workflow bytes to the same exact commit and
  verifying official Test #211 plus its Python 3.13 / Sphinx `>=9,<10` job
  succeeded.
- Closed `SLLM-C01` for the pinned baseline. Broader Python/Sphinx compatibility
  remains A02 work; no downstream adapter/curation/assistant implementation was
  mixed into A01.
- No NVIDIA vendor or runtime production file was modified.

## 2026-08-22 — A02 permanently deferred

- Maintainer decision: the ten-cell downstream parity matrix will not be
  executed, by CircleCI or locally. A02 moves `BLOCKED` -> `DEFERRED_PERMANENTLY`
  under new RULESET rule 51.
- What stays proved: NVIDIA upstream matrix GREEN 10/10 on the pinned commit;
  the downstream config-parity shim source- and unit-GREEN (10 tests); the
  `ifconfig` integration fixture present.
- What is permanently unproved: shim behaviour across Python 3.9-3.14 x
  Sphinx 5-9. `A02-G01` stays open with its risk statement retained.
- `check_trackers.py` gained `mpath()`/`rpath()` so the deliberately filed
  `_maintenance/history/` working set resolves without editing any recorded
  path string or digest, and its seven evidence loads became fail-closed: a
  missing evidence file is now an error, not a skipped check.
- The repository `.circleci/config.yml` requirement is dormant under the
  deferral. Its recorded integrated digest is still enforced whenever the file
  is present, so a drifting workflow is still caught; only absence is tolerated.
  The supplied config was verified: sha256 53861e02... matches the recorded
  `integrated_sha256`, so the semantic rebase of rule 50 is applied and closed.
- A02_MATRIX_PLAN.json, run_a02_matrix.py, run_a02_config_parity.py and
  ci/run_a02_cell.sh are retained deliberately: they define what an equivalent
  matrix would have to prove if the deferral is ever reversed.
