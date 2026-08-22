<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 9 — ANNOY-RNG-002 (guide 6.16) — full RNG state, not seed-only

**Priority:** P1  **Area:** `random/_kiss/kiss_random.{pyx,pxd}`  **Gate tier:** submodule + cross-ring
**Note:** this run changes an observable contract (state restore now RESUMES the
stream) and therefore rewrites the tests that encoded the old restart behavior.

## Finding (verified with the guide's oracle)
`get_state`/`set_state` (and `__reduce__`) saved only `{seed, bit_width, version}`.
A restore re-seeded, so it **restarted** the stream instead of resuming. Confirmed
by the guide's oracle across all five wrappers (draw N, serialize, draw M from the
original, restore, require the restored M to match): all five failed before the fix.

## Root cause
The C++ KISS generators keep four live words (`x, y, z, c`), but they were
commented out in the `.pxd` and never serialized; `__reduce__` also passed `None`
as the pickle state, so pickle reconstructed from the seed alone.

## Fix
- **pxd:** exposed the public `x, y, z, c` members of `CKiss32Random` (uint32) and
  `CKiss64Random` (uint64).
- **pyx:** `Kiss32Random`/`Kiss64Random`/`KissBitGenerator` `get_state` now include
  the live words (bit-width aware for the bit generator); `set_state` restores them
  when present. `Kiss32Random`/`Kiss64Random` `__reduce__` now carry
  `self.get_state()` instead of `None`, so pickle resumes. Generators and the
  random-state wrapper delegate down, so they resume too.
- **Backward compatible:** a legacy seed-only state (no words) still loads —
  `set_state` falls back to the re-seed. Diffs: `out/kiss_random.pyx.run9.diff`,
  `out/kiss_random.pxd.run9.diff`.

## Test rewrites (old contract → correct contract)
13 existing tests asserted the OLD restart behavior (they drew a sequence,
serialized *after*, and expected the restore to reproduce it). Per the workflow,
they were rewritten — not deleted — to the guide's continuation oracle: capture
state, draw the comparison sequence from the ORIGINAL after capture, restore, and
require the restored continuation to match. Rewritten:
`test_pickle_{kiss32,kiss64,bit_generator,generator,random_state}_round_trip`,
`test_{kiss32,kiss64,bit_generator,generator,random_state}_get_set_state`,
`test_set_state`, `test_state_setter`, `test_state_serialization_reproducibility`.
Each carries an `ANNOY-RNG-002 (guide 6.16)` note. Diff: `out/test_kiss_random.py.run9.diff`.

## Verification
- **Independent oracle** (authoritative — not the rewritten tests): a standalone
  probe draws 20, serializes, and requires the next 10 from a restored object to
  equal the original's next 10, for KISS32, KISS64, bit generator, generator, and
  random-state — all pass.
- **New dedicated regression** `tests/test_kiss_state_continuation.py` (12 tests):
  the oracle via both `get_state`/`set_state` and pickle for all five wrappers, the
  presence of the words in the state, and the legacy seed-only fallback.
- **Full kiss suite: 123 passed** (rebuilt). Cross-ring (isolated change): memmap
  49, annoy/_annoy cython 27.

## Reproducibility note
This changes the user-visible pickle/state-restore contract from restart to
resume, matching numpy `BitGenerator`/`Generator` semantics and the guide's
requirement. The RNG *value stream itself* is unchanged; only serialization
fidelity improved.
