<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# CY-002 — unconsumed/stale `annoylib.pxi` removed

**Priority:** P2  **Area:** `annoy/_annoy/annoylib.pxi` + `meson.build`  **Gate tier:** ring

## Finding
`annoylib.pxi` was `fs.copyfile`'d into the build (meson.build) but **never**
pulled in by a Cython `include` directive — its own header even had the
`include` line commented out. Its `DEF` constants were unused (0 references in any
`.pyx`/`.pxd`) AND stale/contradictory: it declared `DEFAULT_SCHEMA_VERSION = 1`
while the live code initialises `schema_version = 0`. Exactly the CY-002 risk:
stale constants that mislead maintainers, on the build/package surface.

## Fix (removal — the CY-002 exit gate)
Removed `annoylib.pxi` and its `fs.copyfile('annoylib.pxi')` line from
`meson.build`. The module rebuilds and links cleanly without it, proving it was
unconsumed. Added `tests/test_no_orphan_pxi.py` (2 tests): asserts the stale file
stays gone AND that any `.pxi` present in the `_annoy` package is consumed by an
`include` directive — preventing reintroduction of the orphan-`.pxi` pattern.

## Verification
Meson reconfigure clean (0 `.pxi` refs); annoy `.so` builds + links (exit 0);
guard test 2 passed; ring green (supported_dtypes 17, metric-aliases 18).
