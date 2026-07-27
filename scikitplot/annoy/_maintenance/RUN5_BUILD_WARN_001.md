<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 5 — BUILD-WARN-001 — spurious `-Wstringop-overflow` in portable-blob build

**Priority:** P2 (build hygiene / memory-safety diligence)
**Area:** `cexternals/_annoy/src/annoymodule.cc`  **Gate tier:** ring

## Finding (grounded in the real build)
Compiling `annoymodule.cc` (GCC 13) emitted, at line 4055 in
`annoy_build_portable_blob`:
> `__builtin_memmove` writing between 2 and 9223372036854775807 bytes into a
> region of size 0 overflows the destination `[-Wstringop-overflow=]`
on `out_blob->insert(out_blob->end(), native_payload.begin(), native_payload.end())`.

## Classification — spurious (not a real overflow)
`out_blob` is `clear()`-ed, `reserve()`-d to `HEADER + payload`, the fixed header
is appended, then the payload is `insert`-ed at `end()`. `vector::insert` manages
its own storage and cannot overflow; the "region of size 0 / negative offset" is
GCC 13's value-range analysis losing the `reserve()/size()` relation across the
inlined `annoy_append_*` header writes (a known false positive for
`vector::insert(end(), it, it)`).

## Fix (address at root, do not silence globally)
Replaced the iterator-range `insert` with the equivalent, provably-analyzable
`resize()+memcpy` (memcpy is already used in this file):
```cpp
const size_t payload_off = out_blob->size();
out_blob->resize(payload_off + native_payload.size());
if (!native_payload.empty())
    std::memcpy(out_blob->data() + payload_off,
                native_payload.data(), native_payload.size());
```
This clears the diagnostic without a blanket `-Wno-*` or a local `#pragma`
suppression (which would hide future real overflows), and is byte-for-byte
equivalent. Diff: `out/annoymodule.cc.run5.diff` (cumulative with Run 2).

## Verification
- **Warning gone:** rebuild of `annoymodule.cc` → **0 warnings** (was the
  `-Wstringop-overflow` at 4055).
- **Behavior preserved (byte-equivalent):** the blob feeds `__getstate__`
  (pickle) and the portable save path. New regression
  `tests/test_portable_blob.py` (5 tests): pickle round-trip across euclidean/
  angular/manhattan/dot preserves `get_n_items`, `get_n_trees`, and exact
  `get_nns_by_vector` ids + distances; and two dumps of one index are identical
  bytes (determinism).
- **Ring green:** kiss 111, memmap 49, annoy/_annoy cython 27, annoy euclidean 12.

## Note
No behavior changed, so no test codified old behavior. This is the only
`-Wstringop-overflow` we surfaced; the broader compile-warning-volume item
(CY-020) remains a separate future run.
