<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# __contains__ / __getitem__ negative-index consistency (R15 follow-up)

**Priority:** P3  **Area:** `annoy/_annoy/annoylib.pyx.in` (`__contains__`)  **Gate tier:** ring

## Finding
`__getitem__` supports Python negative indexing (`self[-1]` is the last item),
but `__contains__` did `if item < 0: return False` — while its own docstring
claimed "True iff `self[item]` is accessible". So `self[-1]` worked yet
`-1 in self` was False, contradicting the documented contract (CY-007/R15 flagged
this as a separate convenience inconsistency).

## Fix
`__contains__` now returns `-len(self) <= item < len(self)`, i.e. True iff the
same-key `__getitem__` would succeed, including negative indexing. Docstring and
Returns section updated to state the `[-len, len)` extent and the equivalence to
`__getitem__` accessibility. Purely widens membership to the negative extent; the
positive-range and sparse-gap semantics are unchanged.

## Verification
`tests/test_contains_getitem_consistency.py` (32): for every key in [-15, 15),
`(k in idx) == (idx[k] accessible)` — zero mismatches; negative specifics
(`-1`/`-10` in, `-11` out), positive boundary (`9` in, `10` out), and empty/no-ptr
(`0`, `-1` not in). Ring green (supported_dtypes 17, save_load 5).
