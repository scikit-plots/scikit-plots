<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 24 — metric-alias error honesty (R21 follow-up)

**Priority:** P2  **Area:** `annoy/_annoy/annoylib.pyx.in` (metric validation errors)  **Gate tier:** ring

## Finding
`parse_metric` accepts 11 documented aliases beyond the 5 canonical metric names
(angular→cosine; euclidean→l2,lstsq; manhattan→l1,cityblock,taxicab;
dot→@,.,dotproduct,inner,innerproduct; hamming). But the two "Invalid metric"
error sites were inconsistent and under-informative: one listed a partial alias
set with "etc.", the other listed no aliases at all — so a user who tried
`metric="cosine"` and mistyped got an error implying only 5 values exist.

## Fix
Both error messages now use identical, honest text: they list the five canonical
names AND state that documented aliases are accepted (with representative
examples: cosine, l2, l1, cityblock, taxicab, dotproduct, inner) and point to the
class docstring for the full list. No behaviour change — purely message honesty +
consistency. Diff: `out/annoylib.pyx.in.run24.diff`.

## Verification
`tests/test_metric_aliases.py` (18): all 5 canonical + all 11 aliases construct
and build; the invalid-metric error lists every canonical name, mentions aliases,
and points to the docstring; and each alias yields identical neighbours to its
canonical metric (cosine≡angular, l2≡euclidean, taxicab≡manhattan). Both source
error strings are identical. Ring green: doc-parity 15, euclidean 12, kiss 123.
