<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Deferred / future work (annoy subsystem)

Substantial, valuable extensions intentionally deferred. Each is self-contained
and does NOT block the current subsystem. Recorded here (and referenced from
`../MAINTAINING.md`) so they are not lost.

## (A) Multiprecision backend for float256 / float512  [DEFERRED]
**Goal.** Make the `float256`/`float512` tiers real, usable dtypes rather than
honestly-reported "unavailable".

**Why deferred.** No native/hardware C++ type is that wide; it requires an optional
third-party dependency (`boost::multiprecision` `cpp_bin_float`, or MPFR) wired
into the Meson build. The registry + type aliases already exist behind
`ANNOY_ENABLE_MULTIPRECISION` (see `annoy_type_support.h`), and
`supported_dtypes()` reports them "unavailable" until a backend is compiled in.

**Scope when picked up.**
1. Add an optional `boost` (headers-only `multiprecision`) dependency to
   `cexternals/_annoy/meson.build` behind a `-Dannoy_multiprecision=enabled` option
   that defines `ANNOY_ENABLE_MULTIPRECISION`.
2. Wire `float256_t`/`float512_t` into the `data_types` matrix (pxd.in + pyx.in),
   `is_valid_data_type`, the HammingWrapper whitelist, and `TypeName<>` — exactly
   the pattern used for float80 (see `FLOAT80_DTYPE.md`, rule L-DUAL-CONFIG-DATA-TYPES).
3. Confirm `annoy_sqrt`/`annoy_fabs` resolve for the multiprecision type (add
   specializations if `std::sqrt` overloads are absent).
4. Tests: `supported_dtypes()` reports them usable when built with the backend;
   end-to-end build/query/save-load across metrics; graceful "unavailable" when
   the backend is off (already covered by `test_type_support.cpp`).

**Blocked by / risk.** Build-time dependency management; compile-time cost of the
extra ~32 instantiations per tier; the (B) bridge below limits their *I/O* benefit.

## (B) Widened public I/O bridge (beyond double)  [DEFERRED]
**Goal.** Let precision wider than `float64` actually reach users through
`add_item`/`get_item`/queries, not just internal distance arithmetic.

**Why deferred.** All public I/O flows through the `double`-typed `_w` bridge
(`add_item_w(..., const double* ...)`), so today float80/float128/float256/512 gain
higher-precision *internal* arithmetic only — input is narrowed to double and output
widened from double (CY-012). Widening the bridge is an ABI + on-disk-format change
and ties into the format-contract question (§6.6).

**Scope when picked up.**
1. Decide the wire type for the widened bridge (e.g. `long double`, or a
   templated/typed path) and its ABI/versioning story.
2. Add typed `_w`-equivalent entry points (or a precision-parameterised bridge)
   without breaking the existing `double` API; bump the persisted format version
   and gate load/save on it (parity with R16/R19 format handling).
3. Buffer-protocol / NumPy interop for the wide types (numpy has no true
   float128; `np.longdouble` maps to float80 on x86).
4. Tests: round-trip that a wide value survives add_item→get_item without the
   double narrowing; format-version compatibility (old files still load).

**Blocked by / risk.** ABI break + on-disk format change; the largest of the
deferred items. Should be sequenced with the §6.6 format-contract decision.

---
*Cross-refs:* `FLOAT_TYPE_ANALYSIS_AND_DESIGN.md` (ladder + tiers),
`SUPPORTED_DTYPES_AND_MULTIPRECISION.md`, `FLOAT80_DTYPE.md`.

## (C) CY-015 — annoymodule.cc structure / HTML asset pipeline / UX  [DEFERRED]
**Goal.** Bring the 11,520-line C++ module up to the Cython side's standard:
single-source HTML/CSS/JS assets (build-time generation from the text files, no
scattered embedded copies), table-driven metric dispatch (add a metric = one row),
a C++ version + capability introspection API (parity with `supported_dtypes()`),
file modularization behind a section-map guard, and a themeable/collapsible/
richer HTML repr.

**Why deferred.** Large structural change to a monolith; best sequenced as
independent PRs. Full analysis + rollout order in
`CY015_ANNOYMODULE_DESIGN.md`. Recommended first concrete step: the single-source
asset pipeline (§2.1) — self-contained and unblocks all UI/UX iteration without
recompiling C++.
