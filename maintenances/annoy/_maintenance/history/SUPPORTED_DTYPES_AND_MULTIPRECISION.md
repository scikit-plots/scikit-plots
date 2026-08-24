<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# supported_dtypes() runtime API + multiprecision tier (CY-016)

## Feature 1 — `supported_dtypes()` runtime API (DONE)
New header `cexternals/_annoy/src/annoy_type_support.h` (namespace `annoy_support`)
turns the compile-time float-type facts into a runtime capability registry and a
`report_json()` surface. Exposed in Cython as the module function
`scikitplot.annoy._annoy.annoylib.supported_dtypes()`, which returns a dict keyed
by dtype name; each value carries `size_bytes`, `mantissa_bits`, `tier`
(native / runtime-dispatched / emulated / unavailable), `available`,
`usable_as_dtype`, `io_precision_capped`, and a `note`.

Verified on the uploaded tree: `test_supported_dtypes.py` **13 passed**;
`test_type_support.cpp` **0 failures**. The reported `usable_as_dtype` set
(float16/32/64/128) is cross-checked to EXACTLY match what `Index(dtype=...)`
accepts. Runtime output on the uploaded build:

    float16   runtime-dispatched  mant=11   usable=True  avail=True
    float32   native              mant=24   usable=True  avail=True
    float64   native              mant=53   usable=True  avail=True
    float80   native              mant=64   usable=False avail=True
    float128  native              mant=113  usable=True  avail=True
    float256  unavailable         mant=0    usable=False avail=False
    float512  unavailable         mant=0    usable=False avail=False

## Feature 2 — Multiprecision 96/256/512 tier (infrastructure DONE)
The registry now spans the full ladder:
- **float80 / float96** (`long double`, 64-bit mantissa): a real, native,
  `available` tier — honestly `usable_as_dtype=False` (the C++ type exists but is
  not yet wired into the Cython dtype dispatch).
- **float256 / float512**: gated behind `ANNOY_ENABLE_MULTIPRECISION` +
  `boost::multiprecision` (via `__has_include`). When the backend is absent they
  report `tier="unavailable"`, `size_bytes=0` — **never a silent long-double
  alias**. Verified: compiling with `-DANNOY_ENABLE_MULTIPRECISION` and no boost
  still builds and reports unavailable (graceful degradation). The type aliases
  (`float256_t`/`float512_t` = `cpp_bin_float<237>`/`<493>`) activate
  automatically when boost is installed.

This delivers the honest, future-oriented capability surface. Making float80/256/512
ACTUAL usable dtypes (not just reported) remains the larger dispatch-wiring effort
(the ~160-specialization change), tracked separately.

## Verification (uploaded tree, vrepo2)
Merged pyx regenerates cleanly; extensions build and link; ring green
(metric-aliases 18, float128 4, extended-dtype 115); supported_dtypes 13 passed;
C++ type-support 0 failures.
