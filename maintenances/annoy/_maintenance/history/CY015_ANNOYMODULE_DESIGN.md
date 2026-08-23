<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# CY-015 — `annoymodule.cc`: structure, HTML/CSS extraction, versioning, UX (design)

The Cython side (`annoy/_annoy/annoylib.pyx.in`) is clean: class-structured, a
data-driven `data_types`/`index_types`/`annoy_metrics` matrix (add a dtype = one
row), a runtime capability API (`supported_dtypes()`), and an installed-but-marked
`.pxd` (CY-003). The C++ side (`cexternals/_annoy/src/annoymodule.cc`, **11,520
lines**) has none of that: it is a monolith, its type dispatch is hand-written, its
HTML/CSS/JS is scattered, and there is no way to introspect its version or
capabilities. This is the design to close that gap. It intentionally mirrors the
Cython side's strengths.

## 1. Current state (grounded)

### 1.1 The HTML repr assets are in THREE places (no single source of truth)
- `_repr_html/{estimator.css, params.css, estimator.js}` — small text files.
- `_repr_html/annoy_repr_assets.{h,cc}` — hand-maintained C++
  (`annoy_repr_assets_append_css/js`) that appends CSS/JS at runtime.
- `annoymodule.cc` `kAnnoyReprCssFallback[]` (~line 10742) — an embedded C++
  fallback used when on-disk assets are missing.
There is **no build step** that derives the C++ from the text files (the meson
`install_subdir('_repr_html')` is even commented out). Editing the UI means
editing several places and recompiling 11.5k lines of C++.

### 1.2 Type dispatch is hand-written
`switch (self->metric_id)` (≈line 1527) plus ~72 metric references. Adding a
metric or dtype means editing switches/factories by hand — the exact opposite of
the Cython `data_types` matrix.

### 1.3 No version / capability introspection
There is a stored `schema_version` marker, but nothing exposes the module's
build version, "stable vs dev" status, or the metric/dtype matrix it was compiled
with. From Python you cannot ask the C++ module what it supports.

## 2. Design (innovative, mirrors the Cython side)

### 2.1 Single-source asset pipeline (the core CY-015 fix)
Make `_repr_html/{estimator.css, params.css, estimator.js}` the ONLY source of
truth and **generate** the C++ from them at build time:

1. Add a tiny generator (`_repr_html/embed_assets.py`) that reads the css/js and
   emits `annoy_repr_assets.gen.cc` as a byte-array/raw-string blob exposing
   `annoy_repr_assets_append_css/js`.
2. Wire it as a meson `custom_target` so any edit to the css/js regenerates the
   embedded C++ deterministically (hash-checked, idempotent — matches the build
   discipline used elsewhere).
3. Delete the duplicated `kAnnoyReprCssFallback` from `annoymodule.cc`; the
   generated blob IS the always-present fallback, and the runtime file-load path
   becomes an optional override for live theming.

Result: iterating on the repr UI is a css/js edit + rebuild — no hand-editing C++
string literals, no drift between the three copies. A guard test can assert the
embedded blob matches the source files (no stale duplication).

### 2.2 Table-driven type dispatch (X-macro registry)
Replace the hand-written metric switch with one table, e.g.:

```cpp
// ANNOY_METRIC_TABLE(X): the single source the factory, switch, parser, and repr
// all iterate. Adding a metric = one row (mirrors Cython `annoy_metrics`).
#define ANNOY_METRIC_TABLE(X) \
  X(ANGULAR,   "angular",   Angular)   \
  X(EUCLIDEAN, "euclidean", Euclidean) \
  X(MANHATTAN, "manhattan", Manhattan) \
  X(DOT,       "dot",       DotProduct)\
  X(HAMMING,   "hamming",   Hamming)
```

The factory (`create index`), the `metric_id` parser, the string round-trip, and
the repr all expand from `ANNOY_METRIC_TABLE`, so adding a metric touches ONE
place and cannot desync the code paths. This is the C++ analogue of the Cython
data-driven matrix.

### 2.3 Version + capability introspection (parity with `supported_dtypes()`)
Add `annoy_c_capabilities()` returning a JSON string (reuse the pattern from
`annoy_type_support.h`) exposed as a module attribute, e.g.
`scikitplot.annoy._c.capabilities()`, reporting:
- module build version + a `dev`/`stable` flag (from a generated version header),
- the metric table (from `ANNOY_METRIC_TABLE`),
- the supported dtypes/index-types the C++ was compiled with,
- feature flags (multithreaded build, mmap, on-disk).
Now "what does the C++ side support / which version is this" is answerable from
Python, exactly like the Cython side.

### 2.4 File modularization (tame the 11.5k-line monolith)
Split by responsibility into separate translation units behind a thin
`annoymodule.cc` that only wires them: `annoy_pytype.*` (the `py_annoy` type +
dunders), `annoy_repr.*` (HTML repr + asset glue), `annoy_dispatch.*` (the
table-driven factory/metric logic), `annoy_convert.*` (the int/size conversion
boundary, CONV-001/CMP-001). Enforce with a section-map guard test that fails if a
file exceeds a line budget or a section header goes missing. Smaller TUs also cut
the compile time that already pushes past one build window.

### 2.5 UI/UX modernization (enabled by 2.1)
Because assets become external + regenerated, the repr can iterate freely:
- **Theming** via CSS custom properties (`--annoy-bg`, `--annoy-accent`) with a
  `prefers-color-scheme` dark/light default and a versioned theme string.
- **Collapsible sections** (params, build stats, on-disk state) using the existing
  scoped-id mechanism (`__ANNOY_REPR_ID__`) so multiple reprs on one page don't
  clash.
- **Richer, at-a-glance stats**: n_items, n_trees, metric, dtype, dimensionality,
  approximate memory, build/loaded state — a compact header card.
- **Copy-to-clipboard** for the reconstruction snippet (`Index(f=…, metric=…)`),
  aiding reproducibility.
- **Responsive + accessible**: semantic roles, keyboard-focusable toggles, no
  layout break in narrow notebook columns.

## 3. Rollout order (low-risk first)
1. **2.1 asset pipeline** — self-contained, high maintainability payoff, guardable;
   unblocks all UI/UX work without recompiling C++.
2. **2.3 capability/version** — additive, mirrors `supported_dtypes()`, no dispatch
   changes.
3. **2.2 table-driven dispatch** — medium risk (touches the factory); big
   extensibility win; do with a golden test that every metric still round-trips.
4. **2.4 modularization** — largest diff; do last, purely structural, behind the
   section-map guard.
5. **2.5 UI/UX** — continuous, once 2.1 lands.

## 4. Why this is the right shape
Each item imports a property the Cython side already proved valuable: single
source of truth (data-driven generation), runtime introspection, one-line
extensibility, and enforceable structure. None of it changes on-disk format or the
Python `Index` API. Sequenced as above, every step is independently shippable and
testable.

*Status: CY-015 design. Implementation deferred; recorded in
`DEFERRED_FUTURE_WORK.md`. Step 2.1 (asset pipeline) is the recommended first
concrete PR.*
