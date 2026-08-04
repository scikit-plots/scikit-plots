<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# CY-003 — installed `.pxd` ABI: declared PRIVATE

The generated `annoylib.pxd` (1502 lines, 218 declarations incl. 160+ AnnoyIndex
typedefs) is installed (devel tag) and exposes the internal template-dispatch
surface, which regenerates whenever the dtype/index/metric matrix changes (float80
alone added 40 typedefs). **Decision:** private implementation detail — committing
to ABI stability for auto-generated dispatch would freeze the internal layout.

**Fix:** added an INTERNAL / NO-ABI-stability policy block to the `pxd.in` header
(carried into the generated + installed `.pxd`), pointing downstream users to the
supported Python `scikitplot.annoy.Index` API; kept the `devel` install tag so
packaging matches the "private/devel" decision. Guard: `test_pxd_abi_policy.py`
asserts the policy statement stays in the template.

**Verification:** generated `.pxd` carries the marker (1 occurrence each of the
policy line + heading); guard test 1 passed.

---

# CY-015 — annoymodule.cc: analysis + innovative design (deferred)

Grounded analysis + design in `_maintenance/CY015_ANNOYMODULE_DESIGN.md` and a
deferred entry in `_maintenance/DEFERRED_FUTURE_WORK.md`. Summary of the plan:
single-source HTML/CSS/JS asset pipeline (build-time generation, kills the 3-way
duplication), table-driven `ANNOY_METRIC_TABLE` dispatch (add a metric = one row),
a C++ capability/version introspection API (parity with `supported_dtypes()`),
file modularization behind a section-map guard, and a themeable/collapsible/richer
repr. Recommended first PR: the asset pipeline (§2.1).
