<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Continuation pointer

The CY register is substantively complete (see `BURNDOWN.md`). If resuming:

1. **Optional cleanup:** remove `_annoy/backup_template/` (6 dead template copies).
2. **CY-015 §2.1 (recommended first real PR):** single-source HTML/CSS/JS asset
   pipeline for annoymodule.cc — see `CY015_ANNOYMODULE_DESIGN.md`.
3. **Deferred (A)/(B):** multiprecision backend; widened I/O bridge —
   `DEFERRED_FUTURE_WORK.md`.
4. **CI-tier:** wire the C++ warning-budget guard (`-Werror=unused-result` …) and a
   ThreadSanitizer concurrency job into CI.

Standing workflow, gate recipe, and all closed-finding evidence are in this folder
and `../MAINTAINING.md`.
