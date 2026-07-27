<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# `annoy/_maintenance/` — review & maintenance hub (dev-only)

Centralized home for the annoy-subsystem deep-review campaign. **Exclude from the
installed wheel** (contributor/CI material, not runtime code).

- `ANNOY_REVIEW_PLAYBOOK.md` — ordered finding ledger (TIER 0–3).
- `RUN0_BASELINE.md`, `RUN0b_MESON_GATE.md` — build/gate baseline + recipe.
- `RUN1_ANNOY_MMAN_001.md`, … — per-run evidence.
- `todo.md`, `lessons.md` — live tracker and prevention rules.
- `gate.py` — scoped-gate runner (rule L-GATE-SCOPE).

Shipped maintainer entry point is one level up: `annoy/MAINTAINING.md`.
