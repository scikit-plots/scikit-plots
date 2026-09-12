# Fresh-chat handoff

Current date: 2026-09-12.

1. Read `MAINTENANCE.json`, `REVIEW.json`, `STATE.json`, and `VERIFICATION.md`.
2. Run `python -B maintenances/seaborn/_maintenance/check_trackers.py --json`.
3. Run `python -B maintenances/seaborn/_maintenance/review_subsystem.py --json`.
4. Run the maintenance regression tests.
5. Never run focused tests directly from this partial snapshot and interpret collection errors as seaborn-runtime failures: without root `scikitplot/__init__.py`, the directory is imported as top-level `seaborn` and shadows the external package.
6. For runtime repair, address the findings independently: local color compatibility seam; real `x_estimator` feature-importance implementation; weighted decile semantics; hue mapping forwarding.
7. Preserve runtime bytes unless the task explicitly asks for a repair. Update evidence only from commands actually run.
