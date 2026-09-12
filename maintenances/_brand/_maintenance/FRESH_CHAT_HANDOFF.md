# Fresh-chat handoff — `_brand`

Current status: **maintenance PASS / runtime FAIL / integration UNAVAILABLE / release BLOCKED**.

Read `STATE.json`, `FAMILY.md`, `VERIFICATION.md`, and `skills/_brand/SKILL.md`. Then run:

```bash
python -B maintenances/_brand/_maintenance/check_trackers.py --json
python -B maintenances/_brand/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/_brand/_maintenance/tests -q -p no:cacheprovider
python -B -m pytest scikitplot/_brand/tests/test__logo.py -q -p no:cacheprovider
```

Open findings: `BRD-EXP-001`, `BRD-CLI-001`, `BRD-RUN-001`, `BRD-BAN-001`. The 271 green logo tests are not banner/figlet release evidence.
