# Fresh-chat handoff — `_brand`

Current status: **maintenance PASS / runtime FAIL / integration UNAVAILABLE / release BLOCKED**.

Read `STATE.json`, `FAMILY.md`, `VERIFICATION.md`, and `plugins/scikitplot-dev/skills/scikitplot-brand-dev/SKILL.md`. Then run:

```bash
python -B maintenances/_brand/_maintenance/check_trackers.py --json
python -B maintenances/_brand/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/_brand/_maintenance/tests -q -p no:cacheprovider
python -B -m pytest scikitplot/_brand/tests/test__logo.py -q -p no:cacheprovider
```

Open findings: `BRD-EXP-001`, `BRD-CLI-001`, `BRD-RUN-001`, `BRD-BAN-001`, `BRD-BAN-002`. `BRD-EUK-001` is fixed (PR 949); the Eukrina work continues with patch 2 (PNG, ICO and favicon export, CLI) and patch 3 (the scikit-plots logo on the kit). The 271 green logo tests are not banner/figlet release evidence.

Also run the kit and Eukrina tests:

```bash
python -B -m pytest scikitplot/_brand/tests/test__kit.py scikitplot/_brand/tests/test__eukrina.py -q -p no:cacheprovider
```
