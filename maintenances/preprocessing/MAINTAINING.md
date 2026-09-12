# Maintaining `scikitplot.preprocessing`

`scikitplot.preprocessing` currently owns two encoders with different contracts: pandas-backed `GetDummies` and sklearn-style `DummyCodeEncoder`. Review them independently even though they share one implementation file.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`. Keep maintenance-tool health, runtime structural truth, isolated behavioral evidence, complete-package integration, and release readiness separate.

The 2026-09-12 onboarding review found two `DummyCodeEncoder` correctness defects (cross-feature category collapse and unapplied infrequent grouping), one `GetDummies` parameter-validation mismatch, and one test-portability inconsistency. Runtime source was not modified by the campaign.
