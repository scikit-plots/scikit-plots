# Submodule structure — `scikitplot.random`

```text
scikitplot/random/
  runtime package, Cython implementation/declarations/stubs, Meson build, tests

maintenances/random/
  MAINTAINING.md
  MAINTENANCE.json
  REVIEW.json
  _maintenance/
    FRESH_CHAT_HANDOFF.md
    STATE.json / TRACKER.json / EVIDENCE.json
    FAMILY.md / VERIFICATION.md / MAINTENANCE_MODEL.md
    tools/       deterministic dev-only checks
    tests/       maintenance-tool regression tests
    evidence/    hashed logs used by EVIDENCE.json
    history/     historical rationale and archived legacy tooling

skills/random/
  SKILL.md       fresh-chat routing and maintainer operating rules
```

Do not add generated source, build products, `__pycache__`, chat transcripts or parallel `FINAL`/date-suffixed copies to the live maintenance plane. Put obsolete maintenance tooling in `history/` as inert text when provenance matters.
