# History

## 2026-10-10 — maintenance baseline

Established a dedicated maintenance/skill/docs/gallery surface around the new
`scikitplot.levenshtein` facade.

Baseline runtime tests: 23 passed in the focused lane.

Two non-correctness findings remain open:

- LV-001: runtime accelerator failure skips the remaining safe accelerator;
- LV-002: ranking resolves backend capability per candidate.

No runtime behavior was changed as part of creating this maintenance baseline.
