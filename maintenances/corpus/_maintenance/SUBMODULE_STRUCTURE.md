# Corpus structure contract

```text
scikitplot/corpus/              runtime source, tests, examples
maintenances/corpus/            maintenance metadata, evidence, tools, history
skills/corpus/SKILL.md          maintainer onboarding/continuation skill
```

Historical source modules named by the prior tracker are now represented as restoration contracts in `MAINTENANCE.json`. They are not recreated by maintenance tooling. Runtime source belongs only under `scikitplot/corpus/` and must not import either developer-only plane.

The historical `checkpoints/` directory is retained only for provenance. New live state goes into `STATE.json`, `REGISTRY.md`, `EVIDENCE.json`, and concise history/changelog records rather than new chat/run-specific checkpoint files.
