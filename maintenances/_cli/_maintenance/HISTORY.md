# History

## 2026-09-12 — maintenance onboarding

Created a dedicated `_cli` maintenance/skill domain. The runtime tree was left
byte-for-byte unchanged. Review identified two structural findings: declared
`CommandSpec.capabilities` metadata has no enforcement path despite its normative
docstring, and runtime source references a missing `EXTENDING.md`.

The archive is also intentionally incomplete for full command tests: project
`scikitplot.config` and `scikitplot.utils` are absent, and no TOML writer is
installed in the review environment. Those environmental/snapshot gaps are kept
separate from `_cli`-owned defects.
