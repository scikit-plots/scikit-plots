# Maintaining the Annoy source subsystem

The maintained owner is `scikitplot/cexternals/_annoy`: shared C/C++ sources,
the native Python C-extension, and its local Python surface. It is not the
Cython wrapper at `scikitplot/annoy/_annoy`.

This pass adds maintenance and skill logic for this owner only. `annoy`, `random`,
`memmap`, `impute`, `corpus`, and `mcp` retain their own implementation and
maintenance ownership. Their dependency evidence is observed here.

Start with the [handoff](_maintenance/FRESH_CHAT_HANDOFF.md), then
[state](_maintenance/STATE.json), [family contract](_maintenance/FAMILY.md), and
[verification guide](_maintenance/VERIFICATION.md). The declarative contract is
[MAINTENANCE.json](MAINTENANCE.json); the project-local skill is
[SKILL.md](../../../skills/cexternals/_annoy/SKILL.md).

From the repository root:

```sh
python -B maintenances/cexternals/_annoy/_maintenance/check_trackers.py
python -B maintenances/cexternals/_annoy/_maintenance/review_subsystem.py --json
```

From another working directory, use the script's absolute path. `--repo-root`
selects a wide checkout explicitly. No installed `scikitplot` is imported.

A maintenance pass does not prove the native build. `--release` requires current
passing evidence for every native/consumer gate; unavailable checks block it.
Read `_maintenance/EVIDENCE.json` for the exact tested scope and limitations.
