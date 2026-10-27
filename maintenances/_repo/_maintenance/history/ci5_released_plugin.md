## Task: CI-5 - the released plugin in scikit-plots/skills, for the directory (2026-10-26)

- Why: the portal refused scikit-plots/scikit-plots (root `.gitattributes`
  `text`/`eol`; ~108 MiB archived, limit 50); `main` skills name six modules
  PyPI 0.4.0.post11 lacks. Rule 166.
- Done: `tools/maint_tools/build_skills_repo.py` + tests; workflow
  `plugins/maintenances/skills-repo/sync.yml` (newest release tag on the
  default branch, no secret, no cache); README data line, both channels; guides.
- Results: `claude plugin validate .` passes, `agentskills` 22/22, installs as
  version 0.5.0; workflow simulated in six cases; actionlint, zizmor clean.
- Open: copy `sync.yml` to scikit-plots/skills; tag v0.5.0; submit in the portal.
