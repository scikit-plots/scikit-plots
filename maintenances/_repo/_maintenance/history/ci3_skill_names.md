<!-- Archived verbatim from maintenances/_repo/todo/todo.md on 2026-10-26 (CI-4). -->

## Task: CI-3 - skill names and folders (Agent Skills specification) (2026-10-26)

- Cause: `name` must equal its folder and use only `a-z0-9-`; folders mirrored
  packages (`skills/_cli/`), so 30 of 31 failed `agentskills validate` and an
  underscore folder could never pass. `mlflow`'s frontmatter did not parse.
- Done: `skills/<name>/SKILL.md`, name = folder, package in
  `metadata.scikitplot-mirror`, name derived from it (Rule 164; three renamed);
  `check_records.py` check `skills` and generated `skills/INDEX.md`; labeler
  `{skill}` satellite; references and checkers updated; family README moved
  to `docs/skills/sphinx_ext.md`.
- Results: 31 of 31 `Valid skill`; `ok skills`; labeler check PASS; related
  suites fail exactly as before the move (55 pre-existing), 33 new tests pass.
