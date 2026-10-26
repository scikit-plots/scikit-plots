# Documentation tasks

Work on the documentation site as a whole: the Sphinx build and
`docs/source/conf.py`, the API reference (`docs/source/apis_reference.py`),
guides and galleries shared by several areas, the release-note build, and
`docs/maintenances/`, `docs/skills/`.

Not here:

- a guide page of one submodule: that submodule's plane
  (`maintenances/<sub>/todo/todo.md`);
- the Sphinx extensions (`scikitplot/_externals/_sphinx_ext/`): their own
  plane, `maintenances/_externals/_sphinx_ext/` (page feedback, the AI
  assistant, the extension stack).

A task that changes both gets a one-line pointer in the other.

- Active tasks: [`todo.md`](todo.md).
- Completed tasks: archived verbatim under `history/` (created with the first
  completed docs-plane task).
- Rules for documentation work:
  [`documentation.md`](../../skills/scikitplot-engineering-rules/rules/documentation.md), plus
  the rules every increment loads ([`SKILL.md`](../../skills/scikitplot-engineering-rules/SKILL.md)).
- The repository-wide procedure: [`docs/maintenances/MAINTAINING.md`](../maintenances/MAINTAINING.md);
  where other records live: [`tasks/README.md`](../../tasks/README.md).

These files are outside the Sphinx source tree (`docs/source/`) and are never
built into the site.
