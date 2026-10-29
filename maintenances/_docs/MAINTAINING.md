# `_docs` maintenance: the documentation site

Work on the documentation site as a whole: the Sphinx build and
`docs/source/conf.py`, the API reference (`docs/source/apis_reference.py`),
guides and galleries shared by several areas, and the release-note build
(`_maintenance/tools/check_towncrier_sections.py` checks its sections).

Not here:

- a guide page of one submodule: that submodule's plane
  (`maintenances/<sub>/todo/todo.md`);
- the Sphinx extensions (`scikitplot/_externals/_sphinx_ext/`): their own
  plane, `maintenances/_externals/_sphinx_ext/` (page feedback, the AI
  assistant, the extension stack).

A task that changes both gets a one-line pointer in the other.

- Active tasks: [`todo.md`](todo/todo.md).
- Completed tasks: archived verbatim under `_maintenance/history/` (created
  with the first completed docs-plane task).
- Rules for documentation work:
  [`documentation.md`](../../plugins/_shared/skills/scikitplot-engineering-rules/rules/documentation.md), plus
  the rules every increment loads ([`SKILL.md`](../../plugins/_shared/skills/scikitplot-engineering-rules/SKILL.md)).
- The repository-wide procedure: [`maintenances/MAINTAINING.md`](../MAINTAINING.md);
  where other records live: [`maintenances/README.md`](../README.md).

This area lives under `maintenances/` like every other (Rule 167), outside
the Sphinx source tree (`docs/source/`), and is never built into the site.
