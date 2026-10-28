---
name: scikitplot-docs-dev
description: "Maintain the scikit-plots documentation site as a whole: the Sphinx build, docs/source/conf.py, the API reference, shared guides and galleries, and the release notes (towncrier). Use for work on docs/ that no single submodule owns, or when a documentation build fails."
metadata:
  scikitplot-mirror: "_docs"
---

# The documentation site (`_docs`)

Work from the checkout that contains `scikitplot/`, `maintenances/` and
`plugins/`.

## Read first

1. [`maintenances/README.md`](../../../../maintenances/README.md) and the
   engineering rules' [`SKILL.md`](../scikitplot-engineering-rules/SKILL.md);
   load `documentation.md` from its rules.
2. [`maintenances/_docs/MAINTAINING.md`](../../../../maintenances/_docs/MAINTAINING.md):
   what this area owns, and what belongs to a submodule or to the Sphinx
   extensions instead.
3. The active tasks in
   [`maintenances/_docs/todo/todo.md`](../../../../maintenances/_docs/todo/todo.md).

## What it owns

- `docs/source/conf.py`, `docs/source/apis_reference.py`, guides and galleries
  shared by several areas, and the release-note build.
- Not a submodule's own guide page (that submodule's area), and not
  `scikitplot/_externals/_sphinx_ext/` (the `scikitplot-sphinx-*-dev` skills).

## Checks

```sh
python -B maintenances/_docs/_maintenance/tools/check_towncrier_sections.py
python -B maintenances/_repo/_maintenance/tools/check_records.py
```

Build the site the way `docs/` documents it before claiming a fix works, and
record the plan and results in `maintenances/_docs/todo/todo.md`.
