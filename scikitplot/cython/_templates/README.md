# scikitplot.cython templates

This folder contains **package data** templates used by `scikitplot.cython`.

Design rules
- Templates are never imported by Sphinx/autodoc.
- Each template may ship a sibling `*.meta.json` metadata file.
- Template IDs are path-based: `"<category>/<name>"`.

Workflows
- `workflow/*` contains runnable script templates (train/hpo/predict/cli).
- Use `scikitplot.cython.copy_workflow()` to copy a workflow anywhere.
