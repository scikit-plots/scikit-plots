# Sphinx extension maintainer skills

Skills route a fresh chat to subsystem-owned maintenance state; they are never runtime
dependencies and do not duplicate the common checker machinery.

```text
plugins/scikitplot-dev/skills/scikitplot-sphinx-ai-assistant-dev/SKILL.md     -> AI/security/runtime maintenance entry
plugins/scikitplot-dev/skills/scikitplot-sphinx-youtube-gallery-dev/SKILL.md  -> YouTube/gallery family maintenance entry
plugins/scikitplot-dev/skills/scikitplot-sphinx-ai-learn-dev/SKILL.md         -> JSON-first Learn materializer/pages/publication entry
plugins/scikitplot-dev/skills/scikitplot-sphinx-feedback-dev/SKILL.md         -> page-feedback contract/service entry
plugins/scikitplot-dev/skills/scikitplot-sphinx-collection-dev/SKILL.md       -> shared collection browser entry
```

Each skill's `metadata.scikitplot-mirror` names its package below
`scikitplot/_externals/_sphinx_ext/` (moved here from
`skills/_externals/_sphinx_ext/README.md` when skills became one folder each,
Rule 164).

The first two subsystems own full maintenance state (`MAINTAINING.md`, `STATE.json`,
trackers). The last three do not yet; their skills route to the current-state notes in
`maintenances/_externals/_sphinx_ext/_notes/`, which every skill here reads for the
stack-wide rules: the two checkouts, how tests find the site, and how tests import the
stack.

Generic structural, typed-dependency, capability-ownership, evidence, and security
hygiene rules live in `maintenances/_externals/_sphinx_ext/_maintenance_core/`.
Domain behavior, evidence, todo and lessons remain owned by each subsystem.

A fresh maintainer can inspect the auto-discovered effective runtime DAG and capability
owners with `_maintenance_core/tools/report_architecture.py`. The report is diagnostic;
`MAINTENANCE.json` plus current runtime source remain authoritative.

## Review orchestration

Each maintained subsystem owns a declarative `_maintenance/REVIEW.json`. The family review
engine can review one subsystem's package targets/lenses independently or review all
subsystems concurrently and reconcile afterward. Skills explain domain ownership; they do
not gain authority to execute arbitrary commands from review metadata.

Use `_maintenance_core/tools/review_subsystem.py` for an isolated review and
`_maintenance_core/tools/review_all.py` for family/PR reconciliation. `PR_READY` is not the
same as release-promotable when optional verification remains `UNAVAILABLE`.
