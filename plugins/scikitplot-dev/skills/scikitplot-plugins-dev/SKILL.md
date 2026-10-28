---
name: scikitplot-plugins-dev
description: "Maintain the Claude plugins of scikit-plots: skills, agents, the marketplace, the shared skills, the usage-skill generator and the release to scikit-plots/skills. Use when changing anything under plugins/, .claude-plugin/, the publish workflow or the skill generators, or when a directory submission reports a problem."
metadata:
  scikitplot-mirror: "_plugins"
---

# Skills, agents and plugins (`_plugins`)

Work from the checkout that contains `scikitplot/`, `maintenances/` and
`plugins/`.

## Read first

1. [`maintenances/README.md`](../../../../maintenances/README.md) and the
   engineering rules' [`SKILL.md`](../scikitplot-engineering-rules/SKILL.md);
   Rules 164 to 168 in `delivery.md` govern this area.
2. [`maintenances/_plugins/README.md`](../../../../maintenances/_plugins/README.md)
   (what each piece is and why) and
   [`maintenances/_plugins/MAINTAINING.md`](../../../../maintenances/_plugins/MAINTAINING.md)
   (every procedure, the gotchas and the release).
3. The active tasks in
   [`maintenances/_plugins/todo/todo.md`](../../../../maintenances/_plugins/todo/todo.md).

## Never edit a generated file by hand

`plugins/INDEX.md`, the agents, `plugins/scikitplot/README.md`, the usage
skills, and every copy of a shared skill are generated. Edit the source:
a module's docstrings and `__all__`, a hand-written development skill, or the
shared skill in `plugins/_shared/skills/<name>/`, then regenerate:

```sh
python tools/maint_tools/generate_usage_skills.py sync
python -B maintenances/_repo/_maintenance/tools/check_records.py --write-index
python tools/maint_tools/generate_pr_labeler.py sync --apply
```

## Checks

```sh
python -B maintenances/_repo/_maintenance/tools/check_records.py      # every line ok
python tools/maint_tools/generate_usage_skills.py check
python tools/maint_tools/build_skills_repo.py check --out <skills checkout> --version <X.Y.Z>
claude plugin validate ./plugins/scikitplot
```

A shared skill that ships programs must tell the user what each does and ask
before every run; the users' README discloses it automatically.
