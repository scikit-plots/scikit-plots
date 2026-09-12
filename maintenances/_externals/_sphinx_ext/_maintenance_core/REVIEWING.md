# Independent review and PR reconciliation

The `_sphinx_ext` maintenance family supports deterministic review lanes that can be
used by a human maintainer, CI, or a future agent orchestrator. Review metadata is
advisory routing; executable evidence remains the authority.

## Safety boundary

`REVIEW.json` may select only registered check IDs. It cannot contain shell commands,
Python snippets, environment overrides, working directories, scripts, or arbitrary
agent actions. This prevents a review profile from becoming an execution surface.

Independent agent reasoning may propose findings, but a finding becomes repository
evidence only after it is represented by a stable finding code and grounded in current
source, state, or an executable gate.

## Review topology

Each subsystem owns `_maintenance/REVIEW.json` and declares:

- `package_targets`: sibling runtime packages reviewed independently;
- `lenses`: independent review concerns such as architecture, fresh-chat skill routing,
  security, and state/evidence consistency;
- `pr_policy`: which severities block review and whether unavailable release evidence
  blocks promotion.

The workflow is intentionally **independent then reconcile**. One reviewer/lens should
not suppress another reviewer before reconciliation.

### One subsystem, parallel lanes

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  maintenances/_externals/_sphinx_ext/_maintenance_core/tools/review_subsystem.py \
  maintenances/_externals/_sphinx_ext/_sphinx_youtube_gallery/MAINTENANCE.json \
  --jobs 4 --format markdown
```

### Whole family, parallel subsystems

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  maintenances/_externals/_sphinx_ext/_maintenance_core/tools/review_all.py \
  --jobs 4 --format markdown
```

Use `--output <path>` with `review_all.py` to create a PR-review artifact. The generated
report is evidence output, not a committed source of truth.

## PR-ready versus release-promotable

`PR_READY` means no required review lens or package target has a blocking finding.
`RELEASE` is stricter: if a profile marks unavailable release evidence as blocking,
release promotion remains blocked until those gates actually execute.

This distinction allows useful code review in a dependency-limited environment without
turning `UNAVAILABLE` into a false failure or a false pass.

For a release gate, add `--require-release`; the command then exits nonzero when release
promotion is blocked even if the code is PR-ready.

## Finding contract

Every deterministic finding has:

- a severity: `ERROR`, `WARNING`, `UNAVAILABLE`, or `INFO`;
- a stable finding code;
- a concise message;
- an evidence path or source;
- a remediation statement.

Stable codes make findings suitable for CI annotations, PR discussion, suppression
review, and future multi-agent reconciliation without relying on free-form prose.
