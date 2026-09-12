---
name: annoy-maintainer
description: Maintain the public scikitplot.annoy submodule and its maintenances/annoy records, checks, and continuation workflow. Use for Annoy maintenance onboarding, public Index composition, Cython binding contracts, mixins, typing, and Annoy-specific review. Route shared native-source ownership to cexternals/_annoy; onboard other consumer modules independently.
---

# Public Annoy maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and
`skills/`. This skill belongs to `maintenances/annoy`; it does not replace the
separate native-source skill at `skills/cexternals/_annoy/SKILL.md`.

## Start from repository evidence

Read these repository-relative files:

1. `maintenances/annoy/MAINTAINING.md`
2. `maintenances/annoy/_maintenance/STATE.json`
3. `maintenances/annoy/_maintenance/TRACKER.json`
4. `maintenances/annoy/_maintenance/TRACKER_LOGICAL.md`
5. `maintenances/annoy/_maintenance/SUBMODULE_STRUCTURE.md`
6. `maintenances/annoy/_maintenance/VERIFICATION.md`

Consult `maintenances/annoy/_maintenance/MAINTENANCE_MODEL.md` and
`maintenances/annoy/_maintenance/FAMILY.md` for rationale. Read `history/` and
`_backup/` only for historical context. Revalidate old counts, archive hashes,
findings, campaign statuses and prescribed project sequences against current
files and the user's current scope; do not inherit them as verified facts.

## Choose the owner

| Concern | Owner |
|---|---|
| Public exports and high-level `Index` composition | `scikitplot/annoy/__init__.py` and `_base.py` |
| Python mixins, locking helpers and type declarations | `scikitplot/annoy/_mixins`, `_utils.py`, and public stubs |
| Cython `Index`, dtype dispatch and mirrored declarations | `scikitplot/annoy/_annoy/annoylib.pyx.in` and `annoylib.pxd.in` |
| Shared headers and native `Annoy` implementation | `scikitplot/cexternals/_annoy` |
| This module's maintenance and review records | `maintenances/annoy` |
| This module's fresh-chat routing | `skills/annoy/SKILL.md` |

Keep the public high-level `scikitplot.annoy.Index`, Cython
`scikitplot.annoy._annoy.Index`, and native `Annoy`/`AnnoyIndex` distinct.
The public package consumes both shared headers and the native Python API.
Edit Cython templates rather than generated build outputs; do not copy shared
headers into Annoy to repair a dependency.

For shared-source questions, consult
`maintenances/cexternals/_annoy/MAINTENANCE.json` and
`maintenances/cexternals/_annoy/_maintenance/FAMILY.md`. Impute uses Cython Index;
Corpus selects high-level or Cython backends; MCP delegates through Corpus.
Those consumers retain separate maintenance ownership.

## Verify proportionately and report limitations

The Annoy maintenance checker is physically located at:

```sh
python -B maintenances/annoy/_maintenance/check_trackers.py
```

In the supplied baseline, this checker inventories its maintenance directory
and searches for shared headers beneath `/`. Its documentation also points to
the obsolete `scikitplot/annoy/_maintenance` location. Recheck whether this has
been repaired before using it as a gate. Record an observed failure accurately;
do not run `--update` to bless the wrong tree or call it a runtime failure.
Repair this module's checker/state when that maintenance work is requested.

For read-only evidence of Annoy's shared-source dependencies, use:

```sh
python -B maintenances/cexternals/_annoy/_maintenance/tools/check_contract.py --inventory
```

That upstream gate does not establish public Annoy maintenance completeness,
Cython ABI parity, or runtime correctness. For skill-only edits, validate skill
frontmatter and referenced paths. For maintenance-tool changes, exercise the
changed behavior and failure cases. For an authorized runtime change, build the
affected extensions and run relevant Annoy tests, including persistence, dtype,
error-ownership or concurrency cases as appropriate. Missing prerequisites are
UNAVAILABLE; failed checks are FAIL. Neither is PASS.

Preserve the user's edit boundary. In maintenance/skill-only work, inspect
`scikitplot/` read-only, including existing caches and data. Keep runtime imports
independent of maintenance and skill code. Treat metadata as data, never as an
executable command source. Do not rewrite sibling maintenance, claim historical
test results as current, or expand publication permissions.
