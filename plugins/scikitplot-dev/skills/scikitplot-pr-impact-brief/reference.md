# Impact brief: scales and scripts

## The four dimensions

Each brief places the pull request on four scales. Pick the level the
evidence supports and cite it; when the evidence is missing, say "could not
determine", never the middle level by default.

### Reach (the one estimate; label it as such)

| Level | Means |
|---|---|
| few | a rare configuration, an optional dependency path, an internal helper |
| some | one public function or option in ordinary but not universal use |
| many | the default path of a commonly used function, installation, import |

Ground it in something countable when there is one: how often the changed
name appears in the gallery and documentation (`grep -rn NAME galleries docs`),
how many issues report the problem, whether the path is the default.

### Severity, for an affected user without the change

| Level | Means |
|---|---|
| cosmetic | output looks wrong or a message is unclear; results are right |
| inconvenience | a workaround exists and is documented or obvious |
| error | an exception or failure with no reasonable workaround |
| wrong result | silent: the code runs and returns or plots something incorrect |

A silent wrong result outranks a loud error at the same reach.

### Counterfactual: can users get this today?

Look, in this order, and cite what you find: a sibling function or class that
already takes the option; a composition of existing public calls; an
upstream library the user already has (scikit-learn, matplotlib, NumPy) that
does it; a documented workaround in an issue or the docs. "Not found" is a
valid result.

### Cost to keep

List what the project would maintain: new public names or options (from the
API change in `pr_context.py`), a new or loosened dependency (and its upper
bound, which needs a written reason), new generated files or skills, tests
and docs to keep current, interaction with work already planned (open ledger
notes under `upcoming_changes/`, `orient.py`).

## Scripts

Run from the repository's checkout; `gh` must be logged in. Each script reads
only; none writes to GitHub.

### `scripts/worklist.py`

```sh
python3 plugins/scikitplot-dev/skills/scikitplot-pr-impact-brief/scripts/worklist.py --repo scikit-plots/scikit-plots fetch
python3 plugins/scikitplot-dev/skills/scikitplot-pr-impact-brief/scripts/worklist.py --repo scikit-plots/scikit-plots list --counts
python3 plugins/scikitplot-dev/skills/scikitplot-pr-impact-brief/scripts/worklist.py --repo scikit-plots/scikit-plots pick --seed 1
python3 plugins/scikitplot-dev/skills/scikitplot-pr-impact-brief/scripts/worklist.py --repo scikit-plots/scikit-plots --briefs ~/briefs pick --seed 1
```

`fetch` keeps the open pull requests in a cache under `~/.cache` (refreshed
when older than `--max-age-hours`, 24 by default, or with `--force`). `list
--counts` shows how each filter stage narrows the list. `pick` draws at random
from the worklist (open, not draft, opened by someone outside the maintainers,
touching `--paths`, no "Needs ..." label, no maintainer engaged, not yet
briefed in `--briefs`); `--seed` makes the draw repeatable. Options such as
`--briefs`, `--paths` and `--include-maintainers` go before the command.

### `scripts/pr_context.py`

```sh
python3 plugins/scikitplot-dev/skills/scikitplot-pr-impact-brief/scripts/pr_context.py 123 --repo scikit-plots/scikit-plots
```

Prints the pull request's metadata and description (the author's claims, to
check), the issues of the same repository it links, with labels, the change
to the public Python API between the merge base and the head (defaults
included; a renamed file is read under its old name), prior art from a
title-word search of pull requests and issues (merged ones marked), and the
diff (capped by `--diff-lines`). It reads only GitHub's REST API, so it
works from a fork or outside a checkout and changes nothing; a part it
cannot read is printed as "could not compute", never as "no change".
`--no-prior-art` skips the search; `--json` gives the raw bundle.
