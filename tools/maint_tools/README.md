# Maintenance tools

Repository-local maintenance helpers live here. They are intended to be safe for
humans, CI, and AI-assisted maintenance without requiring prior chat history.

## CI `.pre-commit-config.yaml` configuration

```yaml
ci:
  # Only tells pre-commit.ci to skip that hook; it does not disable it locally.
  # Your local hook will still run on commit if you’ve installed the Git hook with: `pre-commit install`
  # keeps pre-commit.ci from running these hooks in its environment; it doesn’t stop your local pre-push hooks.
  # [check-records, check-skills, check-affiliated-index, check-path-lengths, check-pr-labeler, check-towncrier-sections, check-apis-reference]
  skip:
    - check-records
    - check-skills
    - check-affiliated-index
    - check-path-lengths
    - check-pr-labeler
    - check-towncrier-sections
    - check-apis-reference


repos:

  # Sync with tools/maint_tools/README.md
  - repo: local
    hooks:
      - id: check-records
        name: Run check records
        entry: python -I -B maintenances/_repo/_maintenance/tools/check_records.py
        language: unsupported
        # They'll run once when you push, rather than on every commit.
        # pre-commit install --hook-type pre-push
        stages: [pre-push]
        # scripts should run on every commit regardless of changed files
        always_run: true
        # types: [python]
        pass_filenames: false
        # files: \.py$

      - id: check-skills
        name: Run check skills
        entry: python -I -B tools/maint_tools/generate_usage_skills.py check
        language: unsupported
        stages: [pre-push]
        always_run: true
        pass_filenames: false

      - id: check-affiliated-index
        name: Run check affiliated index
        entry: python -I -B tools/maint_tools/generate_affiliated_index.py check
        language: unsupported
        stages: [pre-push]
        always_run: true
        pass_filenames: false

      - id: check-path-lengths
        name: Run check path lengths
        entry: python -I -B tools/maint_tools/check_path_lengths.py check
        language: unsupported
        stages: [pre-push]
        always_run: true
        pass_filenames: false

      - id: check-pr-labeler
        name: Run check pr labeler
        entry: python -I -B tools/maint_tools/generate_pr_labeler.py check
        language: unsupported
        stages: [pre-push]
        always_run: true
        pass_filenames: false

      - id: check-towncrier-sections
        name: Run check towncrier sections
        entry: python -I -B tools/maint_tools/generate_towncrier_sections.py check
        language: unsupported
        stages: [pre-push]
        always_run: true
        pass_filenames: false

      - id: check-apis-reference
        name: Run check apis reference
        entry: python -m scikitplot._build_utils.generate_apis_reference check
        language: unsupported
        stages: [pre-push]
        always_run: true
        pass_filenames: false
```

## CLI policy for Python helpers

New Python maintenance CLIs should follow the repository's hardened CLI pattern
used by `scikitplot.mcp`:

- importing the module must not parse arguments or mutate the repository;
- build arguments in a `_parser()` function;
- expose `main(argv=None, *, stdout=None, stderr=None) -> int` when practical;
- finish with `raise SystemExit(main())` only under `if __name__ == "__main__"`;
- make read-only behavior the default;
- require an explicit flag such as `--apply` for mutations;
- return non-zero status for validation failures or refused unsafe actions;
- provide deterministic output and a machine-readable mode when automation
  benefits from it;
- avoid network access unless the command explicitly exists for a network task;
- validate repository-relative paths before writing or deleting files.

These rules make helpers testable, scriptable, and safe to call repeatedly.

## Towncrier ownership helper

`generate_towncrier_sections.py` is the canonical helper for release-note section
coverage. The durable ownership policy is stored in
`[tool.scikitplot.maintenance.towncrier]` in `pyproject.toml`.

Common commands:

```sh
# Validate policy, pyproject configuration, fragment directories and fragments.
python tools/maint_tools/generate_towncrier_sections.py check

# Show the expected sections derived from the current repository tree.
python tools/maint_tools/generate_towncrier_sections.py list

# Preview synchronization. This never writes.
python tools/maint_tools/generate_towncrier_sections.py sync --prune-empty

# Apply the same deterministic synchronization after reviewing the preview.
python tools/maint_tools/generate_towncrier_sections.py sync --apply --prune-empty

# Machine-readable forms for CI/AI callers.
python tools/maint_tools/generate_towncrier_sections.py check --json
python tools/maint_tools/generate_towncrier_sections.py sync --prune-empty --json
```

The helper currently derives owners from:

- immediate Python packages under `scikitplot/`;
- immediate components under `libs/`;
- immediate components under `tools/`;
- explicitly configured top-level owner sections; and
- selected deeper Python packages declared in
  `[tool.scikitplot.maintenance.towncrier.nested_owner_sections]`.

The nested-owner mapping is intentionally opt-in. Keys are exact parent package
paths and values are relative child package paths; dotted child paths are allowed
for deeper ownership boundaries. The normal base owner remains active, so unlisted
nested packages continue to roll up to the nearest default owner. For example:

```toml
[tool.scikitplot.maintenance.towncrier.nested_owner_sections]
"scikitplot.externals" = ["_probscale", "array_api_compat"]
"scikitplot._externals._sphinx_ext" = ["_sphinx_ai_assistant"]
```

The helper validates that every requested parent and child is an importable package
in the checkout. Missing paths, invalid dotted names, duplicate/redundant owners, and
conflicts with `exclude_owner_sections` fail explicitly rather than silently changing
release-note ownership. Explicit exclusions such as test-only or archived trees are
declared in `pyproject.toml`, not hidden in the script.

`sync --apply` may rewrite only the marked auto-generated owner block in
`pyproject.toml`, create missing section directories, and remove stale section
directories only when `--prune-empty` is supplied and those directories are
empty or contain only `.gitkeep`. It refuses to delete populated stale sections.

## API reference generator

`scikitplot/_build_utils/generate_apis_reference/` is the canonical maintenance
package for `docs/source/apis_reference.py`. Its adjacent
`apis_reference.py.in` is the durable, human-editable template; the docs file is
generated output and should be reproducible from that template. The package is
installed with Scikit-Plots, exposes importable parsing/customization helpers,
and keeps runtime symbol verification separate from offline rebuilds.

```sh
# Verify generated output exactly matches the canonical template.
python -m scikitplot._build_utils.generate_apis_reference rebuild --check

# Preview a full rebuild. No package API imports and no write.
python -m scikitplot._build_utils.generate_apis_reference rebuild

# Recreate the generated file even if it was deleted.
python -m scikitplot._build_utils.generate_apis_reference rebuild --apply

# Runtime verification requires an installed/importable Scikit-Plots build.
python -m scikitplot._build_utils.generate_apis_reference check
python -m scikitplot._build_utils.generate_apis_reference check --strict-coverage
python -m scikitplot._build_utils.generate_apis_reference plan
python -m scikitplot._build_utils.generate_apis_reference inventory
```

Do not hand-edit `docs/source/apis_reference.py` for durable changes. Edit
`generate_apis_reference/apis_reference.py.in`, run `rebuild`, and review the
diff. Advanced Python tooling may call `reference_blueprint_copy()` to derive an
isolated mutable mapping from the template and pass it to
`build_reference_source()`. Non-literal trusted expressions such as
`_get_guide(...)` and `_get_submodule(...)` are preserved as validated
`PythonExpression` values; rebuild does not execute them.

The existing `generate` command remains the installed-package symbol-list
planner: it detects stale names and unambiguous new exports without executing
the docs configuration. Its dry-run output is discovery evidence; accepted
symbol changes must be written back to the canonical `.py.in` template and a
clean `rebuild --check` must follow.

Runtime generation remains conservative: prefer `__all__` when present,
otherwise inventory locally defined public objects; exclude module objects by
default; auto-assign only unambiguous symbols; and report optional/compiled
import failures as unverified rather than using them as deletion evidence.

## Affiliated / partial-distribution index generator

`generate_affiliated_index.py` keeps `docs/source/affiliated/index.rst` aligned
with the partial distributions under `libs/` without creating a second package
registry. Ownership is read from `scikitplot/_distributions.py`; packaging facts
are read from `libs/_tools/registry.py`; the surrounding human-authored narrative
lives in `docs/source/affiliated/index.rst.in`.

```sh
# Fail when the generated page is stale.
python tools/maint_tools/generate_affiliated_index.py check

# Render the expected page without writing it.
python tools/maint_tools/generate_affiliated_index.py render

# Preview a synchronization as a diff.
python tools/maint_tools/generate_affiliated_index.py sync

# Apply only after reviewing the preview.
python tools/maint_tools/generate_affiliated_index.py sync --apply
```

The helper does not import `scikitplot`, does not use the network, and validates
that the ownership and packaging registries describe the same distributions.
The page intentionally calls these entries *partial distributions*, not
independent affiliated projects: they are maintained from the same repository
and share the `scikitplot` namespace.

## Path-length helper (Windows checkouts)

`check_path_lengths.py` keeps every tracked path short enough for Git on
Windows, which refuses a full path of 260 characters or more unless
`core.longpaths` is set. A Git-based `pip install ... #subdirectory=libs/<name>`
clones the whole repository, so one long file name anywhere breaks every such
install. The budget is derived, not chosen: 259 usable characters minus the
longest clone directory pip can create (a 20-character account name and the
longest distribution name among `scikit-plots` and `libs/*`).

```sh
python tools/maint_tools/check_path_lengths.py budget                      # the model and the number
python tools/maint_tools/check_path_lengths.py check                       # exit 1 on a new violation
python tools/maint_tools/check_path_lengths.py check --prefix-length 108   # one user's prefix
python tools/maint_tools/check_path_lengths.py suggest                     # rename plan (read-only)
python tools/maint_tools/check_path_lengths.py fix --apply                 # git mv + rewrite references
```

Known debt outside the maintenance planes is listed with a reason in
`path_length_baseline.txt`; a listed path that no longer violates fails the
check, so the list only shrinks. `fix` renames only Markdown under
`maintenances/`, deterministically (shared directory
affixes, then filler words, then trailing words; a leading identifier is
kept), and refuses a name it cannot shorten or that would collide.

Until a checkout is fixed, a Windows user can still install with
`git config --global core.longpaths true` (Git for Windows), or by enabling
long paths in Windows itself.

## Test module names (`check_test_module_names.py`)

pytest's default import mode imports a test file in a directory without
`__init__.py` under its bare file name. Two such files with the same name stop
the whole session at collection ("import file mismatch"), and every partial
distribution's own test run fails the same way.

```bash
python tools/maint_tools/check_test_module_names.py            # scikitplot/
python tools/maint_tools/check_test_module_names.py scikitplot maintenances/cleanprompt
```

Each argument is one test session. The check computes the name exactly as
pytest's `prepend` mode does (dotted package path when the directory has
`__init__.py`, bare stem otherwise) and lists every clash. CI runs it in
`pr_check_test_module_names.yml`. A drop-in archive cannot delete files: when
a file is renamed, delete the old one, or this check fails on the clash.


## Labeler generator (`generate_pr_labeler.py`)

Three files drive labeling, all checked by one tool and by CI
(`pr_check_labeler.yml`):

| file | edited | used by |
|------|--------|---------|
| `.github/labeler-by-module.yml` | generated | `pr_labeler.yml` (`actions/labeler`) |
| `.github/labeler-issue.yml` | generated | `pr_labeler.yml` and `issue_labeler.yml` (`github/issue-labeler`) |
| `.github/labeler.yml` | by hand | `pr_labeler.yml` (`actions/labeler`) |

```bash
python tools/maint_tools/generate_pr_labeler.py check            # read-only; CI runs this
python tools/maint_tools/generate_pr_labeler.py sync             # preview
python tools/maint_tools/generate_pr_labeler.py sync --apply     # write the generated files
python tools/maint_tools/generate_pr_labeler.py explain --title "BUG, TST: x" scikitplot/corpus/_base.py
python tools/maint_tools/generate_pr_labeler.py list --json
```

Sources of truth, all in the repository:

- **Module and library labels** come from the Towncrier owner sections
  (`[tool.scikitplot.maintenance.towncrier]`, the same discovery as
  `generate_towncrier_sections.py`): a new package gets its label as soon as it
  gets its release-note section. Run the Towncrier `sync --apply` first, then
  this one. A `libs/<dist>` label also fires for every path the distribution
  ships (`scikitplot/_distributions.py`). Satellites (user guide, gallery,
  maintenance plane, fragments) are added only when they exist.
- **Title labels** come from one registry,
  `[tool.scikitplot.maintenance.labeler.title_prefixes]` and `title_keywords`.
  A prefix needs an explicit terminator (`BUG: x`, `[Bug]: x`, `[BUG] x`,
  `MAINT(ci): x`, `BUG, TST: x`), case-insensitive; `Fix typo`, `Bug in reader`
  and `DOCKER image` get nothing. Keywords match whole words.
- **Templates** `tools/maint_tools/templates/*.yml.in` hold the file headers;
  `@GENERATED_RULES@` is replaced by the rules, so a deleted generated file is
  recreated exactly by `sync --apply`.

`check` fails on: a stale generated file; a glob outside the supported subset
(`[A-Za-z0-9._-]`, `*`, `?`, whole `**` segments, one leading `!`, no
trailing `/**`); an unknown key (the labeler only logs it); a negated glob in
`any-glob-*` (true for every other file, so the label fires on almost every
pull request); a curated label that is also a module label; a label longer
than GitHub's 50 characters (use `label_aliases`); a label named by an issue
template, a workflow `contains(...labels.*.name, '...')` or the gilesbot
changelog settings that no rule defines (`manual_labels` lists the ones set
by hand). Curated globs that match no file are warnings (errors with
`--strict`).
