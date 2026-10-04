# Test planning for the coverage workflow

`.github/workflows/ci_codecov_test_coverage.yml` runs in three stages:

1. **plan** decides what to test and how many jobs to use;
2. **test** runs one job per shard, each with its own six-hour limit;
3. **report** combines the coverage of every shard, uploads it, and gives the
   run its result under the name `CI Test and Coverage Codecov Reports`.

## What gets tested

| Mode | When | What runs |
|---|---|---|
| `auto` | pull requests; manual runs by default | the submodules the changed files belong to, plus the `tests` directory of every package above them |
| `all` | pushes to `main` and `maintenance/**`; the weekly schedule | every submodule |
| `custom` | manual runs | the submodules you name |

The rules for `auto`, in order:

1. A changed file inside a submodule selects that submodule.
   `scikitplot/corpus/x.py` runs `scikitplot/tests` and `scikitplot/corpus`.
   `scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/x.py` runs
   `scikitplot/tests`, `scikitplot/_externals/tests`,
   `scikitplot/_externals/_sphinx_ext/tests` and the submodule itself
   (directories that do not exist are left out).
2. A changed file directly in a container package, such as
   `scikitplot/_externals/_sphinx_ext/__init__.py`, selects every submodule
   below it.
3. A changed file listed in `full_run_globs` (build files, `pyproject.toml`,
   the planner, the workflow, the install scripts), or a file directly in
   `scikitplot/`, selects everything.
4. Anything else (documentation, galleries, maintenance notes) selects
   nothing. When nothing is selected, no test job starts and the run succeeds.

If the changed files cannot be determined (a new branch, a force push that
removed the previous commit), everything runs.

## Running it by hand

*Actions → CI ☂️ Codecov Test Coverage → Run workflow*:

- **mode** `auto`, `all` or `custom`;
- **submodules**, for `custom`: `corpus`, `scikitplot/cleanprompt`,
  `_externals/_sphinx_ext` (a whole container), or several separated by
  spaces or commas;
- **max_shards**: the most parallel jobs for this run;
- **test_gc**: garbage collection between tests (`young` unless you are
  chasing a leak; see `scikitplot/conftest.py`).

The same plan can be made locally, which is the quickest way to see what a
change would run:

```bash
python .github/scripts/ci_test_plan.py units
python .github/scripts/ci_test_plan.py plan --mode auto --base origin/main --head HEAD
python .github/scripts/ci_test_plan.py plan --mode custom --select "corpus cleanprompt"
```

## `test_plan.json`

| Key | Meaning |
|---|---|
| `package_root` | the package directory |
| `containers` | directories whose children are independent submodules; add a directory here when it starts holding several submodules |
| `full_run_globs` | changed files that make everything run |
| `ignore_globs` | changed files inside the package that select nothing |
| `loose_test_globs` | names of test files; a test file directly in a container is planned by name |
| `default_mode` | `auto` or `all` per event; an event not listed runs `all` |
| `max_shards` | the most test jobs in one run |
| `shard_target_minutes` | a run is split into as many jobs as it takes to keep each near this estimate |
| `shard_timeout_minutes` | `timeout-minutes` of each test job (at most 360 on GitHub-hosted runners) |
| `default_unit_seconds`, `per_test_seconds` | estimate for a submodule with no measurement, and the cost added per test |
| `dependents` | `none`, or `direct` to also test the submodules that import a changed one |

## `test_durations.json`

Per-submodule estimates used only to balance the jobs. Every run publishes a
fresh `test_durations.json` in its `coverage-report` artifact; to adopt it,
copy it over this file and commit.

## Coverage on Codecov

A report is uploaded only by a complete, successful `all` run. Codecov reads a
report as the coverage of the whole project, so a partial run would look like
a collapse. Partial runs still publish their combined `coverage.xml` as an
artifact and their total in the run summary.

## Why the single job used to need six hours

`scikitplot/conftest.py` ran a full `gc.collect()` before and after every
test. With the scientific stack imported, one collection took 0.36 to 0.70 s,
so twenty thousand tests spent more than five hours collecting garbage and
about forty-four minutes testing. The policy is now `young`; see
`SKPLT_TEST_GC` in that file.
