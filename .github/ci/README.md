# Test and coverage workflow: guide

Everything about `.github/workflows/ci_codecov_test_coverage.yml`: what it
runs, how to run it by hand, and how to change it.

## 1. At a glance

| Question | Answer |
|---|---|
| How many test jobs? | **One** (`max_shards: 1`) |
| How long? | About 30 minutes: 6–7 to install and build, about 24 to test |
| What does a pull request test? | Only the submodules its changed files belong to |
| What does a push to `main` test? | Everything |
| Where is the configuration? | `.github/ci/test_plan.json` |
| Which check do I require in branch protection? | `CI Test and Coverage Codecov Reports` |

The workflow has three stages:

1. **plan** decides what to test (`.github/scripts/ci_test_plan.py`). It
   installs nothing and takes a few seconds.
2. **test** installs the dependencies, builds the library and runs `pytest`
   with coverage.
3. **report** combines the coverage, uploads it to Codecov, and gives the run
   its result. It runs even when nothing needed testing, and then succeeds.

## 2. What gets tested

| Mode | Used for | What runs |
|---|---|---|
| `auto` | pull requests; manual runs by default | the submodules the changed files belong to, plus the `tests` directory of every package above them |
| `all` | pushes to `main` and `maintenance/**`; the weekly schedule | every submodule |
| `custom` | manual runs | the submodules you name |

### The rules of `auto`, in order

1. **A changed file inside a submodule selects that submodule.**
   `scikitplot/corpus/x.py` runs `scikitplot/tests` and `scikitplot/corpus`.
   `scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/x.py` runs
   `scikitplot/tests`, `scikitplot/_externals/tests`,
   `scikitplot/_externals/_sphinx_ext/tests` and the submodule itself.
   Directories that do not exist are left out.
2. **A changed file directly in a container package** (for example
   `scikitplot/_externals/_sphinx_ext/__init__.py`) selects every submodule
   below it.
3. **A changed file listed in `full_run_globs`** (build files,
   `pyproject.toml`, `pytest.ini`, the planner, this workflow, the install
   scripts), **or a file directly in `scikitplot/`**, selects everything.
4. **Anything else** (documentation, galleries, maintenance notes) selects
   nothing. No test job starts and the run succeeds.

If the changed files cannot be determined (a new branch, a force push that
removed the previous commit), everything runs.

### What is never tested

A directory listed in `norecursedirs` in `pytest.ini` is not part of the
suite. The planner reads that setting from `pytest.ini` and follows it:

- such a directory is never handed to `pytest`;
- a change inside it runs only the `tests` directories above it;
- it cannot be chosen in `custom` mode (the plan fails and says why).

Today these are `scikitplot/cexternals/_astropy` and `scikitplot/externals`:

```bash
python .github/scripts/ci_test_plan.py units --excluded
```

To bring one into the suite, remove it from `norecursedirs`. Nothing else
needs changing.

## 3. Running it by hand

*Actions → CI ☂️ Codecov Test Coverage → Run workflow.*

| Input | Values | Default | Meaning |
|---|---|---|---|
| `mode` | `auto`, `all`, `custom` | `auto` | see section 2 |
| `submodules` | names separated by spaces or commas | empty | for `custom`: `corpus`, `scikitplot/cleanprompt`, or a whole container such as `_externals/_sphinx_ext` |
| `max_shards` | a whole number | empty: the value in `test_plan.json` | the most parallel test jobs for this run |
| `test_gc` | `young`, `module`, `test`, `off` | `young` | garbage collection between tests, see section 8 |
| `max_failures` | a whole number | empty: `50` | failures after which the job stops; `0` means never |

`pytest.ini` stops at the first failure, which suits a local run. The
workflow overrides that so one run reports everything that is wrong.

### Seeing the plan without running anything

```bash
# what a pull request against main would test
python .github/scripts/ci_test_plan.py plan --mode auto --base origin/main --head HEAD

# named submodules
python .github/scripts/ci_test_plan.py plan --mode custom --select "corpus cleanprompt"

# everything
python .github/scripts/ci_test_plan.py plan --mode all

# what can be selected, and what is excluded
python .github/scripts/ci_test_plan.py units
python .github/scripts/ci_test_plan.py units --excluded
```

The plan is printed as JSON: the mode, the reasons, and the paths each job
gives to `pytest`.

## 4. Changing the setup

Every change below is one edit in `.github/ci/test_plan.json`, unless it says
otherwise. That file is in `full_run_globs`, so the pull request that changes
it runs everything once.

### Run in several parallel jobs

One job is enough today. If the suite grows, or you want results sooner:

```json
"max_shards": 3,
"shard_target_minutes": 12,
```

The number of jobs is the estimated total time divided by
`shard_target_minutes`, rounded up, and never more than `max_shards`. The
estimated total is about 34 minutes today, so:

| `max_shards` | `shard_target_minutes` | Jobs |
|---|---|---|
| `1` | any | 1 |
| `3` | `30` | 2 |
| `3` | `12` | 3 |
| `4` | `9` | 4 |

Each job installs and builds on its own (6–7 minutes), so more jobs finish
sooner but use more runner minutes. Each job has its own six-hour limit.

For a single run, the `max_shards` input does the same without a commit.

### Test everything on every pull request

```json
"default_mode": { "pull_request": "all", ... }
```

### Also test the submodules that import a changed one

```json
"dependents": "direct"
```

A change in `scikitplot/utils` then also runs every submodule with a module
that imports `scikitplot.utils`. One step only: importers of importers are
not followed.

### Add a package whose children are independent submodules

Add its path to `containers`. Its children then become separate units, and a
change in one of them no longer runs its siblings.

### Make a file trigger a full run, or no run

Add a glob to `full_run_globs` (everything runs) or to `ignore_globs`
(a changed file inside the package that selects nothing, for example
`scikitplot/**/*.md`).

### Give the job a shorter time limit

`shard_timeout_minutes` is the `timeout-minutes` of each test job. It is 350,
just under the six-hour limit of a GitHub-hosted runner. A lower value stops
a hung run sooner.

## 5. `test_plan.json` reference

| Key | Now | Meaning |
|---|---|---|
| `package_root` | `scikitplot` | the package directory |
| `pytest_ini` | `pytest.ini` | where `norecursedirs` is read from. `""` if the project has no such file; a file that is named but missing is an error |
| `containers` | 5 paths | directories whose children are independent submodules |
| `full_run_globs` | 12 globs | changed files that make everything run |
| `ignore_globs` | none | changed files inside the package that select nothing |
| `loose_test_globs` | `test_*.py`, `*_test.py` | names of test files |
| `default_mode` | see section 2 | `auto` or `all` per event; an event not listed runs `all` |
| `max_shards` | `1` | the most test jobs in one run |
| `shard_target_minutes` | `30` | a run is split so each job stays near this estimate |
| `shard_timeout_minutes` | `350` | time limit of each test job (at most 360) |
| `default_unit_seconds` | `60` | estimate for a submodule with no measurement |
| `per_test_seconds` | `0` | extra estimate per test; `0` because the measured times already include it |
| `dependents` | `none` | `none`, or `direct` to add importers |

## 6. `test_durations.json`

Measured time per submodule. It is used only to estimate a job's length and
to balance several jobs; it never decides *what* is tested.

Every run publishes a fresh `test_durations.json` in its `coverage-report`
artifact. To adopt it, copy it over `.github/ci/test_durations.json` and
commit. Do this after a complete `all` run, when the suite has changed a lot.
The current file comes from the run of 5 October 2026 (23 049 tests).

## 7. Coverage and Codecov

- Each test job writes `.coverage.<job name>` and a JUnit report, kept as the
  artifact `coverage-<job name>` for 7 days.
- The report job combines them into `coverage.xml`, kept as the artifact
  `coverage-report` for 14 days, and writes the total in the run summary.
- **Codecov receives a report only from a complete, successful `all` run.**
  Codecov reads a report as the coverage of the whole project, so a partial
  run would look like a collapse.
- The upload step does not fail the run. If Codecov shows no report for a
  green run, open the report job's log at *Upload coverage reports to
  Codecov*: a network error there means the upload did not happen. Re-run the
  report job.

## 8. Garbage collection between tests

`scikitplot/conftest.py` collects garbage between tests so that one test's
figures and reference cycles do not reach the next. `SKPLT_TEST_GC` (the
`test_gc` input) chooses how:

| Value | After each test | After each test file | Cost |
|---|---|---|---|
| `young` (default) | young generations only | full collection | about 0.76 s per test file: 403 s of the suite's 1482 s (measured) |
| `module` | nothing | full collection | the same |
| `test` | full collection, before and after | — | about 1 second per test: hours |
| `off` | nothing | nothing | none |

Use `test` only while chasing a memory leak, and only with `custom` mode on
one submodule. It was the old behaviour and the reason the single job needed
six hours: a full collection walks every live object in the process, which
takes 0.4–0.7 s with the scientific stack loaded, and it ran twice per test.
The old job spent 5.9 hours on 17 838 tests and was cancelled at the limit;
the whole suite of 23 049 tests now takes about 23 minutes of test time.

### The check that keeps it from coming back

A cost added to every test is invisible test by test: no test looks slow,
and `--durations` does not list it. So `scikitplot/conftest.py` measures it
and every run ends with:

```text
================================ cost per test =================================
cost of every test: 0.003 s (5% quantile of 22792 tests)
garbage collection (SKPLT_TEST_GC=young): 22792 young in 1.9 s, 531 full in 403.1 s
```

(The single-job run on `main`, 5 October 2026.)

- **First line:** what a test costs even when it does nothing, read from the
  fastest tests of the run. It is 0.003 s today and was 0.73 s and more in
  the six-hour run.
- **Second line:** how many collections ran and how long they took. Expect
  one full collection per test file. They are the larger part of what the
  policy costs: about a quarter of the test time in that run.
- **The workflow sets `SKPLT_TEST_FLOOR_BUDGET=0.1`.** If every test costs
  more than 0.1 s, the test job fails although all tests passed, and says
  so. On your machine the variable is unset: you get a warning, never a
  failure.
- It needs 200 finished tests to judge, and it never fails a run made with
  `SKPLT_TEST_GC=test`.

The rules for anything that runs once per test are in the developer notes of
`scikitplot/conftest.py`. A test also fails if any `conftest.py` of the
package calls `gc.collect` by itself.

## 9. Secrets and the test job

The test job is given **no repository secret**. Two reasons:

- A pull request from a fork has no secrets, so a secret in the test job
  makes a push to `main` test something the pull request did not. That is
  how a suite that was green on the pull request failed on `main`: the
  repository's `HF_TOKEN` was exported, the documentation assistant's proxy
  read it when it was imported, and reported a provider as enabled.
- Every test, and every package installed for the tests, could read it.

Tests do not rely on that alone. The assistant's tests remove every
variable its services read from the process before a service is imported,
and put them back when the session ends
(`_sphinx_ai_assistant/tests/conftest.py`), so an `HF_TOKEN` in your own
shell does not reach them either. A test that needs a token sets a made-up
one itself.

Only the report job uses a secret: `CODECOV_TOKEN`, for the upload.

## 10. When a run fails

| What you see | What it means | What to do |
|---|---|---|
| The plan job fails with `ci_test_plan: ...` | the configuration or a `custom` selection is wrong; the message says which | fix what it names |
| `... cannot be selected: ... excluded from test collection` | the submodule is in `norecursedirs` | see *What is never tested* |
| A test fails on a warning raised inside `site-packages` during an `import` | a third-party package warned, and `filterwarnings = error` made it a failure | see below |
| `SKIPPED ... the '<tier>' tier is INCOMPATIBLE: ...` | a dependency is installed at a version the library refuses | the message names the version and the range |
| `no tests were collected for: ...` (a notice) | the selected submodules have no tests | nothing; the run succeeds |
| `ERROR: every test costs at least ... s before it does any work`, all tests passed | something that runs once per test became expensive: a hook or an `autouse` fixture | see *The check that keeps it from coming back*; find it with `pytest --durations=0 -vv` on a few skipped tests. If the selection really has slow tests only, raise `SKPLT_TEST_FLOOR_BUDGET` in the workflow for that case |
| A test passes on the pull request and fails after the merge, on `main` | the two runs did not have the same environment | see section 9; compare the `env:` block at the top of the two *Run tests* steps |
| The report job fails with *at least one test job did not succeed* | a test job failed or was cancelled | open that test job |

### Warnings from third-party packages

A package may warn once per process, the first time it is imported. Under
`filterwarnings = error` that fails whichever test imports it first, so the
failing test changes when the selection or the order changes. Decide which
case it is:

- **The package is describing the machine** (a CUDA build of PyTorch on a
  runner without CUDA, for example). Exempt that message in `filterwarnings`
  in `pytest.ini`, with a comment saying why.
- **Our code imports something it should not.** Fix the import. The vendored
  `platformdirs` importing `pip` was found this way; see
  `scikitplot/tests/test_vendored_self_contained.py`.

## 11. Files

| File | Role |
|---|---|
| `.github/workflows/ci_codecov_test_coverage.yml` | the workflow |
| `.github/ci/test_plan.json` | the configuration |
| `.github/ci/test_durations.json` | measured time per submodule |
| `.github/scripts/ci_test_plan.py` | the planner |
| `.github/scripts/ci_test_durations.py` | turns JUnit reports into `test_durations.json` |
| `.github/scripts/tests/` | tests of both scripts; the plan job runs them first |
| `docker/scripts/install_nltk.sh`, `install_spacy.sh` | install NLTK data and spaCy models; shared with Docker and the documentation build |
| `pytest.ini` | pytest options, `norecursedirs`, `filterwarnings` |
| `scikitplot/conftest.py` | the garbage-collection policy and the cost-per-test check |
| `scikitplot/tests/test_conftest_gc_policy.py` | tests of both |

The planner's tests run with the standard library alone:

```bash
python -m unittest discover -s .github/scripts/tests -p "test_*.py"
```
