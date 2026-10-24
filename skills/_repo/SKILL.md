---
name: repo-test-performance
description: Measure, analyse and compare where scikit-plots test runs spend time and memory, and how fast the package itself is (benchmarks/), locally, offline and in CI. Use when tests or CI shards are slow or time out, when a profile, a benchmark result or a CI artifact (coverage-*.zip, test-perf.zip, benchmarks-on-demand.zip, a .prof, junit.xml) must be analysed, when a speed-up must be proved before and after, or when performance work continues from a fresh chat.
---

# Test and package performance (`_repo`)

Work from the checkout that contains `scikitplot/`, `maintenances/` and
`skills/`. Everything below is measured, never estimated: a claim names the
field and value it rests on.

## Read first

1. [`tasks/README.md`](../../tasks/README.md) and
   [`skills/_engineering/SKILL.md`](../_engineering/SKILL.md) (rules; load
   `verification.md`, `delivery.md`, `tooling.md`, and `testing.md` for tests).
2. [`.github/ci/README.md`](../../.github/ci/README.md), section 12: what is
   recorded, the commands, the findings and their thresholds; section 13:
   measuring without the coverage workflow; section 8: garbage collection.
   [`benchmarks/README.md`](../../benchmarks/README.md): the package benchmarks.
3. The active task in [`maintenances/_repo/todo/todo.md`](../../maintenances/_repo/todo/todo.md)
   (TP-1 and later) and the open notes under
   [`upcoming_changes/`](../../upcoming_changes/README.md).
4. The docstrings of `scikitplot/_testing/_perf_telemetry.py` (the recorder)
   and `.github/scripts/ci_test_perf.py` (the analyzer): the developer notes
   say what each number means and where it can mislead.

## Given evidence (an artifact, a zip, a profile)

```bash
python .github/scripts/ci_test_perf.py report <files, folders or zips> --out build/perf-report --bundle
```

Read `ANALYZE_ME.md` first, then `report.md`. Without a checkout, the bundle
carries the analyzer: `python tool/ci_test_perf.py report raw --out again`.

Two runs of the same tests in one input are refused (their times would add
up): pass `--run latest`, `--run <id prefix>`, or use `compare`.

## Finding the cost

1. **Record** the selection: `SKPLT_TEST_PERF_DIR=build/test-perf python -m pytest <paths>`
   (or `make test-perf PERF_PATHS=<paths>`). Run long selections in the
   background on a snapshot with a pid file (Rules 9, 87).
2. **Locate** it with `report`: unit -> test file -> test -> phase. A
   setup-dominated test points at a fixture; `fixtures` names it, its scope
   and where it is defined.
3. **Drill down** without editing a test:
   `SKPLT_TEST_PERF_PROFILE='<regex of the node id>'` profiles it from setup to
   teardown; `ci_test_perf.py profile <its .prof> --callees '<fixture>$'` lists
   what the fixture called, edge-exact. Profile it in a run of its own file
   when `module-re-executed` fired: duplicated modules make cProfile records
   incomplete (`profile-records-incomplete`, "not attributed").
4. **Explain** the root cause from the evidence (five whys): what is computed,
   how often, and why it is not reused. Subprocesses: `child_cpu` and
   `programs`. Waiting: `cpu + child_cpu` well below the wall time.
   Memory: `rss`, `peak_rss_growth`, then `SKPLT_TEST_PERF_TRACEMALLOC=1`.
5. **Fix** at the root, without weakening what a test checks (a security
   test still builds and verifies what it claims to).
6. **Prove** it: record again on the same machine and run
   `ci_test_perf.py compare --base <before> --head <after>`; quote both
   speed probes when the machines differ.

## Package benchmarks (how fast scikit-plots is for its users)

The tests measure the test suite; `benchmarks/` measures the package. Choose
by the question: "why is CI slow" is the recorder, "did this change make
`plot_roc` or `import scikitplot` slower" is a benchmark.

```bash
python .github/scripts/ci_bench.py list
python .github/scripts/ci_bench.py run --out build/bench/head.json        # make bench
python .github/scripts/ci_bench.py pair --base-python <base venv>/bin/python \
    --head-python <head venv>/bin/python --out build/bench/pair         # make bench-pair
python .github/scripts/ci_bench.py compare base.json head.json            # make bench-compare
```

- Results are JSON (`skplt-bench`): every sample, median, min, max. A change
  needs the factor (1.1) *and* every round (process) of one side slower
  than every round of the other, 4 rounds a side (`x`: not enough
  evidence; a `--quick` run never is); prefer `pair`,
  which alternates the environments, over two separate `run`s.
- Measure an installed package, not an editable one: an editable meson
  install rebuilds on import and distorts import times.
- New benchmarks use the asv subset described in `benchmarks/README.md`;
  check them with `ci_bench.py list` and, where asv is installed,
  `asv check --python=same`.
- On a clean runner: workflow **CI ⏱️ Performance (on demand)**,
  `benchmarks: true` and `base` (`merge-base` or `default-branch`).

## Pitfalls (each has cost a round)

- cProfile keeps one record per (file, line, function): a module executed
  twice reports one copy's cumulative time, and calls into it vanish from its
  callers (Rule 131).
- On Python 3.11 a second `cProfile.Profile` silently replaces the first; on
  3.12+ it raises. Never nest profilers; the recorder skips and says why
  (Rule 133).
- A failed or timed-out job is the most important evidence: keep it
  (`if: always()`), and read `incomplete-session` for the test that was
  running (Rule 134).
- A required check skipped by its own condition counts as passing (Rule 132).
- Machines differ by several times (this suite: 3x between a laptop and the
  container): compare a machine with itself, or read the speed probe.
- A skipped test is not a measured test (Rule 13).
- Two runs made one after the other can differ by 10-20% on a shared
  machine with no code change (measured here). One run against one run is
  not evidence; `pair` with enough samples, and an A/A run of the same
  method that finds nothing (Rule 135), is.
- Full garbage collections at module boundaries were about a quarter of CI
  test time. `SKPLT_TEST_GC=frozen` removes most of it (section 8), at the
  cost described in `_FreezeSurvivors`; it is opt-in until a CI run proves it.

## Record the work

Plan and results go to the owner's `todo/todo.md` (CI and tooling:
`maintenances/_repo/todo/todo.md`); a finding in another area's runtime goes
to `upcoming_changes/<area>/`; a lesson becomes a numbered rule
(`skills/_engineering/SKILL.md`, then `check_records.py --write-index`).

## Checks

```bash
python -m unittest discover -s .github/scripts/tests -p "test_*.py"
python -m pytest scikitplot/_testing/tests/test__perf_telemetry.py scikitplot/tests/test_conftest_gc_policy.py -p no:cacheprovider
python .github/scripts/ci_bench.py run --quick --out build/bench/smoke.json
python -B maintenances/_repo/_maintenance/tools/check_records.py
```
