# Benchmarks of scikit-plots

How fast the package itself is: import time, the plotting API, annoy. The test
suite's own time and memory are measured by the test recorder instead
(`SKPLT_TEST_PERF_DIR`, see `.github/ci/README.md`, section 12).

The suite in `benchmarks/` is written for [asv](https://asv.readthedocs.io/),
as SciPy's and scikit-learn's are (their suites, kept for reference, are in
`zz_yanked/`). It can be run with asv, or without it.

## Without asv: the installed package of an environment

```bash
make bench                                    # all, into build/bench/head.json
make bench BENCH="Plots" BENCH_ARGS=--quick   # some of them, one sample each (smoke test)
python .github/scripts/ci_bench.py list       # what there is
```

Two environments, for example the main branch and your branch, each in its
own virtualenv, on one machine:

```bash
python .github/scripts/ci_bench.py pair \
    --base-python ../venv-main/bin/python --head-python .venv/bin/python \
    --out build/bench/pair
```

`pair` alternates the two environments for every benchmark (base, head,
head, base), so a machine that slows down during the run does not show as a
change. `compare` (or `pair`) marks a timing slower `+` or faster `-` only
when the medians differ by `--factor` (1.1) *and* every round of one side has
a slower median than every round of the other (a round is one worker
process; `--processes`, 4 by default, at least 4 a side are needed). The
round, not the sample, is the unit: samples within one process vary less
than processes do. `x` is a ratio that reached the factor without that
evidence (run again; a `--quick` run is never evidence), `!` a benchmark that failed or was skipped on a side (with
`--fail-on-slower`, one that newly fails in head fails the command too), `?`
a result on one side only. Memory peaks and `track_` values are compared by
the ratio alone.

The runner needs only the Python standard library and works offline; results
are JSON (`skplt-bench`), readable by people and by an AI assistant alike.

## With asv: each commit built in its own environment

```bash
cd benchmarks
asv machine --yes
asv continuous --factor 1.1 main HEAD     # build both, run, compare
asv run --python=same --quick             # this environment, no build
```

## Writing a benchmark

Use the subset of asv that both runners support (it is checked by
`asv check --python=same` and by `ci_bench.py list`):

- functions or methods named `time_*`, `peakmem_*`, `track_*`, `timeraw_*`
  (inherited methods count);
- class attributes `params`, `param_names`, `timeout`;
- `setup` / `teardown` with the parameters, at module, class or function
  level (all of them run, module first); raise `NotImplementedError` in
  `setup` to skip a combination. `setup_cache` is asv-only: `ci_bench.py`
  reports such a benchmark as failed;
- inputs from `benchmarks/common.py` (fixed seeds, no download), and
  `close_figures()` in `teardown` after plotting.

A benchmark measures the package, so it imports `scikitplot` inside `setup`
or the benchmark, never at module level of a heavy submodule only to list it.
