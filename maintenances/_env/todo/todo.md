# `scikitplot._env` maintenance: active tasks

Rules that apply to every increment: [`skills/_engineering/`](../../../skills/_engineering/SKILL.md).
Release-note PR for this round: **868** (given by the maintainer, Rule 100).

## Task: ENV-1, one command to build the same environment anywhere (2026-10-24)

### Context

- **Goal.** From any Python that has scikit-plots (shell, notebook, CI,
  container, agent), one command or one call builds the same environment:

  ```text
  scikitplot env create --config scikitplot-env.toml
  ```

  The request names *what* (Python version, packages, dependency files,
  channels, target); `_env` decides *how* (uv, venv + pip, micromamba,
  conda/mamba, pipenv), shows the plan before it changes anything, asks only
  through one decision point, runs every tool non-interactively, and leaves a
  receipt.
- **Baseline, read from the uploaded tree.** `_env` is a mirror of
  jupyterlite-xeus: `__init__.py` is a docstring; `create_conda_env.py`
  hard-codes `emscripten-wasm32` and caches `shutil.which("micromamba")` at
  import (ENV-002); `add_on.py` imports jupyterlite-core, empack, requests,
  yaml and traitlets at import (ENV-010); `get_environment_specs` /
  `get_environment_channels` raise `UnboundLocalError` for a `Path` prefix
  (ENV-004); `create_prefix` requires `name` while `create_conda_env_from_env_file`
  defaults it to `"xeus-env"` (ENV-005); the remote empack config is fetched
  with no timeout or status check (ENV-006); `tests/test___init__.py` has no
  active test (ENV-001). Nothing in the repository imports `_env`.
- **Reference, not ground.** `scikitplots_env_architecture_review_v3`
  (supplied) is used as a checklist. Its 12-phase plan is collapsed into one
  round that delivers the core and the native backends with real runs; WASM
  and mobile targets are modelled and refuse execution explicitly.

### Big picture

```text
inputs                      _config / _sources            _core
-----------------------     -----------------------       -------------------------
CLI flags / Python args --> invocation (authority)  -->
scikitplot-env.toml     --> project config (intent) -->   EnvironmentRequest
[tool.scikitplot.env]   -->        + preset overlay       (frozen, digest)
requirements.txt, environment.yml, pyproject.toml,              |
Pipfile, inline list    --> DependencySet (+ conversion notes)  |
                                                               v
                            HostInfo (runtime discovery) --> plan()  --> EnvironmentPlan
                            backends / targets registry        |        (steps, argv, network,
                                                               |         mutations, approvals)
                                                     decisions (one provider:
                                                     never / auto / always)
                                                               |
                                                               v
                                                     execute() under a prefix lock
                                                     -> events (logger + JSONL)
                                                     -> EnvironmentReceipt (JSON)
```

### Invariants

1. **Plan before mutation.** `plan()` never touches disk or network; every
   step of a plan lists its argv, whether it uses the network, what it
   writes, and whether it needs approval.
2. **Intent is not authority.** A config file can ask for bootstrap, network
   or a shell-profile edit; only the invocation (`--yes`, `approve=`) grants
   it. Config keys that would grant authority are refused.
3. **Network is a policy with three values** (`online`, `prefer-offline`,
   `offline`); `offline` never plans a network step and passes each tool's
   offline flag.
4. **One decision point.** Backends never prompt; tools run with prompts off;
   `interaction="never"` turns every unanswered approval into an
   `ApprovalRequired` error that names the flag to pass.
5. **Requested is not effective.** The plan records the backend asked for,
   the backend chosen and why; conversions between ecosystems are `exact`,
   `approximate` or refused.
6. **Runtime discovery.** Tools are found when planning, never at import.
7. **Light import.** `import scikitplot._env` imports the standard library
   only; YAML, jupyterlite and empack load where used.
8. **Machine-clean stdout.** The CLI writes results to stdout (text or JSON)
   and everything else to stderr; tool output is captured and logged.
9. **Honest targets.** `pyodide`, `android` and `ios` are modelled targets that
   refuse execution with a reason; `emscripten-wasm32` is the micromamba
   cross-build lane the Xeus add-on uses.
10. **No silent partial success.** A failed step stops the run; the receipt
    lists the steps that ran, so partial mutation is visible.

### Layout (`scikitplot/_env/`)

| Package | Responsibility |
|---|---|
| `__init__.py` | lazy public facade (`plan`, `create`, `doctor`, `load_config` ...) |
| `__main__.py` | `python -m scikitplot._env` / `scikitplot env` |
| `_core/` | errors, enums, request, host detection, plan/steps, receipt, decisions, events, lock, planner, executor |
| `_config/` | config schema, formats (TOML, JSON, YAML lazily, JSONL batches), presets |
| `_sources/` | dependency sources to one `DependencySet` |
| `_backends/` | one module per tool: argv for create/install/run/remove |
| `_targets/` | target descriptors (host, emscripten-wasm32, pyodide, android, ios) |
| `_bootstrap/` | micromamba / uv installation into a tools directory; shell init |
| `_xeus/` | the JupyterLite Xeus add-on (moved, fixed, lazily imported) |

### Steps

- [x] 1. Plan (this section). [Completed: 2026-10-24]
- [x] 2. `_core`. [Completed: 2026-10-24]
  - Added during build: `_receipt.py`, `_planner.py`, `_executor.py`; state digest
    separate from plan digest (create and update plans differ in steps).
- [x] 3. `_config`, `_sources`. [Completed: 2026-10-24]
  - Deviation: `environment.yml` is parsed once by the builder and reaches every
    backend as specs (no `-f`), so one file gives one environment; pip options
    other than `-r`/`-e` and the `variables` key are refused, not dropped.
  - Deviation: the `prefer-offline` network policy was removed (no backend
    defines it; a cache-then-network fallback is not deterministic).
- [x] 4. `_backends`, `_targets`, `_bootstrap`, planner, executor. [Completed: 2026-10-24]
  - Added: Conda-subdir cross-build targets (`linux-aarch64` ...); `--relocate-prefix`
    only for the browser target; removal is one `remove_tree` action for every backend.
- [x] 5. `_xeus` migration with defect fixes and real tests. [Completed: 2026-10-24]
- [x] 6. CLI command, user guide, Towncrier 868, maintenance plane, skill. [Completed: 2026-10-24]
  - Added: extras `env` (PyYAML) and `env-xeus` in `pyproject.toml`; the reviewed
    delegated-command exemption in `_cli/tests/test_frontend_surface_parity.py`.
- [x] 7. Verification: unit and adversarial tests; real runs of venv, uv,
      micromamba (conda-forge) and pipenv where reachable; Python floor;
      numpydoc lint; independent review (Rule 92). [Completed: 2026-10-25]
  - Independent review: 29 findings (20 confirmed by reproduction). All fixed
    with a regression test each (`TestReviewFindings` in the planner,
    executor, sources and bootstrap tests; CLI, config and xeus tests); the
    reviewer's reproductions were re-run against the fixes.

### Acceptance criteria

- [x] The same TOML, JSON and YAML config normalise to one request with one digest.
- [x] `plan` on a fresh host shows the bootstrap step and its URL; `create`
      without `--yes` in non-interactive mode stops before any mutation.
- [x] `offline` plans no network step; a backend that would need the network is refused.
- [x] Real environments are created and verified with `venv`, `uv` and
      micromamba from one config; a second run is a no-op `ensure`.
- [x] Each Xeus defect above has a regression test.
- [x] `import scikitplot._env` imports no third-party module.

### Risks

- Network allowlist in the build sandbox → micromamba is bootstrapped from
  the conda-forge channel package; lanes that cannot run are recorded as
  unavailable, not as passed.
- Tool flag drift between versions → each backend builds argv from flags
  checked against the installed tool's `--help` in tests.

## Results Review - ENV-1 - 2026-10-24

### Implementation summary
- Replaced the vendored Xeus helper (`add_on.py`, `create_conda_env.py`,
  `_pip.py`, `constants.py`, an all-commented test) with the builder:
  `_core` (request, plan, planner, executor, approvals, events, lock, receipt),
  `_config`, `_sources`, `_backends` (uv, venv, pipenv, micromamba, mamba,
  conda), `_targets`, `_bootstrap`, `_xeus`, `_api`, `_cli`.
- `scikitplot env` delegated in `scikitplot/_cli/registry.py`; extras `env` and
  `env-xeus`; user guide `docs/source/user_guide/_env/` (5 pages + sync test);
  Towncrier `868.feature.rst`, `868.fix.rst`; labeler synced.

### Verification evidence
- Unit + adversarial suite from source:
  `python -m pytest --noconftest -c /dev/null scikitplot/_env/tests docs/source/user_guide/_env`
  -> 164 passed, 13 skipped (add-on tests without jupyterlite-core; network
  tests gated); the one `captured_logger` test runs in installed mode.
- Installed build with the repository `pytest.ini` (warnings are errors) and
  conftest: `pytest --pyargs scikitplot._env scikitplot._cli` -> 341 passed,
  13 skipped.
- Real tools (`SKPLT_ENV_NETWORK_TESTS=1`, 10 passed): venv end-to-end (create,
  no-op re-run, run, remove); uv with a requirements file edited between runs
  (only the install step re-runs); micromamba bootstrapped from conda-forge with
  the pinned SHA-256, Conda + pip packages, offline no-op re-run, `run`; a
  `linux-aarch64` cross-build (`conda-meta` records `subdir: linux-aarch64`);
  pipenv `lock` + `sync`; real uv bootstrap through pip `--require-hashes`.
  Every `--flag` a plan passes is in the installed tool's `--help`
  (uv 0.11.32, micromamba 2.9.0, pip 24.0, pipenv 2026.10.2).
- Xeus add-on tests in a venv with jupyterlite-core 0.8.6 / empack 6.0.1:
  14 passed.
- Python 3.9 and 3.11 (floor venvs): 164 passed, 13 skipped.
- `ruff check` (package): clean; `black`: clean; numpydoc
  GL06/GL07/PR01-04/PR10/RT01/YD01: none; doctests: 17 passed;
  `sphinx-build -W -n` of the guide: clean; codespell: clean;
  `generate_pr_labeler.py check`, `generate_towncrier_sections.py check`,
  `check_test_module_names.py`, `check_path_lengths.py check`: pass.
- Not run here: an emscripten-wasm32 build end to end (prefix.dev is not
  reachable from this sandbox; the failure path was run and reported the
  micromamba error with the step id); conda and mamba executables (not
  installed; plan-tested); Windows and macOS hosts (plan-tested with fake hosts).

### Deviations from plan
- `prefer-offline` removed; `environment.yml` parsed once for all backends;
  Conda-subdir cross-build targets added; `--relocate-prefix` only for the
  browser; bootstrap URL and SHA-256 shown in the plan and checked against
  the pins at execution.

### Technical debt created
- conda/mamba argv are plan-tested only (no executable in the sandbox).
- The Xeus add-on keeps the upstream task structure (doit generators).
