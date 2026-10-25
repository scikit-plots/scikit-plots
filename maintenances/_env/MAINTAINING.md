# Maintaining `scikitplot._env`

`scikitplot._env` builds one described Python environment with whichever tool
the host has (uv, venv + pip, pipenv, micromamba, mamba, conda), for the host,
another Conda platform or the browser. Runtime: `scikitplot/_env/` (standard
library only; PyYAML for YAML configs and `environment.yml`; jupyterlite-core
and empack for the Xeus add-on). Command line: `scikitplot env` (delegated in
`scikitplot/_cli/registry.py`) and `python -m scikitplot._env`. User guide:
`docs/source/user_guide/_env/`. Skill: [`skills/_env/SKILL.md`](../../skills/_env/SKILL.md).

Start a session with [`tasks/README.md`](../../tasks/README.md), then the active
[`todo/todo.md`](todo/todo.md) and this area's [`todo/lessons.md`](todo/lessons.md).

## Flow

```
request (_config, _core._request)
   -> plan (_core._planner: _sources, _targets, _backends, _bootstrap)   reads only
   -> approvals (_core._decisions)                                       before any step
   -> execute (_core._executor, under _core._lock)                       steps + actions
   -> receipt (_core._receipt)                                           state digest
events (_core._events) and errors (_core._errors) throughout
```

## Module map

| Module | Owns |
|---|---|
| `_core/_types.py` | the closed vocabularies (network, interaction, bootstrap, conversion, ecosystem, approval, operation) |
| `_core/_errors.py` | error types with a stable `code` and `hint`; each derives from the built-in a caller expects |
| `_core/_request.py` | `EnvironmentRequest`, `Policy`, `SourceRef` (value objects, digest) |
| `_core/_host.py` | `detect_host` (every input injectable), `env_home` |
| `_core/_plan.py` | `Step`, `EnvironmentPlan` (data, digest, render, redaction) |
| `_core/_planner.py` | backend rules, conversion, targets, existing environments |
| `_core/_executor.py` | approvals first, lock, runner, the action table, post-check, receipt |
| `_core/_decisions.py` | `Decider`: the only place a question is asked |
| `_core/_events.py`, `_lock.py`, `_receipt.py` | JSON Lines events, prefix lock, receipts |
| `_config/` | TOML/JSON/YAML/JSONL readers (strict), presets, layering with origins |
| `_sources/_readers.py` | requirements, environment.yml, pyproject, Pipfile -> `DependencySet` (+ input hashes) |
| `_backends/` | one class per tool, producing steps only; `BACKENDS` registry (lazy) |
| `_targets/` | `Target` data: host, Conda subdirs, emscripten-wasm32, pyodide; mobile refusal |
| `_bootstrap/` | pinned micromamba install (SHA-256), uv requirement, shell start-up block |
| `_xeus/` | browser pure-Python layer (`_pip.py`) and the JupyterLite add-on (`_addon.py`) |
| `_api.py`, `_cli.py` | public functions and the command line (thin) |

## Invariants (each has tests in `scikitplot/_env/tests/`)

1. Planning changes nothing; every change is a plan step (command or named action).
2. Every approval is decided before the first step runs; configuration files
   cannot grant one (`config_authority`).
3. The backend is chosen by the documented rules only, and the plan records the rule.
4. Offline plans contain no network step; a step that would need the network is
   refused at planning time.
5. `environment.yml` is parsed by the builder; no backend receives the file, so
   one file gives one environment for every backend. Unappliable keys are refused.
6. The state digest covers the request (minus behaviour policies), target,
   backend, prefix, Python and the hash of every dependency file read.
7. The builder writes only into an empty directory or a recognised environment
   (`pyvenv.cfg`, `conda-meta` or a receipt; never `bin/python` alone), and
   deletes only a recognised environment (checked again at deletion).
8. An environment built with another backend, target or resolved Python is not
   changed in place; one whose request dropped a dependency is recreated.
9. Downloads are https-only, size-limited and SHA-256 checked before use (uv
   through pip `--require-hashes`).
10. Credentials are refused in every URL-bearing request field and redacted from
    plans, receipts, logs, events and errors.
12. The executor re-checks the planned state under the lock; approvals come only
    from the invocation (CLI, API, a terminal, `SKPLT_ENV_APPROVE` for the add-on).
11. `import scikitplot._env` imports no subpackage and no third-party module.

## Verification

```sh
python -B -m pytest scikitplot/_env docs/source/user_guide/_env -q --doctest-modules -p no:cacheprovider
SKPLT_ENV_NETWORK_TESTS=1 python -B -m pytest scikitplot/_env/tests/test_env_integration.py -q
python -m numpydoc lint $(find scikitplot/_env -name '*.py' -not -path '*/tests/*') | grep -E ': (GL06|GL07|PR0[1-4]|PR10|RT01|YD01) '
python tools/maint_tools/generate_pr_labeler.py check
```

The add-on tests need `jupyterlite-core` and `empack`; run them once in an
environment that has them. Run the suite at the Python floor (3.9) too.

## Moving the micromamba pin

Download the five `micromamba-<v>-0.tar.bz2` archives from
`https://conda.anaconda.org/conda-forge/<subdir>/`, record their SHA-256 in
the `MICROMAMBA` table and `MICROMAMBA_VERSION` of `_bootstrap/__init__.py`,
and run the network tests (`test_micromamba_bootstrap_conda_and_pip`
downloads and checks the host's). For uv: set `UV_REQUIREMENT` and replace
`UV_WHEEL_SHA256` with the SHA-256 of every wheel of that release
(`https://pypi.org/pypi/uv/<v>/json`), then run `test_uv_bootstrap_real`.
