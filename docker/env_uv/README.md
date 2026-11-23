# ⚡ uv lock store

One standalone, non-package uv project per Python version, written from
`requirements/default.txt` (runtime) and `requirements/all.txt` (dev, docs,
tests, build) — the same sources as `../env_pipenv/`.

| Path | Contents |
|---|---|
| `py310/` ... `py315/` | `pyproject.toml` + `uv.lock` for that Python only |
| `pyproject.toml`, `uv.lock` | a copy of the default version (3.11) |

Python 3.8 and 3.9 are not here: several requirements need Python 3.10.

## Use

```bash
uv sync --project docker/env_uv              # default version, dev group included
uv sync --project docker/env_uv/py313 --no-dev
uv run --project docker/env_uv/py312 pytest
```

The environment is created as `.venv` inside the folder (ignored by git).
`--frozen` installs exactly the lock without re-resolving.

There is no `uv.lock` at the repository root, on purpose. A root `uv.lock`
belongs to the root `pyproject.toml` (the scikit-plots package): with this
lock linked there, `uv sync`, `uv lock --check` and `uv sync --frozen` all
fail, and `uv lock` would rewrite the linked file.

## Update

Never edit these files. Change `pyproject.toml` at the repository root,
regenerate `requirements/*.txt` (`python tools/generate_requirements.py`),
then run the workflow **CI 📦 uv Lock Artifacts Update**
(`.github/workflows/ci_uv_artifact.yml`): it opens one pull request per
Python version. Locally, for one version:

```bash
python .github/scripts/flatten_reqs.py requirements/default.txt requirements/all.txt /tmp/r.txt /tmp/d.txt 3.11
python .github/scripts/uv_project_from_reqs.py /tmp/r.txt /tmp/d.txt docker/env_uv/py311 3.11
uv lock --directory docker/env_uv/py311 --python 3.11
```

`.github/scripts/tests/test_uv_project_from_reqs.py` fails when a committed
`pyproject.toml` no longer matches the requirements.
