# scikitplot.mlflow

Adds a **project-level configuration** mechanism so multiple scripts
(e.g., `train.py`, `hpo.py`, `predict.py`) share the exact same MLflow settings,
regardless of current working directory.

## Why this matters

Without a shared config, it is easy for scripts to drift:
- different `backend_store_uri` (db) paths
- different `default_artifact_root` paths
- different tracking URIs / ports
- different env var setup

This module provides **strict, deterministic rules**:
- Args > env > `.env` file (fills missing only)
- Relative local paths are resolved against project root (found via `pyproject.toml` or `.git`)
- Local directories are created deterministically when starting a managed server

## Quiskstart: Beginner workflow demo

This package ships **demo configs** and a **workflow** that demonstrates how to:
- export a config into your project
- customize experiment name
- run train/predict runs
- export your project config to YAML

### Python API

```python
import scikitplot as sp

sp.mlflow.workflow(profile="local", open_ui_seconds=5)
```

### CLI

```bash
python -m scikitplot.mlflow --profile local --open-ui-seconds 5
```

## Config file (TOML, stdlib)

Create `configs/mlflow.toml`:

```toml
[profiles.local]
start_server = true

[profiles.local.session]
# 🌐 optional; if omitted and start_server=true, it will default to http://127.0.0.1:<port>
tracking_uri = "http://127.0.0.1:5000"
registry_uri = ""
env_file = ".env"
startup_timeout_s = 30.0

[profiles.local.server]
host = "127.0.0.1"
port = 5000
backend_store_uri = "sqlite:///./.mlflow/mlflow.db"
default_artifact_root = "./.mlflow/artifacts"
serve_artifacts = true
strict_cli_compat = true
```

### Strict path normalization

- `sqlite:///./.mlflow/mlflow.db` becomes an **absolute** sqlite path based on project root.
- `./.mlflow/artifacts` becomes an **absolute** artifact directory based on project root.

This ensures `train.py` and `predict.py` use the same store even if run from different folders.

## Usage

### train.py / hpo.py / predict.py

```python
import scikitplot as sp

with sp.mlflow.session_from_toml("configs/mlflow.toml", profile="local") as mlflow:
    mlflow.set_experiment("my-project")
    with mlflow.start_run(run_name="train"):
        mlflow.log_param("model", "xgb")
```

## Connect-only (no server spawn, but ensure reachable)

```toml
[profiles.remote]
start_server = false

[profiles.remote.session]
tracking_uri = "http://mlflow.my.company:5000"
ensure_reachable = true
startup_timeout_s = 30.0
```

```python
import scikitplot as sp

with sp.mlflow.session_from_toml("configs/mlflow.toml", profile="remote") as mlflow:
    mlflow.set_experiment("my-project")
```


## YAML alternative (read + write)

If you need a config format that supports **both** read and write with common tooling,
use YAML (`.yaml`/`.yml`). The schema is identical to TOML.

### configs/mlflow.yaml

```yaml
profiles:
  local:
    start_server: true
    session:
      tracking_uri: "http://127.0.0.1:5000"
      env_file: ".env"
      startup_timeout_s: 30.0
    server:
      host: "127.0.0.1"
      port: 5000
      backend_store_uri: "sqlite:///./.mlflow/mlflow.db"
      default_artifact_root: "./.mlflow/artifacts"
      serve_artifacts: true
      strict_cli_compat: true
```

### Usage

```python
import scikitplot as sp

with sp.mlflow.session_from_file("configs/mlflow.yaml", profile="local") as mlflow:
    mlflow.set_experiment("my-project")
```


## Experiment + run defaults (stored in config)

You can store `experiment_name` and default run metadata in the session config.

### TOML

```toml
[profiles.local.session]
experiment_name = "my-project"
create_experiment_if_missing = true
default_run_name = "train"
# For tags, prefer YAML (TOML nested mapping is verbose).
```

### YAML (recommended for tags)

```yaml
profiles:
  local:
    start_server: true
    session:
      experiment_name: "my-project"
      create_experiment_if_missing: true
      default_run_name: "train"
      default_run_tags:
        pipeline: "train"
        owner: "team-ml"
```

Usage:

```python
import scikitplot as sp

with sp.mlflow.session_from_file("configs/mlflow.yaml", profile="local") as mlflow:
    # experiment already set; run defaults applied by handle.start_run
    with mlflow.start_run():
        mlflow.log_param("alpha", 0.1)
```


## Viewing the MLflow UI from Docker / remote notebooks

If your Python kernel runs inside a container or on a remote machine, `127.0.0.1:5000` refers
to the kernel host, not your local browser machine. In that case, set a **public UI URL**
so users can open the correct address in a browser:

### TOML

```toml
[profiles.local.session]
public_tracking_uri = "http://localhost:5000"  # or your forwarded host/port
```

### YAML

```yaml
profiles:
  local:
    session:
      public_tracking_uri: "http://localhost:5000"
```

Inside the session you can print:

```python
with sp.mlflow.session_from_file(..., profile="local") as mlflow:
    print("Open MLflow UI:", mlflow.ui_url)
```
