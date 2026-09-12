# `scikitplot.datasets` family boundaries

`datasets` is one package with three different runtime responsibilities. Do not use evidence from one to bless the others.

| Surface | Owns | Does not own |
|---|---|---|
| `_load_dataset.py` | curated dataset names, cache location, download, dataset-specific normalization | remote GitHub availability, appdirs implementation, pandas parser internals |
| `_data_export.py` | deterministic subset selection, size allocation, streaming hash selection, manifests/profiles, export CLI | parquet engines, pandas/NumPy implementation details |
| `_data_loader.py` | extension dispatch, URL/local/upload/database routing, optional-backend selection | safety of untrusted pickle/SQL/network payloads, DB drivers, root logger/safe_import |
| `_autoscout24_tasks.py` | examples/recipes | authoritative loader/export contracts |

## Trust boundary

`pickle`, `joblib`, and `cloudpickle` loaders can execute code from serialized input. SQL comment stripping is normalization, not sanitization. URL loading crosses the network trust boundary. Never market these helpers as making untrusted input safe.
