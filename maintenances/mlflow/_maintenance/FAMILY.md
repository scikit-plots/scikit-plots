# Ownership family

`scikitplot.mlflow` owns scikitplot-side MLflow composition. The external `mlflow` package owns tracking/server/client behavior. `_cli` may delegate to this submodule but does not own MLflow semantics. Generic package/config utilities remain outside this boundary unless explicitly imported by the runtime.
