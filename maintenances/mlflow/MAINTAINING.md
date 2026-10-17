# Maintaining `scikitplot.mlflow`

`scikitplot.mlflow` is an integration/orchestration layer around the external MLflow package. It owns scikitplot-side configuration, lifecycle, safety policy, compatibility and convenience behavior; it does not own upstream MLflow internals.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, then run the contract, review and maintenance regression suite from any working directory. Keep static/runtime/integration/release truth separate: mock-heavy unit tests can prove scikitplot logic without proving compatibility with a real installed MLflow server.

The current review has three runtime structural findings and one test-maintenance finding. Most importantly, the exported SecurityPolicy is not currently connected to session/server operations even though its documentation says it guards them.
