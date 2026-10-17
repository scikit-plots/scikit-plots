# Verification

Current evidence:

- subsystem imports successfully with upstream `mlflow` absent;
- 657 tests pass in one SDK-free batch when `TestRunDemo` is deselected;
- all 6 `TestRunDemo` tests pass when run separately; four incur the real 10-second default sleep unless that individual test patches sleep;
- bundled TOML and YAML both parse and expose `local` and `remote` profiles;
- a negative security probe proves active `DEFAULT_SECURITY_POLICY` does not currently block `disable_security_middleware=True` in server args;
- upstream `mlflow`, mypy and pyright are unavailable in this environment.

For release, add a real supported-version MLflow matrix, `mlflow server --help` capability checks, managed server launch/readiness/teardown, artifact/model/session integration, packaging + type-checking, and POSIX/Windows evidence.
