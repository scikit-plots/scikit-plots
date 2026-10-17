# Submodule structure

- `_config`, `_project`, `_env`, `_container`: configuration/project/environment plane.
- `_custom`, `_compat`, `_utils`, `_facade`: integration/provider/compatibility plane.
- `_server`, `_readiness`, `_session`, `_security`: lifecycle and safety plane.
- `_workflow`, `__main__`, `_cli_caps`: user workflow and CLI compatibility plane.
- `_configs`: bundled TOML/YAML examples.
- `tests`: unit/contract tests, mostly upstream-MLflow independent through stubs.
