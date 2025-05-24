"""load_mlflow_gateway_config."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os

# from pathlib import Path
import yaml


def load_mlflow_gateway_config(path: str) -> "dict[str, any]":
    """load_mlflow_gateway_config."""
    # path = Path.cwd() / path
    # path = Path(path).expanduser().resolve()
    # path = os.path.abspath(os.path.join(os.getcwd(), path))
    path = os.path.abspath(os.path.expanduser(path))

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    configs = {}
    for endpoint in raw.get("endpoints", []):
        provider = endpoint["model"].get("provider")
        model_id = endpoint["model"].get("name")
        config = endpoint["model"].get("config", {})

        key = next(iter(config.values()), "")
        # Expand $VAR or ${VAR} environment-style names
        if key.startswith(("$", "${")):
            # stays as-is (not a variable ref)
            expanded = os.path.expandvars(key)
            key = expanded if expanded != key else ""

        if provider and model_id:
            if provider not in configs:
                configs[provider] = []
            configs[provider].append({"model_id": model_id, "api_key": key})

    return configs
