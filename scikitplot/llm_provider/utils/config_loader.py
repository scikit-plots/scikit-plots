"""load_mlflow_gateway_config."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os

# from pathlib import Path
import yaml

from ... import logger


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
        provider = endpoint["model"].get("provider", "")
        model_id = endpoint["model"].get("name")
        config = endpoint["model"].get("config", {})

        # get api key
        suffix = "_TOKEN" if provider in ["huggingface"] else "_API_KEY"
        # api_key = next(iter(config.values()), "")
        api_key = config.get(f"{provider.upper()}{suffix}", "")
        # Expand $VAR or ${VAR} environment-style names
        if api_key.startswith(("$", "${")):
            # stays as-is (not a variable ref) (read-only)
            expanded = os.path.expandvars(api_key)
            api_key = expanded if expanded != api_key else ""
            if api_key:
                logger.info(f"Loaded {provider.upper()}{suffix} from Env.")
        if provider and model_id:
            if provider not in configs:
                configs[provider] = []
            configs[provider].append({"model_id": model_id, "api_key": api_key})
    return configs
