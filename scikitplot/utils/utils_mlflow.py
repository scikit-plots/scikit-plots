"""load_mlflow_gateway_config."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-locals

import os as _os

# from pathlib import Path
from .. import logger as _logger
from ..exceptions import ScikitplotException


def load_mlflow_gateway_config(
    path: str,
) -> dict[str, list[dict[str, any]]]:
    """
    Load MLflow Gateway model configuration from a YAML file.

    Parses a configuration file defining multiple model endpoints. Supports
    environment variable expansion and validates expected structure.

    Parameters
    ----------
    path : str
        Path to the MLflow Gateway YAML file. May contain `~` or environment variables.

    Returns
    -------
    dict of str to list of dict
        A dictionary where each key is a provider name and each value is a list of
        dictionaries with keys:
        - 'model_id': str
        - 'api_key': str (may be empty if not resolved)

    Raises
    ------
    ScikitplotException
        If the YAML is malformed, missing fields, or the file doesn't exist.

    Notes
    -----
    - API keys can be hardcoded or passed via environment variables using
      `$VARNAME` or `${VARNAME}`.
    - This loader is tolerant to partial failures but logs all issues.

    Examples
    --------
    >>> cfg = load_mlflow_gateway_config("~/gateway.yaml")
    >>> cfg["openai"][0]["model_id"]
    'gpt-4'
    """
    try:
        import yaml  # Lazy dependency
    except ImportError as e:
        raise ScikitplotException("Missing required dependency: pyyaml") from e

    try:
        # Normalize and resolve the path
        abs_path = _os.path.abspath(_os.path.expanduser(path))
        if not _os.path.exists(abs_path):
            raise FileNotFoundError(f"Configuration file not found: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ScikitplotException("Config root must be a dictionary.")

        endpoints = raw.get("endpoints", [])
        if not isinstance(endpoints, list):
            raise ScikitplotException("'endpoints' field must be a list.")

        configs: dict[str, list[dict[str, any]]] = {}

        for idx, endpoint in enumerate(endpoints):
            model = endpoint.get("model", {})
            if not isinstance(model, dict):
                _logger.warning(f"Skipping endpoint[{idx}]: 'model' must be a dict.")
                continue

            provider = model.get("provider", "").lower()
            model_id = model.get("name", None)
            config = model.get("config", {})

            if not provider or not model_id:
                _logger.warning(
                    f"Skipping endpoint[{idx}]: Missing 'provider' or 'name'."
                )
                continue

            if not isinstance(config, dict):
                _logger.warning(f"Skipping endpoint[{idx}]: 'config' must be a dict.")
                continue

            # get api key
            suffix = "_TOKEN" if provider == "huggingface" else "_API_KEY"
            key_name = f"{provider.upper()}{suffix}"
            # api_key = next(iter(config.values()), "")
            api_key = config.get(key_name, "")

            # Expand environment variable if format is $VAR or ${VAR}
            if isinstance(api_key, str) and api_key.startswith(("$", "${")):
                # stays as-is (not a variable ref) (read-only)
                expanded = _os.path.expandvars(api_key)
                api_key = expanded if expanded != api_key else ""
                if api_key:
                    _logger.info(f"Loaded {key_name} from environment variables.")

            # Initialize provider list if not already present
            configs.setdefault(provider, []).append(
                {"model_id": model_id, "api_key": api_key}
            )

        return configs

    except (yaml.YAMLError, FileNotFoundError, OSError) as exc:
        _logger.exception(f"Failed to load or parse config: {exc}")
        raise ScikitplotException(f"Error loading config: {exc}") from exc
