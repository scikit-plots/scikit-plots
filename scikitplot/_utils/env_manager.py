"""env_manager."""

from ..exceptions import ScikitplotException

LOCAL = "local"
CONDA = "conda"
VIRTUALENV = "virtualenv"
UV = "uv"


def validate(env_manager):
    """Validate."""
    allowed_values = [LOCAL, CONDA, VIRTUALENV, UV]
    if env_manager not in allowed_values:
        raise ScikitplotException(
            f"Invalid value for `env_manager`: {env_manager}. Must be one of {allowed_values}",
            error_code=0,
        )
