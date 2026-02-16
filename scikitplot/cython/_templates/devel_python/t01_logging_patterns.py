"""Developer Python template: logging patterns + structured errors."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be > 0; got {value!r}")
    log.debug("validated %s=%s", name, value)
