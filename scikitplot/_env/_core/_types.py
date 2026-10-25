# scikitplot/_env/_core/_types.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Closed vocabularies of the environment builder.

Notes
-----
**User notes.** Pass the string values (``network="offline"``); each is
checked against the members here, so a misspelling fails with the list of
valid values.

**Developer notes.** :func:`choice` is the one validator every entry point
uses (Rule 84).
"""

from __future__ import annotations

import enum
from typing import Any, TypeVar

from ._errors import ConfigError

__all__ = [
    "Approval",
    "Bootstrap",
    "Conversion",
    "Ecosystem",
    "Interaction",
    "Network",
    "Operation",
]

_E = TypeVar("_E", bound=enum.Enum)


class _StrEnum(str, enum.Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class Network(_StrEnum):
    """
    Network policy.

    ``online``: tools may download. ``offline``: no step may use the
    network; each tool runs with its offline switch (micromamba/uv
    ``--offline``, pip ``--no-index``), and a step that cannot run offline (a
    bootstrap download without a cached archive) is refused at planning time.

    There is deliberately no "prefer offline" mode: no backend offers one
    with a defined meaning, and a fallback from cache to network would make
    the same request behave differently on two machines.
    """

    ONLINE = "online"
    OFFLINE = "offline"


class Interaction(_StrEnum):
    """
    Whether the builder may ask a person.

    ``never``: an unanswered approval is an error naming the flag to pass.
    ``auto``: ask only on an interactive terminal outside CI. ``always``: ask
    on the terminal (an error when there is none).
    """

    NEVER = "never"
    AUTO = "auto"
    ALWAYS = "always"


class Bootstrap(_StrEnum):
    """Whether a missing tool (micromamba, uv) may be installed by the builder."""

    NEVER = "never"
    IF_MISSING = "if-missing"


class Conversion(_StrEnum):
    """
    Whether a dependency may move between ecosystems inexactly.

    ``exact``: a Conda package on a PyPI-only backend (or the reverse) is
    refused. ``allow-approximate``: it is installed by the same name from the
    other ecosystem, and the plan says so.
    """

    EXACT = "exact"
    ALLOW_APPROXIMATE = "allow-approximate"


class Ecosystem(_StrEnum):
    """Where a dependency comes from."""

    PYPI = "pypi"
    CONDA = "conda"


class Approval(_StrEnum):
    """Kinds of step that need an explicit approval."""

    BOOTSTRAP = "bootstrap"
    SHELL_INIT = "shell_init"
    REMOVE = "remove"


class Operation(_StrEnum):
    """What a plan does to the environment."""

    ENSURE = "ensure"
    RECREATE = "recreate"
    REMOVE = "remove"


def choice(value: Any, enum_type: type[_E], *, field: str) -> _E:
    """
    Return the member of ``enum_type`` named by ``value``.

    Parameters
    ----------
    value : str or enum member
        Value to convert; matching is exact.
    enum_type : type
        The enumeration.
    field : str
        Field name used in the error.

    Returns
    -------
    enum member
        The member.

    Raises
    ------
    ConfigError
        If ``value`` is not a string or not a member value.
    """
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string, got {type(value).__name__}")
    try:
        return enum_type(value)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in enum_type)
        raise ConfigError(f"{field}={value!r} is not one of {valid}") from None
