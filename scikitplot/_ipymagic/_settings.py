# scikitplot/_ipymagic/_settings.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Session settings of the magics, changed with ``%scikitplot_config``.

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Meaning
   * - ``raise_on_error``
     - ``False``
     - a non-zero exit raises :py:class:`~.CommandFailedError` (stops "Run
       all"; for CI notebooks)
   * - ``timeout``
     - ``None``
     - seconds a subprocess command may run (None: no limit)
   * - ``isolate``
     - ``False``
     - run every command in a subprocess, native ones too
   * - ``stream``
     - ``True``
     - show subprocess output as it is produced

Notes
-----
**User notes.** ``%scikitplot_config`` alone shows the settings;
``%scikitplot_config raise_on_error=true timeout=600`` changes two. Values are
``true``/``false``, a number, or ``none``; an unknown name or a bad value is
refused and nothing changes.

**Developer notes.** The settings are an immutable value; a change makes a new
one, so a half-applied change cannot exist.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from ._errors import MagicUsageError

__all__ = ["MagicSettings"]

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _as_bool(name: str, text: str) -> bool:
    value = text.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise MagicUsageError(f"{name}={text!r} is not a boolean", hint="use true or false")


def _as_timeout(text: str) -> float | None:
    value = text.strip().lower()
    if value in ("none", "null", ""):
        return None
    try:
        seconds = float(value)
    except ValueError:
        raise MagicUsageError(
            f"timeout={text!r} is not a number", hint="seconds, or none"
        ) from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise MagicUsageError(
            f"timeout={text!r} must be a positive number of seconds",
            hint="seconds > 0, or none for no limit",
        )
    return seconds


@dataclasses.dataclass(frozen=True)
class MagicSettings:  # numpydoc ignore=PR02
    """
    Behaviour of the magics for one IPython session.

    Parameters
    ----------
    raise_on_error : bool, default=False
        Raise :py:class:`~.CommandFailedError` for a non-zero exit status.
    timeout : float or None, default=None
        Seconds a subprocess command may run; None for no limit.
    isolate : bool, default=False
        Run native commands in a subprocess too.
    stream : bool, default=True
        Show subprocess output as it is produced.
    """

    raise_on_error: bool = False
    timeout: float | None = None
    isolate: bool = False
    stream: bool = True

    def __post_init__(self) -> None:  # noqa: D105
        for name in ("raise_on_error", "isolate", "stream"):
            if not isinstance(getattr(self, name), bool):
                raise MagicUsageError(f"{name} must be a bool")
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise MagicUsageError("timeout must be a positive number or None")

    def updated(self, text: str) -> MagicSettings:
        """
        Return the settings with ``key=value`` assignments applied.

        Parameters
        ----------
        text : str
            Space-separated ``key=value`` pairs.

        Returns
        -------
        MagicSettings
            New settings (``self`` is unchanged).

        Raises
        ------
        MagicUsageError
            For an unknown name, a malformed pair or a bad value (no change
            is applied).

        Examples
        --------
        >>> from scikitplot._ipymagic import MagicSettings
        >>> MagicSettings().updated("raise_on_error=true timeout=60").timeout
        60.0
        """
        changes: dict[str, Any] = {}
        known = {f.name for f in dataclasses.fields(self)}
        for pair in text.split():
            name, sep, value = pair.partition("=")
            if not sep or not name:
                raise MagicUsageError(
                    f"{pair!r} is not key=value", hint=f"known keys: {sorted(known)}"
                )
            if name not in known:
                raise MagicUsageError(
                    f"unknown setting {name!r}", hint=f"known keys: {sorted(known)}"
                )
            changes[name] = (
                _as_timeout(value) if name == "timeout" else _as_bool(name, value)
            )
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """
        Return the settings as plain data.

        Returns
        -------
        dict
            Every setting.
        """
        return dataclasses.asdict(self)
