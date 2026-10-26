# scikitplot/_env/_core/_events.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Structured events: one record per thing that happened, to the log and to JSON Lines.

Notes
-----
**User notes.** Every event goes to the module logger (``-v`` on the command
line shows them). Give ``events=`` a path or a writable text stream to also
get one JSON object per line, for CI dashboards and agents::

    {"schema": "scikitplot.env.event/v1", "seq": 3, "time": "...",
     "event": "step.finished", "step": "create", "returncode": 0, ...}

**Developer notes.** Values pass through :py:func:`~._plan.redact` before they
are written, and events never carry environment variables or tool output in
full (Rule 141); tool output is logged at debug level, line by line.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
from typing import Any, TextIO

from ._errors import ConfigError
from ._plan import redact_data as _clean

__all__ = ["EventSink"]

logger = logging.getLogger(__name__)

#: Schema identifier of an event record.
EVENT_SCHEMA = "scikitplot.env.event/v1"


class EventSink:
    """
    Emit events to the logger and, optionally, to a JSON Lines stream.

    Parameters
    ----------
    target : str, os.PathLike, text stream or None, default=None
        Where JSON Lines go: a file path (appended to) or an open text stream.

    Examples
    --------
    >>> import io
    >>> from scikitplot._env._core._events import EventSink
    >>> buffer = io.StringIO()
    >>> sink = EventSink(buffer)
    >>> sink.emit("plan.created", backend="uv")
    >>> '"event":"plan.created"' in buffer.getvalue()
    True
    """

    def __init__(self, target: Any = None) -> None:  # noqa: D107
        self._seq = 0
        self._own = False
        self._stream: TextIO | None = None
        if target is None:
            return
        if isinstance(target, (str, os.PathLike)):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
                # Kept open for the sink's lifetime; closed by close().
                self._stream = open(  # noqa: SIM115
                    target, "a", encoding="utf-8", newline="\n"
                )
            except OSError as exc:
                raise ConfigError(
                    f"cannot open the events file {target}: {exc}"
                ) from exc
            self._own = True
        elif isinstance(target, io.TextIOBase) or hasattr(target, "write"):
            self._stream = target
        else:
            raise TypeError("events must be a path or a writable text stream")

    def emit(self, event: str, **fields: Any) -> None:
        """
        Record one event.

        Parameters
        ----------
        event : str
            Dotted event name (``"step.started"``).
        **fields
            Plain-data details.
        """
        self._seq += 1
        record = {
            "schema": EVENT_SCHEMA,
            "seq": self._seq,
            "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **_clean(fields),
        }
        logger.info(
            "%s %s",
            event,
            json.dumps(_clean(fields), sort_keys=True, ensure_ascii=False),
        )
        if self._stream is not None:
            self._stream.write(
                json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            self._stream.flush()

    def close(self) -> None:
        """Close a stream this sink opened itself."""
        if self._own and self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> EventSink:  # noqa: D105, PYI034
        return self

    def __exit__(self, *exc: object) -> None:  # noqa: D105
        self.close()
