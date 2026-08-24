# scikitplot/corpus/_diagnostics.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Structured, serialisable diagnostics for Corpus operations.

This module provides :class:`ErrorRecord` -- the single diagnostic type every
Corpus operation uses to report *what went wrong* without either raising or
staying silent.

Notes
-----
**User-focused.**  When an operation completes but did not do everything it was
asked to do, it hands back one :class:`ErrorRecord` per problem.  Each record is
plain data: it can be printed, counted, filtered by ``code``, written to a build
manifest, or sent across a process boundary.

**Developer-focused.**  Two independent findings motivate this type.

*Serialisability.*  ``BuildResult.errors`` was previously
``list[tuple[str, Exception]]``.  A live exception is not JSON-serialisable, so
a :class:`~scikitplot.corpus.BuildResult` could not be written to a run manifest
or crossed over an adapter boundary without bespoke handling (finding
F-R02-03).

*Memory.*  A live exception holds ``__traceback__``, which references every
frame in the failing call stack, which references that frame's locals.  Holding
one per failure retains those object graphs for the lifetime of the result.
Measured at 2000 failures carrying 50 KB frame locals: **97.0 MB retained versus
0.7 MB** for the equivalent structured records -- a factor of 139, and the
retention is worst exactly when a build is going worst (finding F-R11-02).

:class:`ErrorRecord` therefore captures the exception's *type name* and
*message* as strings and deliberately **does not retain the exception object**.
A formatted traceback may be attached explicitly via ``traceback_text`` when a
caller opts in.

See Also
--------
scikitplot.corpus._schema.ErrorPolicy : chooses *what to do* when an error occurs.
"""

from __future__ import annotations

import dataclasses
import traceback as _traceback
from enum import unique
from typing import Any

from ._schema import _StrEnumBase

__all__: list[str] = [
    "ErrorCategory",
    "ErrorRecord",
]


@unique
class ErrorCategory(_StrEnumBase):
    """Coarse classification of a diagnostic, for counting and filtering.

    Notes
    -----
    Categories are intentionally few and stable.  Fine-grained meaning belongs
    in :attr:`ErrorRecord.code`, which is a free-form stable string; the
    category exists so a caller can answer "how many *input* problems did this
    build have?" without enumerating every code.

    Examples
    --------
    >>> ErrorCategory.SOURCE == "source"
    True
    """

    SOURCE = "source"
    """The input could not be obtained: missing file, refused URL, bad archive."""

    PARSE = "parse"
    """The input was obtained but could not be interpreted."""

    VALIDATION = "validation"
    """The input was interpreted but violated a schema or contract."""

    CAPABILITY = "capability"
    """A required component was absent, broken or misconfigured."""

    SECURITY = "security"
    """The input was refused by a security control (traversal, SSRF, integrity)."""

    RESOURCE = "resource"
    """A declared budget or limit was exhausted: bytes, files, time, retries."""

    INTERNAL = "internal"
    """An unexpected failure inside Corpus itself."""


@dataclasses.dataclass(frozen=True)
class ErrorRecord:
    """One structured, serialisable diagnostic.

    Parameters
    ----------
    code : str
        Stable machine-readable identifier, e.g. ``"ARCHIVE_MEMBER_REFUSED"``.
        Stable across releases so callers may branch on it.
    category : ErrorCategory
        Coarse classification, for counting and filtering.
    message : str
        Human-readable description. Should name the specific subject.
    stage : str or None, optional
        Pipeline stage that produced the record, e.g. ``"read"``, ``"embed"``.
    source_id : str or None, optional
        Identifier of the affected input: a ``doc_id``, path, URL or member name.
    exception_type : str or None, optional
        Class *name* of the originating exception, as a string. The exception
        object itself is deliberately not retained -- see the module notes.
    traceback_text : str or None, optional
        Formatted traceback, attached only when a caller explicitly opts in via
        ``include_traceback=True``.
    details : dict, optional
        Additional JSON-compatible context.

    Notes
    -----
    **Developer.**  ``frozen=True`` because a diagnostic describes something that
    already happened; mutating it after the fact would make a build manifest
    disagree with the run it describes.

    Examples
    --------
    >>> rec = ErrorRecord(
    ...     code="PARSE_FAILED",
    ...     category=ErrorCategory.PARSE,
    ...     message="malformed XML",
    ...     source_id="doc-1",
    ... )
    >>> rec.code
    'PARSE_FAILED'
    >>> sorted(rec.to_dict())[:3]
    ['category', 'code', 'details']

    A record built from an exception keeps the type name, not the object:

    >>> try:
    ...     raise ValueError("boom")
    ... except ValueError as exc:
    ...     rec = ErrorRecord.from_exception(
    ...         exc, code="X", category=ErrorCategory.INTERNAL
    ...     )
    >>> rec.exception_type
    'ValueError'
    >>> rec.traceback_text is None
    True
    """

    code: str
    category: ErrorCategory
    message: str
    stage: str | None = None
    source_id: str | None = None
    exception_type: str | None = None
    traceback_text: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        code: str,
        category: ErrorCategory | str,
        stage: str | None = None,
        source_id: str | None = None,
        include_traceback: bool = False,
        details: dict[str, Any] | None = None,
    ) -> ErrorRecord:
        """Build a record from ``exc`` **without retaining the exception**.

        Parameters
        ----------
        exc : BaseException
            The originating exception. Only its type name and ``str()`` are kept.
        code : str
            Stable machine-readable identifier.
        category : ErrorCategory or str
            Coarse classification.
        stage : str or None, optional
            Pipeline stage.
        source_id : str or None, optional
            Identifier of the affected input.
        include_traceback : bool, optional
            When ``True``, format and attach the traceback as text. Default
            ``False``: a formatted traceback is a string and therefore safe to
            retain, but it is verbose, so it is opt-in.
        details : dict or None, optional
            Additional JSON-compatible context.

        Returns
        -------
        ErrorRecord
            A record holding no reference to ``exc`` or its frames.

        Notes
        -----
        **Developer.**  This method is the reason the 139x retention measured in
        F-R11-02 does not recur: the exception is read and dropped inside this
        call, so no frame locals survive it.
        """
        tb = None
        if include_traceback and exc.__traceback__ is not None:
            tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            code=code,
            category=ErrorCategory(category),
            message=str(exc),
            stage=stage,
            source_id=source_id,
            exception_type=type(exc).__name__,
            traceback_text=tb,
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping.

        Returns
        -------
        dict
            All fields, with :attr:`category` rendered as its string value.

        Examples
        --------
        >>> import json
        >>> rec = ErrorRecord(code="C", category=ErrorCategory.SOURCE, message="m")
        >>> _ = json.dumps(rec.to_dict())
        """
        data = dataclasses.asdict(self)
        data["category"] = str(self.category.value)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ErrorRecord
        """
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["category"] = ErrorCategory(kwargs["category"])
        kwargs.setdefault("details", {})
        return cls(**kwargs)

    def __str__(self) -> str:
        """Return a compact single-line rendering."""
        where = self.source_id or self.stage or "-"
        return f"[{self.category.value}/{self.code}] {where}: {self.message}"
