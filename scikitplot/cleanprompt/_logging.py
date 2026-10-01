"""
Logging, with one rule: a removed value never reaches a log record.

Notes
-----
**User notes.** Nothing is logged until you ask for it::

    python -m scikitplot.cleanprompt redact --in p.txt --vault v.json --log-level debug
    python -m scikitplot.cleanprompt inspect --in p.txt --log-level info --log-format json

From Python::

    from scikitplot.cleanprompt import configure_logging

    configure_logging("debug")

Logs say *what happened* — which engine ran, how many entities of each kind,
how long it took, which capability decision was taken. They never say *what was
found*. That is deliberate: a log file is copied into tickets, shipped to
aggregators and kept far longer than the data it describes.

**Developer notes — why this module exists at all.**

A redaction tool that logs the values it removed has defeated itself. The
personal data leaves the process through the log instead of the prompt, and it
leaves in a form that is *more* durable: log lines are shipped off the machine,
retained for months and indexed.

So the rule is absolute and it is enforced twice.

*First, by construction.* No call site passes a secret. Every log statement in
this submodule takes counts, kinds, labels, offsets and durations. A label is
``[PERSON-1]`` and is already in the text being sent; a *value* never appears.

*Second, by a scrubbing filter.* :class:`SecretFilter` holds the surfaces a
redaction removed and replaces any occurrence in a formatted record with
``<redacted>``. That is defence in depth against a future call site, not a
licence for one: the filter is attached only while a redaction is in flight,
and it cannot see a secret the pipeline never held.

The two are not redundant. Construction is the guarantee; the filter is what
catches the mistake that construction will eventually make.

**Library manners.** A :class:`logging.NullHandler` is attached at import, so
importing this submodule never produces output and never configures the root
logger. :func:`configure_logging` is opt-in and touches only this submodule's
logger, because a library that reconfigures root logging fights whatever the
application already set up.

See Also
--------
scikitplot.cleanprompt._diagnostics : What to report to a user, rather than a log.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import types
from typing import IO, Any, Iterable

__all__ = [
    "AUDIT_LOGGER_NAME",
    "LOGGER_NAME",
    "JsonFormatter",
    "SecretFilter",
    "VaultScrubber",
    "audit",
    "configure_logging",
    "get_logger",
    "log_level_from_env",
    "redacting",
]

#: Root logger name for this submodule.
LOGGER_NAME = "scikitplot.cleanprompt"

#: Logger for audit events: what left, never what was removed.
AUDIT_LOGGER_NAME = LOGGER_NAME + ".audit"

#: Environment variables that set the level, most specific first.
_ENV_VARS = ("CLEANPROMPT_LOG_LEVEL", "SKPLT_LOGGING_LEVEL")

#: What a scrubbed value is replaced with.
REDACTION_MARK = "<redacted>"

_root = logging.getLogger(LOGGER_NAME)
_root.addHandler(logging.NullHandler())

#: Scrubbing filters currently in force. Every logger in this namespace carries
#: each of them, including loggers created while they are active.
_ACTIVE: list[logging.Filter] = []


def _namespace_loggers() -> list[logging.Logger]:
    """
    Return every existing logger in this submodule's namespace.

    Notes
    -----
    **Developer notes — why a filter must go on every one of them.** A filter
    attached to a *logger* is consulted only for records created by that
    logger. A record created by ``scikitplot.cleanprompt._api`` propagates to
    the handlers of ``scikitplot.cleanprompt``, but not through its filters.
    Every module here logs through its own child logger, so a filter on the
    parent alone scrubbed nothing that mattered (``CP-059``, measured).
    """
    found = [_root]
    for name, item in sorted(logging.Logger.manager.loggerDict.items()):
        if name.startswith(LOGGER_NAME + ".") and isinstance(item, logging.Logger):
            found.append(item)
    return found


def _attach(filt: logging.Filter) -> None:
    """Put ``filt`` on every logger in the namespace, now and later."""
    if filt not in _ACTIVE:
        _ACTIVE.append(filt)
    for logger in _namespace_loggers():
        if filt not in logger.filters:
            logger.addFilter(filt)


def _detach(filt: logging.Filter) -> None:
    """Remove ``filt`` from every logger in the namespace."""
    if filt in _ACTIVE:
        _ACTIVE.remove(filt)
    for logger in _namespace_loggers():
        logger.removeFilter(filt)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger under this submodule's namespace.

    Parameters
    ----------
    name : str, optional
        A module's ``__name__``. The submodule prefix is stripped so that
        ``scikitplot.cleanprompt._engine`` logs under ``…cleanprompt._engine``
        rather than being nested twice.

    Returns
    -------
    logging.Logger
        The logger.

    Examples
    --------
    >>> get_logger("scikitplot.cleanprompt._engine").name
    'scikitplot.cleanprompt._engine'
    >>> get_logger().name
    'scikitplot.cleanprompt'
    """
    if not name or name == LOGGER_NAME:
        return _root
    if name.startswith(LOGGER_NAME + "."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger("{}.{}".format(LOGGER_NAME, name.rsplit(".", 1)[-1]))
    for filt in _ACTIVE:
        if filt not in logger.filters:
            logger.addFilter(filt)
    return logger


#: Attributes every :class:`logging.LogRecord` carries; anything else was
#: attached by a call site with ``extra=``.
_STANDARD_FIELDS = frozenset(
    set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}
)


class SecretFilter(logging.Filter):
    """
    Replace known secret surfaces in a record with :data:`REDACTION_MARK`.

    Parameters
    ----------
    secrets : iterable of str, optional
        Surfaces to scrub. Short strings are ignored; see the notes.

    Notes
    -----
    **Developer notes.** Defence in depth, not the primary guarantee. The
    primary guarantee is that no call site passes a secret; this catches the
    call site that eventually will.

    Two deliberate limits.

    *Short surfaces are not scrubbed.* A two-character secret would match
    inside ordinary words and turn every message into noise, which trains
    people to ignore logs. The floor is four characters.

    *Scrubbing happens on the formatted message*, including arguments, because
    a secret passed as a ``%s`` argument is not in ``record.msg``. The record's
    ``args`` are cleared afterwards so the handler does not re-interpolate them
    and undo the work.

    Examples
    --------
    >>> import logging
    >>> filt = SecretFilter(["topsecret@example.com"])
    >>> rec = logging.LogRecord(
    ...     "x", logging.INFO, "f", 1, "found %s", ("topsecret@example.com",), None
    ... )
    >>> filt.filter(rec) and "topsecret" not in rec.getMessage()
    True
    """

    #: Surfaces shorter than this are not scrubbed. Scrubbing ``"Ann"`` would
    #: also rewrite ``"Annual"`` and ``"Planning"`` in every record, making logs
    #: unreadable; this filter is the second line of defence — the submodule
    #: never passes a value to a logger in the first place.
    MIN_LENGTH = 4

    def __init__(self, secrets: Iterable[str] | None = None) -> None:
        super().__init__()
        self._secrets: list[str] = []
        self.add(secrets or ())

    def add(self, secrets: Iterable[str]) -> None:
        """Add surfaces to scrub, longest first."""
        for secret in secrets:
            if isinstance(secret, str) and len(secret) >= self.MIN_LENGTH:
                self._secrets.append(secret)
        # Longest first, so a secret containing another is scrubbed whole.
        self._secrets.sort(key=len, reverse=True)

    def scrub(self, text: str) -> str:
        """
        Return ``text`` with every held surface replaced.

        Parameters
        ----------
        text : str
            Any text.

        Returns
        -------
        str
            The scrubbed text.
        """
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTION_MARK)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Scrub the record in place. Always returns ``True``.

        Notes
        -----
        **Developer notes.** Three places carry text out of a record, and all
        three are scrubbed: the formatted message; string fields a call site
        attached with ``extra=`` (which :class:`JsonFormatter` emits); and the
        rendered traceback, which is formatted here once and stored as
        ``exc_text`` so no handler renders the raw one afterwards.
        """
        if not self._secrets:
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad record must still be emitted
            return True
        scrubbed = self.scrub(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        for key, value in list(vars(record).items()):
            if (
                key in _STANDARD_FIELDS
                or key.startswith("_")
                or not isinstance(value, str)
            ):
                continue
            cleaned = self.scrub(value)
            if cleaned != value:
                setattr(record, key, cleaned)
        if record.exc_info and record.exc_info[0] is not None:
            rendered = record.exc_text or logging.Formatter().formatException(
                record.exc_info
            )
            record.exc_text = self.scrub(rendered)
        elif record.exc_text:
            record.exc_text = self.scrub(record.exc_text)
        return True


class JsonFormatter(logging.Formatter):
    """
    Format a record as one JSON object per line.

    Notes
    -----
    **Developer notes.** For shipping into a log aggregator, where a text
    format has to be re-parsed with a regular expression. Keys are stable;
    anything a call site attaches with ``extra=`` is merged in, so a caller can
    add structure without a new formatter.
    """

    #: Attributes :class:`logging.LogRecord` always carries, excluded from the
    #: merged ``extra`` payload.
    _STANDARD = _STANDARD_FIELDS

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a JSON object."""
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in self._STANDARD and not key.startswith("_"):
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    value = repr(value)  # ruff: ignore[redefined-loop-name]
                payload[key] = value
        if record.exc_info or record.exc_text:
            # exc_text first: a SecretFilter stores the scrubbed rendering there.
            payload["exception"] = record.exc_text or self.formatException(
                record.exc_info
            )
        return json.dumps(payload, ensure_ascii=False)


def log_level_from_env(default: str = "warning") -> str:
    """
    Return the log level named by the environment, or ``default``.

    Parameters
    ----------
    default : str, default='warning'
        Used when no variable is set or the value is not a level.

    Returns
    -------
    str
        A lower-case level name.

    Notes
    -----
    **Developer notes.** An unrecognised value falls back rather than raising.
    These variables are commonly set once for a whole shell, and refusing to
    run because an inherited value is misspelt would be obstructive.

    Examples
    --------
    >>> log_level_from_env("info")
    'info'
    """
    for name in _ENV_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value in ("critical", "error", "warning", "info", "debug"):
            return value
    return default


def configure_logging(
    level: str = "warning",
    fmt: str = "text",
    stream: IO[str] | None = None,
) -> logging.Logger:
    """
    Attach a handler to this submodule's logger.

    Parameters
    ----------
    level : str, default='warning'
        One of ``critical``, ``error``, ``warning``, ``info``, ``debug``.
    fmt : str, default='text'
        ``"text"`` or ``"json"``.
    stream : file-like, optional
        Destination. Defaults to :data:`sys.stderr`, because standard output
        carries results and a log line in a pipe would corrupt them.

    Returns
    -------
    logging.Logger
        This submodule's logger.

    Raises
    ------
    ValueError
        If ``level`` or ``fmt`` is unknown.

    Notes
    -----
    **Developer notes.** Replaces any handler this function previously
    attached, and leaves handlers the application attached alone. Calling it
    twice therefore does not double every line — the duplicate-handler defect
    that makes library logging notorious.

    ``propagate`` is left ``True`` so an application that configured root
    logging still receives these records; the handler attached here is
    additional, not exclusive. That does mean an application which has
    configured root logging *and* calls this will see records twice, which is
    why this function exists only as an explicit opt-in.
    """
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    if level not in levels:
        msg = "unknown log level {!r}; choose from {}".format(
            level,
            ", ".join(sorted(levels)),
        )
        raise ValueError(msg)
    if fmt not in ("text", "json"):
        raise ValueError(f"unknown log format {fmt!r}; choose text or json")

    for handler in list(_root.handlers):
        if getattr(handler, "_cleanprompt_owned", False):
            _root.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    handler._cleanprompt_owned = True  # noqa: SLF001 - our marker on our handler
    _root.addHandler(handler)
    _root.setLevel(levels[level])
    return _root


class redacting:  # noqa: N801 - a context manager reads better lower-case
    """
    Scrub known secrets from log records for the duration of a block.

    Parameters
    ----------
    secrets : iterable of str, optional
        Surfaces to scrub.
    logger : logging.Logger, optional
        Logger to attach to. Defaults to this submodule's root logger.

    Notes
    -----
    **User notes.** Wrap anything that logs while holding personal data::

        with redacting(vault.export().values()):
            do_work()

    **Developer notes.** The filter is attached to *every logger in this
    namespace*, not to a handler, so it applies however the application has
    configured its output. Attaching it to the namespace root alone looked
    equivalent and was not: a logger's filters see only that logger's own
    records, never its children's (``CP-059``).
    It is removed on exit, including when the block raises, so a long-lived
    process does not accumulate filters holding the secrets of every request it
    has ever served — which would be the very leak this is meant to prevent.

    Examples
    --------
    >>> with redacting(["topsecret@example.com"]):
    ...     pass
    """

    __slots__ = ("_filter", "_logger")

    def __init__(
        self,
        secrets: Iterable[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._filter = SecretFilter(secrets)
        self._logger = logger if logger is not None else _root

    def __enter__(self) -> SecretFilter:
        if self._logger is _root:
            _attach(self._filter)
        else:
            self._logger.addFilter(self._filter)
        return self._filter

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        if self._logger is _root:
            _detach(self._filter)
        else:
            self._logger.removeFilter(self._filter)


class _SharedSecrets(SecretFilter):
    """
    One scrubbing filter for every live vault, counting each surface.

    Notes
    -----
    **Developer notes — why one filter and not one per vault.** A web server
    holds a session per visitor. One filter per session meant every log record
    was checked by every filter in turn, and each filter was attached to every
    logger in the namespace: the cost of one log line grew with the number of
    visitors (``CP-065``). One shared filter holds a *count* per surface, so a
    value two sessions hold is scrubbed until both release it, and it is
    attached to the loggers once — when the first surface arrives.

    **Developer notes — why the cost does not grow with what is held.** Adding
    or releasing a surface is O(1): nothing is copied or recompiled (a
    compiled alternation, rebuilt per add, made 5000 sessions take six minutes
    to create; copying the surface list per add made one cleaner with 240 000
    values quadratic — both measured). Finding which held surfaces occur in a
    record uses the fact that every surface is at least :attr:`MIN_LENGTH`
    characters: each is indexed by its first ``MIN_LENGTH`` characters, with
    the lengths held under that prefix. A record is read once; at each
    position whose prefix is held, one dictionary lookup per recorded length
    decides membership exactly. Up to :attr:`SCAN_LIMIT` surfaces a substring
    test per surface, which runs in C, is cheaper and is used instead; both
    find exactly the surfaces that occur, and a test holds them equal.

    **Thread safety.** Writers take a lock. The scrubber, which runs inside
    ``logging`` on any thread, takes none: it reads single keys and iterates
    only snapshots (``tuple(dict)`` is taken under the interpreter lock).
    """

    #: Up to this many held surfaces, one substring test each is cheaper than
    #: indexing the record. Measured on a 268-character line: 15 µs against  # ruff: ignore[ambiguous-unicode-character-comment]
    #: 24 µs at 200 surfaces, 30 against 24 at 400; both grow with the line.  # ruff: ignore[ambiguous-unicode-character-comment]
    SCAN_LIMIT = 256

    def __init__(self) -> None:
        import threading  # ruff: ignore[import-outside-top-level]

        logging.Filter.__init__(self)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        #: prefix -> {length: number of held surfaces with that prefix and length}
        self._lengths: dict[str, dict[int, int]] = {}

    @property
    def _secrets(self) -> dict[str, int]:
        """dict: The held surfaces (as keys); what :meth:`filter` tests."""
        return self._counts

    def add(self, secrets: Iterable[str]) -> None:
        """Hold these surfaces once more each."""
        with self._lock:
            was_empty = not self._counts
            for secret in secrets:
                if not isinstance(secret, str) or len(secret) < self.MIN_LENGTH:
                    continue
                count = self._counts.get(secret, 0)
                self._counts[secret] = count + 1
                if count == 0:
                    by_length = self._lengths.setdefault(secret[: self.MIN_LENGTH], {})
                    by_length[len(secret)] = by_length.get(len(secret), 0) + 1
            if was_empty and self._counts:
                _attach(self)

    def release(self, secrets: Iterable[str]) -> None:
        """Hold these surfaces once less each; forget any no longer held."""
        with self._lock:
            was_empty = not self._counts
            for secret in secrets:
                count = self._counts.get(secret)
                if count is None:
                    continue
                if count > 1:
                    self._counts[secret] = count - 1
                    continue
                del self._counts[secret]
                prefix = secret[: self.MIN_LENGTH]
                by_length = self._lengths[prefix]
                if by_length[len(secret)] > 1:
                    by_length[len(secret)] -= 1
                else:
                    del by_length[len(secret)]
                    if not by_length:
                        del self._lengths[prefix]
            if not was_empty and not self._counts:
                _detach(self)

    def present(self, text: str) -> set[str]:
        """
        Return the held surfaces that occur in ``text``.

        Parameters
        ----------
        text : str
            Any text.

        Returns
        -------
        set of str
            Exactly the held surfaces that are substrings of ``text``.
        """
        counts = self._counts
        if len(counts) <= self.SCAN_LIMIT:
            return {secret for secret in tuple(counts) if secret in text}
        found: set[str] = set()
        lengths = self._lengths
        width = self.MIN_LENGTH
        for start in range(len(text) - width + 1):
            by_length = lengths.get(text[start : start + width])
            if by_length is None:
                continue
            for length in tuple(by_length):
                candidate = text[start : start + length]
                if len(candidate) == length and candidate in counts:
                    found.add(candidate)
        return found

    def scrub(self, text: str) -> str:
        """
        Return ``text`` with every held surface replaced, longest first.

        Parameters
        ----------
        text : str
            Any text.

        Returns
        -------
        str
            The scrubbed text.
        """
        for secret in sorted(self.present(text), key=len, reverse=True):
            text = text.replace(secret, REDACTION_MARK)
        return text

    @property
    def held(self) -> int:
        """int: Distinct surfaces currently held."""
        return len(self._counts)


#: The filter every :class:`VaultScrubber` contributes to.
_SHARED = _SharedSecrets()


class VaultScrubber:
    """
    Keep a vault's values out of every log record for as long as it holds them.

    Notes
    -----
    **User notes.** You do not create one: :class:`~scikitplot.cleanprompt.Cleaner`,
    :class:`~scikitplot.cleanprompt.Guard` and
    :class:`~scikitplot.cleanprompt.Session` do, and it lives exactly as long
    as they hold values. While it lives, any record from this submodule that
    would contain a removed value carries ``<redacted>`` instead.

    **Developer notes.** Surfaces go into one shared, counted filter. They are
    released by :meth:`close` and, if the owner is dropped without being
    closed, by a :func:`weakref.finalize` on the owner — a surface left in the
    filter would otherwise keep the secrets of every cleaner the process ever
    made, which is the leak it exists to prevent.
    """

    __slots__ = ("__weakref__", "_finalizer", "_held")

    def __init__(self, owner: object | None = None) -> None:
        import weakref  # ruff: ignore[import-outside-top-level]

        self._held: list[str] = []
        # The finalizer holds the list, not this object, so it releases
        # whatever the list contains at the moment the owner is collected.
        self._finalizer = (
            weakref.finalize(owner, _SHARED.release, self._held)
            if owner is not None
            else None
        )

    def add(self, secrets: Iterable[str]) -> None:
        """Scrub these surfaces too."""
        fresh = [
            one
            for one in secrets
            if isinstance(one, str) and len(one) >= SecretFilter.MIN_LENGTH
        ]
        self._held.extend(fresh)
        _SHARED.add(fresh)

    def close(self) -> None:
        """Stop scrubbing this vault's surfaces and forget them."""
        _SHARED.release(self._held)
        self._held.clear()
        if self._finalizer is not None:
            self._finalizer.detach()


def audit(event: str, **fields: Any) -> None:
    """
    Emit one structured audit event on :data:`AUDIT_LOGGER_NAME`.

    Parameters
    ----------
    event : str
        What happened: ``"encoded"``, ``"decoded"``, ``"blocked"``, ...
    **fields
        Counts, kinds, formats, fingerprints and digests. **Never a value.**

    Notes
    -----
    **User notes.** Audit events answer "what left this machine, under which
    plan?" without keeping what was removed. Send them where compliance wants
    them::

        import logging

        logging.getLogger("scikitplot.cleanprompt.audit").addHandler(my_handler)

    With ``configure_logging(level="info", fmt="json")`` each event is one JSON
    line whose fields are the keywords given here.

    **Developer notes.** Emitted at ``INFO`` through the ordinary logger, so it
    is off unless asked for and every scrubbing filter applies to it as well.
    """
    get_logger(AUDIT_LOGGER_NAME).info("%s", event, extra={"event": event, **fields})
