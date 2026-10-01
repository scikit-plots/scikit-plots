"""
Encode a prompt, send it anywhere, decode the reply.

The model-agnostic surface: three functions and a context manager.

Notes
-----
**User notes.**

.. code-block:: python

    from scikitplot.cleanprompt import encode, decode

    safe, handle = encode("Mail ada@example.com about the Acme deal")
    # -> 'Mail [EMAIL-1] about the Acme deal'

    reply = your_model(safe)  # Claude, Gemini, ChatGPT, Ollama, anything
    decode(reply, handle)  # your values come back

For a conversation, a :class:`Session` keeps the placeholders consistent across
turns and clears itself when the block ends:

.. code-block:: python

    from scikitplot.cleanprompt import session

    with session(profile="strict", ner=True) as chat:
        answer = chat.roundtrip("Draft a reply to ada@example.com", send=your_model)
        follow = chat.roundtrip("Now make it shorter", send=your_model)
    # the vault is cleared here, whatever happened inside

``send`` is any callable taking a string and returning a string. That is the
whole integration contract, which is why this works with every provider without
this submodule knowing about any of them.

**Developer notes — why the seam is a callable.**

An adapter per provider would mean shipping, versioning and testing an HTTP
client for each, and being wrong about all of them within a release. A callable
inverts it: the caller already has a working client, already handles its auth,
retries and streaming, and hands over a function. The submodule stays
model-agnostic by construction rather than by effort.

:meth:`Session.roundtrip` exists because the encode-send-decode sequence has one
correct order and a failure mode in the middle. Written by hand, an exception
from the model leaves the vault dangling; here the vault's lifetime is the
block's.

**On the handle.** :class:`Handle` is the unsafe half — it holds the values that
were removed. It is a separate object from the safe text so that no serializer
can emit both by accident, its ``repr`` reports counts rather than values, and
:meth:`Handle.export` is the one named way to get the secrets out.

See Also
--------
scikitplot.cleanprompt._engine : The pipeline underneath.
scikitplot.cleanprompt._vault : Where the removed values live.
"""

from __future__ import annotations

import functools
import threading
import time
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

if TYPE_CHECKING:
    # Annotations are strings under ``from __future__ import annotations``, so
    # ``Self`` is needed only by a type checker. Importing it at runtime made
    # the base tier depend on typing_extensions, which Python 3.11+ does not
    # ship and nothing here declares (CP-054).
    from typing_extensions import Self


from ._detectors import DetectorRegistry, default_registry
from ._diagnostics import describe_outcome, diagnose, suggest_terms
from ._engine import Redactor, restore
from ._engines import DEFAULT_ENGINE, build_detectors
from ._exceptions import CleanPromptError, PolicyError
from ._logging import VaultScrubber, get_logger
from ._policy import DEFAULT_POLICY, RedactionPolicy
from ._policy import profile as _profile
from ._types import Entry, RedactionResult
from ._vault import Vault

__all__ = [
    "EncodedPrompt",
    "Handle",
    "Session",
    "decode",
    "encode",
    "session",
]

logger = get_logger(__name__)

#: Callable that takes the safe text and returns the model's reply.
Sender = Callable[[str], str]


def _synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Run ``method`` holding its instance's ``_lock``.

    Parameters
    ----------
    method : callable
        A method of an object with a :class:`threading.RLock` in ``_lock``.
        Never a generator: the lock would be released when the generator
        object is returned, before any of its work ran.

    Returns
    -------
    callable
        The wrapped method.

    Notes
    -----
    **Developer notes — why every vault owner is locked (``CP-067``).**
    Encoding reads the vault as a seed, issues the next free labels and
    stores them. Two threads doing that at once read the same seed and issue
    the same label to two different values; a reply then restores one
    person's value where the other's was meant — measured on a shared Guard
    with eight threads. The lock is re-entrant because public methods call
    one another (a learning pass encodes many files).
    """

    @functools.wraps(method)
    def locked(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


@dataclass(frozen=True)
class Handle:
    """
    Everything needed to decode a reply. **Holds the removed values**.

    Parameters
    ----------
    vault : Vault
        Label-to-value mapping.
    policy : RedactionPolicy
        The policy that produced the labels; supplies the placeholder grammar.
    entries : tuple of Entry
        Ordered record of what was replaced, used to seed a following turn.

    Notes
    -----
    **User notes.** Treat a handle like a password: keep it on the machine that
    created it and let it go out of scope when you are done. Printing or
    logging it discloses nothing — its representation reports counts, not
    values.

    **Developer notes.** Deliberately *not* a plain tuple with the safe text.
    Keeping the safe half and the unsafe half in different objects means a
    caller who serializes "the result" cannot accidentally serialize the
    secrets, which is the mistake the upstream web tier made by putting the
    mapping in a cookie.
    """

    vault: Vault
    policy: RedactionPolicy
    entries: tuple[Entry, ...] = ()

    def __repr__(self) -> str:
        """Return a secret-free representation."""
        return f"Handle(entries={len(self.entries)}, policy={self.policy.fingerprint})"

    __str__ = __repr__

    @property
    def labels(self) -> tuple[str, ...]:
        """Tuple of str: Every placeholder issued. Labels are not secrets."""
        return tuple(entry.label for entry in self.entries)

    def export(self) -> dict[str, Any]:
        """
        Return a JSON-safe document containing the values.

        Returns
        -------
        dict
            ``{"grammar": ..., "policy": ..., "entries": {label: value}}``.

        Raises
        ------
        PolicyError
            If the vault has been cleared.

        Notes
        -----
        **User notes.** This is the deliberate, named way to move a handle
        across a process boundary — a web request, a queue, a subprocess. It
        contains the removed values in clear text. Nothing in this submodule
        calls it implicitly.
        """
        return {
            "grammar": self.policy.tag_style.as_dict(),
            "policy_fingerprint": self.policy.fingerprint,
            "entries": self.vault.export(),
        }

    @classmethod
    def load(
        cls, document: Mapping[str, Any], policy: RedactionPolicy | None = None
    ) -> Handle:
        """
        Rebuild a handle from :meth:`export` output.

        Parameters
        ----------
        document : mapping
            The exported document.
        policy : RedactionPolicy, optional
            Policy to decode under. Defaults to
            :data:`~scikitplot.cleanprompt._policy.DEFAULT_POLICY`.

        Returns
        -------
        Handle
            A handle with no entry history, which is enough to decode but not
            to seed a following turn.

        Raises
        ------
        PolicyError
            If the document is malformed, or its grammar does not match
            ``policy``.
        """
        entries = document.get("entries")
        if not isinstance(entries, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in entries.items()
        ):
            raise PolicyError("handle document has a malformed 'entries' object")
        active = policy if policy is not None else DEFAULT_POLICY
        grammar = document.get("grammar")
        if isinstance(grammar, dict) and grammar != active.tag_style.as_dict():
            raise PolicyError(
                "this handle was written under a different placeholder grammar; "
                "decode it with the policy that produced it"
            )
        return cls(
            vault=Vault(entries, grammar_fingerprint=active.tag_style.fingerprint),
            policy=active,
        )

    def clear(self) -> None:
        """Drop every value this handle holds."""
        self.vault.clear()


@dataclass(frozen=True)
class EncodedPrompt:
    """
    The result of :func:`encode`: a safe text and the handle to undo it.

    Parameters
    ----------
    text : str
        Safe to send.
    handle : Handle
        **Not** safe to send.
    result : RedactionResult
        The full result, for callers that want the detail.
    report : dict
        What happened, and what was not looking; see
        :func:`~scikitplot.cleanprompt._diagnostics.describe_outcome`.

    Notes
    -----
    **User notes.** Unpacks as a pair, so the common case reads well::

        safe, handle = encode(text)

    Check ``report["level"]``: ``"alert"`` means nothing was redacted *and*
    part of the detection surface is switched off, which is the one case where
    an empty result should not be read as "there was nothing to find".
    """

    text: str
    handle: Handle
    result: RedactionResult = field(repr=False)
    report: dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self):
        """Yield ``text`` then ``handle``, so the pair unpacks."""
        yield self.text
        yield self.handle

    @property
    def level(self) -> str:
        """str: ``"ok"``, ``"warning"`` or ``"alert"``."""
        return str(self.report.get("level", "ok"))

    @property
    def count(self) -> int:
        """int: How many distinct values were replaced."""
        return self.result.stats.entries


def _make_policy(  # ruff: ignore[too-many-positional-arguments]
    policy: RedactionPolicy | None = None,
    profile: str | None = None,
    hide: Iterable[str] | None = None,
    allow: Iterable[str] | None = None,
    ignore_case: bool | None = None,
    kinds: Iterable[str] | None = None,
) -> RedactionPolicy:
    """Assemble a policy from the keyword surface these functions accept."""
    base = (
        policy
        if policy is not None
        else (_profile(profile) if profile else DEFAULT_POLICY)
    )
    changes: dict[str, Any] = {}
    if allow is not None:
        changes["allow"] = tuple(allow)
    if ignore_case is not None:
        changes["case_insensitive"] = bool(ignore_case)
    if kinds is not None:
        changes["kinds"] = tuple(kinds)
    del hide  # applied per call, not part of the policy
    return base.evolve(**changes) if changes else base


def _make_registry(  # ruff: ignore[too-many-positional-arguments]
    policy: RedactionPolicy,
    registry: DetectorRegistry | None = None,
    ner: bool = False,
    engine: str = DEFAULT_ENGINE,
    language: str = "en",
    model: str | None = None,
    size: str = "sm",
) -> DetectorRegistry:
    """Build the detector registry for a policy, adding NER when asked."""
    if registry is not None:
        return registry
    built = default_registry(kinds=policy.kinds)
    if ner:
        for detector in build_detectors(
            mode=engine, language=language, model=model, size=size, required=True
        ):
            built.add(detector)
    return built


def encode(
    text: str,
    *,
    policy: RedactionPolicy | None = None,
    profile: str | None = None,
    hide: Iterable[str] | None = None,
    allow: Iterable[str] | None = None,
    kinds: Iterable[str] | None = None,
    ignore_case: bool | None = None,
    ner: bool = False,
    engine: str = DEFAULT_ENGINE,
    language: str = "en",
    model: str | None = None,
    size: str = "sm",
    registry: DetectorRegistry | None = None,
    word_boundary: bool = False,
    suggest: bool = True,
) -> EncodedPrompt:
    """
    Replace sensitive values in ``text`` with placeholders.

    Parameters
    ----------
    text : str
        The prompt to make safe.
    policy : RedactionPolicy, optional
        Full policy. Overrides ``profile`` and the individual keywords.
    profile : str, optional
        Named bundle: ``"minimal"``, ``"balanced"`` or ``"strict"``.
    hide : iterable of str, optional
        Extra exact strings to remove.
    allow : iterable of str, optional
        Surfaces that must never be removed.
    kinds : iterable of str, optional
        Detection kinds to enable.
    ignore_case : bool, optional
        Treat values differing only in case as one entity.
    ner : bool, default=False
        Also run named-entity detection.
    engine : str, default='auto'
        ``"auto"``, ``"spacy"``, ``"nltk"``, ``"both"`` or ``"none"``.
    language : str, default='en'
        Language code for the entity engine.
    model : str, optional
        Explicit spaCy model, overriding ``language`` and ``size``.
    size : str, default='sm'
        Preferred spaCy model size.
    registry : DetectorRegistry, optional
        Detectors to use, bypassing ``ner`` and the engine keywords.
    word_boundary : bool, default=False
        Whether ``hide`` terms match only at word boundaries.
    suggest : bool, default=True
        Include candidate terms in the report.

    Returns
    -------
    EncodedPrompt
        Unpacks as ``(text, handle)``.

    Raises
    ------
    TypeError
        If ``text`` is not a string.
    CapabilityError
        If ``ner`` was asked for and the chosen engine is unavailable.
    CleanPromptError
        For any other configuration or limit failure.

    Notes
    -----
    **User notes.** Send the text; keep the handle. If
    ``result.level == "alert"``, nothing was redacted *and* something was not
    looking — read the report before sending.

    Examples
    --------
    >>> safe, handle = encode("Mail ada@example.com")
    >>> safe
    'Mail [EMAIL-1]'
    >>> decode("Reply to [EMAIL-1]", handle)
    'Reply to ada@example.com'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}")

    active = _make_policy(policy, profile, hide, allow, ignore_case, kinds)
    detectors = _make_registry(active, registry, ner, engine, language, model, size)
    started = time.monotonic()
    result = Redactor(policy=active, registry=detectors).redact(
        text, extra_terms=tuple(hide or ()) or None, word_boundary=word_boundary
    )
    elapsed = time.monotonic() - started

    report = describe_outcome(
        result,
        diagnose(active, detectors),
        suggest_terms(text, result) if suggest else (),
    )
    logger.info(
        "encoded %d characters in %.3fs: %d value(s) across %d kind(s), level=%s",
        len(text),
        elapsed,
        result.stats.entries,
        len(result.stats.by_kind),
        report["level"],
    )
    return EncodedPrompt(
        text=result.text,
        handle=Handle(vault=result.vault, policy=active, entries=result.entries),
        result=result,
        report=report,
    )


def decode(reply: str, handle: Handle, *, strict: bool = False) -> str:
    """
    Put the original values back into a model's reply.

    Parameters
    ----------
    reply : str
        Text containing placeholders.
    handle : Handle
        The handle :func:`encode` returned.
    strict : bool, default=False
        Raise when the reply contains a placeholder the handle never issued.

    Returns
    -------
    str
        The reply with values restored.

    Raises
    ------
    TypeError
        If ``handle`` is not a :class:`Handle`.
    PolicyError
        If the handle has been cleared.
    RestorationError
        Under ``strict``, when a placeholder is unknown.

    Notes
    -----
    **User notes.** ``strict`` is off by default because a model can invent a
    placeholder that was never issued, and failing the whole reply over one
    hallucinated token is rarely what you want. Use
    :meth:`Session.decode_report` when you need to know.

    Examples
    --------
    >>> safe, handle = encode("Call +1 555 010 4477")
    >>> decode("Ring [PHONE-1] tomorrow", handle)
    'Ring +1 555 010 4477 tomorrow'
    """
    if not isinstance(handle, Handle):
        raise TypeError(f"handle must be a Handle, got {type(handle).__name__!r}")
    outcome = restore(reply, handle.vault, policy=handle.policy, strict=strict)
    if outcome.unknown:
        logger.warning(
            "reply contained %d placeholder(s) this handle never issued: %s",
            len(outcome.unknown),
            ", ".join(outcome.unknown),
        )
    return outcome.text


class Session:
    """
    A conversation: consistent placeholders, and a vault with a lifetime.

    Parameters
    ----------
    **options
        Any keyword :func:`encode` accepts. They apply to every turn.

    Notes
    -----
    **User notes.** Use it as a context manager so the values are cleared when
    the block ends, however it ends::

        with session(ner=True) as chat:
            reply = chat.roundtrip("Summarise this for ada@example.com", send=model)

    Across turns the same value keeps the same placeholder, so a model reading
    the thread sees one consistent participant rather than a new stranger each
    time.

    **Developer notes.** Continuity is carried by passing the previous turn's
    entries into the next :meth:`Redactor.redact` call as a ``seed``. The
    redactor itself stays stateless — that is invariant I4 and the upstream
    defect ``CP-003`` — and the session is the object that deliberately owns
    the link between two documents.

    A session is not thread-safe: it is a conversation, and a conversation has
    one voice. Use one session per thread.

    Examples
    --------
    >>> with session() as chat:
    ...     first = chat.encode("Mail ada@example.com")
    ...     second = chat.encode("Ping ada@example.com again")
    >>> first, second
    ('Mail [EMAIL-1]', 'Ping [EMAIL-1] again')
    """

    __slots__ = (
        "__weakref__",
        "_entries",
        "_lock",
        "_options",
        "_policy",
        "_registry",
        "_scrubber",
        "_turns",
        "_vault",
    )

    def __init__(self, **options: Any) -> None:
        policy = _make_policy(
            options.get("policy"),
            options.get("profile"),
            options.get("hide"),
            options.get("allow"),
            options.get("ignore_case"),
            options.get("kinds"),
        )
        self._policy = policy
        self._registry = _make_registry(
            policy,
            options.get("registry"),
            options.get("ner", False),
            options.get("engine", DEFAULT_ENGINE),
            options.get("language", "en"),
            options.get("model"),
            options.get("size", "sm"),
        )
        self._options = options
        self._lock = threading.RLock()
        self._entries: tuple[Entry, ...] = ()
        self._vault = Vault(grammar_fingerprint=policy.tag_style.fingerprint)
        # Keep this conversation's values out of every log record while the
        # session holds them (CP-059); released by clear() or collection.
        self._scrubber = VaultScrubber(owner=self)
        self._turns = 0

    # -- the conversation -------------------------------------------------

    @_synchronized
    def encode(self, text: str) -> str:
        """
        Redact ``text``, keeping placeholders consistent with earlier turns.

        Parameters
        ----------
        text : str
            This turn's prompt.

        Returns
        -------
        str
            Safe to send.

        Raises
        ------
        TypeError
            If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}")
        result = Redactor(policy=self._policy, registry=self._registry).redact(
            text,
            extra_terms=tuple(self._options.get("hide") or ()) or None,
            word_boundary=bool(self._options.get("word_boundary", False)),
            seed=self._entries,
        )
        merged = {entry.label: entry for entry in self._entries}
        for entry in result.entries:
            merged.setdefault(entry.label, entry)
        self._entries = tuple(merged.values())
        for label, value in result.vault.items():
            self._vault.add(label, value)
        self._scrubber.add(value for value in result.vault.values())
        self._turns += 1
        logger.debug(
            "session turn %d: %d new value(s), %d carried",
            self._turns,
            result.stats.entries,
            len(self._entries),
        )
        return result.text

    @_synchronized
    def decode(self, reply: str, *, strict: bool = False) -> str:
        """Restore values in ``reply`` from everything seen this session."""
        return decode(reply, self.handle, strict=strict)

    @_synchronized
    def decode_report(self, reply: str):
        """
        Restore ``reply`` and report what was and was not resolved.

        Returns
        -------
        RestorationResult
            Carries ``restored``, ``unknown`` and ``unused``.
        """
        return restore(reply, self._vault, policy=self._policy)

    def roundtrip(self, text: str, send: Sender, *, strict: bool = False) -> str:
        """
        Encode, hand to ``send``, decode the answer.

        Parameters
        ----------
        text : str
            This turn's prompt.
        send : callable
            Takes the safe text, returns the model's reply. Any provider.
        strict : bool, default=False
            Passed to :meth:`decode`.

        Returns
        -------
        str
            The reply with your values restored.

        Raises
        ------
        CleanPromptError
            If ``send`` does not return a string. Anything ``send`` itself
            raises propagates unchanged: it is the caller's client and the
            caller's error.

        Examples
        --------
        >>> with session() as chat:
        ...     chat.roundtrip(
        ...         "Mail ada@example.com",
        ...         send=lambda safe: "Sent to " + safe.split()[-1],
        ...     )
        'Sent to ada@example.com'
        """
        safe = self.encode(text)
        reply = send(safe)
        if not isinstance(reply, str):
            raise CleanPromptError(
                f"send must return str, got {type(reply).__name__!r}. The callable receives the "
                "redacted prompt and must return the model's reply as "
                "text."
            )
        return self.decode(reply, strict=strict)

    # -- state ------------------------------------------------------------

    @property
    def handle(self) -> Handle:
        """Handle: everything seen this session."""
        return Handle(vault=self._vault, policy=self._policy, entries=self._entries)

    @property
    def policy(self) -> RedactionPolicy:
        """RedactionPolicy: the policy every turn uses."""
        return self._policy

    @property
    def turns(self) -> int:
        """int: How many prompts have been encoded."""
        return self._turns

    def report(self) -> dict[str, Any]:
        """
        Return what this session has done and what it can and cannot detect.

        Returns
        -------
        dict
            ``turns``, ``entries``, ``labels`` (placeholder to category, in
            issue order) and ``detection``, which is the full
            :func:`~scikitplot.cleanprompt.diagnose` report.

        Notes
        -----
        **User notes.** Safe to print, log or attach to a ticket: it names
        placeholders and categories, never values.

        **Developer notes.** This returned the diagnosis alone, which is what
        the session *can* detect and says nothing about what it *did*. On a
        session object ``report()`` reads as "tell me about this session", and
        a caller who wanted the turn count had to reach past the method to
        ``.turns``. The diagnosis is still here, under ``detection``; the
        session's own state is now the top level.

        ``labels`` maps each placeholder to its category and deliberately
        stops there. The placeholder is already in the text that was sent, so
        it discloses nothing, while the value it stands for is exactly what
        this submodule exists to keep out of anything that gets copied
        elsewhere.
        """
        return {
            "turns": self._turns,
            "entries": len(self._entries),
            "labels": {entry.label: entry.kind for entry in self._entries},
            "detection": diagnose(self._policy, self._registry).as_dict(),
        }

    @_synchronized
    def clear(self) -> None:
        """Drop every value this session holds."""
        self._vault.clear()
        self._entries = ()
        self._scrubber.close()

    # -- lifetime ---------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.clear()

    def __repr__(self) -> str:
        return f"Session(turns={self._turns}, values={len(self._entries)}, policy={self._policy.fingerprint})"


def session(**options: Any) -> Session:
    """
    Open a :class:`Session`.

    Parameters
    ----------
    **options
        Any keyword :func:`encode` accepts.

    Returns
    -------
    Session
        Usable directly or as a context manager.

    Examples
    --------
    >>> with session(profile="minimal") as chat:
    ...     chat.encode("Mail ada@example.com")
    'Mail [EMAIL-1]'
    """
    return Session(**options)
