# scikitplot/corpus/_metrics/_llm.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One call shape for the language models that judge answers.

:func:`make_llm` returns an :class:`LLM` whose :meth:`LLM.complete` sends one
prompt (and an optional system instruction) and returns a :class:`Completion`:
the text, the provider and model, the token counts the provider reported, the
finish reason and the latency. Four providers share it:

``"openai"``
    ``client.chat.completions.create(model=..., messages=[...], **options)``
    of the ``openai`` package, or any server that speaks the same API
    (pass ``client=openai.OpenAI(base_url=...)``). ``max_tokens`` is sent as
    ``max_completion_tokens``.
``"anthropic"``
    ``client.messages.create(model=..., max_tokens=..., system=...,
    messages=[...], **options)`` of the ``anthropic`` package. The Messages
    API requires ``max_tokens``.
``"gemini"``
    ``client.models.generate_content(model=..., contents=..., config={...})``
    of the ``google-genai`` package; ``system`` becomes
    ``system_instruction``, ``max_tokens`` becomes ``max_output_tokens`` and
    ``options`` are further ``GenerateContentConfig`` fields.
a callable
    ``fn(prompt, system) -> str`` (or a :class:`Completion`): a local model, a
    proxy, a test double.

Notes
-----
**User note.** Nothing is sent that the caller did not ask for: no
temperature, seed or response format is added, because models differ in
which they accept (``messages.create`` in ``anthropic`` 1.13 has no
``temperature`` parameter). Pass them in
``options`` where your model takes them. A completion that stopped for any
reason other than a normal end (``finish_reason`` ``"length"``,
``stop_reason`` ``"max_tokens"``, a content filter) raises
:class:`IncompleteCompletionError`, because a cut-off verdict is not a
verdict.

The SDK is imported when the :class:`LLM` is made, never when
:mod:`scikitplot.corpus` is imported, and its default client reads the API
key from the environment variable the SDK documents (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``). A key is
never stored in :meth:`LLM.identity`, a cache key or a report.

**Developer note.** ``cache`` is any mutable mapping (a ``dict``, a
``shelve`` file). The key is the SHA-256 of the canonical request (provider,
model, ``max_tokens``, ``options``, the client's ``base_url``, system and
prompt), so a cached answer is only reused for an identical request to the
same endpoint. Access is serialised by a lock; the provider call is not, so
several threads can wait on the network at once. A mapping bound to one
thread (``shelve`` on Python 3.13+, which uses SQLite) cannot be used with
``n_jobs > 1``; its failure raises :class:`CacheError` instead of being
mistaken for a model failure.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import numbers
import threading
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .. import _serialize

__all__ = [
    "LLM",
    "CacheError",
    "Completion",
    "IncompleteCompletionError",
    "make_llm",
]

#: Providers :func:`make_llm` accepts by name.
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini")

#: Name under which a callable provider is reported.
CALLABLE_PROVIDER = "callable"

#: Finish reasons that mean "the model finished its answer", per provider.
#: Any other reported reason raises :class:`IncompleteCompletionError`; a
#: provider that reports none (``None``) is accepted.
_PROVIDER_FINISH: Mapping[str, frozenset] = {
    "openai": frozenset({"stop"}),
    "anthropic": frozenset({"end_turn", "stop_sequence"}),
    "gemini": frozenset({"STOP"}),
}
#: A callable may report any provider's normal reason (or none).
NORMAL_FINISH: Mapping[str, frozenset] = {
    **_PROVIDER_FINISH,
    CALLABLE_PROVIDER: frozenset().union(*_PROVIDER_FINISH.values()),
}

#: Options whose values are transport details that may carry credentials
#: (gateway or Azure ``api-key`` headers): their values are redacted in
#: :meth:`LLM.identity` and ``repr``, and only hashed into cache keys.
REDACTED_OPTIONS: frozenset = frozenset({"extra_headers", "extra_query"})
_REDACTED = "<redacted>"

#: Request arguments each adapter sets itself; ``options`` may not override them.
_RESERVED_OPTIONS: Mapping[str, frozenset] = {
    "openai": frozenset({"model", "messages", "max_completion_tokens", "stream"}),
    "anthropic": frozenset({"model", "messages", "system", "max_tokens", "stream"}),
    "gemini": frozenset({"system_instruction", "max_output_tokens"}),
}

#: Usage counters copied into :attr:`Completion.usage`, per provider.
_USAGE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "openai": ("prompt_tokens", "completion_tokens", "total_tokens"),
    "anthropic": (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ),
    "gemini": (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
    ),
}

#: (input, output) counter names behind :attr:`Completion.input_tokens` and
#: :attr:`Completion.output_tokens`.
_TOKEN_FIELDS: Mapping[str, tuple[str, str]] = {
    "openai": ("prompt_tokens", "completion_tokens"),
    "anthropic": ("input_tokens", "output_tokens"),
    "gemini": ("prompt_token_count", "candidates_token_count"),
}

_SDK_HINTS: Mapping[str, str] = {
    "openai": "pip install openai",
    "anthropic": "pip install anthropic",
    "gemini": "pip install google-genai",
}

_CACHE_KEY_VERSION = 2


class CacheError(RuntimeError):
    """
    The response cache failed (a configuration problem, not a model failure).

    Notes
    -----
    A ``shelve`` file opened on Python 3.13 or later uses SQLite, whose
    connection works only in the thread that opened it; with ``n_jobs > 1``
    use a ``dict`` or another mapping that any thread may use.
    """


class IncompleteCompletionError(RuntimeError):
    """
    A completion that did not end normally (cut off, filtered, refused).

    Parameters
    ----------
    completion : Completion
        What the provider returned, tokens included.
    """

    def __init__(self, completion: Completion) -> None:
        self.completion = completion
        super().__init__(
            f"{completion.provider} model {completion.model!r} stopped with "
            f"finish reason {completion.finish_reason!r}, not a normal end; "
            "raise max_tokens if it was cut off."
        )


@dataclass(frozen=True)
class Completion:
    """
    One model response.

    Parameters
    ----------
    text : str
        The response text.
    provider : str
        ``"openai"``, ``"anthropic"``, ``"gemini"`` or ``"callable"``.
    model : str
        Model name as requested.
    finish_reason : str or None, default=None
        The provider's reason for stopping (OpenAI ``finish_reason``,
        Anthropic ``stop_reason``, Gemini ``finish_reason`` name).
    input_tokens, output_tokens : int or None, default=None
        The provider's own counts: OpenAI ``prompt_tokens`` and
        ``completion_tokens`` (which include reasoning tokens), Anthropic
        ``input_tokens`` and ``output_tokens``, Gemini ``prompt_token_count``
        and ``candidates_token_count`` (thinking tokens are in ``usage``).
    latency_s : float, default=0.0
        Seconds spent in the provider call; ``0.0`` for a cache hit.
    cached : bool, default=False
        Whether the response came from the cache.
    usage : mapping, default={}
        Every integer usage counter the provider reported, by its own name.
    """

    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0
    cached: bool = False
    usage: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "cached": self.cached,
            "usage": dict(self.usage),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Completion:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            text=data["text"],
            provider=data["provider"],
            model=data["model"],
            finish_reason=data.get("finish_reason"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            latency_s=float(data.get("latency_s", 0.0)),
            cached=bool(data.get("cached", False)),
            usage=dict(data.get("usage", {})),
        )


def _count(value: Any) -> int | None:
    """Return an integer counter, or ``None`` for anything else (absent, mock, bool)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _usage(provider: str, usage: Any) -> tuple[int | None, int | None, dict[str, int]]:
    counters: dict[str, int] = {}
    if usage is not None:
        for name in _USAGE_FIELDS[provider]:
            value = _count(getattr(usage, name, None))
            if value is not None:
                counters[name] = value
    first, second = _TOKEN_FIELDS[provider]
    return counters.get(first), counters.get(second), counters


def _finish_name(value: Any) -> str | None:
    """Return a finish reason as a string (Gemini reports an enum member)."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def _check_text(text: Any, provider: str) -> str:
    if not isinstance(text, str):
        raise TypeError(
            f"The {provider} response held no text (got {type(text).__name__!r}); "
            "the model may have refused or returned only non-text content."
        )
    return text


class LLM:
    """
    A language model behind one call shape; made by :func:`make_llm`.

    Attributes
    ----------
    provider : str
        ``"openai"``, ``"anthropic"``, ``"gemini"`` or ``"callable"``.
    model : str
        Model name.
    max_tokens : int or None
        Output-token limit sent with every request, if any.

    See Also
    --------
    make_llm : The constructor to use.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        client: Any,
        max_tokens: int | None,
        options: Mapping[str, Any],
        cache: MutableMapping | None,
        clock: Callable[[], float],
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._options = dict(options)
        self._cache = cache
        self._lock = threading.Lock()
        self._clock = clock

    @property
    def options(self) -> dict[str, Any]:
        """A copy of the extra request options."""
        return dict(self._options)

    def identity(self) -> dict[str, Any]:
        """
        Return what identifies this judge model in a report.

        Returns
        -------
        dict
            ``provider``, ``model``, ``max_tokens`` (only when set, so the
            identity holds no ``None`` and survives every report format) and
            ``options``, with the values of :data:`REDACTED_OPTIONS`
            (``extra_headers``, ``extra_query``) replaced by
            ``"<redacted>"``. The client, and so its API key, is never
            included.
        """
        identity: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.max_tokens is not None:
            identity["max_tokens"] = self.max_tokens
        identity["options"] = _serialize.to_builtin(_redacted(self._options))
        return identity

    def __repr__(self) -> str:
        parts = [f"provider={self.provider!r}", f"model={self.model!r}"]
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        if self._options:
            parts.append(f"options={_redacted(self._options)!r}")
        if self._cache is not None:
            parts.append("cache=...")
        return f"LLM({', '.join(parts)})"

    def cache_key(self, prompt: str, system: str | None = None) -> str:
        """
        Return the cache key of one request (hex SHA-256).

        Parameters
        ----------
        prompt : str
            User prompt.
        system : str or None, default=None
            System instruction.

        Returns
        -------
        str

        Notes
        -----
        The key covers the provider, model, ``max_tokens``, every option
        (unredacted, so two gateways' headers never share an entry) and the
        client's ``base_url`` where the client exposes one (OpenAI and
        Anthropic clients do), so two endpoints serving the same model name
        never share an entry. Only the hash is stored.
        """
        base_url = getattr(self._client, "base_url", None)
        request = {
            "v": _CACHE_KEY_VERSION,
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "options": _serialize.to_builtin(self._options),
            "base_url": None if base_url is None else str(base_url),
            "system": system,
            "prompt": prompt,
        }
        canonical = json.dumps(
            request,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """
        Send one prompt and return the response.

        Parameters
        ----------
        prompt : str
            User prompt.
        system : str or None, default=None
            System instruction.

        Returns
        -------
        Completion

        Raises
        ------
        TypeError
            If ``prompt`` or ``system`` is not a string, or a normally ended
            response holds no text.
        IncompleteCompletionError
            If the model stopped for any reason but a normal end, whether or
            not it returned text (the completion keeps its token counts).
        CacheError
            If the cache mapping fails.
        Exception
            Whatever the provider's client raises (authentication, rate limit,
            network); it is not caught here.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"prompt must be a str, got {type(prompt).__name__!r}.",
            )
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"system must be a str or None, got {type(system).__name__!r}.",
            )
        key = None
        if self._cache is not None:
            key = self.cache_key(prompt, system)
            hit = self._cache_io(lambda: self._cache.get(key))
            if hit is not None:
                data = dict(hit)
                data.update(cached=True, latency_s=0.0)
                return Completion.from_dict(data)
        start = self._clock()
        completion = self._call(prompt, system)
        completion = replace(completion, latency_s=self._clock() - start)
        if (
            completion.finish_reason is not None
            and completion.finish_reason not in NORMAL_FINISH[self.provider]
        ):
            raise IncompleteCompletionError(
                completion
                if isinstance(completion.text, str)
                else replace(completion, text="")
            )
        _check_text(completion.text, self.provider)
        if key is not None:
            self._cache_io(lambda: self._cache.__setitem__(key, completion.to_dict()))
        return completion

    def _cache_io(self, operation: Callable[[], Any]) -> Any:
        """Run one cache operation under the lock; its failure is a :class:`CacheError`."""
        try:
            with self._lock:
                return operation()
        # any mapping's failure is a cache failure
        except Exception as exc:  # noqa: BLE001
            raise CacheError(
                f"The response cache failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _call(self, prompt: str, system: str | None) -> Completion:
        if self.provider == "openai":
            return self._openai(prompt, system)
        if self.provider == "anthropic":
            return self._anthropic(prompt, system)
        if self.provider == "gemini":
            return self._gemini(prompt, system)
        return self._callable(prompt, system)

    def _openai(self, prompt: str, system: str | None) -> Completion:
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_tokens
        kwargs.update(self._options)
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tokens_in, tokens_out, usage = _usage(
            "openai",
            getattr(response, "usage", None),
        )
        return Completion(
            text=choice.message.content,
            provider="openai",
            model=self.model,
            finish_reason=_finish_name(getattr(choice, "finish_reason", None)),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            usage=usage,
        )

    def _anthropic(self, prompt: str, system: str | None) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        kwargs.update(self._options)
        response = self._client.messages.create(**kwargs)
        texts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        tokens_in, tokens_out, usage = _usage(
            "anthropic",
            getattr(response, "usage", None),
        )
        return Completion(
            text="".join(texts) if texts else None,
            provider="anthropic",
            model=self.model,
            finish_reason=_finish_name(getattr(response, "stop_reason", None)),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            usage=usage,
        )

    def _gemini(self, prompt: str, system: str | None) -> Completion:
        config: dict[str, Any] = dict(self._options)
        if system is not None:
            config["system_instruction"] = system
        if self.max_tokens is not None:
            config["max_output_tokens"] = self.max_tokens
        kwargs: dict[str, Any] = {"model": self.model, "contents": prompt}
        if config:
            kwargs["config"] = config
        response = self._client.models.generate_content(**kwargs)
        candidates = getattr(response, "candidates", None) or ()
        finish = (
            _finish_name(getattr(candidates[0], "finish_reason", None))
            if candidates
            else None
        )
        tokens_in, tokens_out, usage = _usage(
            "gemini",
            getattr(response, "usage_metadata", None),
        )
        return Completion(
            text=response.text,
            provider="gemini",
            model=self.model,
            finish_reason=finish,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            usage=usage,
        )

    def _callable(self, prompt: str, system: str | None) -> Completion:
        out = self._client(prompt, system)
        if isinstance(out, Completion):
            return replace(out, provider=CALLABLE_PROVIDER, model=self.model)
        return Completion(text=out, provider=CALLABLE_PROVIDER, model=self.model)


def _redacted(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``options`` with the values inside :data:`REDACTED_OPTIONS` replaced."""
    out = dict(options)
    for name in REDACTED_OPTIONS & set(out):
        value = out[name]
        out[name] = (
            dict.fromkeys(value, _REDACTED) if isinstance(value, Mapping) else _REDACTED
        )
    return out


def _check_gemini_options(names: list[str]) -> None:
    """Refuse options that are not ``GenerateContentConfig`` fields (when google-genai is importable)."""
    try:
        from google.genai import types  # noqa: PLC0415
    # without the SDK (an injected client) the client decides
    except Exception:  # noqa: BLE001
        return
    fields = set(getattr(types.GenerateContentConfig, "model_fields", ()))
    unknown = [name for name in names if fields and name not in fields]
    if unknown:
        raise ValueError(
            f"options {unknown} are not GenerateContentConfig fields of the installed google-genai.",
        )


def _request_function(provider: str, client: Any) -> Callable[..., Any] | None:
    """Return the client's request function for a keyword-argument provider, else ``None``."""
    try:
        if provider == "openai":
            return client.chat.completions.create
        if provider == "anthropic":
            return client.messages.create
    except AttributeError:
        return None
    return None


def _check_accepted(provider: str, client: Any, names: list[str]) -> None:
    """
    Refuse, at construction, request arguments the client's function does not take.

    A function with ``**kwargs``, or one whose signature cannot be read, is
    trusted; an explicit signature without the argument is not, because every
    request would then fail. ``openai`` added ``max_completion_tokens`` in 1.45.
    """
    func = _request_function(provider, client)
    if func is None:
        return
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return
    unknown = [name for name in names if name not in parameters]
    if unknown:
        hint = (
            " openai < 1.45 has no max_completion_tokens: upgrade, or pass "
            "max_tokens=None and options={'max_tokens': n}."
            if "max_completion_tokens" in unknown
            else ""
        )
        raise ValueError(
            f"This {provider} client does not accept {unknown}.{hint}",
        )


def _default_client(provider: str) -> Any:
    """Construct the provider SDK's default client (reads the API key from the environment)."""
    try:
        if provider == "openai":
            import openai  # noqa: PLC0415

            return openai.OpenAI()
        if provider == "anthropic":
            import anthropic  # noqa: PLC0415

            return anthropic.Anthropic()
        from google import genai  # noqa: PLC0415

        return genai.Client()
    except ImportError as exc:
        raise ImportError(
            f"provider={provider!r} needs its SDK: {_SDK_HINTS[provider]} "
            "(or pass client=, or a callable)."
        ) from exc


def _check_common_args(
    max_tokens: Any, options: Any, cache: Any, clock: Any
) -> dict[str, Any]:
    """Validate the provider-independent arguments of :func:`make_llm`; return the options."""
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, numbers.Integral)
        or max_tokens < 1
    ):
        raise ValueError(
            f"max_tokens must be an int >= 1 or None, got {max_tokens!r}.",
        )
    opts = {} if options is None else options
    if not isinstance(opts, Mapping) or not all(isinstance(k, str) for k in opts):
        raise TypeError("options must be a mapping with str keys or None.")
    try:
        _serialize.to_builtin(dict(opts))
    except TypeError as exc:
        raise TypeError(
            f"options must be JSON-compatible (they are recorded): {exc}",
        ) from None
    if cache is not None and not isinstance(cache, MutableMapping):
        raise TypeError("cache must be a mutable mapping (dict, shelve) or None.")
    if clock is not None and not callable(clock):
        raise TypeError("clock must be callable or None.")
    return dict(opts)


def make_llm(
    provider: str | Callable[..., Any],
    model: str | None = None,
    *,
    client: Any = None,
    max_tokens: int | None = None,
    options: Mapping[str, Any] | None = None,
    cache: MutableMapping | None = None,
    clock: Callable[[], float] | None = None,
) -> LLM:
    """
    Make an :class:`LLM` for OpenAI, Anthropic, Gemini or a callable.

    Parameters
    ----------
    provider : {"openai", "anthropic", "gemini"} or callable
        A provider name, or ``fn(prompt, system) -> str`` (or ->
        :class:`Completion`).
    model : str or None, default=None
        Model name, required for a named provider (no default: model names
        change). For a callable it is the name reported (default: the
        callable's ``__name__``).
    client : object or None, default=None
        A client with the provider SDK's surface (``openai.OpenAI(...)``,
        ``anthropic.Anthropic(...)``, ``google.genai.Client(...)``, or a
        compatible object). ``None`` constructs the SDK's default client,
        which reads the API key from the environment.
    max_tokens : int or None, default=None
        Output-token limit for every request. Required for ``"anthropic"``.
    options : mapping or None, default=None
        Further request arguments, sent unchanged: keyword arguments of
        ``chat.completions.create`` (OpenAI) or ``messages.create``
        (Anthropic), fields of ``GenerateContentConfig`` (Gemini), for example
        ``{"temperature": 0}`` where the model accepts it. Must be
        JSON-compatible: they are part of the judge's recorded identity.
    cache : mutable mapping or None, default=None
        Response cache keyed by :meth:`LLM.cache_key`.
    clock : callable or None, default=None
        ``clock() -> float`` seconds for latency; ``time.perf_counter`` when
        ``None``.

    Returns
    -------
    LLM

    Raises
    ------
    ValueError
        For an unknown provider, a missing model or ``max_tokens``, an option
        that the adapter sets itself, or a request argument that the client's
        explicit signature does not take (``openai`` before 1.45 has no
        ``max_completion_tokens``).
    TypeError
        For arguments of the wrong type, or options that are not
        JSON-compatible.
    ImportError
        If ``client`` is ``None`` and the provider's SDK is not installed.

    See Also
    --------
    LLMJudge : Judges answers with an :class:`LLM`.

    Notes
    -----
    **User note.** For an OpenAI-compatible server (vLLM, a gateway), pass
    ``client=openai.OpenAI(base_url=..., api_key=...)``. If it does not know
    ``max_completion_tokens``, leave ``max_tokens=None`` and pass
    ``options={"max_tokens": n}``.

    Examples
    --------
    >>> llm = make_llm(lambda prompt, system: '{"score": 5}', model="stub")
    >>> llm.complete("Rate this.").text
    '{"score": 5}'
    >>> llm.identity()
    {'provider': 'callable', 'model': 'stub', 'options': {}}
    """
    opts = _check_common_args(max_tokens, options, cache, clock)
    timer = time.perf_counter if clock is None else clock
    if callable(provider) and not isinstance(provider, str):
        name = (
            getattr(provider, "__name__", type(provider).__name__)
            if model is None
            else model
        )
        if not isinstance(name, str) or not name:
            raise TypeError("model must be a non-empty str.")
        if opts:
            raise ValueError(
                "options are for named providers; a callable receives (prompt, system) only.",
            )
        return LLM(
            CALLABLE_PROVIDER,
            name,
            client=provider,
            max_tokens=max_tokens,
            options={},
            cache=cache,
            clock=timer,
        )
    if provider not in PROVIDERS:
        raise ValueError(
            f"provider must be one of {PROVIDERS} or a callable, got {provider!r}.",
        )
    if not isinstance(model, str) or not model:
        raise ValueError(
            f"provider={provider!r} needs model= (a non-empty str), got {model!r}.",
        )
    if provider == "anthropic" and max_tokens is None:
        raise ValueError(
            "provider='anthropic' requires max_tokens (the Messages API has no default).",
        )
    clash = sorted(set(opts) & _RESERVED_OPTIONS[provider])
    if clash:
        raise ValueError(
            f"options may not set {clash}: the {provider} adapter sets them from its arguments.",
        )
    if provider == "openai" and max_tokens is not None and "max_tokens" in opts:
        raise ValueError(
            "Set the token limit once: max_tokens= or options={'max_tokens': ...}, not both.",
        )
    resolved = _default_client(provider) if client is None else client
    if provider == "gemini":
        _check_gemini_options(list(opts))
    sent = list(opts)
    if max_tokens is not None and provider != "gemini":
        sent.append("max_completion_tokens" if provider == "openai" else "max_tokens")
    _check_accepted(provider, resolved, sent)
    return LLM(
        provider,
        model,
        client=resolved,
        max_tokens=None if max_tokens is None else int(max_tokens),
        options=opts,
        cache=cache,
        clock=timer,
    )
