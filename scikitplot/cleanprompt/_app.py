"""
Optional local web interface (tier: ``web``).

Notes
-----
**User notes.** Install the tier and run the app::

    pip install "flask>=2.2,<4"
    export CLEANPROMPT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"
    python -c "from scikitplot.cleanprompt import create_app; create_app().run()"

The app binds to ``127.0.0.1`` by default. It is a local tool, not a service:
anything you type into it is personal data, and it has no authentication.

**Developer notes — what changed and why.**

*No module-level state.* :func:`create_app` is a factory. Upstream built the
Flask app, a ``PromptCleaner`` and a Fernet key at import time. The shared
cleaner meant one visitor's detections influenced another's numbering, and the
two keys, being generated per process, made a multi-worker deployment
invalidate its own sessions at random.

*The vault never reaches the browser.* Upstream encrypted the mapping and stored
it in the session cookie, so every request carried the user's personal data to
and from the client, subject to a four-kilobyte ceiling that fails silently when
crossed. Here the cookie carries an opaque token and the vault stays in
:class:`SessionStore`, server side, with a time-to-live and a bound on how many
sessions are retained.

*No ``eval``.* Upstream serialized the mapping with ``str(dict)`` and read it
back with :func:`eval`, which turns any compromise of the cookie or the key into
arbitrary code execution. Nothing here evaluates input.

*The secret key is configuration.* It comes from the caller or from
``CLEANPROMPT_SECRET_KEY``. When neither is set the app refuses to start unless
``ephemeral_secret_key=True`` is passed explicitly, which is honest about what a
per-process random key means: sessions do not survive a restart and do not work
across workers.

*Cross-site request forgery.* Every form carries a per-session token, compared
with :func:`hmac.compare_digest`.

This module imports :mod:`flask` inside :func:`create_app`, after the tier
check, so importing ``scikitplot.cleanprompt`` never imports Flask.

See Also
--------
scikitplot.cleanprompt._engine : The redaction pipeline this exposes.
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from typing import Any

from ._capabilities import require
from ._engine import Redactor, restore
from ._exceptions import CleanPromptError, PolicyError
from ._policy import DEFAULT_POLICY, RedactionPolicy
from ._types import RedactionResult
from ._vault import Vault

__all__ = [
    "SessionStore",
    "create_app",
]

#: Default time-to-live for a stored vault, in seconds.
DEFAULT_TTL_SECONDS = 3600

#: Default cap on retained sessions.
DEFAULT_MAX_SESSIONS = 256


class SessionStore:
    """
    Bounded, expiring, server-side store of vaults.

    Parameters
    ----------
    ttl_seconds : int, default=3600
        How long a vault is retained after its last use.
    max_sessions : int, default=256
        Most sessions retained. When full, the least recently used session is
        evicted and its vault cleared.

    Raises
    ------
    PolicyError
        If a bound is not a positive integer.

    Notes
    -----
    **Developer notes.** Both bounds exist because this store holds personal
    data indefinitely otherwise: a long-running local app would accumulate every
    vault every visitor ever created. Eviction calls
    :meth:`~scikitplot.cleanprompt._vault.Vault.clear`, so an evicted vault stops
    referencing its secrets rather than merely becoming unreachable.

    A :class:`threading.Lock` guards the mapping because a Flask development
    server is threaded by default. The lock is held only for dictionary
    operations, never across a redaction.

    Tokens come from :func:`secrets.token_urlsafe`, so they are unguessable;
    a session identifier is the only thing standing between one browser tab and
    another's vault.

    Examples
    --------
    >>> store = SessionStore(ttl_seconds=60, max_sessions=4)
    >>> token = store.put(Vault({"[EMAIL-1]": "a@b.co"}))
    >>> store.get(token)["[EMAIL-1]"]
    'a@b.co'
    """

    __slots__ = ("_entries", "_lock", "_max", "_ttl")

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        for name, value in (
            ("ttl_seconds", ttl_seconds),
            ("max_sessions", max_sessions),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise PolicyError(
                    f"SessionStore.{name} must be a positive int, got {value!r}"
                )
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, Vault]] = {}

    def put(self, vault: Vault, token: str | None = None) -> str:
        """
        Store ``vault`` and return its token.

        Parameters
        ----------
        vault : Vault
            The vault to retain.
        token : str, optional
            Reuse an existing token, replacing its vault. A replaced vault is
            cleared.

        Returns
        -------
        str
            The token identifying this vault.
        """
        key = token or secrets.token_urlsafe(32)
        with self._lock:
            self._expire_locked()
            previous = self._entries.get(key)
            if previous is not None:
                previous[1].clear()
            self._entries[key] = (time.monotonic(), vault)
            while len(self._entries) > self._max:
                oldest = min(self._entries, key=lambda k: self._entries[k][0])
                self._entries.pop(oldest)[1].clear()
        return key

    def get(self, token: str | None) -> Vault | None:
        """
        Return the vault for ``token``, refreshing its expiry.

        Parameters
        ----------
        token : str or None
            The token from the session cookie.

        Returns
        -------
        Vault or None
            The vault, or ``None`` when the token is unknown or expired.
        """
        if not token:
            return None
        with self._lock:
            self._expire_locked()
            found = self._entries.get(token)
            if found is None:
                return None
            self._entries[token] = (time.monotonic(), found[1])
            return found[1]

    def drop(self, token: str | None) -> None:
        """Remove and clear the vault for ``token``, if present."""
        if not token:
            return
        with self._lock:
            found = self._entries.pop(token, None)
        if found is not None:
            found[1].clear()

    def _expire_locked(self) -> None:
        """Evict expired sessions. The caller must hold the lock."""
        deadline = time.monotonic() - self._ttl
        stale = [key for key, (seen, _) in self._entries.items() if seen < deadline]
        for key in stale:
            self._entries.pop(key)[1].clear()

    def __len__(self) -> int:
        with self._lock:
            self._expire_locked()
            return len(self._entries)


def _require_csrf(session: Any, submitted: str | None) -> None:
    """
    Compare a submitted CSRF token with the session's, in constant time.

    Raises
    ------
    CleanPromptError
        If the token is missing or does not match.

    Notes
    -----
    **Developer notes.** :func:`hmac.compare_digest` rather than ``==`` so the
    comparison does not leak the matching prefix length through timing.
    """
    expected = session.get("csrf_token")
    if (
        not expected
        or not submitted
        or not hmac.compare_digest(str(expected), str(submitted))
    ):
        raise CleanPromptError("CSRF token missing or invalid; reload the page")


def create_app(  # ruff: ignore[too-many-positional-arguments]
    policy: RedactionPolicy | None = None,
    secret_key: str | None = None,
    ephemeral_secret_key: bool = False,
    store: SessionStore | None = None,
    enable_ner: bool = False,
    ner_model: str | None = None,
    ner_engine: str = "auto",
    language: str = "en",
    model_size: str = "sm",
    hide_terms: tuple[str, ...] = (),
    word_boundary: bool = False,
    session_cookie_secure: bool = False,
    config: dict[str, Any] | None = None,
) -> Any:
    """
    Build the Flask application.

    Parameters
    ----------
    policy : RedactionPolicy, optional
        Redaction configuration. Defaults to
        :data:`~scikitplot.cleanprompt._policy.DEFAULT_POLICY`.
    secret_key : str, optional
        Flask session signing key. Falls back to ``CLEANPROMPT_SECRET_KEY``.
    ephemeral_secret_key : bool, default=False
        Permit a per-process random key when none is configured. Sessions then
        do not survive a restart and do not work across workers.
    store : SessionStore, optional
        Vault store. A new bounded store is created when omitted.
    enable_ner : bool, default=False
        Also run named-entity detection. Requires the ``ner`` tier.
    ner_model : str, optional
        Explicit spaCy model, overriding ``language`` and ``model_size``.
    ner_engine : str, default='auto'
        Entity engine: ``auto``, ``spacy``, ``nltk``, ``both`` or ``none``.
    language : str, default='en'
        Language code for entity detection.
    model_size : str, default='sm'
        Preferred spaCy model size.
    hide_terms : tuple of str, default=()
        Exact strings hidden on every request, in addition to whatever the
        visitor types into the form.
    word_boundary : bool, default=False
        Whether ``hide_terms`` match only at word boundaries.
    session_cookie_secure : bool, default=False
        Set the ``Secure`` cookie flag. Defaults to ``False`` because the app
        is served over plain HTTP on loopback; set it when placing the app
        behind TLS.
    config : dict, optional
        Extra Flask configuration applied last.

    Returns
    -------
    flask.Flask
        The configured application.

    Raises
    ------
    CapabilityError
        If the ``web`` tier — or, with ``enable_ner``, the ``ner`` tier — is
        unavailable.
    PolicyError
        If no secret key is configured and ``ephemeral_secret_key`` is not set.

    Notes
    -----
    **Developer notes.** Every route is registered on a local ``app``; there is
    no module-level application object, so importing this module has no side
    effects and a test can build as many isolated apps as it needs.
    """
    require("web")  # raises CapabilityError before flask is imported

    from flask import (  # noqa: PLC0415 - deliberately deferred to call time
        Flask,
        redirect,
        render_template,
        request,
        session,
        url_for,
    )

    active_policy = policy if policy is not None else DEFAULT_POLICY
    key = secret_key or os.environ.get("CLEANPROMPT_SECRET_KEY")
    if not key:
        if not ephemeral_secret_key:
            raise PolicyError(
                "no session secret key is configured. Set CLEANPROMPT_SECRET_KEY "
                "(for example to the output of "
                '`python -c "import secrets;print(secrets.token_hex(32))"`), '
                "pass secret_key=..., or pass ephemeral_secret_key=True to "
                "accept a per-process key that does not survive a restart and "
                "does not work across workers."
            )
        key = secrets.token_hex(32)

    sessions = store if store is not None else SessionStore()

    from ._detectors import default_registry  # ruff: ignore[import-outside-top-level]

    registry = default_registry(kinds=active_policy.kinds)
    if enable_ner:
        from ._ner import spacy_detector  # ruff: ignore[import-outside-top-level]

        registry.add(spacy_detector(model=ner_model))
    redactor = Redactor(policy=active_policy, registry=registry)

    from ._diagnostics import (  # ruff: ignore[import-outside-top-level]
        describe_outcome,
        diagnose,
        suggest_terms,
    )

    # Computed once: the answer depends on the installation and the policy, not
    # on the request, and probing per request would put a filesystem metadata
    # lookup on every page load for an answer that cannot have changed.
    standing_diagnosis = diagnose(active_policy, registry)

    from ._engines import describe_engines  # ruff: ignore[import-outside-top-level]
    from ._languages import language_report  # ruff: ignore[import-outside-top-level]

    standing_engines = describe_engines(language, ner_engine)
    standing_language = language_report(language, model_size)

    app = Flask(__name__, template_folder="_templates", static_folder="_static")
    app.secret_key = key
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(session_cookie_secure),
        MAX_CONTENT_LENGTH=active_policy.limits.max_input_chars * 4,
        CLEANPROMPT_POLICY=active_policy,
        CLEANPROMPT_STORE=sessions,
    )
    if config:
        app.config.update(config)

    def _csrf_token() -> str:
        """Return this session's CSRF token, creating it on first use."""
        token = session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    def _render(**context: Any) -> Any:
        """
        Render the single page.

        Notes
        -----
        **Developer notes.** ``entries`` carries labels, kinds and counts —
        never the original values. The page shows the user what *categories*
        were removed; echoing the secrets back into the HTML would put them in
        the browser's cache, history and any screenshot.

        The visitor's *own* text does appear twice — in the textarea they typed
        it into, and in a hidden field that carries it across the restore step
        so it is not lost. That is their input echoed back to the same local
        process, not a disclosure: the invariant is that no removed value
        appears in the redacted output, the mapping table or the suggestion
        list, which ``test__app.TestRoutes`` asserts region by region.

        ``diagnosis`` is passed on every render, including the first GET. That
        is the fix for the defect that prompted this rewrite: a visitor must be
        able to see that name detection is switched off *before* deciding the
        output is safe to send, not only after pasting something and getting it
        back unchanged.
        """
        payload: dict[str, Any] = {
            "csrf_token": _csrf_token(),
            "original_text": "",
            "processed_text": "",
            "llm_response": "",
            "reverted_text": "",
            "entries": [],
            "suggestions": [],
            "additional_words": "",
            "error": "",
            "outcome": None,
            "diagnosis": standing_diagnosis,
            "engines": standing_engines,
            "language": standing_language,
            "blind_spots": [
                spot
                for spot in standing_diagnosis.blind_spots
                if spot.severity == "high"
            ],
        }
        payload.update(context)
        return render_template("index.html", **payload)

    def _entry_rows(result: RedactionResult) -> list[dict[str, Any]]:
        """Describe a result for the template, without its secrets."""
        return [
            {
                "label": entry.label,
                "kind": entry.kind,
                "count": entry.count,
                "detector": entry.detector,
                "confidence": entry.confidence,
            }
            for entry in result.entries
        ]

    def _terms_from(raw: str) -> list[str]:
        """Split the visitor's comma-separated term box."""
        return [part.strip() for part in raw.split(",") if part.strip()]

    @app.route("/", methods=["GET", "POST"])
    def index() -> Any:
        """Redact a submitted text, or restore a reply."""
        if request.method == "GET":
            return _render()

        try:
            _require_csrf(session, request.form.get("csrf_token"))

            if "process_text" in request.form:
                original = request.form.get("text", "")
                raw_terms = request.form.get("additional_words", "")
                terms = list(hide_terms) + _terms_from(raw_terms)
                result = redactor.redact(
                    original,
                    extra_terms=terms or None,
                    word_boundary=word_boundary,
                )
                token = sessions.put(result.vault, token=session.get("vault_token"))
                session["vault_token"] = token
                suggestions = suggest_terms(original, result)
                return _render(
                    original_text=original,
                    processed_text=result.text,
                    entries=_entry_rows(result),
                    suggestions=[item.as_dict() for item in suggestions],
                    additional_words=raw_terms,
                    outcome=describe_outcome(result, standing_diagnosis, suggestions),
                )

            if "revert_text" in request.form:
                reply = request.form.get("llm_response", "")
                vault = sessions.get(session.get("vault_token"))
                if vault is None:
                    return _render(
                        original_text=request.form.get("original_text", ""),
                        processed_text=request.form.get("processed_text", ""),
                        llm_response=reply,
                        error=(
                            "This session has no stored vault; it may have "
                            "expired. Redact the text again."
                        ),
                    )
                outcome = restore(reply, vault, policy=active_policy)
                note = f"Restored {len(outcome.restored)} placeholder(s)."
                if outcome.unknown:
                    note += " {} placeholder(s) were not in the vault: {}. The model may have invented them.".format(
                        len(outcome.unknown), ", ".join(outcome.unknown)
                    )
                return _render(
                    original_text=request.form.get("original_text", ""),
                    processed_text=request.form.get("processed_text", ""),
                    llm_response=reply,
                    reverted_text=outcome.text,
                    outcome={
                        "level": "warning" if outcome.unknown else "ok",
                        "headline": note,
                        "detail": "",
                        "actions": [],
                    },
                )
        except CleanPromptError as exc:
            return _render(error=str(exc))

        return _render(error="Unrecognised form submission.")

    @app.route("/api/doctor")
    def api_doctor() -> Any:
        """
        Return the diagnosis as JSON.

        Notes
        -----
        **Developer notes.** The same payload the ``doctor`` subcommand emits,
        so a deployment can be checked over HTTP without a shell in the
        container. It reports configuration and capability only — never a
        vault, never a text, never anything a visitor pasted.
        """
        return dict(
            standing_diagnosis.as_dict(),
            entity_engines=standing_engines,
            languages=standing_language,
        )

    @app.route("/reset", methods=["POST"])
    def reset() -> Any:
        """
        Discard this session's vault.

        Notes
        -----
        **Developer notes.** ``POST`` only, and CSRF-checked. Upstream exposed
        reset as a ``GET`` link, which any page — or a prefetching browser —
        could trigger, and which discarded the mapping the user still needed.
        """
        try:
            _require_csrf(session, request.form.get("csrf_token"))
        except CleanPromptError as exc:
            return _render(error=str(exc))
        sessions.drop(session.pop("vault_token", None))
        return redirect(url_for("index"))

    @app.route("/healthz")
    def healthz() -> Any:
        """Report liveness and the number of retained sessions."""
        return {"status": "ok", "sessions": len(sessions)}

    return app
