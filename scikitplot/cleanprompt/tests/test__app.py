"""
Tests for :mod:`scikitplot.cleanprompt._app`.

Notes
-----
**Developer notes.** :class:`SessionStore` is pure standard library and is tested
unconditionally. Everything that needs Flask is skipped when the ``web`` tier is
unavailable, and the evidence record marks that lane accordingly rather than
claiming it.
"""

from __future__ import annotations

import re

import pytest

from .. import PolicyError, Vault
from .. import _capabilities as caps
from .._app import DEFAULT_MAX_SESSIONS, DEFAULT_TTL_SECONDS, SessionStore
from ._tiers import skip_reason

WEB_AVAILABLE = caps.probe("web").available
pytestmark_web = pytest.mark.skipif(
    not WEB_AVAILABLE, reason=skip_reason("web")
)


class TestSessionStore:
    """Server-side vault retention, bounded and expiring."""

    def test_put_and_get(self):
        store = SessionStore()
        token = store.put(Vault({"[A-1]": "x"}))
        assert store.get(token)["[A-1]"] == "x"

    def test_tokens_are_unguessable(self):
        store = SessionStore()
        tokens = {store.put(Vault()) for _ in range(50)}
        assert len(tokens) == 50
        assert all(len(token) >= 32 for token in tokens)

    def test_unknown_token_returns_none(self):
        assert SessionStore().get("nope") is None

    def test_empty_token_returns_none(self):
        assert SessionStore().get("") is None
        assert SessionStore().get(None) is None

    def test_reusing_a_token_clears_the_previous_vault(self):
        store = SessionStore()
        first = Vault({"[A-1]": "x"})
        token = store.put(first)
        store.put(Vault({"[A-1]": "y"}), token=token)
        assert first.closed is True
        assert store.get(token)["[A-1]"] == "y"

    def test_drop_clears_and_removes(self):
        store = SessionStore()
        vault = Vault({"[A-1]": "x"})
        token = store.put(vault)
        store.drop(token)
        assert store.get(token) is None
        assert vault.closed is True

    def test_drop_of_an_unknown_token_is_harmless(self):
        SessionStore().drop("nope")

    def test_expiry_clears_the_vault(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(
            "scikitplot.cleanprompt._app.time.monotonic", lambda: clock[0]
        )
        store = SessionStore(ttl_seconds=1)
        vault = Vault({"[A-1]": "x"})
        token = store.put(vault)
        clock[0] += 10
        assert store.get(token) is None
        assert vault.closed is True

    def test_access_refreshes_the_expiry(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(
            "scikitplot.cleanprompt._app.time.monotonic", lambda: clock[0]
        )
        store = SessionStore(ttl_seconds=10)
        token = store.put(Vault({"[A-1]": "x"}))
        clock[0] += 8
        assert store.get(token) is not None
        clock[0] += 8
        assert store.get(token) is not None  # refreshed, not 16 seconds old

    def test_capacity_evicts_the_least_recently_used(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(
            "scikitplot.cleanprompt._app.time.monotonic", lambda: clock[0]
        )
        store = SessionStore(max_sessions=3)
        tokens = []
        for index in range(3):
            tokens.append(store.put(Vault({"[A-1]": str(index)})))
            clock[0] += 1
        store.get(tokens[0])  # refresh the oldest
        clock[0] += 1
        store.put(Vault({"[A-1]": "new"}))
        assert store.get(tokens[1]) is None  # now the least recent
        assert store.get(tokens[0]) is not None

    def test_evicted_vault_is_cleared(self, monkeypatch):
        store = SessionStore(max_sessions=1)
        first = Vault({"[A-1]": "x"})
        store.put(first)
        store.put(Vault({"[A-1]": "y"}))
        assert first.closed is True

    def test_len_reports_live_sessions(self):
        store = SessionStore()
        store.put(Vault())
        store.put(Vault())
        assert len(store) == 2

    @pytest.mark.parametrize("field", ["ttl_seconds", "max_sessions"])
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True])
    def test_invalid_bound_is_refused(self, field, bad):
        with pytest.raises(PolicyError, match="positive int"):
            SessionStore(**{field: bad})

    def test_defaults_are_bounded(self):
        assert DEFAULT_TTL_SECONDS > 0
        assert DEFAULT_MAX_SESSIONS > 0

    def test_concurrent_writes_are_safe(self):
        import threading

        store = SessionStore(max_sessions=1000)
        tokens = []
        lock = threading.Lock()

        def work(index):
            token = store.put(Vault({"[A-1]": str(index)}))
            with lock:
                tokens.append(token)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(set(tokens)) == 40


@pytestmark_web
class TestAppFactory:
    """Construction-time configuration."""

    def test_refuses_an_unset_secret_key(self, monkeypatch):
        from .. import create_app

        monkeypatch.delenv("CLEANPROMPT_SECRET_KEY", raising=False)
        with pytest.raises(PolicyError, match="CLEANPROMPT_SECRET_KEY"):
            create_app()

    def test_accepts_an_explicit_key(self):
        from .. import create_app

        assert create_app(secret_key="k" * 32) is not None

    def test_accepts_an_environment_key(self, monkeypatch):
        from .. import create_app

        monkeypatch.setenv("CLEANPROMPT_SECRET_KEY", "k" * 32)
        assert create_app() is not None

    def test_ephemeral_key_must_be_explicit(self, monkeypatch):
        from .. import create_app

        monkeypatch.delenv("CLEANPROMPT_SECRET_KEY", raising=False)
        assert create_app(ephemeral_secret_key=True) is not None

    def test_no_module_level_application(self):
        """Importing must not build an app, a cleaner or a key."""
        from .. import _app

        assert not [
            name
            for name, value in vars(_app).items()
            if type(value).__name__ == "Flask"
        ]

    def test_each_call_builds_an_isolated_app(self):
        from .. import create_app

        first = create_app(secret_key="a" * 32)
        second = create_app(secret_key="b" * 32)
        assert first is not second
        assert first.secret_key != second.secret_key

    def test_cookie_flags(self):
        from .. import create_app

        app = create_app(secret_key="k" * 32)
        assert app.config["SESSION_COOKIE_HTTPONLY"] is True
        assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
        assert app.config["SESSION_COOKIE_SECURE"] is False

    def test_secure_cookie_is_configurable(self):
        from .. import create_app

        app = create_app(secret_key="k" * 32, session_cookie_secure=True)
        assert app.config["SESSION_COOKIE_SECURE"] is True

    def test_request_body_is_bounded(self):
        from .. import create_app

        app = create_app(secret_key="k" * 32)
        assert app.config["MAX_CONTENT_LENGTH"] > 0

    def test_extra_config_is_applied_last(self):
        from .. import create_app

        app = create_app(secret_key="k" * 32, config={"SESSION_COOKIE_SAMESITE": "Strict"})
        assert app.config["SESSION_COOKIE_SAMESITE"] == "Strict"


@pytest.fixture()
def client():
    """Return a test client for a freshly built app."""
    if not WEB_AVAILABLE:
        pytest.skip(skip_reason("web"))
    from .. import create_app

    app = create_app(secret_key="k" * 32, store=SessionStore())
    app.config.update(TESTING=True)
    return app.test_client()


def _csrf(html):
    """Extract the CSRF token from a rendered page."""
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None, "no CSRF token in the page"
    return match.group(1)


@pytestmark_web
class TestRoutes:
    """End-to-end behaviour through the test client."""

    def test_get_renders_the_form(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"CleanPrompt" in response.data

    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"

    def test_redact_then_restore(self, client):
        page = client.get("/").get_data(as_text=True)
        token = _csrf(page)

        redacted = client.post(
            "/",
            data={
                "csrf_token": token,
                "process_text": "1",
                "text": "mail ada@example.com about Acme",
                "additional_words": "Acme",
            },
        ).get_data(as_text=True)
        assert "[EMAIL-1]" in redacted
        assert "[CUSTOM-1]" in redacted

        restored = client.post(
            "/",
            data={
                "csrf_token": token,
                "revert_text": "1",
                "llm_response": "Contact [EMAIL-1] at [CUSTOM-1].",
            },
        ).get_data(as_text=True)
        assert "Contact ada@example.com at Acme." in restored

    def test_the_secret_never_appears_in_the_page(self, client):
        page = client.get("/").get_data(as_text=True)
        redacted = client.post(
            "/",
            data={
                "csrf_token": _csrf(page),
                "process_text": "1",
                "text": "mail topsecret@example.com",
            },
        ).get_data(as_text=True)
        # It appears once, echoed back into the input the user typed, and
        # nowhere in the mapping table.
        table = redacted.split("What was removed")[-1].split("Send this")[0]
        assert "topsecret@example.com" not in table

    def test_the_cookie_carries_no_personal_data(self, client):
        page = client.get("/").get_data(as_text=True)
        response = client.post(
            "/",
            data={
                "csrf_token": _csrf(page),
                "process_text": "1",
                "text": "mail topsecret@example.com",
            },
        )
        cookies = "".join(
            value for key, value in response.headers if key == "Set-Cookie"
        )
        assert "topsecret" not in cookies

    def test_missing_csrf_token_is_refused(self, client):
        client.get("/")
        response = client.post("/", data={"process_text": "1", "text": "x"})
        assert "CSRF token missing or invalid" in response.get_data(as_text=True)

    def test_wrong_csrf_token_is_refused(self, client):
        client.get("/")
        response = client.post(
            "/", data={"csrf_token": "forged", "process_text": "1", "text": "x"}
        )
        assert "CSRF token missing or invalid" in response.get_data(as_text=True)

    def test_reset_requires_a_post(self, client):
        assert client.get("/reset").status_code == 405

    def test_reset_requires_a_csrf_token(self, client):
        client.get("/")
        response = client.post("/reset", data={})
        assert "CSRF token missing or invalid" in response.get_data(as_text=True)

    def test_reset_discards_the_vault(self, client):
        page = client.get("/").get_data(as_text=True)
        token = _csrf(page)
        client.post(
            "/", data={"csrf_token": token, "process_text": "1", "text": "mail a@x.com"}
        )
        assert client.post("/reset", data={"csrf_token": token}).status_code == 302
        restored = client.post(
            "/",
            data={"csrf_token": token, "revert_text": "1", "llm_response": "[EMAIL-1]"},
        ).get_data(as_text=True)
        assert "no stored vault" in restored

    def test_expired_session_is_reported_not_crashed(self, client):
        page = client.get("/").get_data(as_text=True)
        response = client.post(
            "/",
            data={
                "csrf_token": _csrf(page),
                "revert_text": "1",
                "llm_response": "[EMAIL-1]",
            },
        )
        assert "no stored vault" in response.get_data(as_text=True)

    def test_unrecognised_submission_is_reported(self, client):
        page = client.get("/").get_data(as_text=True)
        response = client.post("/", data={"csrf_token": _csrf(page)})
        assert "Unrecognised form submission" in response.get_data(as_text=True)

    def test_two_clients_do_not_share_state(self, client):
        from .. import create_app

        app = create_app(secret_key="k" * 32, store=SessionStore())
        app.config.update(TESTING=True)
        first, second = app.test_client(), app.test_client()

        token_a = _csrf(first.get("/").get_data(as_text=True))
        token_b = _csrf(second.get("/").get_data(as_text=True))
        first.post(
            "/", data={"csrf_token": token_a, "process_text": "1", "text": "a@one.com"}
        )
        second.post(
            "/", data={"csrf_token": token_b, "process_text": "1", "text": "b@two.com"}
        )
        restored_a = first.post(
            "/",
            data={"csrf_token": token_a, "revert_text": "1", "llm_response": "[EMAIL-1]"},
        ).get_data(as_text=True)
        assert "a@one.com" in restored_a
        assert "b@two.com" not in restored_a

    def test_numbering_does_not_accumulate_across_requests(self, client):
        """``CP-003`` at the HTTP layer."""
        page = client.get("/").get_data(as_text=True)
        token = _csrf(page)
        for _ in range(3):
            body = client.post(
                "/",
                data={"csrf_token": token, "process_text": "1", "text": "mail a@x.com"},
            ).get_data(as_text=True)
            assert "[EMAIL-1]" in body
            assert "[EMAIL-2]" not in body

    def test_oversized_body_is_rejected(self, client):
        page = client.get("/").get_data(as_text=True)
        huge = "x" * (client.application.config["MAX_CONTENT_LENGTH"] + 1000)
        response = client.post(
            "/", data={"csrf_token": _csrf(page), "process_text": "1", "text": huge}
        )
        assert response.status_code == 413


@pytestmark_web
class TestTemplates:
    """The shipped page is self-contained."""

    def test_no_remote_resources(self):
        import pathlib

        package = pathlib.Path(__file__).resolve().parent.parent
        for path in list((package / "_templates").rglob("*.html")) + list(
            (package / "_static").rglob("*")
        ):
            if path.is_dir():
                continue
            text = path.read_text(encoding="utf-8")
            # No exemptions. An earlier version waived every ``https://`` in a
            # file that mentioned one particular host anywhere, which would
            # have let any remote resource through beside it; no shipped file
            # needed the waiver.
            for marker in ("http://", "https://", "//cdn", "integrity="):
                assert marker not in text, "{0} references {1}".format(path.name, marker)

    def test_no_inline_event_handlers(self):
        """So the page needs no 'unsafe-inline' in a content security policy."""
        import pathlib

        package = pathlib.Path(__file__).resolve().parent.parent
        for path in (package / "_templates").rglob("*.html"):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"\son[a-z]+\s*=", text), path.name


@pytestmark_web
class TestEntityDetectionUsesTheSharedBuilder:
    """
    ``CP-094``: the web app builds entity detectors the way everything else does.

    Notes
    -----
    **Developer notes.** Reproduced before the fix: ``create_app(enable_ner=
    True, ner_engine="nltk", language="tr", model_size="lg")`` called
    ``spacy_detector(model=None)``. The page then described one configuration
    while each request ran another.
    """

    def test_every_argument_reaches_the_builder(self, monkeypatch):
        from .. import _engines
        from .._app import create_app

        calls = []
        monkeypatch.setattr(
            _engines, "build_detectors", lambda **kw: calls.append(kw) or []
        )
        create_app(
            ephemeral_secret_key=True,
            enable_ner=True,
            ner_engine="nltk",
            language="en",
            model_size="lg",
            ner_model="custom_pipeline",
        )
        assert calls == [
            {
                "mode": "nltk",
                "language": "en",
                "model": "custom_pipeline",
                "size": "lg",
                "required": True,
            }
        ]

    def test_an_unmeetable_request_refuses_to_start(self, monkeypatch):
        """NLTK cannot read Turkish: the app says so instead of serving."""
        from .. import CapabilityError
        from .._app import create_app

        real = caps._installed_version
        engines = {"nltk": "3.9", "spacy": None}
        monkeypatch.setattr(
            caps,
            "_installed_version",
            lambda name: engines[name] if name in engines else real(name),
        )
        with pytest.raises(CapabilityError, match="does not support language 'tr'"):
            create_app(
                ephemeral_secret_key=True,
                enable_ner=True,
                ner_engine="nltk",
                language="tr",
            )

    def test_no_ner_builds_nothing(self, monkeypatch):
        from .. import _engines
        from .._app import create_app

        calls = []
        monkeypatch.setattr(_engines, "build_detectors", lambda **kw: calls.append(kw) or [])
        create_app(ephemeral_secret_key=True)
        assert calls == []
