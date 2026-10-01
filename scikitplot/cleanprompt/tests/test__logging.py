"""
Tests for :mod:`scikitplot.cleanprompt._logging`.

Notes
-----
**Developer notes.** :class:`TestNoSecretReachesALog` is the reason this module
exists. It drives the real pipeline with a capturing handler attached and
inspects every emitted record, so a future call site that passes a value is
caught by the suite rather than by a customer reading a log file.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from .._logging import (
    LOGGER_NAME,
    REDACTION_MARK,
    JsonFormatter,
    SecretFilter,
    configure_logging,
    get_logger,
    log_level_from_env,
    redacting,
)


@pytest.fixture()
def capture():
    """Attach a capturing handler to the submodule logger and clean up."""
    logger = logging.getLogger(LOGGER_NAME)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


class TestGetLogger:
    """Naming."""

    def test_root(self):
        assert get_logger().name == LOGGER_NAME

    def test_submodule_name_is_kept(self):
        assert get_logger(LOGGER_NAME + "._engine").name == LOGGER_NAME + "._engine"

    def test_foreign_name_is_nested_under_the_submodule(self):
        assert get_logger("somewhere.else").name == LOGGER_NAME + ".else"

    def test_every_logger_is_under_one_namespace(self):
        """So an application can silence this submodule with one call."""
        for name in (None, "a.b", LOGGER_NAME + "._api"):
            assert get_logger(name).name.startswith(LOGGER_NAME)


class TestLibraryManners:
    """A library must not configure logging behind an application's back."""

    def test_a_null_handler_is_attached_at_import(self):
        logger = logging.getLogger(LOGGER_NAME)
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_configure_is_idempotent(self):
        """Calling twice must not double every line."""
        buf = io.StringIO()
        configure_logging("info", stream=buf)
        configure_logging("info", stream=buf)
        logger = logging.getLogger(LOGGER_NAME)
        owned = [h for h in logger.handlers if getattr(h, "_cleanprompt_owned", False)]
        assert len(owned) == 1
        configure_logging("critical")

    def test_unknown_level_is_refused(self):
        with pytest.raises(ValueError, match="unknown log level"):
            configure_logging("chatty")

    def test_unknown_format_is_refused(self):
        with pytest.raises(ValueError, match="unknown log format"):
            configure_logging("info", "xml")

    def test_defaults_to_stderr(self):
        """Results go to stdout; a log line there would corrupt a pipe."""
        configure_logging("info")
        logger = logging.getLogger(LOGGER_NAME)
        owned = [h for h in logger.handlers if getattr(h, "_cleanprompt_owned", False)]
        import sys

        assert owned[0].stream is sys.stderr
        configure_logging("critical")


class TestEnvironment:
    """Level from the environment."""

    def test_default_when_unset(self, monkeypatch):
        for var in ("CLEANPROMPT_LOG_LEVEL", "SKPLT_LOGGING_LEVEL"):
            monkeypatch.delenv(var, raising=False)
        assert log_level_from_env() == "warning"

    def test_submodule_variable(self, monkeypatch):
        monkeypatch.setenv("CLEANPROMPT_LOG_LEVEL", "debug")
        assert log_level_from_env() == "debug"

    def test_project_variable(self, monkeypatch):
        monkeypatch.delenv("CLEANPROMPT_LOG_LEVEL", raising=False)
        monkeypatch.setenv("SKPLT_LOGGING_LEVEL", "info")
        assert log_level_from_env() == "info"

    def test_garbage_falls_back(self, monkeypatch):
        """An inherited misspelling must not stop the command running."""
        monkeypatch.setenv("CLEANPROMPT_LOG_LEVEL", "verbose")
        assert log_level_from_env() == "warning"


class TestSecretFilter:
    """Defence in depth against a future call site."""

    def _record(self, msg, args=()):
        return logging.LogRecord("t", logging.INFO, "f", 1, msg, args, None)

    def test_scrubs_the_message(self):
        record = self._record("found topsecret@example.com")
        SecretFilter(["topsecret@example.com"]).filter(record)
        assert "topsecret" not in record.getMessage()
        assert REDACTION_MARK in record.getMessage()

    def test_scrubs_interpolated_arguments(self):
        """A secret passed as %s is not in record.msg."""
        record = self._record("found %s", ("topsecret@example.com",))
        SecretFilter(["topsecret@example.com"]).filter(record)
        assert "topsecret" not in record.getMessage()

    def test_short_surfaces_are_left_alone(self):
        """Scrubbing two-letter values would make every message noise."""
        record = self._record("the ab thing")
        SecretFilter(["ab"]).filter(record)
        assert "ab" in record.getMessage()

    def test_longest_first(self):
        record = self._record("x ada@example.com y")
        SecretFilter(["ada@example.com", "example.com"]).filter(record)
        assert record.getMessage().count(REDACTION_MARK) == 1

    def test_always_emits(self):
        assert SecretFilter(["x" * 9]).filter(self._record("nothing")) is True

    def test_no_secrets_is_a_noop(self):
        record = self._record("plain message")
        SecretFilter().filter(record)
        assert record.getMessage() == "plain message"


class TestRedactingContext:
    """Attached for a block, removed afterwards."""

    def test_scrubs_inside_the_block(self, capture):
        logger = get_logger()
        with redacting(["topsecret@example.com"]):
            logger.info("saw topsecret@example.com")
        assert "topsecret" not in capture[0].getMessage()

    def test_filter_is_removed_on_exit(self):
        logger = logging.getLogger(LOGGER_NAME)
        before = len(logger.filters)
        with redacting(["topsecret@example.com"]):
            assert len(logger.filters) == before + 1
        assert len(logger.filters) == before

    def test_filter_is_removed_when_the_block_raises(self):
        """Otherwise a long-lived process accumulates every request's secrets."""
        logger = logging.getLogger(LOGGER_NAME)
        before = len(logger.filters)
        with pytest.raises(ValueError):
            with redacting(["topsecret@example.com"]):
                raise ValueError("boom")
        assert len(logger.filters) == before


class TestJsonFormatter:
    """Structured output."""

    def test_emits_one_json_object(self):
        record = logging.LogRecord("n", logging.INFO, "f", 1, "hello", (), None)
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "n"

    def test_merges_extra(self):
        record = logging.LogRecord("n", logging.INFO, "f", 1, "m", (), None)
        record.entries = 3
        assert json.loads(JsonFormatter().format(record))["entries"] == 3

    def test_unserialisable_extra_is_repr_ed(self):
        record = logging.LogRecord("n", logging.INFO, "f", 1, "m", (), None)
        record.thing = object()
        assert "object" in json.loads(JsonFormatter().format(record))["thing"]


class TestNoSecretReachesALog:
    """The rule: logs say what happened, never what was found."""

    SECRETS = (
        "topsecret@example.com",
        "+1 555 010 4477",
        "4242 4242 4242 4242",
        "Mustafa Kemal",
    )

    def _text(self):
        return (
            "Mustafa Kemal wrote to topsecret@example.com, called "
            "+1 555 010 4477 and paid with 4242 4242 4242 4242."
        )

    def test_encode_logs_no_secret(self, capture):
        from .. import encode

        encode(self._text(), hide=["Mustafa Kemal"])
        assert capture, "expected at least one record"
        blob = " ".join(record.getMessage() for record in capture)
        for secret in self.SECRETS:
            assert secret not in blob

    def test_session_logs_no_secret(self, capture):
        from .. import session

        with session() as chat:
            chat.encode(self._text())
            chat.decode("Reply to [EMAIL-1]")
        blob = " ".join(record.getMessage() for record in capture)
        for secret in self.SECRETS:
            assert secret not in blob

    def test_redaction_pipeline_logs_no_secret(self, capture):
        from .. import Redactor, restore

        result = Redactor().redact(self._text())
        restore(result.text, result.vault)
        blob = " ".join(record.getMessage() for record in capture)
        for secret in self.SECRETS:
            assert secret not in blob

    def test_placeholders_may_be_logged(self, capture):
        """A label is already in the text being sent; it is not a secret."""
        from .. import decode, encode

        safe, handle = encode(self._text())
        decode("see [EMAIL-9]", handle)
        blob = " ".join(record.getMessage() for record in capture)
        assert "[EMAIL-9]" in blob

    def test_json_records_carry_no_secret(self):
        from .. import encode

        buf = io.StringIO()
        configure_logging("debug", "json", stream=buf)
        try:
            encode(self._text())
        finally:
            configure_logging("critical")
        for secret in self.SECRETS:
            assert secret not in buf.getvalue()

    def test_nothing_is_emitted_by_default(self):
        """Importing and using the library must be silent."""
        from .. import encode

        logger = logging.getLogger(LOGGER_NAME)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            encode("mail ada@example.com")
        finally:
            logger.removeHandler(handler)
        assert buf.getvalue() == ""


class TestNamespaceScrubbing:
    """CP-059: a filter on the namespace root never saw a child's records."""

    def _capture(self):
        import io

        from .. import configure_logging

        buffer = io.StringIO()
        configure_logging("debug", "json", stream=buffer)
        return buffer

    def test_child_logger_records_are_scrubbed(self):
        from .._logging import get_logger

        buffer = self._capture()
        with redacting(["topsecret@example.com"]):
            get_logger("scikitplot.cleanprompt._api").warning(
                "found %s", "topsecret@example.com"
            )
            get_logger("scikitplot.cleanprompt._created_inside").warning(
                "topsecret@example.com"
            )
        get_logger("scikitplot.cleanprompt._api").warning("after topsecret@example.com")
        lines = buffer.getvalue().splitlines()
        assert "topsecret" not in lines[0] and "topsecret" not in lines[1]
        assert "topsecret" in lines[2]  # the filter is gone after the block

    def test_extra_fields_and_tracebacks_are_scrubbed(self):
        from .._logging import get_logger

        buffer = self._capture()
        with redacting(["topsecret@example.com"]):
            logger = get_logger("scikitplot.cleanprompt._api")
            logger.warning("x", extra={"who": "topsecret@example.com"})
            try:
                raise ValueError("bad topsecret@example.com")
            except ValueError:
                logger.exception("failed")
        assert "topsecret" not in buffer.getvalue()

    def test_a_vault_scrubber_lives_as_long_as_its_owner(self):
        import gc

        from .._logging import _ACTIVE, VaultScrubber

        class Owner:
            pass

        owner = Owner()
        scrubber = VaultScrubber(owner=owner)
        scrubber.add(["topsecret@example.com"])
        assert any("topsecret@example.com" in f._secrets for f in _ACTIVE)
        del owner
        gc.collect()
        assert not any("topsecret@example.com" in f._secrets for f in _ACTIVE)

    def test_audit_events_are_structured(self):
        import json

        from .._logging import AUDIT_LOGGER_NAME, audit

        buffer = self._capture()
        audit("encoded", values=2, kinds={"EMAIL": 2})
        record = json.loads(buffer.getvalue().splitlines()[-1])
        assert record["logger"] == AUDIT_LOGGER_NAME
        assert record["event"] == "encoded" and record["kinds"] == {"EMAIL": 2}


class TestSharedScrubber:
    """Every vault's values are scrubbed by one filter, counted per holder."""

    def test_a_value_held_twice_survives_one_release(self):
        from .._logging import _SHARED, VaultScrubber

        first, second = VaultScrubber(), VaultScrubber()
        first.add(["shared@example.com", "only-first@example.com"])
        second.add(["shared@example.com"])
        first.close()
        assert _SHARED.scrub("shared@example.com") == REDACTION_MARK
        assert _SHARED.scrub("only-first@example.com") == "only-first@example.com"
        second.close()
        assert _SHARED.scrub("shared@example.com") == "shared@example.com"

    def test_adding_the_same_value_twice_needs_one_release(self):
        from .._logging import _SHARED, VaultScrubber

        scrubber = VaultScrubber()
        scrubber.add(["twice@example.com"])
        scrubber.add(["twice@example.com"])
        scrubber.close()
        assert _SHARED.scrub("twice@example.com") == "twice@example.com"

    def test_the_longer_value_is_replaced_first(self):
        from .._logging import _SHARED, VaultScrubber

        scrubber = VaultScrubber()
        scrubber.add(["Marion", "Marion Holt", "Ann"])
        try:
            assert (
                _SHARED.scrub("Marion Holt and Marion, Ann")
                == f"{REDACTION_MARK} and {REDACTION_MARK}, Ann"
            )  # "Ann" is below MIN_LENGTH: it would rewrite "Annual" too
        finally:
            scrubber.close()

    @pytest.mark.parametrize("seed", range(40))
    def test_both_lookups_find_exactly_the_surfaces_present(self, seed):
        """The substring scan and the prefix index agree with brute force."""
        import random

        from .._logging import _SharedSecrets

        rng = random.Random(seed)
        alphabet = "ab0@. "
        values = {
            "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 9)))
            for _ in range(rng.randint(0, 600))
        }
        shared = _SharedSecrets()
        shared.add(values)
        dropped = [value for value in values if rng.random() < 0.3]
        shared.release(dropped)
        held = values - set(dropped)
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))
        try:
            truth = {value for value in held if value in text}
            shared.SCAN_LIMIT = len(values) + 1
            assert shared.present(text) == truth
            shared.SCAN_LIMIT = -1
            assert shared.present(text) == truth
        finally:
            shared.release(list(held))
        assert shared.held == 0 and shared._lengths == {}

    def test_many_holders_stay_fast(self):
        import time

        from .._logging import VaultScrubber

        buffer = TestNamespaceScrubbing()._capture()
        holders = [VaultScrubber() for _ in range(2000)]
        for index, holder in enumerate(holders):
            holder.add([f"user{index}@example.com"])
        started = time.perf_counter()
        logger = get_logger("scikitplot.cleanprompt._api")
        for index in range(200):
            logger.warning("about user%d@example.com", index)
        elapsed = time.perf_counter() - started
        for holder in holders:
            holder.close()
        assert "@example.com" not in buffer.getvalue()
        assert elapsed < 5.0, elapsed  # was quadratic before round 15
