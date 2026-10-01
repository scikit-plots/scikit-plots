"""Tests for :mod:`scikitplot.cleanprompt._api`."""

from __future__ import annotations

import json

import pytest

from .. import CleanPromptError, Handle, PolicyError, Session, decode, encode, session


class TestEncode:
    """The safe half."""

    def test_returns_a_pair(self):
        safe, handle = encode("Mail ada@example.com")
        assert safe == "Mail [EMAIL-1]"
        assert isinstance(handle, Handle)

    def test_result_object_carries_detail(self):
        result = encode("Mail ada@example.com")
        assert result.count == 1
        assert result.level in ("ok", "warning", "alert")
        assert result.result.stats.entries == 1

    def test_rejects_non_string(self):
        with pytest.raises(TypeError, match="must be str"):
            encode(None)

    def test_profile_is_honoured(self):
        safe, _ = encode("Mustafa Kemal founded it", profile="strict")
        assert "[TITLE_CASE-1]" in safe

    def test_hide_terms(self):
        safe, _ = encode("Acme shipped it", hide=["Acme"])
        assert safe == "[CUSTOM-1] shipped it"

    def test_allowlist(self):
        safe, _ = encode(
            "a@x.com and b@x.com", allow=["b@x.com"]
        )
        assert "b@x.com" in safe

    def test_kinds_narrow_detection(self):
        safe, _ = encode("a@x.com +1 555 010 4477", kinds=["EMAIL"])
        assert "[EMAIL-1]" in safe and "555" in safe

    def test_empty_text(self):
        safe, handle = encode("")
        assert safe == ""
        assert handle.labels == ()

    def test_report_explains_an_empty_result(self):
        result = encode("nothing here")
        assert "Nothing was redacted" in result.report["headline"]

    def test_suggestions_can_be_skipped(self):
        assert encode("Ada Lovelace", suggest=False).report["suggestions"] == []


class TestDecode:
    """The inverse."""

    def test_round_trip(self):
        text = "Mail ada@example.com or call +1 555 010 4477"
        safe, handle = encode(text)
        assert decode(safe, handle) == text

    def test_partial_reply(self):
        safe, handle = encode("Mail ada@example.com")
        assert decode("Reply to [EMAIL-1] please", handle) == (
            "Reply to ada@example.com please"
        )

    def test_unknown_placeholder_is_left_alone(self):
        safe, handle = encode("Mail ada@example.com")
        assert "[EMAIL-9]" in decode("see [EMAIL-9]", handle)

    def test_strict_raises_on_unknown(self):
        from .. import RestorationError

        safe, handle = encode("Mail ada@example.com")
        with pytest.raises(RestorationError):
            decode("see [EMAIL-9]", handle, strict=True)

    def test_rejects_a_non_handle(self):
        with pytest.raises(TypeError, match="must be a Handle"):
            decode("text", {"not": "a handle"})

    def test_cleared_handle_is_refused(self):
        safe, handle = encode("Mail ada@example.com")
        handle.clear()
        with pytest.raises(PolicyError, match="cleared"):
            decode(safe, handle)


class TestHandle:
    """The unsafe half."""

    def test_repr_hides_values(self):
        _, handle = encode("Mail topsecret@example.com")
        assert "topsecret" not in repr(handle)
        assert "entries=1" in repr(handle)

    def test_labels_are_not_secrets(self):
        _, handle = encode("Mail ada@example.com")
        assert handle.labels == ("[EMAIL-1]",)

    def test_export_is_the_named_way_out(self):
        _, handle = encode("Mail ada@example.com")
        document = handle.export()
        assert document["entries"] == {"[EMAIL-1]": "ada@example.com"}

    def test_export_is_json_safe(self):
        _, handle = encode("Mail ada@example.com")
        assert json.loads(json.dumps(handle.export()))

    def test_survives_a_process_boundary(self):
        """A web request or a queue must be able to carry it."""
        safe, handle = encode("Mail ada@example.com")
        wire = json.dumps(handle.export())
        revived = Handle.load(json.loads(wire))
        assert decode(safe, revived) == "Mail ada@example.com"

    def test_load_rejects_a_malformed_document(self):
        with pytest.raises(PolicyError, match="malformed"):
            Handle.load({"entries": ["not", "a", "mapping"]})

    def test_load_rejects_a_foreign_grammar(self):
        from .. import DEFAULT_POLICY, TagStyle

        other = DEFAULT_POLICY.evolve(tag_style=TagStyle(prefix="<<", suffix=">>"))
        _, handle = encode("Mail ada@example.com", policy=other)
        with pytest.raises(PolicyError, match="different placeholder grammar"):
            Handle.load(handle.export())

    def test_clear_drops_the_values(self):
        _, handle = encode("Mail ada@example.com")
        handle.clear()
        assert handle.vault.closed


class TestSession:
    """A conversation."""

    def test_placeholders_are_consistent_across_turns(self):
        with session() as chat:
            first = chat.encode("Mail ada@example.com and bob@example.com")
            second = chat.encode("Just ping bob@example.com")
        assert "[EMAIL-2]" in first
        assert second == "Just ping [EMAIL-2]"

    def test_decode_resolves_labels_from_any_turn(self):
        with session() as chat:
            chat.encode("Mail ada@example.com")
            chat.encode("And bob@example.com")
            out = chat.decode("Tell [EMAIL-2] that [EMAIL-1] agreed")
        assert out == "Tell bob@example.com that ada@example.com agreed"

    def test_turns_are_counted(self):
        with session() as chat:
            chat.encode("a@x.com")
            chat.encode("b@x.com")
            assert chat.turns == 2

    def test_the_vault_is_cleared_on_exit(self):
        with session() as chat:
            chat.encode("Mail ada@example.com")
        assert chat.handle.vault.closed

    def test_the_vault_is_cleared_when_the_block_raises(self):
        chat = Session()
        with pytest.raises(ValueError):
            with chat:
                chat.encode("Mail ada@example.com")
                raise ValueError("boom")
        assert chat.handle.vault.closed

    def test_roundtrip_calls_the_sender(self):
        seen = {}

        def sender(safe):
            seen["got"] = safe
            return "Echo: " + safe

        with session() as chat:
            out = chat.roundtrip("Mail ada@example.com", send=sender)
        assert seen["got"] == "Mail [EMAIL-1]"
        assert out == "Echo: Mail ada@example.com"

    def test_roundtrip_refuses_a_non_string_reply(self):
        with session() as chat:
            with pytest.raises(CleanPromptError, match="must return str"):
                chat.roundtrip("Mail ada@example.com", send=lambda _s: 42)

    def test_a_sender_exception_propagates_unchanged(self):
        """It is the caller's client and the caller's error."""

        class MyClientError(RuntimeError):
            pass

        def boom(_safe):
            raise MyClientError("rate limited")

        with session() as chat:
            with pytest.raises(MyClientError, match="rate limited"):
                chat.roundtrip("Mail ada@example.com", send=boom)

    def test_decode_report_names_unknown_labels(self):
        with session() as chat:
            chat.encode("Mail ada@example.com")
            report = chat.decode_report("see [EMAIL-9]")
        assert report.unknown == ("[EMAIL-9]",)

    def test_report_describes_the_configuration(self):
        with session() as chat:
            assert "active_kinds" in chat.report()["detection"]

    def test_report_describes_the_session_itself(self):
        """``report()`` on a session must say what the session did."""
        with session() as chat:
            chat.encode("Mail ada@example.com")
            chat.encode("Mail ada@example.com again from 192.168.1.10.")
            report = chat.report()
        assert report["turns"] == 2
        assert report["entries"] == 2
        assert sorted(report["labels"].values()) == ["EMAIL", "IPV4"]

    def test_report_names_no_value(self):
        with session() as chat:
            chat.encode("Mail topsecret@example.com")
            assert "topsecret" not in repr(chat.report())

    def test_report_is_json_safe(self):
        import json

        with session() as chat:
            chat.encode("Mail ada@example.com")
            assert "ada@example.com" not in json.dumps(chat.report())

    def test_repr_hides_values(self):
        with session() as chat:
            chat.encode("Mail topsecret@example.com")
            assert "topsecret" not in repr(chat)

    def test_options_apply_to_every_turn(self):
        with session(profile="minimal") as chat:
            out = chat.encode("Mustafa Kemal mailed a@x.com")
        assert "[EMAIL-1]" in out
        assert "Mustafa" in out

    def test_rejects_non_string(self):
        with session() as chat:
            with pytest.raises(TypeError):
                chat.encode(None)

    def test_usable_without_the_context_manager(self):
        chat = Session()
        assert chat.encode("Mail ada@example.com") == "Mail [EMAIL-1]"
        chat.clear()


class TestStatelessnessIsPreserved:
    """Continuity is explicit; two independent calls stay independent."""

    def test_module_level_encode_does_not_carry(self):
        first = encode("Mail ada@example.com")
        second = encode("Mail bob@example.com")
        assert first.text == second.text == "Mail [EMAIL-1]"

    def test_two_sessions_do_not_share(self):
        with session() as one, session() as two:
            one.encode("Mail ada@example.com")
            assert two.encode("Mail bob@example.com") == "Mail [EMAIL-1]"


class TestSessionConcurrency:
    """CP-067: a session shared by threads (a threaded web server) stays exact."""

    @pytest.mark.parametrize("trial", range(3))
    def test_threads_sharing_a_session(self, trial):
        from ._concurrency import hammer

        with session() as shared:
            assert hammer(shared.encode, shared.decode) == []
