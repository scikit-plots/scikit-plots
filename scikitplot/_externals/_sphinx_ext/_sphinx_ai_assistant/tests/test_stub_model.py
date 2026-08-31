"""
Tests for the Path 0 deterministic stub responder.

The stub exists so security properties can be asserted cheaply.  These tests
assert the stub's OWN security properties first, because a test rig that leaks
is worse than no rig: it would be trusted while being wrong.

SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_PROXY_DIR = pathlib.Path(__file__).resolve().parents[1] / "_hf_spaces_proxy"
if str(_PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(_PROXY_DIR))

from _utils._stub_model import (  # noqa: E402
    build_stub_reply,
    classify_secret,
    is_stub_model,
    parse_stub_mode,
    scan_for_secrets,
    stub_payload,
    stub_sse_frames,
    summarize_headers,
)

# A token that must never appear in any output, in any mode.
_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz0123456789"
_AWS = "AKIAIOSFODNN7EXAMPLE"
_OPENAI = "sk-abcdefghijklmnopqrstuvwxyz"

_HEADERS = {
    "Authorization": f"Bearer {_TOKEN}",
    "Cookie": "session=supersecretvalue",
    "X-Api-Key": _OPENAI,
    "Content-Type": "application/json",
    "Origin": "https://docs.example.org",
}


def _body(**over):
    payload = {
        "model": "stub/echo",
        "max_tokens": 1000,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Documentation page context."},
            {"role": "user", "content": "ping"},
        ],
    }
    payload.update(over)
    return payload


# ===========================================================================
# 1. Model-id routing
# ===========================================================================


class TestStubRouting:
    """Only `stub/` ids are intercepted, and the mode parse is total."""

    @pytest.mark.parametrize(
        "model",
        ["stub/echo", "stub/qa", "STUB/ECHO", " stub/error:404 ", "stub/"],
    )
    def test_recognised(self, model):
        assert is_stub_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "openai/gpt-4",
            "stub",
            "notstub/echo",
            "",
            None,
            123,
            {},
            ["stub/echo"],
        ],
    )
    def test_not_recognised(self, model):
        assert is_stub_model(model) is False

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("stub/echo", ("echo", "")),
            ("stub/qa", ("qa", "")),
            ("stub/hostile", ("hostile", "")),
            ("stub/error:503", ("error", "503")),
            ("stub/slow:250", ("slow", "250")),
            # A typo must yield a usable report, not a stack trace: the rig is
            # what you reach for when something is already wrong.
            ("stub/nonsense", ("echo", "")),
            ("stub/", ("echo", "")),
            ("not-a-stub", ("echo", "")),
        ],
    )
    def test_mode_parse_is_total(self, model, expected):
        assert parse_stub_mode(model) == expected


# ===========================================================================
# 2. The stub must not leak — its own most important property
# ===========================================================================


class TestNoCredentialLeak:
    """No credential value may appear in any response, in any mode."""

    @pytest.mark.parametrize(
        "model",
        ["stub/echo", "stub/qa", "stub/hostile", "stub/error:500", "stub/slow:0"],
    )
    def test_token_never_echoed(self, model):
        _status, doc = stub_payload(model, _body(model=model), _HEADERS)
        blob = json.dumps(doc)
        assert _TOKEN not in blob
        assert "supersecretvalue" not in blob
        assert _OPENAI not in blob

    def test_streaming_never_echoes_a_token(self):
        frames = stub_sse_frames("stub/echo", _body(stream=True), _HEADERS)
        blob = "".join(frames)
        assert _TOKEN not in blob
        assert "supersecretvalue" not in blob

    def test_credential_presence_is_still_reported(self):
        """Redaction must not become silence — the whole point is visibility."""
        _status, doc = stub_payload("stub/echo", _body(), _HEADERS)
        creds = doc["stub_report"]["headers"]["credentials"]
        assert creds["authorization"]["present"] is True
        assert creds["authorization"]["length"] == len(f"Bearer {_TOKEN}")
        assert creds["authorization"]["scheme"] == "Bearer"
        assert "huggingface_token" in creds["authorization"]["matched_patterns"]

    def test_prefix_class_is_too_short_to_use(self):
        info = classify_secret(f"Bearer {_TOKEN}")
        assert len(info["prefix_class"].rstrip("\u2026")) <= 3

    def test_non_secret_headers_are_reported_in_full(self):
        """A test needs to assert on origin and content-type."""
        summary = summarize_headers(_HEADERS)
        assert summary["other"]["origin"] == "https://docs.example.org"
        assert summary["other"]["content-type"] == "application/json"
        assert "authorization" not in summary["other"]

    def test_header_summary_survives_a_non_mapping(self):
        summary = summarize_headers(None)
        assert summary == {"names": [], "credentials": {}, "other": {}}


# ===========================================================================
# 3. Secret detection in the prompt — the leakage question
# ===========================================================================


class TestSecretScanning:
    def test_finds_structured_secrets(self):
        findings = scan_for_secrets(f"key {_AWS} and {_OPENAI} here")
        names = {f["pattern"] for f in findings}
        assert "aws_access_key_id" in names
        assert "openai_key" in names

    def test_reports_offset_not_content(self):
        """A leak report that quotes the leak has moved the problem."""
        findings = scan_for_secrets(f"prefix {_AWS}")
        assert findings
        blob = json.dumps(findings)
        assert _AWS not in blob
        assert findings[0]["first_offset"] == 7

    @pytest.mark.parametrize("text", ["", None, 42, {}, "no secrets here at all"])
    def test_clean_input_yields_nothing(self, text):
        assert scan_for_secrets(text) == []

    def test_secrets_in_page_context_are_surfaced(self):
        payload = _body(
            messages=[
                {"role": "system", "content": f"Page says {_AWS}"},
                {"role": "user", "content": "hi"},
            ]
        )
        _status, doc = stub_payload("stub/echo", payload, {})
        found = doc["stub_report"]["secrets_in_system_prompt"]
        assert [f["pattern"] for f in found] == ["aws_access_key_id"]

    def test_secrets_in_the_user_message_are_surfaced(self):
        payload = _body(
            messages=[{"role": "user", "content": f"my key is {_OPENAI}"}]
        )
        _status, doc = stub_payload("stub/echo", payload, {})
        assert doc["stub_report"]["secrets_in_user_message"]


# ===========================================================================
# 4. Reasoning controls — "what do Effort and Thinking actually send?"
# ===========================================================================


class TestReasoningVisibility:
    def test_absent_when_not_sent(self):
        _status, doc = stub_payload("stub/echo", _body(), {})
        report = doc["stub_report"]["reasoning"]
        assert report["sent"] == []
        assert "reasoning_effort" in report["absent"]

    def test_effort_is_reported_with_its_value(self):
        payload = _body(reasoning_effort="high")
        _status, doc = stub_payload("stub/echo", payload, {})
        report = doc["stub_report"]["reasoning"]
        assert report["sent"] == ["reasoning_effort"]
        assert report["values"]["reasoning_effort"] == "high"

    def test_thinking_budget_is_reported_verbatim(self):
        thinking = {"type": "enabled", "budget_tokens": 5000}
        _status, doc = stub_payload("stub/echo", _body(thinking=thinking), {})
        assert doc["stub_report"]["reasoning"]["values"]["thinking"] == thinking

    def test_the_reply_text_states_them_too(self):
        """A maintainer reading the panel must see this without a debugger."""
        payload = _body(reasoning_effort="low")
        _status, doc = stub_payload("stub/echo", payload, {})
        text = doc["choices"][0]["message"]["content"]
        assert "reasoning_effort" in text
        assert "low" in text

    def test_absent_and_default_are_distinguishable(self):
        """The distinction that matters when a control appears to do nothing."""
        _s1, with_field = stub_payload("stub/echo", _body(reasoning_effort="medium"), {})
        _s2, without = stub_payload("stub/echo", _body(), {})
        assert with_field["stub_report"]["reasoning"]["sent"] == ["reasoning_effort"]
        assert without["stub_report"]["reasoning"]["sent"] == []

    @pytest.mark.parametrize("value", ["", 0, False, None, {}])
    def test_present_but_falsy_still_counts_as_sent(self, value):
        """
        Membership, not truthiness.

        A field sent as an empty string or zero WAS sent, and reporting it as
        absent would send a maintainer hunting a client bug that does not
        exist -- while hiding the server-side one that does. This is exactly
        the case a truthiness check gets wrong.
        """
        _status, doc = stub_payload("stub/echo", _body(reasoning_effort=value), {})
        report = doc["stub_report"]["reasoning"]
        assert report["sent"] == ["reasoning_effort"]
        assert report["values"]["reasoning_effort"] == value
        assert "reasoning_effort" not in report["absent"]


# ===========================================================================
# 5. Response shape and modes
# ===========================================================================


class TestResponseShape:
    def test_uses_the_shape_the_client_already_parses(self):
        _status, doc = stub_payload("stub/echo", _body(), {})
        assert doc["object"] == "chat.completion"
        assert isinstance(doc["choices"][0]["message"]["content"], str)
        assert doc["choices"][0]["finish_reason"] == "stop"

    def test_report_marks_that_nothing_was_forwarded(self):
        _status, doc = stub_payload("stub/echo", _body(), {})
        assert doc["stub_report"]["upstream_called"] is False
        assert doc["stub_report"]["credentials_read"] is False

    def test_response_is_json_serialisable(self):
        """Never HTML — the stub must not become a reflected-XSS oracle."""
        _status, doc = stub_payload("stub/hostile", _body(), _HEADERS)
        assert json.loads(json.dumps(doc)) == doc

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("stub/error:404", 404),
            ("stub/error:503", 503),
            ("stub/error:429", 429),
            # Out of HTTP space, or nonsense: clamped rather than passed
            # through, so a request field cannot set an arbitrary status.
            ("stub/error:99", 500),
            ("stub/error:700", 500),
            ("stub/error:abc", 500),
            ("stub/error", 500),
        ],
    )
    def test_error_status_is_clamped(self, model, expected):
        status, doc = stub_payload(model, _body(model=model), {})
        assert status == expected
        assert doc["error"]["type"] == "stub_error"

    def test_success_modes_return_200(self):
        for model in ("stub/echo", "stub/qa", "stub/hostile", "stub/slow:0"):
            status, _doc = stub_payload(model, _body(model=model), {})
            assert status == 200

    def test_qa_fixtures_match_the_last_user_turn(self):
        payload = _body(
            model="stub/qa",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "ping"},
            ],
        )
        _status, doc = stub_payload("stub/qa", payload, {})
        assert doc["choices"][0]["message"]["content"] == "pong"

    def test_qa_miss_lists_what_is_available(self):
        payload = _body(model="stub/qa", messages=[{"role": "user", "content": "zzz"}])
        _status, doc = stub_payload("stub/qa", payload, {})
        assert "Known fixtures" in doc["choices"][0]["message"]["content"]

    def test_hostile_payload_is_returned_as_ordinary_reply_text(self):
        """It must take the client's normal rendering path, not a special one."""
        _status, doc = stub_payload("stub/hostile", _body(), {})
        text = doc["choices"][0]["message"]["content"]
        assert "<script>" in text
        assert doc["choices"][0]["message"]["role"] == "assistant"

    def test_anthropic_content_blocks_are_understood(self):
        payload = _body(
            messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}]
        )
        _status, doc = stub_payload("stub/qa", payload, {})
        assert doc["choices"][0]["message"]["content"] == "pong"

    def test_top_level_system_field_is_understood(self):
        payload = {
            "model": "stub/echo",
            "system": f"anthropic-style system with {_AWS}",
            "messages": [{"role": "user", "content": "hi"}],
        }
        _status, doc = stub_payload("stub/echo", payload, {})
        assert doc["stub_report"]["secrets_in_system_prompt"]


# ===========================================================================
# 6. Streaming
# ===========================================================================


class TestStreaming:
    def test_multiple_frames_are_emitted(self):
        """A single-frame stream would pass while a real one failed."""
        frames = stub_sse_frames("stub/echo", _body(stream=True), {})
        assert len(frames) > 2

    def test_every_frame_is_wellformed_sse(self):
        frames = stub_sse_frames("stub/qa", _body(stream=True), {})
        for frame in frames:
            assert frame.startswith("data: ")
            assert frame.endswith("\n\n")

    def test_terminated_with_done(self):
        frames = stub_sse_frames("stub/qa", _body(stream=True), {})
        assert frames[-1] == "data: [DONE]\n\n"

    def test_deltas_reassemble_to_the_full_reply(self):
        payload = _body(model="stub/qa", stream=True,
                        messages=[{"role": "user", "content": "ping"}])
        frames = stub_sse_frames("stub/qa", payload, {})
        text = ""
        for frame in frames[:-1]:
            chunk = json.loads(frame[len("data: "):])
            text += chunk["choices"][0]["delta"].get("content", "")
        assert text == "pong"

    def test_final_frame_carries_the_report(self):
        frames = stub_sse_frames("stub/echo", _body(stream=True), _HEADERS)
        final = json.loads(frames[-2][len("data: "):])
        assert final["stub_report"]["stub"] is True
        assert final["choices"][0]["finish_reason"] == "stop"


# ===========================================================================
# 7. Determinism — the property that makes it a rig rather than a toy
# ===========================================================================


class TestDeterminism:
    def test_identical_inputs_give_identical_output(self):
        a = stub_payload("stub/echo", _body(), _HEADERS, request_id="fixed", created=0)
        b = stub_payload("stub/echo", _body(), _HEADERS, request_id="fixed", created=0)
        assert json.dumps(a[1], sort_keys=True) == json.dumps(b[1], sort_keys=True)

    def test_build_reply_is_pure(self):
        args = ("echo", "", _body(), _HEADERS)
        first = build_stub_reply(*args, request_id="fixed")
        second = build_stub_reply(*args, request_id="fixed")
        assert first == second

    def test_malformed_payloads_do_not_raise(self):
        for payload in (None, [], "text", 42, {}):
            status, doc = stub_payload("stub/echo", payload, {})
            assert status == 200
            assert "choices" in doc


# ===========================================================================
# 8. Mode registry — the extension point
# ===========================================================================


class TestModeRegistry:
    """Adding a mode must be one entry, not an edit in four places."""

    def test_builtin_modes_are_registered(self):
        from _utils._stub_model import stub_modes

        modes = stub_modes()
        assert set(modes) == {"echo", "qa", "hostile", "error", "slow"}
        assert all(isinstance(v, str) and v for v in modes.values())

    def test_parser_reads_the_registry(self):
        """A registered mode becomes parseable with no parser edit."""
        from _utils._stub_model import parse_stub_mode, register_stub_mode

        register_stub_mode(
            "probe", lambda arg, payload, report: "probe reply", "Test-only mode."
        )
        try:
            assert parse_stub_mode("stub/probe") == ("probe", "")
            _status, doc = stub_payload("stub/probe", _body(model="stub/probe"), {})
            assert doc["choices"][0]["message"]["content"] == "probe reply"
        finally:
            from _utils._stub_model import _STUB_MODES

            _STUB_MODES.pop("probe", None)

    def test_a_registered_mode_is_advertised(self):
        from _utils._stub_model import _STUB_MODES, register_stub_mode, stub_modes

        register_stub_mode(
            "probe2", lambda a, p, r: "x", "Advertised summary."
        )
        try:
            assert stub_modes()["probe2"] == "Advertised summary."
        finally:
            _STUB_MODES.pop("probe2", None)

    def test_duplicate_registration_is_refused(self):
        """Silent overwrite would let two deployments disagree about a mode."""
        from _utils._stub_model import register_stub_mode

        with pytest.raises(ValueError, match="already registered"):
            register_stub_mode("echo", lambda a, p, r: "x", "dupe")

    @pytest.mark.parametrize(
        "name", ["Echo", "with space", "1bad", "_bad", "", "with:colon", "x" * 40]
    )
    def test_malformed_mode_names_are_refused(self, name):
        from _utils._stub_model import register_stub_mode

        with pytest.raises(ValueError):
            register_stub_mode(name, lambda a, p, r: "x", "bad")

    def test_non_callable_handler_is_refused(self):
        from _utils._stub_model import register_stub_mode

        with pytest.raises(ValueError, match="callable"):
            register_stub_mode("nothandler", "not a function", "bad")

    def test_echo_lists_the_available_modes(self):
        """Discoverable from the panel, without reading the source."""
        _status, doc = stub_payload("stub/echo", _body(), {})
        text = doc["choices"][0]["message"]["content"]
        assert "available modes" in text
        assert "hostile" in text

    def test_delay_clamp_has_one_home(self):
        """Both proxies call this; a second copy is how one goes unbounded."""
        from _utils._stub_model import stub_delay_ms

        assert stub_delay_ms("stub/slow:250") == 250
        assert stub_delay_ms("stub/slow:999999") == 60_000
        assert stub_delay_ms("stub/slow:-5") == 0
        assert stub_delay_ms("stub/slow:abc") == 0
        assert stub_delay_ms("stub/echo") == 0
        assert stub_delay_ms("openai/gpt-4") == 0


# ===========================================================================
# 9. Cross-language parity — the same secrets, both sides of the wire
# ===========================================================================


class TestSecretPatternParity:
    """
    The browser redacts before sending; the stub scans after receiving.

    Two implementations in two languages is unavoidable — the check has to run
    where the text is, and the text is in both places. What is avoidable is
    them drifting apart, which would produce the worst possible outcome: a
    pattern the client believes it strips and the server never sees, or one the
    server reports that the client never removes. This test is the only thing
    standing between "one list" and "two lists that used to agree".
    """

    @staticmethod
    def _js_pattern_names() -> list[str]:
        """Read the pattern names out of the shipped JS, not a copy of them."""
        js = (
            pathlib.Path(__file__).resolve().parents[1]
            / "_static"
            / "ai-assistant.js"
        ).read_text(encoding="utf-8")
        start = js.index("var _SECRET_PATTERNS = [")
        end = js.index("];", start)
        block = js[start:end]
        import re as _re

        return _re.findall(r"name:\s*'([a-z0-9_]+)'", block)

    def test_the_two_lists_name_the_same_patterns(self):
        from _utils._stub_model import _SECRET_PATTERNS

        py_names = [name for name, _pattern in _SECRET_PATTERNS]
        assert sorted(self._js_pattern_names()) == sorted(py_names)

    def test_neither_list_is_empty(self):
        from _utils._stub_model import _SECRET_PATTERNS

        assert len(_SECRET_PATTERNS) >= 5
        assert len(self._js_pattern_names()) >= 5

    def test_names_are_unique_on_both_sides(self):
        from _utils._stub_model import _SECRET_PATTERNS

        py_names = [name for name, _pattern in _SECRET_PATTERNS]
        js_names = self._js_pattern_names()
        assert len(set(py_names)) == len(py_names)
        assert len(set(js_names)) == len(js_names)

    @pytest.mark.parametrize(
        ("name", "sample"),
        [
            ("aws_access_key_id", "AKIAIOSFODNN7EXAMPLE"),
            ("openai_key", "sk-abcdefghijklmnopqrstuvwx"),
            ("github_token", "ghp_abcdefghijklmnopqrstuvwxyz01"),
            ("huggingface_token", "hf_abcdefghijklmnopqrstuvwxyz01"),
            ("private_key_block", f"-----BEGIN RSA {'PRIVATE' + ' KEY'}-----"),
        ],
    )
    def test_shared_samples_are_detected_server_side(self, name, sample):
        """The same strings the JS harness redacts must be seen here."""
        findings = scan_for_secrets(f"context {sample} more")
        assert name in {f["pattern"] for f in findings}

    @pytest.mark.parametrize(
        "text",
        [
            "Set your key with export OPENAI_API_KEY=...",
            "Tokens look like sk-... in the docs.",
            "AKIA is the prefix used by AWS access key ids.",
            "A JWT has three dot-separated parts: header.payload.signature",
            "ghp_ tokens are classic personal access tokens.",
        ],
    )
    def test_documentation_prose_is_not_flagged_server_side(self, text):
        """
        Same false-positive corpus as the JS harness, same verdict.

        An ML library's docs discuss credential formats constantly. A scanner
        that fires on prose about keys is a scanner people disable.
        """
        assert scan_for_secrets(text) == []


def test_structured_contract_stub_reads_user_message() -> None:
    payload = {
        "contract": "scikitplot-chat-v1",
        "model": "stub/qa",
        "user_message": "ping",
        "context": {"page_text": "docs"},
        "max_tokens": 1000,
        "stream": False,
    }
    text, report = build_stub_reply("qa", "", payload, {})
    assert report["user_message_chars"] == len("ping")
    assert "No fixture matched" not in text
