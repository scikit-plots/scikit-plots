# scikitplot/corpus/_metrics/tests/test__llm.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The provider adapters: what each sends, what each reads, and what it refuses."""

from __future__ import annotations

import inspect
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import ClassVar

import pytest

from .._judge import LLMJudge
from .._llm import LLM, CacheError, Completion, IncompleteCompletionError, make_llm


class Recorder:
    """A fake SDK client: records keyword arguments, returns a canned response."""

    def __init__(self, response):
        self.calls = []
        self.response = response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(generate_content=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def openai_response(text='{"ok": true}', finish="stop", usage=(7, 3)):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=None if usage is None else SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)),
    )


def anthropic_response(blocks=(("text", '{"ok": true}'),), stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type=kind, text=text) for kind, text in blocks],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )


class FinishReason:
    """Stands in for google.genai.types.FinishReason (an enum member)."""

    def __init__(self, name):
        self.name = name


def gemini_response(text='{"ok": true}', finish="STOP"):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=FinishReason(finish))],
        usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=3, thoughts_token_count=5),
    )


class TestRequests:
    def test_openai(self):
        client = Recorder(openai_response())
        llm = make_llm("openai", "m", client=client, max_tokens=50, options={"temperature": 0})
        out = llm.complete("p", system="s")
        assert client.calls == [{
            "model": "m",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}],
            "max_completion_tokens": 50,
            "temperature": 0,
        }]
        assert (out.text, out.provider, out.finish_reason) == ('{"ok": true}', "openai", "stop")
        assert (out.input_tokens, out.output_tokens) == (7, 3)
        assert dict(out.usage) == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    def test_openai_sends_nothing_unrequested(self):
        client = Recorder(openai_response())
        make_llm("openai", "m", client=client).complete("p")
        assert client.calls == [{"model": "m", "messages": [{"role": "user", "content": "p"}]}]

    def test_anthropic(self):
        client = Recorder(anthropic_response(blocks=(("thinking", "hmm"), ("text", "a"), ("text", "b"))))
        out = make_llm("anthropic", "m", client=client, max_tokens=9).complete("p", system="s")
        assert client.calls == [{"model": "m", "max_tokens": 9, "messages": [{"role": "user", "content": "p"}], "system": "s"}]
        assert out.text == "ab"
        assert (out.input_tokens, out.output_tokens, out.finish_reason) == (7, 3, "end_turn")

    def test_gemini(self):
        client = Recorder(gemini_response())
        out = make_llm("gemini", "m", client=client, max_tokens=9, options={"seed": 1}).complete("p", system="s")
        assert client.calls == [{"model": "m", "contents": "p",
                                 "config": {"seed": 1, "system_instruction": "s", "max_output_tokens": 9}}]
        assert out.finish_reason == "STOP"
        assert (out.input_tokens, out.output_tokens) == (7, 3)
        assert out.usage["thoughts_token_count"] == 5

    def test_gemini_without_config(self):
        client = Recorder(gemini_response())
        make_llm("gemini", "m", client=client).complete("p")
        assert client.calls == [{"model": "m", "contents": "p"}]

    def test_callable(self):
        seen = []
        llm = make_llm(lambda prompt, system: seen.append((prompt, system)) or "x", model="local")
        assert llm.complete("p", system="s").text == "x"
        assert seen == [("p", "s")]
        assert llm.identity() == {"provider": "callable", "model": "local", "options": {}}
        assert make_llm(lambda p, s: "x", model="local", max_tokens=8).identity()["max_tokens"] == 8

    def test_callable_may_report_tokens(self):
        def fn(prompt, system):
            return Completion("x", "anything", "anything", input_tokens=4, output_tokens=2)

        out = make_llm(fn, model="local").complete("p")
        assert (out.provider, out.model, out.input_tokens, out.output_tokens) == ("callable", "local", 4, 2)


class TestFailures:
    @pytest.mark.parametrize(
        ("provider", "response"),
        [
            ("openai", openai_response(finish="length")),
            ("openai", openai_response(finish="content_filter")),
            ("anthropic", anthropic_response(stop="max_tokens")),
            ("anthropic", anthropic_response(stop="refusal")),
            ("gemini", gemini_response(finish="MAX_TOKENS")),
            ("gemini", gemini_response(finish="SAFETY")),
        ],
    )
    def test_incomplete_raises_with_the_completion(self, provider, response):
        llm = make_llm(provider, "m", client=Recorder(response), max_tokens=5)
        with pytest.raises(IncompleteCompletionError, match="not a normal end") as info:
            llm.complete("p")
        assert info.value.completion.output_tokens == 3

    def test_no_finish_reason_is_accepted(self):
        out = make_llm("openai", "m", client=Recorder(openai_response(finish=None, usage=None))).complete("p")
        assert (out.finish_reason, out.input_tokens) == (None, None)

    @pytest.mark.parametrize(
        ("provider", "response"),
        [
            ("openai", openai_response(text=None)),
            ("anthropic", anthropic_response(blocks=(("tool_use", None),))),
            ("gemini", gemini_response(text=None)),
        ],
    )
    def test_no_text_is_a_type_error(self, provider, response):
        with pytest.raises(TypeError, match="no text"):
            make_llm(provider, "m", client=Recorder(response), max_tokens=5).complete("p")

    def test_provider_errors_propagate(self):
        class Broken(Recorder):
            def _create(self, **kwargs):
                raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            make_llm("openai", "m", client=Broken(None)).complete("p")


class TestConstruction:
    @pytest.mark.parametrize(
        ("args", "kwargs", "error", "match"),
        [
            (("nope", "m"), {}, ValueError, "provider"),
            (("openai",), {}, ValueError, "model"),
            (("openai", ""), {}, ValueError, "model"),
            (("anthropic", "m"), {}, ValueError, "max_tokens"),
            (("openai", "m"), {"max_tokens": 0}, ValueError, "max_tokens"),
            (("openai", "m"), {"max_tokens": True}, ValueError, "max_tokens"),
            (("openai", "m"), {"options": {"messages": []}}, ValueError, "may not set"),
            (("anthropic", "m"), {"max_tokens": 1, "options": {"system": "x"}}, ValueError, "may not set"),
            (("gemini", "m"), {"options": {"max_output_tokens": 3}}, ValueError, "may not set"),
            (("openai", "m"), {"options": {"x": object()}}, TypeError, "JSON-compatible"),
            (("openai", "m"), {"options": {1: 2}}, TypeError, "str keys"),
            (("openai", "m"), {"cache": []}, TypeError, "mutable mapping"),
            (("openai", "m"), {"clock": 3}, TypeError, "clock"),
        ],
    )
    def test_refusals(self, args, kwargs, error, match):
        with pytest.raises(error, match=match):
            make_llm(*args, client=Recorder(None), **kwargs)

    def test_arguments_the_client_cannot_take_fail_at_construction(self):
        class OldCompletions:
            def create(self, *, model, messages, max_tokens=None, temperature=None):
                raise AssertionError("never called")

        old = SimpleNamespace(chat=SimpleNamespace(completions=OldCompletions()))
        with pytest.raises(ValueError, match=r"does not accept \['max_completion_tokens'\].*1.45"):
            make_llm("openai", "m", client=old, max_tokens=5)
        with pytest.raises(ValueError, match=r"does not accept \['seed'\]"):
            make_llm("openai", "m", client=old, options={"seed": 1})
        llm = make_llm("openai", "m", client=old, options={"max_tokens": 5, "temperature": 0})
        assert llm.options == {"max_tokens": 5, "temperature": 0}

    def test_callable_refuses_options(self):
        with pytest.raises(ValueError, match="named providers"):
            make_llm(lambda p, s: "x", options={"temperature": 0})

    @pytest.mark.parametrize(("provider", "module"), [("openai", "openai"), ("anthropic", "anthropic"), ("gemini", "google")])
    def test_missing_sdk_names_the_package(self, monkeypatch, provider, module):
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(ImportError, match="pip install"):
            make_llm(provider, "m", max_tokens=5)

    def test_prompt_types(self):
        llm = make_llm(lambda p, s: "x")
        with pytest.raises(TypeError, match="prompt"):
            llm.complete(b"p")
        with pytest.raises(TypeError, match="system"):
            llm.complete("p", system=1)

    def test_repr_and_identity_hold_no_client(self):
        client = Recorder(None)
        client.api_key = "sk-secret"
        llm = make_llm("openai", "m", client=client, max_tokens=5, options={"seed": 1})
        assert repr(llm) == "LLM(provider='openai', model='m', max_tokens=5, options={'seed': 1})"
        assert "secret" not in json.dumps(llm.identity()) + repr(llm)
        assert isinstance(llm, LLM)


class TestCache:
    def test_hit_skips_the_call(self):
        client = Recorder(openai_response())
        cache = {}
        llm = make_llm("openai", "m", client=client, cache=cache)
        first, second = llm.complete("p"), llm.complete("p")
        assert len(client.calls) == 1
        assert (first.cached, second.cached, second.latency_s, second.text) == (False, True, 0.0, first.text)
        assert len(cache) == 1

    def test_key_covers_the_request(self):
        llm = make_llm("openai", "m", client=Recorder(None), options={"seed": 1})
        other = make_llm("openai", "m", client=Recorder(None), options={"seed": 2})
        keys = {llm.cache_key("p"), llm.cache_key("p", "s"), llm.cache_key("q"), other.cache_key("p")}
        assert len(keys) == 4
        assert all(len(k) == 64 for k in keys)

    def test_incomplete_is_not_cached(self):
        cache = {}
        llm = make_llm("openai", "m", client=Recorder(openai_response(finish="length")), cache=cache)
        with pytest.raises(IncompleteCompletionError):
            llm.complete("p")
        assert cache == {}

    def test_latency_uses_the_clock(self):
        ticks = iter([10.0, 12.5])
        out = make_llm(lambda p, s: "x", clock=lambda: next(ticks)).complete("p")
        assert out.latency_s == 2.5

    def test_concurrent_use(self):
        llm = make_llm(lambda p, s: p.upper(), cache={})
        results = []
        threads = [threading.Thread(target=lambda i=i: results.append(llm.complete(f"p{i % 4}").text)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) == sorted(f"P{i % 4}" for i in range(32))


def test_completion_round_trip():
    c = Completion("t", "openai", "m", "stop", 1, 2, 0.5, False, {"prompt_tokens": 1})
    assert Completion.from_dict(json.loads(json.dumps(c.to_dict()))) == c


# --------------------------------------------------------------------------
# The real SDKs, over HTTP, against a local server (skipped without them).
# --------------------------------------------------------------------------

class _Server(BaseHTTPRequestHandler):
    seen: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            out = {"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}}
        elif self.path.endswith("/messages"):
            out = {"id": "m", "type": "message", "role": "assistant", "model": body["model"],
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 11, "output_tokens": 5}}
        else:
            out = {"candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def local_server(monkeypatch):
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    _Server.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.mark.thread_unsafe  # environment variables are process-wide
@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_real_sdk_over_http(local_server, provider):
    if provider == "openai":
        sdk = pytest.importorskip("openai")
        client = sdk.OpenAI(base_url=local_server + "/v1", api_key="test-not-a-key", max_retries=0)
    elif provider == "anthropic":
        sdk = pytest.importorskip("anthropic")
        client = sdk.Anthropic(base_url=local_server, api_key="test-not-a-key", max_retries=0)
    else:
        genai = pytest.importorskip("google.genai")
        client = genai.Client(api_key="test-not-a-key", http_options={"base_url": local_server})
    limit = {"max_tokens": 64}
    if provider == "openai" and "max_completion_tokens" not in inspect.signature(client.chat.completions.create).parameters:
        with pytest.raises(ValueError, match=r"1\.45"):  # openai < 1.45
            make_llm(provider, "model-x", client=client, max_tokens=64)
        limit = {"options": {"max_tokens": 64}}
    out = make_llm(provider, "model-x", client=client, **limit).complete("Rate it.", system="Judge.")
    path, body = _Server.seen[-1]
    assert (out.text, out.input_tokens, out.output_tokens) == ("ok", 11, 5)
    if provider == "openai":
        assert path == "/v1/chat/completions"
        assert body == {"model": "model-x", ("max_tokens" if "options" in limit else "max_completion_tokens"): 64,
                        "messages": [{"role": "system", "content": "Judge."}, {"role": "user", "content": "Rate it."}]}
    elif provider == "anthropic":
        assert path == "/v1/messages"
        assert body == {"model": "model-x", "max_tokens": 64, "system": "Judge.",
                        "messages": [{"role": "user", "content": "Rate it."}]}
    else:
        assert path.endswith("/models/model-x:generateContent")
        assert body["contents"] == [{"parts": [{"text": "Rate it."}], "role": "user"}]
        assert body["generationConfig"] == {"maxOutputTokens": 64}
        assert body["systemInstruction"]["parts"] == [{"text": "Judge."}]
    assert "test-not-a-key" not in json.dumps(body)
    assert os.environ.get("NO_PROXY") == "127.0.0.1,localhost"


class TestReviewFixes:
    """Increment 2b independent review."""

    @pytest.mark.parametrize(
        ("provider", "response"),
        [
            ("openai", openai_response(text=None, finish="length")),
            ("openai", openai_response(text=None, finish="content_filter")),
            ("anthropic", anthropic_response(blocks=(("thinking", "spent the budget"),), stop="max_tokens")),
            ("anthropic", anthropic_response(blocks=(), stop="max_tokens")),
            ("gemini", gemini_response(text=None, finish="MAX_TOKENS")),
            ("gemini", gemini_response(text=None, finish="SAFETY")),
        ],
    )
    def test_cut_off_without_text_is_incomplete_not_a_type_error(self, provider, response):
        """Review #2: the text check ran before the finish-reason check."""
        llm = make_llm(provider, "m", client=Recorder(response), max_tokens=5)
        with pytest.raises(IncompleteCompletionError) as info:
            llm.complete("p")
        assert info.value.completion.text == ""
        assert info.value.completion.output_tokens == 3

    @pytest.mark.parametrize("finish", ["length", "max_tokens", "MAX_TOKENS", "content_filter"])
    def test_callable_completion_is_checked_too(self, finish):
        """Review #5: a callable's own finish reason bypassed the check."""
        llm = make_llm(lambda p, s: Completion("{}", "x", "x", finish_reason=finish), model="local", cache={})
        with pytest.raises(IncompleteCompletionError):
            llm.complete("p")

    @pytest.mark.parametrize("finish", [None, "stop", "end_turn", "stop_sequence", "STOP"])
    def test_callable_normal_ends(self, finish):
        assert make_llm(lambda p, s: Completion("ok", "x", "x", finish_reason=finish)).complete("p").text == "ok"

    def test_a_failing_cache_is_a_cache_error(self):
        """Review #1: a thread-bound cache failed as a 'provider' error in every judgement."""

        class ThreadBound(dict):
            def __init__(self):
                super().__init__()
                self.owner = threading.get_ident()

            def get(self, key, default=None):
                if threading.get_ident() != self.owner:
                    raise RuntimeError("objects created in a thread can only be used in that same thread")
                return super().get(key, default)

        llm = make_llm(lambda p, s: '{"score": 4, "reason": "r"}', cache=ThreadBound())
        assert llm.complete("p").text  # same thread: fine
        errors = []
        thread = threading.Thread(target=lambda: _capture(errors, lambda: LLMJudge(llm, on_error="record").answer_relevance("q", "a")))
        thread.start()
        thread.join()
        assert len(errors) == 1
        assert isinstance(errors[0], CacheError)
        assert "same thread" in str(errors[0])

    def test_header_values_are_redacted(self):
        """Review #6: a gateway key in extra_headers reached repr, identity and reports."""
        llm = make_llm("openai", "m", client=Recorder(None),
                       options={"extra_headers": {"api-key": "sk-secret"}, "extra_query": {"key": "q-secret"}, "seed": 1})
        shown = repr(llm) + json.dumps(llm.identity())
        assert "secret" not in shown
        assert llm.identity()["options"] == {"extra_headers": {"api-key": "<redacted>"},
                                             "extra_query": {"key": "<redacted>"}, "seed": 1}
        other = make_llm("openai", "m", client=Recorder(None),
                         options={"extra_headers": {"api-key": "sk-other"}, "extra_query": {"key": "q-secret"}, "seed": 1})
        assert llm.cache_key("p") != other.cache_key("p")
        assert llm.options["extra_headers"] == {"api-key": "sk-secret"}  # still sent

    def test_token_limit_set_once(self):
        """Review #7: max_tokens= and options['max_tokens'] were both sent."""
        with pytest.raises(ValueError, match="once"):
            make_llm("openai", "m", client=Recorder(None), max_tokens=5, options={"max_tokens": 5})

    def test_gemini_options_are_config_fields(self):
        """Review #7: a misspelt Gemini option failed on every call instead of at construction."""
        pytest.importorskip("google.genai")
        with pytest.raises(ValueError, match="GenerateContentConfig"):
            make_llm("gemini", "m", client=Recorder(None), options={"temperatures": 0})
        assert make_llm("gemini", "m", client=Recorder(None), options={"temperature": 0}).options == {"temperature": 0}

    def test_cache_key_names_the_endpoint(self):
        """Review #8: two endpoints serving the same model name shared cache entries."""
        first, second = Recorder(None), Recorder(None)
        first.base_url, second.base_url = "https://gateway-a/v1", "https://gateway-b/v1"
        a = make_llm("openai", "m", client=first)
        b = make_llm("openai", "m", client=second)
        assert a.cache_key("p") != b.cache_key("p")


def _capture(errors, func):
    try:
        func()
    except Exception as exc:  # noqa: BLE001 - the test inspects it
        errors.append(exc)
