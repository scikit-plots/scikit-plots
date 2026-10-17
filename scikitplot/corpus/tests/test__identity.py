# corpus/tests/test__identity.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Durable identities (R2-3, decision D3): 64-hex SHA-256 everywhere, opt-in UUIDv7.

Covers the RFC 9562 layout and monotonicity of the UUIDv7 generator (same
millisecond, clock steps back, counter rollover, threads, fork), the id-scheme
vocabulary, the batch assignment with parent links, the width of every durable
SHA-256 identity, the refusal of payloads written before the widening, and both
producer paths (builder and pipeline).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

import pytest

from scikitplot.corpus import (
    IDENTITY_HEX_WIDTH,
    BuilderConfig,
    CorpusBuilder,
    CorpusDocument,
    CorpusPipeline,
    DefaultFilter,
    DocIdScheme,
    EmbeddingManifest,
    ExportFormat,
    IncompatibleSchemaError,
    IndexGeneration,
    SentenceChunker,
    SourceType,
    UUID7Generator,
    assign_doc_ids,
    doc_id_scheme_of,
    uuid7,
    uuid7_unix_ms,
)
from scikitplot.corpus._identity import require_full_width
from scikitplot.corpus._types import PreprocessingStep, PreprocessingTrace

_HEX64 = re.compile(r"[0-9a-f]{64}")
#: RFC 9562 appendix A.6, "Example of a UUIDv7 Value".
_RFC_VECTOR = "017f22e2-79b0-7cc3-98c4-dc0c0c07398f"


class _Bytes:
    """A deterministic ``random_bytes`` source that records what it handed out."""

    def __init__(self, fill: int = 0x5A) -> None:
        self.fill = fill
        self.calls: list[int] = []

    def __call__(self, n: int) -> bytes:
        self.calls.append(n)
        return bytes([self.fill]) * n


def _fields(value: uuid.UUID) -> dict[str, int]:
    """Split a UUIDv7 into its RFC 9562 section 5.7 fields."""
    n = value.int
    return {
        "unix_ts_ms": n >> 80,
        "ver": (n >> 76) & 0xF,
        "rand_a": (n >> 64) & 0xFFF,
        "var": (n >> 62) & 0b11,
        "rand_b": n & ((1 << 62) - 1),
    }


# ---------------------------------------------------------------------------
# RFC 9562 layout
# ---------------------------------------------------------------------------


class TestLayout:
    def test_fields_match_rfc_9562_section_5_7(self):
        make = UUID7Generator(
            clock_ns=lambda: 1_645_557_742_000_123_456, random_bytes=_Bytes()
        )
        value = make()
        f = _fields(value)
        assert f["unix_ts_ms"] == 1_645_557_742_000
        assert f["ver"] == 7
        assert f["var"] == 0b10
        assert value.version == 7
        assert value.variant == uuid.RFC_4122
        # 80 random bits 0x5a...: counter = bits 32..72 with the top bit of 42
        # clear, tail = low 32 bits.
        rand = int.from_bytes(b"\x5a" * 10, "big")
        counter = (rand >> 32) & ((1 << 41) - 1)
        assert f["rand_a"] == counter >> 30
        assert f["rand_b"] >> 32 == counter & ((1 << 30) - 1)
        assert value.int & 0xFFFF_FFFF == rand & 0xFFFF_FFFF

    def test_the_rfc_test_vector_decodes(self):
        # RFC 9562 appendix A.6: the shared test time 2022-02-22 19:22:22 UTC.
        assert (
            uuid7_unix_ms("017f22e2-79b0-7cc3-98c4-dc0c0c07398f") == 1_645_557_742_000
        )

    def test_the_text_form_is_canonical_lowercase(self):
        text = str(uuid7())
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", text
        )

    def test_the_module_generator_reads_the_real_clock(self):
        before = time.time_ns() // 1_000_000
        value = uuid7()
        after = time.time_ns() // 1_000_000
        # The process generator may run ahead after a burst (rollover), never
        # behind the clock.
        assert uuid7_unix_ms(value) >= before
        assert uuid7_unix_ms(value) <= after + 1


# ---------------------------------------------------------------------------
# Monotonicity (RFC 9562, section 6.2)
# ---------------------------------------------------------------------------


class TestMonotonic:
    def test_a_burst_within_one_millisecond_counts_up(self):
        make = UUID7Generator(clock_ns=lambda: 7_000_000)
        values = [make() for _ in range(2_000)]
        assert {uuid7_unix_ms(v) for v in values} == {7}
        assert values == sorted(values)
        assert len(set(values)) == len(values)
        # The counter (rand_a plus the first 30 bits of rand_b) steps by one.
        counters = [
            (v.int >> 64 & 0xFFF) << 30 | (v.int >> 32 & 0x3FFF_FFFF) for v in values
        ]
        assert all(b - a == 1 for a, b in zip(counters, counters[1:]))

    def test_the_order_holds_as_int_bytes_and_text(self):
        values = [uuid7() for _ in range(5_000)]
        assert [v.int for v in values] == sorted(v.int for v in values)
        assert [v.bytes for v in values] == sorted(v.bytes for v in values)
        assert [str(v) for v in values] == sorted(str(v) for v in values)
        assert len(set(values)) == len(values)

    def test_a_clock_step_backwards_keeps_the_order(self):
        ticks = iter(
            [9_000_000, 9_000_000, 3_000_000, 2_000_000, 9_000_000, 10_000_000]
        )
        make = UUID7Generator(clock_ns=lambda: next(ticks))
        values = [make() for _ in range(6)]
        assert values == sorted(values)
        stamps = [uuid7_unix_ms(v) for v in values]
        assert stamps == [9, 9, 9, 9, 9, 10]

    def test_counter_rollover_advances_the_timestamp(self):
        # Every seed is all ones: counter = 2**41 - 1, so 2**41 increments fit
        # before 2**42 - 1 is passed. Start the generator just below the top.
        make = UUID7Generator(clock_ns=lambda: 4_000_000, random_bytes=_Bytes(0xFF))
        first = make()
        make._last_counter = (1 << 42) - 2
        top = make()
        rolled = make()
        assert first < top < rolled
        assert uuid7_unix_ms(top) == 4
        assert uuid7_unix_ms(rolled) == 5  # the timestamp ran ahead by one
        later = make()  # the clock still reads 4 ms: stay at 5, count on
        assert rolled < later and uuid7_unix_ms(later) == 5

    def test_threads_never_share_an_identifier(self):
        make = UUID7Generator()
        per_thread: list[list[uuid.UUID]] = [[] for _ in range(8)]

        def work(slot: list[uuid.UUID]) -> None:
            for _ in range(3_000):
                slot.append(make())

        threads = [threading.Thread(target=work, args=(slot,)) for slot in per_thread]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        everything = [v for slot in per_thread for v in slot]
        assert len(set(everything)) == len(everything)
        for slot in per_thread:
            assert slot == sorted(slot)

    @staticmethod
    def _run_forking_child(code: str) -> list[str]:
        """Run ``code`` in a fresh single-threaded interpreter; return its words.

        The fork happens in that child, never in the multi-threaded test
        process.
        """
        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.split()

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    @pytest.mark.parametrize("which", ["module", "private"])
    def test_a_forked_child_starts_a_new_sequence(self, which):
        # Parent and child read the same clock. The child is given a random
        # source of zeros: a reseed then gives counter 0, while continuing the
        # parent's sequence would give the parent's counter plus one.
        make = (
            "m._DEFAULT_GENERATOR"
            if which == "module"
            else "m.UUID7Generator(clock_ns=lambda: 1_000_000_000)"
        )
        code = (
            "import os\n"
            "from scikitplot.corpus import _identity as m\n"
            f"g = {make}\n"
            "g._clock_ns = lambda: 1_000_000_000\n"
            "g()\n"
            "r, w = os.pipe()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.close(r)\n"
            "    g._random_bytes = lambda n: bytes(n)\n"
            "    v = g()\n"
            "    os.write(w, str(v).encode())\n"
            "    os._exit(0)\n"
            "os.close(w)\n"
            "before = g._last_counter\n"
            "os.waitpid(pid, 0)\n"
            "print(os.read(r, 200).decode(), before)\n"
        )
        child_value, parent_counter = self._run_forking_child(code)
        child = uuid.UUID(child_value)
        child_counter = (child.int >> 64 & 0xFFF) << 30 | (
            child.int >> 32 & 0x3FFF_FFFF
        )
        assert child_counter == 0
        assert int(parent_counter) != 0  # so 0 cannot be the parent's next value

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    def test_a_lock_held_at_the_fork_does_not_hang_the_child(self):
        code = (
            "import os, signal, threading\n"
            "from scikitplot.corpus import _identity as m\n"
            "g = m.UUID7Generator()\n"
            "g()\n"
            "held, release = threading.Event(), threading.Event()\n"
            "def hold():\n"
            "    with g._lock:\n"
            "        held.set()\n"
            "        release.wait()\n"
            "t = threading.Thread(target=hold)\n"
            "t.start()\n"
            "held.wait()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    signal.alarm(10)\n"
            "    g()\n"
            "    os._exit(0)\n"
            "release.set()\n"
            "t.join()\n"
            "_, status = os.waitpid(pid, 0)\n"
            "print(os.waitstatus_to_exitcode(status) if hasattr(os, 'waitstatus_to_exitcode') else status)\n"
        )
        (exit_code,) = self._run_forking_child(code)
        assert exit_code == "0"


# ---------------------------------------------------------------------------
# Generator contract
# ---------------------------------------------------------------------------


class TestGeneratorContract:
    def test_a_short_random_source_is_refused(self):
        make = UUID7Generator(
            clock_ns=lambda: 1_000_000, random_bytes=lambda n: b"\x00"
        )
        with pytest.raises(ValueError, match="must return 10 bytes"):
            make()

    def test_a_clock_before_the_epoch_is_refused(self):
        make = UUID7Generator(clock_ns=lambda: -5_000_000)
        with pytest.raises(OverflowError, match="Unix epoch"):
            make()

    def test_a_timestamp_beyond_48_bits_is_refused(self):
        make = UUID7Generator(clock_ns=lambda: (1 << 48) * 1_000_000)
        with pytest.raises(OverflowError, match="48 bits"):
            make()

    def test_a_failed_call_leaves_the_sequence_unchanged(self):
        good = iter([b"\x01" * 10])
        make = UUID7Generator(
            clock_ns=lambda: 2_000_000, random_bytes=lambda n: next(good)
        )
        first = make()
        with pytest.raises(StopIteration):
            make()  # the tail's random source fails
        assert make._last_counter == (first.int >> 64 & 0xFFF) << 30 | (
            first.int >> 32 & 0x3FFF_FFFF
        )

    def test_reset_forgets_the_sequence(self):
        make = UUID7Generator(clock_ns=lambda: 2_000_000)
        make()
        make.reset()
        assert make._last_ms == -1
        assert make._last_counter == 0


# ---------------------------------------------------------------------------
# Scheme vocabulary
# ---------------------------------------------------------------------------


class TestSchemes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0" * 64, DocIdScheme.SHA256),
            ("a" * 64, DocIdScheme.SHA256),
            ("A" * 64, None),
            ("0" * 63, None),
            ("0" * 65, None),
            ("0" * 16, None),
            ("017f22e2-79b0-7cc3-98c4-dc0c0c07398f", DocIdScheme.UUID7),
            ("017F22E2-79B0-7CC3-98C4-DC0C0C07398F", None),
            ("017f22e279b07cc398c4dc0c0c07398f", None),
            (str(uuid.UUID(int=(4 << 76) | (0b10 << 62))), None),  # version 4
            ("017f22e2-79b0-7cc3-c8c4-dc0c0c07398f", None),  # variant 0b110
            ("mechanism", None),
            ("", None),
            (b"0" * 64, None),
            (None, None),
            (12, None),
        ],
    )
    def test_the_scheme_is_read_from_the_exact_form(self, value, expected):
        assert doc_id_scheme_of(value) is expected

    def test_every_minted_id_is_recognised(self):
        assert all(
            doc_id_scheme_of(str(uuid7())) is DocIdScheme.UUID7 for _ in range(500)
        )

    def test_coerce(self):
        assert DocIdScheme.coerce("uuid7") is DocIdScheme.UUID7
        assert DocIdScheme.coerce(DocIdScheme.SHA256) is DocIdScheme.SHA256
        with pytest.raises(ValueError, match="'sha256', 'uuid7'"):
            DocIdScheme.coerce("uuid4")
        with pytest.raises(TypeError, match="DocIdScheme or str"):
            DocIdScheme.coerce(7)

    def test_unix_ms_refuses_what_is_not_a_uuid7(self):
        with pytest.raises(ValueError, match="not a UUID version 7"):
            uuid7_unix_ms(uuid.uuid4())
        with pytest.raises(ValueError, match="is not a UUID"):
            uuid7_unix_ms("not-a-uuid")
        with pytest.raises(TypeError):
            uuid7_unix_ms(1234)
        value = uuid7()
        assert uuid7_unix_ms(value) == uuid7_unix_ms(str(value))


# ---------------------------------------------------------------------------
# Batch assignment
# ---------------------------------------------------------------------------


def _family():
    parent = CorpusDocument.create("a.txt", 0, "Part one.")
    child = CorpusDocument.create("a.txt", 1, "Part one, section one.").replace(
        parent_doc_id=parent.doc_id
    )
    grandchild = CorpusDocument.create("a.txt", 2, "A detail.").replace(
        parent_doc_id=child.doc_id
    )
    return [parent, child, grandchild]


class TestAssign:
    def test_sha256_keeps_the_documents(self):
        rows = _family()
        out = assign_doc_ids(rows)
        assert all(a is b for a, b in zip(out, rows))
        assert assign_doc_ids(iter(rows), "sha256") == rows

    def test_uuid7_names_every_document_in_order_and_relinks_parents(self):
        rows = _family()
        out = assign_doc_ids(rows, "uuid7")
        ids = [d.doc_id for d in out]
        assert all(doc_id_scheme_of(i) is DocIdScheme.UUID7 for i in ids)
        assert ids == sorted(ids) and len(set(ids)) == 3
        assert out[0].parent_doc_id is None
        assert out[1].parent_doc_id == out[0].doc_id
        assert out[2].parent_doc_id == out[1].doc_id
        # Content identity is untouched; the inputs are unchanged.
        assert [d.content_hash for d in out] == [d.content_hash for d in rows]
        assert [d.doc_id for d in rows] == [d.doc_id for d in _family()]

    def test_a_parent_after_its_child_is_relinked_too(self):
        rows = list(reversed(_family()))
        out = assign_doc_ids(rows, "uuid7")
        assert out[0].parent_doc_id == out[1].doc_id
        assert out[1].parent_doc_id == out[2].doc_id

    def test_a_parent_outside_the_batch_is_kept_verbatim(self):
        # An external parent names a document stored elsewhere (an earlier
        # build, another source); this call cannot rename it.
        family = _family()
        out = assign_doc_ids(family[1:], "uuid7")
        assert out[0].parent_doc_id == family[0].doc_id
        assert out[1].parent_doc_id == out[0].doc_id

    def test_an_ambiguous_parent_is_refused(self):
        parent, child, _ = _family()
        twin = parent.replace(text="Part one.")  # same doc_id as parent
        with pytest.raises(ValueError, match="ambiguous"):
            assign_doc_ids([parent, twin, child], "uuid7")

    def test_duplicates_without_children_get_distinct_ids(self):
        doc = CorpusDocument.create("a.txt", 0, "Same.")
        out = assign_doc_ids([doc, doc], "uuid7")
        assert out[0].doc_id != out[1].doc_id

    def test_a_generator_that_is_not_uuid7_is_refused(self):
        with pytest.raises(ValueError, match="version-7"):
            assign_doc_ids(_family(), "uuid7", generator=uuid.uuid4)

    @pytest.mark.parametrize("step", [0, -1])
    def test_a_generator_that_repeats_or_steps_back_is_refused(self, step):
        start = uuid.UUID(_RFC_VECTOR).int  # counter_lo far from 0: no borrow

        def mint(state={"n": 0}):  # noqa: B006 - deliberate per-test counter
            state["n"] += 1
            return uuid.UUID(int=start + step * state["n"] * (1 << 32))

        with pytest.raises(ValueError, match="strictly increase"):
            assign_doc_ids(_family(), "uuid7", generator=mint)

    def test_a_private_generator_is_used(self):
        make = UUID7Generator(clock_ns=lambda: 1_000_000_000)
        out = assign_doc_ids(_family(), "uuid7", generator=make)
        assert {uuid7_unix_ms(d.doc_id) for d in out} == {1_000}

    def test_an_unknown_scheme_is_refused(self):
        with pytest.raises(ValueError, match="unknown doc_id_scheme"):
            assign_doc_ids([], "sha1")
        assert assign_doc_ids([], "uuid7") == []


# ---------------------------------------------------------------------------
# Widths of every durable SHA-256 identity
# ---------------------------------------------------------------------------


def _v2_doc_id(path: str, index: int, text: str, source_type: SourceType) -> str:
    """The id a build before R2-3 assigned (16 hex over a 32-hex content hash)."""
    content = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    raw = f"v2:{source_type.value}:{path}:{index}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class TestWidths:
    def test_doc_id_and_content_hash_are_the_full_digest(self):
        doc = CorpusDocument.create(
            "f.txt", 3, "Hello world.", source_type=SourceType.BOOK
        )
        assert IDENTITY_HEX_WIDTH == 64
        assert _HEX64.fullmatch(doc.doc_id)
        assert doc.content_hash == hashlib.sha256(b"Hello world.").hexdigest()
        assert doc_id_scheme_of(doc.doc_id) is DocIdScheme.SHA256

    def test_doc_id_is_deterministic_and_shares_no_prefix_with_v2(self):
        args = ("f.txt", 3, "Hello world.", SourceType.BOOK)
        assert CorpusDocument.make_doc_id(*args) == CorpusDocument.make_doc_id(*args)
        assert not CorpusDocument.make_doc_id(*args).startswith(_v2_doc_id(*args))

    def test_the_raw_media_sentinel_is_full_width(self):
        assert CorpusDocument.make_content_hash() == "0" * 64
        assert (
            CorpusDocument.make_content_hash(raw_bytes=b"")
            == hashlib.sha256(b"").hexdigest()
        )

    def test_manifest_and_generation_fingerprints_are_the_full_digest(self):
        manifest = EmbeddingManifest(provider="p", model="m", dimension=8)
        generation = IndexGeneration("3.0", manifest.fingerprint, "2-abc", "bruteforce")
        assert _HEX64.fullmatch(manifest.fingerprint)
        assert _HEX64.fullmatch(generation.fingerprint)

    def test_the_preprocessing_trace_fingerprint_is_sha256(self):
        step = PreprocessingStep(
            name="nfc", description="NFC", changed=False, char_delta=0
        )
        trace = PreprocessingTrace.build(raw_text="x", steps=[step], final_text="x")
        assert trace.pipeline_fingerprint == hashlib.sha256(b"nfc").hexdigest()


# ---------------------------------------------------------------------------
# Payloads written before the widening are refused, with the remedy
# ---------------------------------------------------------------------------


class TestLegacyPayloads:
    def test_a_16_hex_generation_is_refused_with_rebuild_advice(self):
        data = IndexGeneration("2.0", None, "2-abc", "bruteforce").to_dict()
        data["fingerprint"] = data["fingerprint"][:16]
        with pytest.raises(ValueError, match=r"16 hex characters.*rebuild"):
            IndexGeneration.from_dict(data)

    def test_a_16_hex_manifest_is_refused_with_rebuild_advice(self):
        data = EmbeddingManifest(provider="p", model="m").to_dict()
        data["fingerprint"] = "0123456789abcdef"
        with pytest.raises(ValueError, match="before identities were widened"):
            EmbeddingManifest.from_dict(data)

    def test_a_tampered_full_width_fingerprint_is_still_called_altered(self):
        data = EmbeddingManifest(provider="p", model="m").to_dict()
        data["fingerprint"] = "0" * 64
        with pytest.raises(ValueError, match="altered"):
            EmbeddingManifest.from_dict(data)

    @pytest.mark.parametrize("declared", ["", "XYZ", "0" * 65, 7, None, "0" * 16 + "-"])
    def test_a_malformed_fingerprint_is_refused(self, declared):
        with pytest.raises(ValueError, match="not 64 lowercase hex"):
            require_full_width(declared, "owner")

    def test_a_schema_2_document_payload_is_refused(self):
        payload = CorpusDocument.create("f.txt", 0, "Hi.").to_dict()
        assert payload["schema_version"] == "3.0"
        payload["schema_version"] = "2.0"
        payload["doc_id"] = "0123456789abcdef"
        with pytest.raises(ValueError, match="incompatible with this build"):
            CorpusDocument.from_dict(payload)

    def test_the_current_payload_round_trips(self):
        doc = CorpusDocument.create("f.txt", 0, "Hi.")
        assert CorpusDocument.from_dict(json.loads(json.dumps(doc.to_dict()))) == doc


# ---------------------------------------------------------------------------
# Producer paths: builder and pipeline (Rule 82)
# ---------------------------------------------------------------------------


_TEXT = "Alpha sentence one here. Beta sentence two here. Gamma sentence three here."


def _builder_config(scheme):
    return BuilderConfig(
        filter_kwargs={"min_words": 1, "min_chars": 1}, doc_id_scheme=scheme
    )


class TestProducers:
    def test_the_builder_default_is_content_addressed_and_idempotent(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_bytes(_TEXT.encode("utf-8"))
        with CorpusBuilder(_builder_config("sha256")) as builder:
            first = [d.doc_id for d in builder.build(path).documents]
        with CorpusBuilder(
            BuilderConfig(filter_kwargs={"min_words": 1, "min_chars": 1})
        ) as builder:
            second = builder.build(path).documents
        assert len(first) == 3
        assert first == [d.doc_id for d in second]
        for doc in second:
            assert doc.doc_id == CorpusDocument.make_doc_id(
                doc.input_path, doc.chunk_index, doc.text, doc.source_type
            )

    def test_the_builder_mints_uuid7_in_build_order(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_bytes(_TEXT.encode("utf-8"))
        with CorpusBuilder(_builder_config("uuid7")) as builder:
            first = builder.build(path).documents
            added = builder.add(path).documents
        with CorpusBuilder(_builder_config("uuid7")) as builder:
            again = builder.build(path).documents
        ids = [d.doc_id for d in first]
        assert all(doc_id_scheme_of(i) is DocIdScheme.UUID7 for i in ids)
        assert ids == sorted(ids)
        assert [d.chunk_index for d in first] == [0, 1, 2]
        # add() keeps the old ids and mints newer ones for the new documents.
        assert [d.doc_id for d in added[:3]] == ids
        assert min(d.doc_id for d in added[3:]) > max(ids)
        # A rebuild mints new ids; the content identity is what stays.
        assert not set(ids) & {d.doc_id for d in again}
        assert [d.content_hash for d in first] == [d.content_hash for d in again]

    def test_the_builder_refuses_an_unknown_scheme_at_construction(self):
        with pytest.raises(ValueError, match="unknown doc_id_scheme 'uuid8'"):
            BuilderConfig(doc_id_scheme="uuid8")
        assert BuilderConfig().doc_id_scheme is DocIdScheme.SHA256

    def test_the_index_and_search_see_the_minted_ids(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_bytes(_TEXT.encode("utf-8"))
        config = BuilderConfig(
            filter_kwargs={"min_words": 1, "min_chars": 1},
            build_index=True,
            doc_id_scheme="uuid7",
        )
        with CorpusBuilder(config) as builder:
            result = builder.build(path)
            hits = builder.search("Beta sentence", match_mode="keyword")
        ids = {d.doc_id for d in result.documents}
        assert hits and all(h.doc.doc_id in ids for h in hits)

    @pytest.mark.parametrize("scheme", ["sha256", "uuid7"])
    def test_the_pipeline_applies_the_scheme(self, tmp_path, scheme):
        path = tmp_path / "a.txt"
        path.write_bytes(_TEXT.encode("utf-8"))
        pipeline = CorpusPipeline(
            chunker=SentenceChunker(),
            filter_=DefaultFilter(min_words=1, min_chars=1),
            format=None,
            doc_id_scheme=scheme,
        )
        docs = pipeline.run(path).documents
        assert len(docs) == 3
        assert {doc_id_scheme_of(d.doc_id) for d in docs} == {DocIdScheme(scheme)}
        batch = pipeline.run_batch([path, path])
        assert {doc_id_scheme_of(d.doc_id) for r in batch for d in r.documents} == {
            DocIdScheme(scheme)
        }

    def test_the_pipeline_refuses_an_unknown_scheme_at_construction(self):
        with pytest.raises(ValueError, match="unknown doc_id_scheme"):
            CorpusPipeline(doc_id_scheme="random")

    def test_assignment_after_construction_is_validated(self):
        config = BuilderConfig()
        config.doc_id_scheme = "uuid7"
        assert config.doc_id_scheme is DocIdScheme.UUID7
        with pytest.raises(ValueError, match="unknown doc_id_scheme"):
            config.doc_id_scheme = "uuid8"
        pipeline = CorpusPipeline()
        pipeline.doc_id_scheme = "uuid7"
        assert pipeline.doc_id_scheme is DocIdScheme.UUID7
        with pytest.raises(ValueError, match="unknown doc_id_scheme"):
            pipeline.doc_id_scheme = "uuid8"

    def test_builder_and_pipeline_name_per_source(self, tmp_path):
        # One scope for both producers: ids are minted per source, in
        # reading order, right after the source is read.
        paths = []
        for name in ("a.txt", "b.txt"):
            path = tmp_path / name
            path.write_bytes(_TEXT.encode("utf-8"))
            paths.append(path)
        with CorpusBuilder(_builder_config("uuid7")) as builder:
            built = builder.build(paths).documents
        ran = CorpusPipeline(
            chunker=SentenceChunker(),
            filter_=DefaultFilter(min_words=1, min_chars=1),
            format=None,
            doc_id_scheme="uuid7",
        ).run_batch(paths)
        for docs in (built, [d for r in ran for d in r.documents]):
            ids = [d.doc_id for d in docs]
            assert len(ids) == 6 and ids == sorted(ids)

    def test_every_producer_takes_the_scheme(self, tmp_path):
        from scikitplot.corpus import FluentCorpus, create_corpus  # noqa: PLC0415
        from scikitplot.corpus._custom_hooks import (  # noqa: PLC0415
            HookableCorpusPipeline,
        )

        path = tmp_path / "a.txt"
        path.write_bytes(_TEXT.encode("utf-8"))
        hookable = HookableCorpusPipeline(
            filter_=DefaultFilter(min_words=1, min_chars=1),
            format=None,
            doc_id_scheme="uuid7",
        )
        assert hookable.doc_id_scheme is DocIdScheme.UUID7
        (doc,) = hookable.run(path).documents
        assert doc_id_scheme_of(doc.doc_id) is DocIdScheme.UUID7
        result = create_corpus(
            path,
            tmp_path / "out.jsonl",
            filter_=DefaultFilter(min_words=1, min_chars=1),
            format=ExportFormat.JSONL,
            doc_id_scheme="uuid7",
        )
        assert doc_id_scheme_of(result.documents[0].doc_id) is DocIdScheme.UUID7
        runtime = FluentCorpus().identity("uuid7").materialize()
        try:
            assert runtime.pipeline.doc_id_scheme is DocIdScheme.UUID7
        finally:
            runtime.close()
        plain = FluentCorpus().materialize()
        try:
            assert plain.pipeline.doc_id_scheme is DocIdScheme.SHA256
        finally:
            plain.close()
        with pytest.raises(ValueError, match="unknown doc_id_scheme"):
            FluentCorpus().identity("uuid8").materialize()
        # The scheme decides what gets built, so it moves the ingest identity.
        assert (
            FluentCorpus().identity("uuid7").plan().ingest_fingerprint
            != FluentCorpus().plan().ingest_fingerprint
        )


# ---------------------------------------------------------------------------
# Data written by another generation is refused, never read as empty
# ---------------------------------------------------------------------------


def _as_generation_2(document):
    """A document payload as a build before R2-3 stored it."""
    payload = document.to_dict()
    payload["schema_version"] = "2.0"
    payload["doc_id"] = payload["doc_id"][:16]
    payload["content_hash"] = payload["content_hash"][:32]
    return payload


class TestOtherGenerations:
    def test_an_old_jsonl_store_is_refused_when_opened(self, tmp_path):
        from scikitplot.corpus import JSONLStorage  # noqa: PLC0415

        path = tmp_path / "store.jsonl"
        rows = [_as_generation_2(d) for d in _family()]
        path.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))
        with pytest.raises(IncompatibleSchemaError, match="line 1.*incompatible"):
            JSONLStorage(path)

    def test_an_old_sqlite_store_is_refused_when_opened_and_read(self, tmp_path):
        import sqlite3  # noqa: PLC0415
        from contextlib import closing  # noqa: PLC0415

        from scikitplot.corpus import SQLiteStorage, StorageQuery  # noqa: PLC0415

        path = tmp_path / "store.db"
        store = SQLiteStorage(path)
        try:
            store.save_batch(_family())
        finally:
            store.close()
        # Rewrite every row as a build before R2-3 stored it.
        with closing(sqlite3.connect(path)) as conn, conn:
            for doc in _family():
                conn.execute(
                    "UPDATE corpus_documents SET json_data = ? WHERE doc_id = ?",
                    (json.dumps(_as_generation_2(doc)), doc.doc_id),
                )
        with pytest.raises(IncompatibleSchemaError, match="SQLiteStorage"):
            SQLiteStorage(path)
        # A mixed store opens (its first row is current) but no read path
        # skips an old row as malformed: each raises.
        with closing(sqlite3.connect(path)) as conn, conn:
            first = _family()[0]
            conn.execute(
                "UPDATE corpus_documents SET json_data = ? WHERE doc_id = ?",
                (json.dumps(first.to_dict()), first.doc_id),
            )
            conn.execute(
                "DELETE FROM corpus_documents WHERE doc_id != ?", (first.doc_id,)
            )
        store = SQLiteStorage(path)
        try:
            store.save(_family()[1])
        finally:
            store.close()
        with closing(sqlite3.connect(path)) as conn, conn:
            second = _family()[1]
            conn.execute(
                "UPDATE corpus_documents SET json_data = ? WHERE doc_id = ?",
                (json.dumps(_as_generation_2(second)), second.doc_id),
            )
        store = SQLiteStorage(path)
        try:
            with pytest.raises(IncompatibleSchemaError):
                store.query(StorageQuery(limit=10))
            with pytest.raises(IncompatibleSchemaError):
                store.get(second.doc_id)
        finally:
            store.close()

    def test_an_empty_store_opens(self, tmp_path):
        from scikitplot.corpus import SQLiteStorage  # noqa: PLC0415

        store = SQLiteStorage(tmp_path / "empty.db")
        try:
            assert store.count() == 0
        finally:
            store.close()

    def test_an_old_text_export_keeps_the_error_type(self, tmp_path):
        from scikitplot.corpus import load_documents  # noqa: PLC0415

        path = tmp_path / "old.jsonl"
        path.write_bytes(
            (json.dumps(_as_generation_2(_family()[0])) + "\n").encode("utf-8")
        )
        with pytest.raises(IncompatibleSchemaError, match="record 0"):
            load_documents(path)

    def test_a_checkpoint_records_its_generation(self, tmp_path):
        from scikitplot.corpus import PipelineGuard  # noqa: PLC0415

        path = tmp_path / "run.ckpt"
        with PipelineGuard(checkpoint_path=path, checkpoint_every=1) as guard:
            assert len(list(guard.iter(iter(_family())))) == 3
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        assert {line["schema_version"] for line in lines} == {"3.0"}
        with PipelineGuard(checkpoint_path=path) as guard:
            assert list(guard.iter(iter(_family()))) == []  # all resumed

    @pytest.mark.parametrize("version", [None, "2.0"])
    def test_an_old_checkpoint_is_refused(self, tmp_path, version):
        from scikitplot.corpus import PipelineGuard  # noqa: PLC0415

        path = tmp_path / "old.ckpt"
        line = {"doc_id": "0123456789abcdef", "content_hash": "0" * 32}
        if version is not None:
            line["schema_version"] = version
        path.write_bytes((json.dumps(line) + "\n").encode("utf-8"))
        with pytest.raises(IncompatibleSchemaError, match="checkpoint"):
            PipelineGuard(checkpoint_path=path)


# ---------------------------------------------------------------------------
# Text identity: distinct texts never share a hash
# ---------------------------------------------------------------------------


class TestTextIdentity:
    def test_lone_surrogates_are_hashed_not_replaced(self):
        texts = [
            "key \ud800",
            # f"key {chr(0xd800)}",
            "key \udfff",
            # f"key {chr(0xdfff)}",
            "key ?",
            f"key {chr(0xfffd)}",
        ]
        hashes = {CorpusDocument.make_content_hash(text=t) for t in texts}
        ids = {CorpusDocument.make_doc_id("f.txt", 0, t) for t in texts}
        assert len(hashes) == len(texts)
        assert len(ids) == len(texts)

    def test_an_undecodable_file_name_still_has_an_id(self):
        # os.fsdecode maps undecodable bytes to lone surrogates.
        name = b"\xe9t\xe9.txt".decode("utf-8", "surrogateescape")
        assert _HEX64.fullmatch(CorpusDocument.make_doc_id(name, 0, "x"))

    def test_valid_text_hashes_as_strict_utf8(self):
        text = "naïve 日本"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert CorpusDocument.make_content_hash(text=text) == expected

    def test_the_scheme_prints_as_its_value_on_every_python(self):
        assert str(DocIdScheme.UUID7) == "uuid7"
        assert f"{DocIdScheme.SHA256}" == "sha256"
