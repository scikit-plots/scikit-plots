# scikitplot/corpus/_chunkers/tests/test__chunker_bridge.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._chunkers._chunker_bridge
======================================================

Coverage
--------
* :class:`ChunkerBridge` — abstract base; ``chunk()`` delegation contract;
  ``_to_tuples()`` with offsets present (``start_char`` path) and absent
  (cursor-scan fallback); ``chunks`` attribute absent (empty-result guard).
* :class:`SentenceChunkerBridge` — strategy, ``_call_inner`` signature.
* :class:`ParagraphChunkerBridge` — strategy, ``_call_inner`` signature.
* :class:`FixedWindowChunkerBridge` — strategy, ``_call_inner`` signature.
* :class:`WordChunkerBridge` — strategy (CUSTOM), ``_call_inner`` signature.
* :func:`bridge_chunker` — already-bridged pass-through; auto-detection for
  all four new-style chunker class names; unknown chunker warning; missing
  ``metadata`` parameter detection.

All inner chunkers and ``ChunkResult`` objects are constructed from real
``Chunk`` / ``ChunkResult`` types so that offset logic is exercised
against the actual dataclass API.

Run with::

    pytest corpus/_chunkers/tests/test__chunker_bridge.py -v
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from .. import _chunker_bridge  # Import the module to access its logger
from .._chunker_bridge import (
    ChunkerBridge,
    FixedWindowChunkerBridge,
    ParagraphChunkerBridge,
    SentenceChunkerBridge,
    WordChunkerBridge,
    bridge_chunker,
)
from ..._schema import ChunkingStrategy
from ..._types import Chunk, ChunkResult

# 1. Access the logger instance directly from the module
# 2. Force propagation so caplog (which hooks into the hierarchy) catches it
_chunker_bridge.logger.propagate = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, start: int) -> Chunk:
    """Return a ``Chunk`` with populated ``start_char`` / ``end_char``."""
    return Chunk(text=text, start_char=start, end_char=start + len(text))


def _make_chunk_no_offset(text: str) -> Chunk:
    """Return a ``Chunk`` with zero offsets (offset-disabled convention)."""
    return Chunk(text=text, start_char=0, end_char=0)


def _make_result(*pairs: tuple[str, int]) -> ChunkResult:
    """Build a ``ChunkResult`` from (text, start) pairs."""
    chunks = [_make_chunk(t, s) for t, s in pairs]
    return ChunkResult(chunks=chunks)


def _make_result_no_offsets(*texts: str) -> ChunkResult:
    """Build a ``ChunkResult`` where all chunks have zero-based offsets."""
    chunks = [_make_chunk_no_offset(t) for t in texts]
    return ChunkResult(chunks=chunks)


# ---------------------------------------------------------------------------
# Concrete bridge for testing (since ChunkerBridge is abstract)
# ---------------------------------------------------------------------------


class _StubInner:
    """Minimal inner chunker for testing; records calls."""

    def __init__(self, result: ChunkResult) -> None:
        self._result = result
        self.calls: list[tuple[str, Any]] = []

    def chunk(self, text: str, extra_metadata: Any = None) -> ChunkResult:
        self.calls.append((text, extra_metadata))
        return self._result


class _Concretebridge(ChunkerBridge):
    """Minimal concrete subclass for abstract-base tests."""

    strategy = ChunkingStrategy.PARAGRAPH

    def _call_inner(self, text: str, metadata: Any) -> Any:
        return self.inner.chunk(text, extra_metadata=metadata)


# ===========================================================================
# ChunkerBridge — abstract base behaviour
# ===========================================================================


class TestChunkerBridgeBase:
    """Tests for shared ``ChunkerBridge`` logic."""

    def test_chunk_delegates_to_call_inner(self) -> None:
        """``chunk()`` must call ``_call_inner`` and convert the result."""
        inner = _StubInner(_make_result(("Hello.", 0)))
        bridge = _Concretebridge(inner)
        result = bridge.chunk("Hello.", metadata={})
        assert result == [(0, "Hello.")]
        assert len(inner.calls) == 1

    def test_chunk_passes_metadata_to_inner(self) -> None:
        """Metadata dict must reach the inner chunker unchanged."""
        meta = {"source": "test.txt", "index": 3}
        inner = _StubInner(_make_result(("Some text.", 0)))
        bridge = _Concretebridge(inner)
        bridge.chunk("Some text.", metadata=meta)
        _, passed_meta = inner.calls[0]
        assert passed_meta == meta

    def test_chunk_metadata_none(self) -> None:
        """``metadata=None`` must not raise; inner receives ``None``."""
        inner = _StubInner(_make_result(("OK.", 0)))
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk("OK.", metadata=None)
        assert isinstance(pairs, list)
        assert pairs[0][1] == "OK."

    def test_inner_stored(self) -> None:
        """The wrapped inner chunker must be accessible via ``.inner``."""
        inner = _StubInner(_make_result(("x", 0)))
        bridge = _Concretebridge(inner)
        assert bridge.inner is inner

    def test_repr_contains_strategy_and_inner_class(self) -> None:
        inner = _StubInner(_make_result(("x", 0)))
        bridge = _Concretebridge(inner)
        r = repr(bridge)
        assert "_Concretebridge" in r
        assert "_StubInner" in r

    # ------------------------------------------------------------------
    # _to_tuples: offset path
    # ------------------------------------------------------------------

    def test_to_tuples_uses_start_char_when_present(self) -> None:
        """Chunks with ``start_char`` must use the fast offset path."""
        result = _make_result(("foo", 0), ("bar", 5))
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk("foo  bar", metadata=None)
        # The offset path returns start_char exactly as stored.
        assert pairs[0] == (0, "foo")
        assert pairs[1] == (5, "bar")

    def test_to_tuples_cursor_fallback_when_no_start_char_attribute(self) -> None:
        """Cursor-scan fallback triggers only when chunk has NO ``start_char`` attr."""
        source = "Hello world today."
        # MagicMock(spec=["text"]) has no start_char, no char_start -> fallback.
        chunks = []
        for word in ("Hello", "world", "today."):
            m = MagicMock(spec=["text"])
            m.text = word
            chunks.append(m)
        fake_result = MagicMock()
        fake_result.chunks = chunks
        inner2 = MagicMock()
        inner2.chunk.return_value = fake_result
        bridge = _Concretebridge(inner2)
        pairs = bridge.chunk(source, metadata=None)
        for offset, text in pairs:
            assert 0 <= offset < len(source)
            assert source[offset : offset + len(text)] == text

    def test_to_tuples_zero_start_char_used_as_valid_offset(self) -> None:
        """``start_char=0`` is a real offset; bridge must NOT treat it as absent."""
        source = "Hello world."
        result = _make_result(("Hello", 0), ("world.", 6))
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert pairs == [(0, "Hello"), (6, "world.")]

    def test_to_tuples_cursor_fallback_correct_order(self) -> None:
        """Cursor scan must not rewind — offsets must be non-decreasing."""
        source = "The cat sat on the mat."
        result = _make_result_no_offsets("The", "cat", "sat", "on", "the", "mat.")
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        offsets = [p[0] for p in pairs]
        assert offsets == sorted(offsets)

    def test_to_tuples_chunk_not_found_uses_cursor(self) -> None:
        """If ``find`` returns -1 (normalised text), use cursor as fallback offset."""
        source = "abc"
        # Chunk text is not literally in source → triggers find == -1 branch.
        chunk = Chunk(text="xyz", start_char=0, end_char=0)
        result = ChunkResult(chunks=[chunk])
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert len(pairs) == 1
        assert pairs[0][0] >= 0  # Must not raise; cursor fallback used.

    def test_to_tuples_empty_chunk_list(self) -> None:
        """Empty ``ChunkResult`` must return an empty list of pairs."""
        result = ChunkResult(chunks=[])
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk("Some text.", metadata=None)
        assert pairs == []

    def test_to_tuples_no_chunks_attribute(self) -> None:
        """When inner returns an object without ``chunks``, fall back to single chunk."""
        source = "Fallback text."
        inner = MagicMock()
        inner.chunk.return_value = object()  # No .chunks attribute.
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert pairs == [(0, source)]

    def test_to_tuples_single_chunk(self) -> None:
        """Single chunk with known offset must round-trip cleanly."""
        source = "One sentence only."
        result = _make_result(("One sentence only.", 0))
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert pairs == [(0, "One sentence only.")]

    def test_to_tuples_preserves_chunk_text(self) -> None:
        """Returned text must exactly match the ``Chunk.text`` value."""
        source = "Paragraph one.\n\nParagraph two."
        result = _make_result(("Paragraph one.", 0), ("Paragraph two.", 16))
        inner = _StubInner(result)
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert pairs[0][1] == "Paragraph one."
        assert pairs[1][1] == "Paragraph two."

    def test_to_tuples_char_start_alias_also_works(self) -> None:
        """Legacy ``char_start`` attribute (non-Chunk objects) is also respected."""
        # Simulate an object with old-style `char_start` (not `start_char`).
        legacy_chunk = MagicMock()
        legacy_chunk.text = "Legacy."
        del legacy_chunk.start_char        # No start_char attribute.
        legacy_chunk.char_start = 7
        result = MagicMock()
        result.chunks = [legacy_chunk]
        inner = MagicMock()
        inner.chunk.return_value = result
        bridge = _Concretebridge(inner)
        pairs = bridge.chunk("0123456Legacy.", metadata=None)
        assert pairs[0] == (7, "Legacy.")


# ===========================================================================
# Concrete bridges — strategy and _call_inner
# ===========================================================================


class TestSentenceChunkerBridge:
    def test_strategy_is_sentence(self) -> None:
        bridge = SentenceChunkerBridge(MagicMock())
        assert bridge.strategy is ChunkingStrategy.SENTENCE

    def test_call_inner_passes_extra_metadata(self) -> None:
        inner = MagicMock()
        inner.chunk.return_value = ChunkResult(chunks=[])
        bridge = SentenceChunkerBridge(inner)
        meta = {"key": "value"}
        bridge.chunk("Hello.", metadata=meta)
        inner.chunk.assert_called_once_with("Hello.", extra_metadata=meta)

    def test_returns_list_of_tuples(self) -> None:
        inner = MagicMock()
        inner.chunk.return_value = _make_result(("Hello.", 0))
        bridge = SentenceChunkerBridge(inner)
        result = bridge.chunk("Hello.", metadata=None)
        assert isinstance(result, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)


class TestParagraphChunkerBridge:
    def test_strategy_is_paragraph(self) -> None:
        bridge = ParagraphChunkerBridge(MagicMock())
        assert bridge.strategy is ChunkingStrategy.PARAGRAPH

    def test_call_inner_passes_extra_metadata(self) -> None:
        inner = MagicMock()
        inner.chunk.return_value = ChunkResult(chunks=[])
        bridge = ParagraphChunkerBridge(inner)
        meta = {"page": 1}
        bridge.chunk("Para.", metadata=meta)
        inner.chunk.assert_called_once_with("Para.", extra_metadata=meta)

    def test_multi_paragraph(self) -> None:
        source = "Para one.\n\nPara two."
        result = _make_result(("Para one.", 0), ("Para two.", 11))
        inner = _StubInner(result)
        bridge = ParagraphChunkerBridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert len(pairs) == 2
        assert pairs[0][1] == "Para one."
        assert pairs[1][1] == "Para two."


class TestFixedWindowChunkerBridge:
    def test_strategy_is_fixed_window(self) -> None:
        bridge = FixedWindowChunkerBridge(MagicMock())
        assert bridge.strategy is ChunkingStrategy.FIXED_WINDOW

    def test_call_inner_passes_extra_metadata(self) -> None:
        inner = MagicMock()
        inner.chunk.return_value = ChunkResult(chunks=[])
        bridge = FixedWindowChunkerBridge(inner)
        bridge.chunk("Text here.", metadata={"window": 3})
        inner.chunk.assert_called_once_with("Text here.", extra_metadata={"window": 3})

    def test_windowed_offsets(self) -> None:
        source = "AABBCCDD"
        result = _make_result(("AABB", 0), ("BBCC", 2), ("CCDD", 4))
        inner = _StubInner(result)
        bridge = FixedWindowChunkerBridge(inner)
        pairs = bridge.chunk(source, metadata=None)
        assert pairs[0] == (0, "AABB")
        assert pairs[1] == (2, "BBCC")
        assert pairs[2] == (4, "CCDD")


class TestWordChunkerBridge:
    def test_strategy_is_custom(self) -> None:
        bridge = WordChunkerBridge(MagicMock())
        assert bridge.strategy is ChunkingStrategy.CUSTOM

    def test_call_inner_passes_extra_metadata(self) -> None:
        inner = MagicMock()
        inner.chunk.return_value = ChunkResult(chunks=[])
        bridge = WordChunkerBridge(inner)
        bridge.chunk("Word doc.", metadata={"n": 1})
        inner.chunk.assert_called_once_with("Word doc.", extra_metadata={"n": 1})


# ===========================================================================
# bridge_chunker factory
# ===========================================================================


class TestBridgeChunker:
    """Tests for the :func:`bridge_chunker` auto-detection factory."""

    # ------------------------------------------------------------------
    # Already bridged / already ChunkerBase-compatible
    # ------------------------------------------------------------------

    def test_already_compatible_returned_unchanged(self) -> None:
        """Object with ``strategy`` + ``chunk(text, metadata=...)`` passes through."""

        class _AlreadyOK:
            strategy = ChunkingStrategy.SENTENCE

            def chunk(self, text: str, metadata=None):
                return []

        obj = _AlreadyOK()
        assert bridge_chunker(obj) is obj

    def test_existing_bridge_instance_returned_unchanged(self) -> None:
        """A ``ChunkerBridge`` instance is itself already compatible."""
        inner = MagicMock()
        inner.chunk.return_value = ChunkResult(chunks=[])
        bridge = SentenceChunkerBridge(inner)
        assert bridge_chunker(bridge) is bridge

    def test_missing_metadata_param_triggers_wrapping(self) -> None:
        """If ``chunk`` exists but has no ``metadata`` param, a bridge is created."""

        class _NoMetadataParam:
            strategy = ChunkingStrategy.PARAGRAPH

            def chunk(self, text: str):
                return []

        # Must not be returned as-is; the class name doesn't match a bridge
        # so should log a warning and return the object.
        obj = _NoMetadataParam()
        result = bridge_chunker(obj)
        # Falls through to the unknown path — returned as-is.
        assert result is obj

    # ------------------------------------------------------------------
    # Auto-detection for the four new-style chunker class names
    # ------------------------------------------------------------------

    def test_detects_sentence_chunker(self) -> None:
        class SentenceChunker:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        obj = SentenceChunker()
        result = bridge_chunker(obj)
        assert isinstance(result, SentenceChunkerBridge)
        assert result.inner is obj

    def test_detects_paragraph_chunker(self) -> None:
        class ParagraphChunker:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        obj = ParagraphChunker()
        result = bridge_chunker(obj)
        assert isinstance(result, ParagraphChunkerBridge)

    def test_detects_fixed_window_chunker(self) -> None:
        class FixedWindowChunker:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        obj = FixedWindowChunker()
        result = bridge_chunker(obj)
        assert isinstance(result, FixedWindowChunkerBridge)

    def test_detects_word_chunker(self) -> None:
        class WordChunker:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        obj = WordChunker()
        result = bridge_chunker(obj)
        assert isinstance(result, WordChunkerBridge)

    def test_unknown_chunker_returns_original_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unrecognised chunker must be returned as-is with a WARNING log.

        Notes
        -----
        ``caplog.records`` stores raw ``LogRecord`` objects; the ``.message``
        attribute only exists after ``Formatter.format()`` has been called.
        Use ``caplog.messages`` (which calls ``r.getMessage()`` on each
        record) instead.  We also set the level on both possible logger
        hierarchy roots so the test is robust regardless of whether the
        module was imported as ``scikitplot.corpus.*``.
        """

        class WeirdChunker:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        obj = WeirdChunker()
        # We patch the logger instance directly in the target module.
        # This intercepts the call before it even reaches the logging handlers.
        patch_path = "scikitplot.corpus._chunkers._chunker_bridge.logger"
        with patch(patch_path) as mock_logger:
            result = bridge_chunker(obj)
        assert result is obj
        # Verify logger.warning was called
        assert mock_logger.warning.called, "logger.warning() was not called"
        # Verify the content of the arguments
        # The first arg is the format string, subsequent are values for %r
        args, _ = mock_logger.warning.call_args
        assert any("WeirdChunker" in str(arg) for arg in args), (
            f"Expected 'WeirdChunker' in log arguments, but got: {args}"
        )

    def test_no_chunk_method_returned_unchanged(self) -> None:
        """Object with no ``chunk`` method at all goes through the unknown path."""

        class NoChunk:
            strategy = ChunkingStrategy.PARAGRAPH

        obj = NoChunk()
        # Has strategy but no chunk method, so sig_ok=False → goes to bridge_map lookup.
        result = bridge_chunker(obj)
        assert result is obj

    def test_bridged_inner_strategy_matches_class(self) -> None:
        """Each auto-bridged object must carry the correct strategy."""
        expected = {
            "SentenceChunker": ChunkingStrategy.SENTENCE,
            "ParagraphChunker": ChunkingStrategy.PARAGRAPH,
            "FixedWindowChunker": ChunkingStrategy.FIXED_WINDOW,
            "WordChunker": ChunkingStrategy.CUSTOM,
        }
        for cls_name, expected_strategy in expected.items():
            # Dynamically create a class with the given name.
            cls = type(cls_name, (), {"chunk": lambda self, t, extra_metadata=None: ChunkResult(chunks=[])})
            obj = cls()
            result = bridge_chunker(obj)
            assert result.strategy is expected_strategy, (
                f"{cls_name}: expected {expected_strategy}, got {result.strategy}"
            )

    # ------------------------------------------------------------------
    # Integration: bridge_chunker + real ParagraphChunker
    # ------------------------------------------------------------------

    def test_real_paragraph_chunker_integration(self) -> None:
        """End-to-end: bridge real ParagraphChunker → tuple list."""
        from .._paragraph import ParagraphChunker  # noqa: PLC0415

        chunker = ParagraphChunker()
        bridged = bridge_chunker(chunker)
        assert isinstance(bridged, ParagraphChunkerBridge)

        source = "First paragraph here.\n\nSecond paragraph here."
        pairs = bridged.chunk(source, metadata={})
        assert len(pairs) >= 2
        for offset, text in pairs:
            assert isinstance(offset, int)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_real_fixed_window_chunker_integration(self) -> None:
        """End-to-end: bridge real FixedWindowChunker → tuple list.

        Uses CHARS unit with window_size=10, step_size=5 so a 33-char
        source produces multiple non-empty windows.
        """
        from .._fixed_window import FixedWindowChunker, FixedWindowChunkerConfig, WindowUnit  # noqa: PLC0415

        cfg = FixedWindowChunkerConfig(window_size=10, step_size=5, unit=WindowUnit.CHARS)
        chunker = FixedWindowChunker(cfg)
        bridged = bridge_chunker(chunker)
        assert isinstance(bridged, FixedWindowChunkerBridge)
        source = "one two three four five six seven"   # 33 chars -> >=3 windows
        pairs = bridged.chunk(source, metadata=None)
        assert len(pairs) >= 1
        for offset, text in pairs:
            assert 0 <= offset <= len(source)
            assert len(text) > 0

# ===========================================================================
# register_bridge / unregister_bridge
# ===========================================================================


class TestRegisterBridge:
    """Tests for :func:`register_bridge`."""

    def setup_method(self) -> None:
        """Ensure any custom bridge registered in a test is removed after."""
        from .._chunker_bridge import _BRIDGE_MAP  # noqa: PLC0415

        self._bridge_map_snapshot = dict(_BRIDGE_MAP)

    def teardown_method(self) -> None:
        """Restore _BRIDGE_MAP to pre-test state to prevent cross-test pollution."""
        from .._chunker_bridge import _BRIDGE_MAP  # noqa: PLC0415

        keys_to_remove = [k for k in _BRIDGE_MAP if k not in self._bridge_map_snapshot]
        for k in keys_to_remove:
            del _BRIDGE_MAP[k]

    def test_register_then_bridge_chunker_uses_new_bridge(self) -> None:
        """Registered bridge must be used by :func:`bridge_chunker`."""
        from .._chunker_bridge import bridge_chunker, register_bridge  # noqa: PLC0415

        class _MyChunker:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        class _MyBridge(ChunkerBridge):
            strategy = ChunkingStrategy.CUSTOM

            def _call_inner(self, text, metadata):
                return self.inner.chunk(text, extra_metadata=metadata)

        register_bridge(_MyChunker, _MyBridge)
        result = bridge_chunker(_MyChunker())
        assert isinstance(result, _MyBridge)

    def test_register_non_bridge_subclass_raises_type_error(self) -> None:
        """Passing a non-ChunkerBridge class must raise TypeError."""
        from .._chunker_bridge import register_bridge  # noqa: PLC0415

        class _FakeChunker:
            pass

        class _NotABridge:
            pass

        with pytest.raises(TypeError, match="ChunkerBridge subclass"):
            register_bridge(_FakeChunker, _NotABridge)  # type: ignore[arg-type]

    def test_register_overwrite_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Re-registering an existing name must not raise; it overwrites silently."""
        from .._chunker_bridge import _BRIDGE_MAP, register_bridge  # noqa: PLC0415

        class _MyChunker2:
            pass

        class _BridgeA(ChunkerBridge):
            strategy = ChunkingStrategy.PARAGRAPH

            def _call_inner(self, text, metadata):
                return self.inner.chunk(text)

        class _BridgeB(ChunkerBridge):
            strategy = ChunkingStrategy.SENTENCE

            def _call_inner(self, text, metadata):
                return self.inner.chunk(text)

        register_bridge(_MyChunker2, _BridgeA)
        register_bridge(_MyChunker2, _BridgeB)  # Overwrite — must not raise
        assert _BRIDGE_MAP["_MyChunker2"] is _BridgeB

    def test_register_uses_class_name_as_key(self) -> None:
        """Bridge must be keyed by ``chunker_class.__name__``."""
        from .._chunker_bridge import _BRIDGE_MAP, register_bridge  # noqa: PLC0415

        class _UniqueNamedChunker:
            pass

        class _AnyBridge(ChunkerBridge):
            strategy = ChunkingStrategy.FIXED_WINDOW

            def _call_inner(self, text, metadata):
                return ChunkResult(chunks=[])

        register_bridge(_UniqueNamedChunker, _AnyBridge)
        assert "_UniqueNamedChunker" in _BRIDGE_MAP
        assert _BRIDGE_MAP["_UniqueNamedChunker"] is _AnyBridge


class TestUnregisterBridge:
    """Tests for :func:`unregister_bridge`."""

    def setup_method(self) -> None:
        from .._chunker_bridge import _BRIDGE_MAP  # noqa: PLC0415

        self._bridge_map_snapshot = dict(_BRIDGE_MAP)

    def teardown_method(self) -> None:
        from .._chunker_bridge import _BRIDGE_MAP  # noqa: PLC0415

        keys_to_remove = [k for k in _BRIDGE_MAP if k not in self._bridge_map_snapshot]
        for k in keys_to_remove:
            del _BRIDGE_MAP[k]
        # Restore any that were removed during test
        for k, v in self._bridge_map_snapshot.items():
            if k not in _BRIDGE_MAP:
                _BRIDGE_MAP[k] = v

    def test_unregister_removes_bridge(self) -> None:
        """Unregistered class must no longer appear in ``_BRIDGE_MAP``."""
        from .._chunker_bridge import (  # noqa: PLC0415
            _BRIDGE_MAP,
            register_bridge,
            unregister_bridge,
        )

        class _TempChunker:
            pass

        class _TempBridge(ChunkerBridge):
            strategy = ChunkingStrategy.CUSTOM

            def _call_inner(self, text, metadata):
                return ChunkResult(chunks=[])

        register_bridge(_TempChunker, _TempBridge)
        assert "_TempChunker" in _BRIDGE_MAP
        unregister_bridge(_TempChunker)
        assert "_TempChunker" not in _BRIDGE_MAP

    def test_unregister_unknown_raises_key_error(self) -> None:
        """Unregistering an unknown class must raise ``KeyError``."""
        from .._chunker_bridge import unregister_bridge  # noqa: PLC0415

        class _NeverRegistered:
            pass

        with pytest.raises(KeyError):
            unregister_bridge(_NeverRegistered)

    def test_unregister_then_bridge_chunker_falls_through_to_warning(self) -> None:
        """After unregistering, :func:`bridge_chunker` must not use the old bridge."""
        from .._chunker_bridge import (  # noqa: PLC0415
            bridge_chunker,
            register_bridge,
            unregister_bridge,
        )
        from unittest.mock import patch  # noqa: PLC0415

        class _ChunkerX:
            def chunk(self, text, extra_metadata=None):
                return ChunkResult(chunks=[])

        class _BridgeX(ChunkerBridge):
            strategy = ChunkingStrategy.CUSTOM

            def _call_inner(self, text, metadata):
                return self.inner.chunk(text, extra_metadata=metadata)

        register_bridge(_ChunkerX, _BridgeX)
        unregister_bridge(_ChunkerX)

        obj = _ChunkerX()
        patch_path = "scikitplot.corpus._chunkers._chunker_bridge.logger"
        with patch(patch_path) as mock_logger:
            result = bridge_chunker(obj)
        assert result is obj
        assert mock_logger.warning.called
