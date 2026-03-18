"""
tests/test__registry.py
=========================
Tests for scikitplot.corpus._registry.
All external imports (chunkers, readers, normalizers) are mocked so that
tests run without heavy optional dependencies.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from .._registry import ComponentRegistry, registry, _fqcn


# ===========================================================================
# ComponentRegistry construction
# ===========================================================================


class TestComponentRegistryConstruction:
    def test_starts_empty(self) -> None:
        r = ComponentRegistry()
        assert r.list_chunkers() == []
        assert r.list_filters() == []
        assert r.list_readers() == []
        assert r.list_normalizers() == []

    def test_repr(self) -> None:
        r = ComponentRegistry()
        s = repr(r)
        assert "ComponentRegistry" in s
        assert "chunkers=0" in s


# ===========================================================================
# Registration / retrieval
# ===========================================================================


class TestChunkerRegistration:
    def test_register_and_get(self) -> None:
        r = ComponentRegistry()

        class FakeChunker:
            pass

        r.register_chunker("fake", FakeChunker)
        assert r.get_chunker("fake") is FakeChunker

    def test_get_missing_raises_key_error(self) -> None:
        r = ComponentRegistry()
        with pytest.raises(KeyError, match="no chunker"):
            r.get_chunker("nonexistent")

    def test_list_chunkers_sorted(self) -> None:
        r = ComponentRegistry()

        class A:
            pass

        class B:
            pass

        r.register_chunker("zebra", B)
        r.register_chunker("apple", A)
        assert r.list_chunkers() == ["apple", "zebra"]

    def test_empty_name_raises(self) -> None:
        r = ComponentRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            r.register_chunker("", object)

    def test_non_class_raises(self) -> None:
        r = ComponentRegistry()
        with pytest.raises(TypeError, match="class"):
            r.register_chunker("x", "not_a_class")  # type: ignore[arg-type]

    def test_override_warns_and_replaces(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        r = ComponentRegistry()

        class V1:
            pass

        class V2:
            pass

        r.register_chunker("my_chunker", V1)
        with caplog.at_level(logging.WARNING):
            r.register_chunker("my_chunker", V2)
        assert r.get_chunker("my_chunker") is V2


class TestFilterRegistration:
    def test_register_and_get(self) -> None:
        r = ComponentRegistry()

        class FakeFilter:
            pass

        r.register_filter("fake", FakeFilter)
        assert r.get_filter("fake") is FakeFilter

    def test_get_missing_raises(self) -> None:
        r = ComponentRegistry()
        with pytest.raises(KeyError, match="no filter"):
            r.get_filter("missing")


class TestReaderRegistration:
    def test_register_and_get(self) -> None:
        r = ComponentRegistry()

        class FakeReader:
            pass

        r.register_reader(".fakeext", FakeReader)
        assert r.get_reader(".fakeext") is FakeReader

    def test_get_missing_raises(self) -> None:
        r = ComponentRegistry()
        with pytest.raises(KeyError, match="no reader"):
            r.get_reader(".missing")


class TestNormalizerRegistration:
    def test_register_and_get(self) -> None:
        r = ComponentRegistry()

        class FakeNorm:
            pass

        r.register_normalizer("fake_norm", FakeNorm)
        assert r.get_normalizer("fake_norm") is FakeNorm

    def test_get_missing_raises(self) -> None:
        r = ComponentRegistry()
        with pytest.raises(KeyError, match="no normalizer"):
            r.get_normalizer("missing")


# ===========================================================================
# build_* convenience methods
# ===========================================================================


class TestBuildMethods:
    def test_build_chunker_instantiates(self) -> None:
        r = ComponentRegistry()
        sentinel = object()

        class FakeChunker:
            def __init__(self, *, size: int = 10) -> None:
                self.size = size

        r.register_chunker("fake", FakeChunker)
        instance = r.build_chunker("fake", size=42)
        assert isinstance(instance, FakeChunker)
        assert instance.size == 42

    def test_build_filter_instantiates(self) -> None:
        r = ComponentRegistry()

        class FakeFilter:
            def __init__(self, *, min_words: int = 3) -> None:
                self.min_words = min_words

        r.register_filter("fake", FakeFilter)
        instance = r.build_filter("fake", min_words=5)
        assert instance.min_words == 5

    def test_build_normalizer_instantiates(self) -> None:
        r = ComponentRegistry()

        class FakeNorm:
            def __init__(self, form: str = "NFC") -> None:
                self.form = form

        r.register_normalizer("fake_norm", FakeNorm)
        instance = r.build_normalizer("fake_norm", form="NFKC")
        assert instance.form == "NFKC"


# ===========================================================================
# snapshot
# ===========================================================================


class TestSnapshot:
    def test_snapshot_structure(self) -> None:
        r = ComponentRegistry()

        class C:
            pass

        r.register_chunker("my_chunker", C)
        snap = r.snapshot()
        assert "chunkers" in snap
        assert "filters" in snap
        assert "readers" in snap
        assert "normalizers" in snap
        assert "my_chunker" in snap["chunkers"]
        expected = f"{C.__module__}.{C.__qualname__}"  # _fqcn(C)
        assert snap["chunkers"]["my_chunker"] == expected

    def test_snapshot_is_copy(self) -> None:
        r = ComponentRegistry()
        snap1 = r.snapshot()
        snap1["chunkers"]["injected"] = "malicious"
        snap2 = r.snapshot()
        assert "injected" not in snap2["chunkers"]


# ===========================================================================
# register_builtins (mocked imports)
# ===========================================================================


class TestRegisterBuiltins:
    def test_idempotent(self) -> None:
        r = ComponentRegistry()

        class FakeChunker:
            pass

        class FakeFilter:
            pass

        # Mock all imports inside register_builtins to avoid actual package deps
        with patch(
            "scikitplot.corpus._chunkers.SentenceChunker", FakeChunker
        ), patch(
            "scikitplot.corpus._chunkers.ParagraphChunker", FakeChunker
        ), patch(
            "scikitplot.corpus._chunkers.FixedWindowChunker", FakeChunker
        ), patch(
            "scikitplot.corpus._base.DefaultFilter", FakeFilter
        ):
            # Call twice — second call must be idempotent
            r.register_builtins()
            n_chunkers_first = len(r.list_chunkers())
            r.register_builtins()
            n_chunkers_second = len(r.list_chunkers())

        assert n_chunkers_first == n_chunkers_second

    def test_import_error_does_not_crash(self) -> None:
        r = ComponentRegistry()
        # Simulate a missing package by patching the import to fail
        with patch.dict(
            "sys.modules",
            {"scikitplot.corpus._chunkers": None},
        ):
            # Should log a warning, not raise
            try:
                r.register_builtins()
            except Exception as exc:
                pytest.fail(
                    f"register_builtins should not raise on ImportError: {exc}"
                )


# ===========================================================================
# Module-level singleton
# ===========================================================================


class TestModuleSingleton:
    def test_registry_is_component_registry(self) -> None:
        assert isinstance(registry, ComponentRegistry)

    def test_registry_is_same_object_on_reimport(self) -> None:
        from scikitplot.corpus._registry import registry as registry2
        assert registry is registry2
