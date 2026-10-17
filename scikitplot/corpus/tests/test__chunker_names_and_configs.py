# scikitplot/corpus/tests/test__chunker_names_and_configs.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One name table and strict configs for ingestion (Run 3: R3-1, R3-2, R3-4).

* R3-1: ``CorpusBuilder`` resolves chunker names through the
  ``ComponentRegistry`` (where ``word`` and ``semantic`` are registered and a
  user's chunker becomes reachable by name); ``chunker_kwargs`` builds the
  chunker's config, and an unknown key raises naming the valid fields; the
  registry snapshot round-trips every built-in.
* R3-2: a chunker config's enum field accepts the member or its value and
  refuses anything else at construction.
* R3-4: an unknown normaliser step raises; components resolve before any
  source is read, so a configuration error raises from ``build``.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from unittest.mock import MagicMock

import pytest

from .._chunkers import (
    ParagraphChunker,
    ProtocolChunkerBridge,
    SentenceChunker,
    SentenceChunkerBridge,
    bridge_chunker,
)
from .._chunkers._fixed_window import FixedWindowChunkerConfig
from .._chunkers._paragraph import ParagraphChunkerConfig
from .._chunkers._recursive import ParentChildChunkerConfig, RecursiveChunkerConfig
from .._chunkers._semantic import SemanticChunkerConfig
from .._chunkers._sentence import SentenceChunkerConfig
from .._chunkers._structure import StructureChunkerConfig
from .._chunkers._word import WordChunkerConfig
from .._chunkers._writing_system import WritingSystemAdapterConfig
from .._corpus_builder import BuilderConfig, CorpusBuilder
from .._custom_hooks import BuilderFactories
from .._registry import registry as _registry
from .._registry._registry import ComponentRegistry
from .._runtime import _resolve_chunker
from .._types import Chunk, ChunkerConfig, ChunkResult

_BUILTINS = {
    "sentence": SentenceChunkerConfig,
    "paragraph": ParagraphChunkerConfig,
    "fixed_window": FixedWindowChunkerConfig,
    "word": WordChunkerConfig,
    "semantic": SemanticChunkerConfig,
    "recursive": RecursiveChunkerConfig,
    "parent_child": ParentChildChunkerConfig,
    "structure": StructureChunkerConfig,
}
_TEXT = "The first sentence is here.\n\nA second paragraph follows. It ends now."


@pytest.fixture(autouse=True)
def _quiet():
    """Optional-segmenter notices are R3-7's subject, not this module's."""
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def fresh():
    registry = ComponentRegistry()
    registry.register_builtins()
    return registry


@pytest.fixture
def shared_registry(monkeypatch):
    """The module-level registry the builder uses, restored afterwards."""
    registry = _registry
    registry.register_builtins()
    monkeypatch.setattr(registry, "_chunkers", dict(registry._chunkers))
    monkeypatch.setattr(registry, "_chunker_configs", dict(registry._chunker_configs))
    monkeypatch.setattr(registry, "_normalizers", dict(registry._normalizers))
    return registry


class _ProtocolChunker:
    """A third-party chunker that satisfies ChunkerProtocol, nothing more."""

    def chunk(self, text, doc_id=None, extra_metadata=None):
        return ChunkResult(
            chunks=[Chunk(text=text, start_char=0, end_char=len(text))],
            metadata={"chunker": "protocol"},
        )

    def chunk_batch(self, texts, doc_ids=None, extra_metadata=None):
        return [self.chunk(text) for text in texts]


def _shipped(snapshot):
    """Keep the readers scikit-plots ships.

    ``DocumentReader`` keeps one process-wide registry, and other test modules
    register throwaway readers there (classes that cannot be imported by
    name), so a snapshot taken in the full suite holds entries no snapshot
    could ever load.
    """
    snapshot["readers"] = {
        name: fqcn
        for name, fqcn in snapshot["readers"].items()
        if fqcn.startswith("scikitplot.corpus._readers.")
    }
    return snapshot


def _build(tmp_path, **config):
    source = tmp_path / "doc.txt"
    source.write_text(_TEXT, encoding="utf-8")
    config.setdefault("filter_kwargs", {"min_words": 0, "min_chars": 0})
    return CorpusBuilder(BuilderConfig(**config)).build(source)


# ---------------------------------------------------------------------------
# R3-1: the registry is the one name table
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_builtin_is_registered_with_its_config(self, fresh):
        assert fresh.list_chunkers() == sorted(_BUILTINS)
        for name, config_cls in _BUILTINS.items():
            assert fresh.get_chunker_config(name) is config_cls

    @pytest.mark.parametrize("name", sorted(_BUILTINS))
    def test_every_builtin_round_trips_by_name(self, fresh, name):
        """The plan's gate: build by name, chunk, through the bridge too."""
        chunker = fresh.build_chunker(name)
        assert isinstance(chunker.config, _BUILTINS[name])
        bridged = bridge_chunker(chunker)
        assert bridged.chunk(_TEXT, metadata={}).chunks

    def test_the_snapshot_round_trips_every_section(self, fresh):
        snapshot = _shipped(fresh.snapshot())
        assert set(snapshot["chunker_configs"]) == set(_BUILTINS)
        loaded = ComponentRegistry.load_from_snapshot(snapshot)
        assert loaded.snapshot() == snapshot
        assert loaded.get_chunker_config("word") is WordChunkerConfig

    def test_a_snapshot_without_configs_still_loads(self, fresh):
        snapshot = _shipped(fresh.snapshot())
        del snapshot["chunker_configs"]
        loaded = ComponentRegistry.load_from_snapshot(snapshot)
        # Review F6: each built-in declares its config_class, so a snapshot
        # written before R3-1 still builds configs from options.
        assert loaded.get_chunker_config("sentence") is SentenceChunkerConfig
        assert loaded.build_chunker("sentence", overlap=1).config.overlap == 1

    @pytest.mark.parametrize(
        ("mutate", "error"),
        [
            (lambda s: s["chunker_configs"].update(nope="x.y.Z"), ValueError),
            (lambda s: s.update(extra={}), ValueError),
            (lambda s: s.pop("readers"), ValueError),
        ],
        ids=["config_for_unknown_chunker", "unknown_section", "missing_section"],
    )
    def test_a_malformed_snapshot_is_refused(self, fresh, mutate, error):
        snapshot = fresh.snapshot()
        mutate(snapshot)
        with pytest.raises(error):
            ComponentRegistry.load_from_snapshot(snapshot)

    def test_options_build_the_config_and_typos_raise(self, fresh):
        chunker = fresh.build_chunker("sentence", overlap=1, backend="regex")
        assert chunker.config.overlap == 1
        with pytest.raises(
            TypeError, match=r"unknown option\(s\) \['overlaps'\].*overlap"
        ):
            fresh.build_chunker("sentence", overlaps=1)

    def test_a_ready_config_is_accepted_and_a_wrong_one_refused(self, fresh):
        config = ParagraphChunkerConfig(min_length=3)
        assert fresh.build_chunker("paragraph", config=config).config is config
        with pytest.raises(TypeError, match="ParagraphChunkerConfig"):
            fresh.build_chunker("paragraph", config=SentenceChunkerConfig())

    def test_a_config_must_be_a_dataclass(self, fresh):
        with pytest.raises(TypeError, match="dataclass"):
            fresh.register_chunker("odd", _ProtocolChunker, config=dict)

    def test_without_a_config_kwargs_reach_the_constructor(self, fresh):
        class Plain(_ProtocolChunker):
            def __init__(self, size=1):
                self.size = size

        fresh.register_chunker("plain", Plain)
        assert fresh.get_chunker_config("plain") is None
        assert fresh.build_chunker("plain", size=7).size == 7

    def test_replacing_a_chunker_drops_its_old_config(self, fresh):
        fresh.register_chunker("sentence", _ProtocolChunker, replace=True)
        assert fresh.get_chunker_config("sentence") is None


# ---------------------------------------------------------------------------
# Bridges
# ---------------------------------------------------------------------------


class TestBridges:
    def test_a_subclass_uses_its_base_bridge(self):
        class MySentences(SentenceChunker):
            pass

        assert isinstance(bridge_chunker(MySentences()), SentenceChunkerBridge)

    def test_a_protocol_chunker_gets_the_generic_bridge(self):
        bridged = bridge_chunker(_ProtocolChunker())
        assert isinstance(bridged, ProtocolChunkerBridge)
        assert [c.text for c in bridged.chunk("abc", metadata={}).chunks] == ["abc"]


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class TestBuilder:
    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("sentence", {"overlap": 1}),
            ("paragraph", {"min_length": 5}),
            ("fixed_window", {"window_size": 50, "step_size": 25}),
            ("word", {"lowercase": True}),
            ("semantic", {"backend": "morphological"}),
        ],
    )
    def test_every_named_chunker_takes_its_options(self, tmp_path, name, kwargs):
        """The review's probe: every one of these raised TypeError."""
        result = _build(tmp_path, chunker=name, chunker_kwargs=kwargs)
        assert result.documents

    def test_a_registered_chunker_is_reachable_by_name(self, tmp_path, shared_registry):
        shared_registry.register_chunker("mine", _ProtocolChunker)
        result = _build(tmp_path, chunker="mine")
        assert [d.text for d in result.documents] == [_TEXT]

    def test_a_typo_raises_from_build_before_reading(self, tmp_path):
        missing = tmp_path / "never-read.txt"
        with pytest.raises(TypeError, match="overlaps"):
            CorpusBuilder(
                BuilderConfig(chunker="sentence", chunker_kwargs={"overlaps": 1})
            ).build(missing)

    def test_an_unknown_name_lists_the_registered_ones(self, tmp_path):
        with pytest.raises(
            ValueError, match=r"Unknown chunker: 'sentences'.*'semantic'"
        ):
            _build(tmp_path, chunker="sentences")

    def test_options_for_an_instance_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="cannot apply to a chunker instance"):
            _build(
                tmp_path, chunker=ParagraphChunker(), chunker_kwargs={"min_length": 2}
            )

    def test_options_without_a_chunker_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="needs a chunker name"):
            _build(tmp_path, chunker=None, chunker_kwargs={"overlap": 1})

    def test_an_unknown_normalizer_step_raises(self, tmp_path):
        """The review's probe: `lowercsae` was dropped with a warning."""
        with pytest.raises(ValueError, match=r"\['lowercsae'\].*'lowercase'"):
            _build(tmp_path, normalizer_steps=["unicode", "lowercsae"])

    def test_normalizer_steps_come_from_the_registry(self, tmp_path):
        result = _build(tmp_path, normalizer_steps=["unicode", "lowercase"])
        assert result.documents[0].normalized_text == result.documents[0].text.lower()


# ---------------------------------------------------------------------------
# R3-2: enum fields of every chunker config
# ---------------------------------------------------------------------------


def _enum_fields():
    for config_cls in _BUILTINS.values():
        for field in dataclasses.fields(config_cls):
            if isinstance(field.default, enum.Enum):
                yield config_cls, field.name, type(field.default)


@pytest.mark.parametrize(
    ("config_cls", "name", "enum_cls"),
    list(_enum_fields()),
    ids=lambda value: getattr(value, "__name__", value),
)
class TestEnumFields:
    def test_every_value_is_coerced_to_its_member(self, config_cls, name, enum_cls):
        for member in enum_cls:
            try:
                config = config_cls(**{name: member.value})
            except ValueError as exc:
                if "must be one of" in str(exc):
                    raise
                continue  # the member needs other options (e.g. a custom callable)
            assert getattr(config, name) is member

    def test_an_unknown_value_raises_at_construction(self, config_cls, name, enum_cls):
        with pytest.raises(ValueError, match=f"{name} must be one of"):
            config_cls(**{name: "no-such-value"})

    def test_replace_coerces_too(self, config_cls, name, enum_cls):
        member = getattr(config_cls(), name)
        replaced = dataclasses.replace(config_cls(), **{name: member.value})
        assert getattr(replaced, name) is member


# ---------------------------------------------------------------------------
# RuntimeCorpus: a config selects its chunker through the same table
# ---------------------------------------------------------------------------


class TestRuntimeResolution:
    @pytest.mark.parametrize("name", sorted(_BUILTINS))
    def test_a_builtin_config_selects_its_chunker(self, name):
        config = _BUILTINS[name]()
        chunker = _resolve_chunker(config, None)
        assert chunker.config is config
        # ...also with a caller's registry that lacks the built-ins.
        assert _resolve_chunker(config, ComponentRegistry()).config is config

    def test_a_registered_config_selects_its_chunker(self):
        @dataclasses.dataclass(frozen=True)
        class MyConfig(ChunkerConfig):
            size: int = 3

        class Mine(_ProtocolChunker):
            def __init__(self, config=None):
                self.config = config

        registry = ComponentRegistry()
        registry.register_chunker("mine", Mine, config=MyConfig)
        config = MyConfig(size=5)
        assert _resolve_chunker(config, registry).config is config
        with pytest.raises(TypeError, match="no registered chunker takes a MyConfig"):
            _resolve_chunker(config, None)


# ---------------------------------------------------------------------------
# Independent review of 3a: one test per finding
# ---------------------------------------------------------------------------


class TestReviewFindings:
    def test_builtins_never_replace_an_earlier_registration(self, fresh):
        """F1: the builder calls register_builtins on the shared registry."""
        registry = ComponentRegistry()

        class MyWord(_ProtocolChunker):
            pass

        registry.register_chunker("word", MyWord)
        registry.register_builtins()
        assert registry.get_chunker("word") is MyWord
        assert registry.get_chunker("sentence") is SentenceChunker
        assert "word" not in registry._builtin_keys["chunker"]
        assert "sentence" in registry._builtin_keys["chunker"]

    def test_a_user_chunker_registered_before_the_first_build_is_kept(
        self, tmp_path, shared_registry, monkeypatch
    ):
        monkeypatch.setattr(shared_registry, "_builtins_registered", False)
        monkeypatch.setattr(shared_registry, "_chunkers", {})
        monkeypatch.setattr(shared_registry, "_chunker_configs", {})
        shared_registry.register_chunker("word", _ProtocolChunker)
        result = _build(tmp_path, chunker="word")
        assert [d.text for d in result.documents] == [_TEXT]

    def test_plans_and_the_builder_share_one_table(self, shared_registry):
        """F2: a chunker registered on the shared registry resolves in plans."""
        shared_registry.register_chunker("mine", _ProtocolChunker)
        assert isinstance(_resolve_chunker("mine", None), _ProtocolChunker)

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("word", {"gensim_dictionary": None}),
            (
                "fixed_window",
                {"config": FixedWindowChunkerConfig(), "multilang_config": None},
            ),
            ("semantic", {"adapter": None, "backend": "morphological"}),
        ],
    )
    def test_constructor_options_still_reach_the_constructor(self, fresh, name, kwargs):
        """F3: options the constructor takes (not the config) keep working."""
        assert fresh.build_chunker(name, **kwargs) is not None

    def test_a_field_beside_config_is_refused(self, fresh):
        with pytest.raises(TypeError, match="with config="):
            fresh.build_chunker("sentence", config=SentenceChunkerConfig(), overlap=1)

    def test_config_none_means_the_default(self, fresh):
        """F14."""
        assert (
            fresh.build_chunker("sentence", config=None).config
            == SentenceChunkerConfig()
        )

    def test_normalizer_kwargs_configure_their_step(self, tmp_path):
        """F4: the field used to be read nowhere."""
        builder = CorpusBuilder(
            BuilderConfig(
                normalizer_steps=["unicode"],
                normalizer_kwargs={"unicode": {"form": "NFKC"}},
            )
        )
        (step,) = builder._get_normalizer_pipeline().steps
        assert step.form == "NFKC"

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"form": "NFKC"}, ValueError, "not in normalizer_steps"),
            ({"unicode": "NFKC"}, TypeError, "mapping"),
            ({"unicode": {"forme": "NFKC"}}, TypeError, "forme"),
        ],
    )
    def test_bad_normalizer_kwargs_raise(self, tmp_path, kwargs, error, match):
        with pytest.raises(error, match=match):
            _build(tmp_path, normalizer_steps=["unicode"], normalizer_kwargs=kwargs)

    def test_a_chunker_class_is_refused(self, tmp_path):
        """F5: a class (not an instance) used to bridge and fail per source."""
        with pytest.raises(TypeError, match="not an instance"):
            _build(tmp_path, chunker=SentenceChunker)

    def test_a_chunker_with_the_wrong_call_is_refused_at_registration(self, fresh):
        class MetadataOnly(_ProtocolChunker):
            def chunk(self, text, metadata=None):
                return super().chunk(text)

        with pytest.raises(TypeError, match="extra_metadata"):
            fresh.register_chunker("meta", MetadataOnly)

    def test_any_registered_config_selects_its_chunker(self):
        """F7: a registered plain dataclass config resolves in plans too."""

        @dataclasses.dataclass(frozen=True)
        class Plain:
            size: int = 1

        class Mine(_ProtocolChunker):
            def __init__(self, config=None):
                self.config = config

        registry = ComponentRegistry()
        registry.register_chunker("mine", Mine, config=Plain)
        assert _resolve_chunker(Plain(4), registry).config == Plain(4)

    def test_a_mismatched_config_is_refused_at_registration(self, fresh):
        """F8: and so a snapshot pairing them is refused at load."""
        with pytest.raises(TypeError, match="takes a SentenceChunkerConfig"):
            fresh.register_chunker("odd", SentenceChunker, config=WordChunkerConfig)
        snapshot = _shipped(fresh.snapshot())
        snapshot["chunker_configs"][
            "sentence"
        ] = "scikitplot.corpus._chunkers._word.WordChunkerConfig"
        with pytest.raises(TypeError, match="takes a SentenceChunkerConfig"):
            ComponentRegistry.load_from_snapshot(snapshot)

    def test_the_protocol_bridge_keeps_a_declared_strategy(self):
        """F9 (a)."""

        class Declared(_ProtocolChunker):
            strategy = "paragraph"

        assert bridge_chunker(Declared()).strategy.value == "paragraph"

    def test_a_mock_is_not_taken_for_a_chunker(self):
        """F9 (b): the check is static, so it does not vary by Python version."""
        mock = MagicMock()
        assert not isinstance(bridge_chunker(mock), ProtocolChunkerBridge)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: SemanticChunkerConfig(adapter_unit="nonsense"),
            lambda: WritingSystemAdapterConfig(unit="nonsense"),
        ],
        ids=["semantic", "adapter"],
    )
    def test_an_unknown_adapter_unit_raises(self, make):
        """F10."""
        with pytest.raises(ValueError, match="nonsense"):
            make()

    def test_options_with_a_factory_are_refused(self, tmp_path):
        """F12."""
        source = tmp_path / "doc.txt"
        source.write_text(_TEXT, encoding="utf-8")
        builder = CorpusBuilder(
            BuilderConfig(chunker_kwargs={"overlap": 1}),
            factories=BuilderFactories(chunker_factory=SentenceChunker),
        )
        with pytest.raises(ValueError, match="chunker_factory"):
            builder.build(source)
