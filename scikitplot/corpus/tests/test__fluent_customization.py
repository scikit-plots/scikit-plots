from __future__ import annotations

import pytest

from scikitplot.corpus import ConfigConflictError, FluentCorpus


def test_from_config_is_order_independent() -> None:
    a = FluentCorpus.from_config({"storage": "memory", "chunker": "paragraph"})
    b = FluentCorpus.from_config({"chunker": "paragraph", "storage": "memory"})
    assert a.plan() == b.plan()


def test_configure_multiple_domains_preserves_error_by_default() -> None:
    base = FluentCorpus().chunker("sentence")
    with pytest.raises(ConfigConflictError):
        base.configure({"chunker": "paragraph", "storage": "memory"})
    assert base.plan().get("storage") is None


def test_with_overrides_is_explicit_multi_domain_replace() -> None:
    base = FluentCorpus().chunker("sentence").storage("disk")
    tuned = base.with_overrides(chunker="paragraph", storage="memory")
    assert base.plan().get("chunker") == "sentence"
    assert tuned.plan().get("chunker") == "paragraph"
    assert tuned.plan().get("storage") == "memory"


def test_without_removes_only_selected_domains() -> None:
    base = FluentCorpus().chunker("sentence").storage("memory")
    result = base.without("storage")
    assert result.plan().get("chunker") == "sentence"
    assert result.plan().get("storage") is None


def test_explain_is_json_like_and_reports_validation() -> None:
    explanation = FluentCorpus().chunker("sentence").explain()
    assert explanation["valid"] is True
    assert explanation["configured"] == ["chunker"]
    assert isinstance(explanation["problems"], list)
    assert len(explanation["fingerprint"]) == 64


def test_diff_reports_domain_change_without_materializing() -> None:
    left = FluentCorpus().chunker("sentence").storage("memory")
    right = left.with_overrides(chunker="paragraph")
    diff = left.diff(right)
    assert set(diff["changed"]) == {"chunker"}
    assert diff["stages_changed"] is False


def test_variants_generate_bounded_canonical_grid() -> None:
    base = FluentCorpus().storage("memory")
    variants = base.variants(
        max_variants=4,
        chunker=["sentence", "paragraph"],
        retrieval=["lexical", "hybrid"],
    )
    assert len(variants) == 4
    assert [v.plan().get("chunker") for v in variants] == [
        "sentence",
        "sentence",
        "paragraph",
        "paragraph",
    ]
    assert len({v.plan().fingerprint for v in variants}) == 4


def test_variants_refuse_explosive_grid_before_generation() -> None:
    with pytest.raises(ValueError, match="exceeding max_variants"):
        FluentCorpus().variants(max_variants=3, chunker=[1, 2], storage=[1, 2])


def test_variants_treat_string_as_one_fragment_not_character_axis() -> None:
    variants = FluentCorpus().variants(chunker="sentence")
    assert len(variants) == 1
    assert variants[0].plan().get("chunker") == "sentence"


def test_variants_treat_mapping_fragment_as_one_choice() -> None:
    fragment = {"kind": "custom", "window": 8}
    variants = FluentCorpus().variants(reader=fragment)
    assert len(variants) == 1
    assert variants[0].plan().get("reader") == fragment


def test_variants_accept_scalar_non_iterable_as_one_choice() -> None:
    variants = FluentCorpus().variants(storage=123)
    assert len(variants) == 1
    assert variants[0].plan().get("storage") == 123


def test_iter_variants_is_lazy_after_bounded_preflight() -> None:
    base = FluentCorpus().storage("memory")
    generated = base.iter_variants(
        max_variants=4,
        chunker=["sentence", "paragraph"],
        retrieval=["lexical", "hybrid"],
    )
    assert iter(generated) is generated
    first = next(generated)
    assert first.plan().get("chunker") == "sentence"
    assert first.plan().get("retrieval") == "lexical"
    remaining = list(generated)
    assert len(remaining) == 3


def test_iter_variants_refuses_large_grid_before_first_yield() -> None:
    generated = FluentCorpus().iter_variants(
        max_variants=3,
        chunker=[1, 2],
        storage=[1, 2],
    )
    with pytest.raises(ValueError, match="exceeding max_variants"):
        next(generated)


def test_variants_and_iter_variants_have_identical_fingerprint_order() -> None:
    base = FluentCorpus().storage("memory")
    kwargs = {
        "chunker": ["sentence", "paragraph"],
        "retrieval": ["lexical", "hybrid"],
    }
    eager = base.variants(**kwargs)
    lazy = tuple(base.iter_variants(**kwargs))
    assert [item.plan().fingerprint for item in eager] == [
        item.plan().fingerprint for item in lazy
    ]
