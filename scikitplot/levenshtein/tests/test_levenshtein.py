from __future__ import annotations

import pytest

from .. import _core as core
from .. import (
    backend_info,
    closest,
    distance,
    normalized_similarity,
    rank,
)


def test_classic_distance() -> None:
    assert distance("kitten", "sitting", backend="python") == 3


def test_sequence_distance() -> None:
    assert distance(["spam", "egg"], ["spam", "ham"], backend="python") == 1


def test_normalized_similarity_bounds() -> None:
    assert normalized_similarity("abc", "abc", backend="python") == 1.0
    assert normalized_similarity("", "", backend="python") == 1.0
    assert normalized_similarity("abc", "xyz", backend="python") == 0.0


def test_auto_backend_is_always_available() -> None:
    assert backend_info("auto").available is True


def test_missing_explicit_backend_can_fallback(monkeypatch) -> None:
    real = core._backend_info

    def fake(name):
        if name == "levenshtein":
            return core.BackendInfo(
                name="levenshtein",
                available=False,
                version=None,
                license="GPL-2.0-or-later",
                implementation="Levenshtein",
                reason="missing",
            )
        return real(name)

    monkeypatch.setattr(core, "_backend_info", fake)
    messages = []
    monkeypatch.setattr(core.logger, "warning", lambda msg, *args: messages.append(msg % args))
    assert distance("a", "b", backend="levenshtein", strict=False) == 1
    assert any("falling back" in message for message in messages)


def test_missing_explicit_backend_strict_raises(monkeypatch) -> None:
    real = core._backend_info

    def fake(name):
        if name == "levenshtein":
            return core.BackendInfo(
                name="levenshtein",
                available=False,
                version=None,
                license="GPL-2.0-or-later",
                implementation="Levenshtein",
                reason="missing",
            )
        return real(name)

    monkeypatch.setattr(core, "_backend_info", fake)
    with pytest.raises(ImportError):
        distance("a", "b", backend="levenshtein", strict=True)


def test_rank_is_stable_on_tie() -> None:
    values = ["cat", "bat", "hat"]
    result = rank("mat", values, backend="python")
    assert [m.choice for m in result] == values


def test_closest_empty_returns_none() -> None:
    assert closest("x", [], backend="python") is None


def test_rapidfuzz_backend_when_available_matches_python() -> None:
    info = backend_info("rapidfuzz")
    if not info.available:
        pytest.skip("RapidFuzz not installed")
    assert distance("kitten", "sitting", backend="rapidfuzz") == 3


def test_rank_reports_actual_runtime_fallback_backend(monkeypatch) -> None:
    monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "rapidfuzz")

    def broken_impl(_name):
        def fail(a, b):
            raise RuntimeError("accelerator broke")
        return fail

    monkeypatch.setattr(core, "_distance_impl", broken_impl)
    monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: None)
    match = core.rank("kitten", ["sitting"], backend="auto")[0]
    assert match.distance == 3
    assert match.backend == "python"


def test_rapidfuzz_matches_python_on_deterministic_unicode_grid() -> None:
    info = backend_info("rapidfuzz")
    if not info.available:
        pytest.skip("RapidFuzz not installed")
    values = ["", "a", "kitten", "sitting", "café", "cafe\u0301", "İstanbul", "東京", "🙂🙃"]
    for left in values:
        for right in values:
            assert distance(left, right, backend="rapidfuzz") == distance(
                left, right, backend="python"
            )


def test_rank_limit_streams_top_k_without_requiring_sequence() -> None:
    seen: list[str] = []

    def choices():
        for value in ("sitting", "kitten", "bitten", "written"):
            seen.append(value)
            yield value

    matches = rank("kitten", choices(), limit=2, backend="python")
    assert [m.choice for m in matches] == ["kitten", "bitten"]
    assert seen == ["sitting", "kitten", "bitten", "written"]


def test_rank_zero_limit_does_not_consume_choices() -> None:
    consumed = False

    def choices():
        nonlocal consumed
        consumed = True
        yield "anything"

    assert rank("query", choices(), limit=0, backend="python") == []
    assert consumed is False


def test_import_facade_does_not_import_optional_accelerators() -> None:
    import subprocess
    import sys
    import textwrap

    source = textwrap.dedent(
        """
        import sys
        import scikitplot
        watched = {"rapidfuzz", "Levenshtein", "scikitplot.corpus"}
        before = {m for m in watched if m in sys.modules}
        import scikitplot.levenshtein
        after = {m for m in watched if m in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    loaded = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert loaded == ""


def test_levenshtein_never_imports_corpus() -> None:
    """The adapter lives in corpus; this algorithm imports nothing above it."""
    import ast
    from pathlib import Path

    root = Path(core.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            offenders += [
                f"{path.name}:{node.lineno}"
                for name in names
                if name.startswith("scikitplot.corpus")
                or (isinstance(node, ast.ImportFrom) and node.level and "corpus" in name)
            ]
    assert offenders == []
    assert not hasattr(core, "make_corpus_scorer")


# -- one execution plan per call (LV-002, LV-001) -----------------------------


#: Choices per call in the LV-002 tests: enough that per-candidate work shows.
_MANY = 50
_MANY_SAME = 200


class _Counter:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)


def test_rank_resolves_the_backend_once_per_call(monkeypatch) -> None:
    counter = _Counter(core._resolve_backend_name)
    monkeypatch.setattr(core, "_resolve_backend_name", counter)
    matches = rank("kitten", [f"kitten{i}" for i in range(_MANY)], backend="auto")
    assert len(matches) == _MANY
    assert counter.calls == 1


def _break(monkeypatch, broken: str) -> list:
    """Make ``broken`` raise at runtime; return the list warnings land in."""
    real = core._distance_impl

    def impl(name):
        if name == broken:
            def fail(a, b):
                raise RuntimeError("accelerator broke")
            return fail
        return real(name)

    warnings_seen: list = []
    monkeypatch.setattr(core, "_distance_impl", impl)
    monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: warnings_seen.append(args))
    return warnings_seen


def test_a_broken_accelerator_warns_once_per_rank_call(monkeypatch) -> None:
    monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "internal")
    seen = _break(monkeypatch, "internal")
    matches = rank("kitten", ["sitting", "kitten", "mitten"] * 20, backend="auto")
    assert len(seen) == 1
    assert {m.backend for m in matches} <= {"rapidfuzz", "python"}
    assert [m.distance for m in matches[:3]] == [0, 0, 0]


def test_a_runtime_failure_tries_the_next_safe_backend(monkeypatch) -> None:
    if not backend_info("rapidfuzz").available:
        pytest.skip("RapidFuzz not installed")
    monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "internal")
    _break(monkeypatch, "internal")
    assert {m.backend for m in rank("kitten", ["sitting", "mitten"])} == {"rapidfuzz"}


def test_strict_runtime_failure_still_raises(monkeypatch) -> None:
    monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "internal")
    _break(monkeypatch, "internal")
    with pytest.raises(RuntimeError, match="accelerator broke"):
        rank("kitten", ["sitting"], backend="internal", strict=True)


def test_an_unavailable_backend_warns_once_per_rank_call(monkeypatch) -> None:
    real = core._backend_info

    def info(name):
        if name == "rapidfuzz":
            return core.BackendInfo("rapidfuzz", False, None, "MIT", "rapidfuzz", "simulated absence")
        return real(name)

    seen: list = []
    monkeypatch.setattr(core, "_backend_info", info)
    monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: seen.append(args))
    matches = rank("kitten", ["sitting"] * _MANY_SAME, backend="rapidfuzz")
    assert len(matches) == _MANY_SAME
    assert len(seen) == 1


def test_invalid_backends_fail_even_without_choices(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        rank("kitten", [], backend="nope")
    monkeypatch.setattr(
        core,
        "_backend_info",
        lambda name: core.BackendInfo(name, name == "python", None, "MIT", name, "simulated"),
    )
    with pytest.raises(ImportError, match="unavailable"):
        rank("kitten", [], backend="rapidfuzz", strict=True)

def test_rank_score_cutoff_filters_distant_matches() -> None:
    matches = rank(
        "kitten",
        ["kitten", "bitten", "completely different"],
        backend="python",
        score_cutoff=0.7,
    )
    assert [m.choice for m in matches] == ["kitten", "bitten"]


def test_rank_score_cutoff_validates_bounds() -> None:
    with pytest.raises(ValueError, match="score_cutoff"):
        rank("a", ["a"], score_cutoff=1.1, backend="python")


def test_closest_respects_score_cutoff() -> None:
    assert closest("kitten", ["xxxxxx"], score_cutoff=0.8, backend="python") is None


def test_match_to_dict_reports_backend_provenance() -> None:
    match = rank("kitten", ["sitting"], backend="python")[0]
    assert match.to_dict()["backend"] == "python"


def test_top_level_scikitplot_lazy_attribute_exposes_levenshtein() -> None:
    import scikitplot

    assert "levenshtein" in scikitplot._submodules
    assert scikitplot.levenshtein.distance(
        "kitten",
        "sitting",
        backend="python",
    ) == 3


# -- exact element equality across backends ----------------------------------
#
# Accelerated backends compare element hashes; CPython collides
# hash(-1) == hash(-2) and hash(0) == hash(2**61 - 1). Before the facade coded
# inputs exactly, distance([-1], [-2]) was 0 on the default backend and 1 on
# the Python one.

_HASH_COLLISIONS = [
    ([-1], [-2]),
    ([0], [2**61 - 1]),
    ([(-1,)], [(-2,)]),
]


def _available_backends():
    return [info.name for info in core.available_backends() if info.available]


@pytest.mark.parametrize(("a", "b"), _HASH_COLLISIONS)
def test_equal_hashes_are_not_equal_elements_on_any_backend(a, b) -> None:
    assert hash(a[0]) == hash(b[0])  # the premise this guards against
    for name in _available_backends():
        assert distance(a, b, backend=name, strict=True) == 1, name


def test_the_default_backend_agrees_with_python_on_hash_collisions() -> None:
    for a, b in _HASH_COLLISIONS:
        assert distance(a, b) == distance(a, b, backend="python")


def test_container_equality_is_shared_by_every_backend() -> None:
    nan = float("nan")
    cases = [
        (([nan, "a"], [nan, "b"]), 1),  # the same object matches itself
        (([float("nan")], [float("nan")]), 1),  # distinct NaNs do not
        (([1, 2], [1.0, 2]), 0),  # 1 == 1.0
        (("ab", ["a", "b"]), 0),  # a str and its characters
    ]
    for (a, b), expected in cases:
        for name in _available_backends():
            assert distance(a, b, backend=name, strict=True) == expected, (name, a, b)


def test_rank_orders_integer_sequences_exactly() -> None:
    match = rank([-1], [[-2], [-1]], backend="auto")
    assert [m.choice for m in match] == [[-1], [-2]]
    assert match[0].distance == 0
    assert match[1].distance == 1


def test_iterators_are_consumed_once() -> None:
    assert distance(iter([1, 2, 3]), iter([1, 2, 4])) == 1


def test_unhashable_elements_still_fall_back_to_python(monkeypatch) -> None:
    monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: None)
    value, used = core._distance_with_backend([[1], [2]], [[1], [3]], backend="auto")
    assert value == 1
    assert used == "python"
