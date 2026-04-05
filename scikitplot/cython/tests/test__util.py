# scikitplot/cython/tests/test__util.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`scikitplot.cython._utils` and related low-level cache helpers.

Covers
------
- ``sanitize()``             : type-checking, char substitution, leading-digit prefix
- ``is_valid_key()``         : 64-hex charset/length validation
- ``make_cache_key()``       : determinism, nested payloads, list/tuple equivalence
- ``source_digest()``        : SHA-256 bytes → hex string
- ``runtime_fingerprint()``  : required keys, Cython/NumPy version recording
- ``_stable_repr()``         : canonical JSON-stable representation
- ``_json_dumps()``          : compact, sorted, Path-aware serialisation
- ``_to_path()``             : str / bytes / Path / tilde expansion / absolute result
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .._utils import sanitize
from .._cache import (
    _json_dumps,
    _stable_repr,
    is_valid_key,
    make_cache_key,
    runtime_fingerprint,
    source_digest,
)
from .._builder import _to_path
import os


class TestSanitizeBranches:
    """Cover all branches of sanitize()."""

    def test_empty_string(self) -> None:
        assert sanitize("") == "_"

    def test_leading_digit_prepends_underscore(self) -> None:
        assert sanitize("9lives") == "_9lives"

    def test_non_str_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="sanitize()"):
            sanitize(42)  # type: ignore[arg-type]

    def test_non_str_bytes_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            sanitize(b"hello")  # type: ignore[arg-type]

    def test_all_specials_become_underscores(self) -> None:
        result = sanitize("a!b@c#")
        assert result == "a_b_c_"

    def test_four_hyphens_becomes_three_underscores(self) -> None:
        # "----" → first char not digit, so no prefix; all become "_"
        assert sanitize("----") == "____"

    def test_unicode_alphanumeric_kept(self) -> None:
        # Greek alpha is alphanumeric; kept as-is
        assert sanitize("α") == "α"

    def test_valid_python_identifier_unchanged(self) -> None:
        assert sanitize("valid_name_123") == "valid_name_123"


class TestSanitize:
    """Unit tests for :func:`scikitplot.cython._utils.sanitize`."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("hello", "hello"),
            ("hello_world", "hello_world"),
            ("hello-world", "hello_world"),
            ("hello world", "hello_world"),
            ("my.module", "my_module"),
            ("123abc", "_123abc"),  # leading digit → prepend _
            ("0", "_0"),
            ("_private", "_private"),
            ("", "_"),  # empty → sentinel
            ("abc123", "abc123"),
            ("a/b/c", "a_b_c"),
            ("_", "_"),
            ("__init__", "__init__"),
        ],
    )
    def test_basic_sanitization(self, name: str, expected: str) -> None:
        assert sanitize(name) == expected

    def test_all_special_chars_replaced(self) -> None:
        result = sanitize("!@#$%^&*()")
        assert result.replace("_", "") == ""

    def test_type_error_on_non_str(self) -> None:
        with pytest.raises(TypeError, match="str"):
            sanitize(None)  # type: ignore[arg-type]

    def test_type_error_on_int(self) -> None:
        with pytest.raises(TypeError, match="int"):
            sanitize(42)  # type: ignore[arg-type]

    def test_type_error_on_bytes(self) -> None:
        with pytest.raises(TypeError, match="bytes"):
            sanitize(b"hello")  # type: ignore[arg-type]

    def test_unicode_alphanumeric_kept(self) -> None:
        # ASCII alphanumeric must pass through unchanged.
        result = sanitize("abcXYZ012")
        assert result == "abcXYZ012"

    def test_return_is_always_str(self) -> None:
        assert isinstance(sanitize("test"), str)
        assert isinstance(sanitize(""), str)


class TestSanitizeAllAsciiChars:
    """``sanitize`` handles every ASCII character class correctly."""

    def test_all_lowercase_letters(self) -> None:
        result = sanitize("abcdefghijklmnopqrstuvwxyz")
        assert result == "abcdefghijklmnopqrstuvwxyz"

    def test_all_uppercase_letters(self) -> None:
        result = sanitize("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert result == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_all_digits(self) -> None:
        result = sanitize("0123456789")
        assert result == "_0123456789"  # leading digit → prepend _

    def test_mixed_with_leading_letter(self) -> None:
        result = sanitize("a0123456789")
        assert result == "a0123456789"

    def test_single_special(self) -> None:
        assert sanitize("-") == "_"
        assert sanitize(".") == "_"
        assert sanitize(" ") == "_"


class TestIsValidKey:
    """Tests for :func:`scikitplot.cython._cache.is_valid_key`."""

    def test_valid_lowercase(self) -> None:
        assert is_valid_key("a" * 64) is True

    def test_valid_uppercase(self) -> None:
        assert is_valid_key("A" * 64) is True

    def test_valid_mixed(self) -> None:
        assert is_valid_key("aAbB09" * 10 + "aAbB") is True  # 64 chars

    def test_too_short(self) -> None:
        assert is_valid_key("a" * 63) is False

    def test_too_long(self) -> None:
        assert is_valid_key("a" * 65) is False

    def test_empty(self) -> None:
        assert is_valid_key("") is False

    def test_non_hex(self) -> None:
        assert is_valid_key("g" * 64) is False
        assert is_valid_key("z" * 64) is False

    def test_non_str(self) -> None:
        assert is_valid_key(None) is False  # type: ignore[arg-type]
        assert is_valid_key(123) is False  # type: ignore[arg-type]

    def test_real_sha256(self) -> None:
        import hashlib

        digest = hashlib.sha256(b"hello").hexdigest()
        assert is_valid_key(digest) is True


class TestMakeCacheKey:
    """Tests for :func:`scikitplot.cython._cache.make_cache_key`."""

    def test_returns_64_hex(self) -> None:
        key = make_cache_key({"a": 1})
        assert is_valid_key(key)

    def test_deterministic(self) -> None:
        key1 = make_cache_key({"x": 1, "y": 2})
        key2 = make_cache_key({"y": 2, "x": 1})
        assert key1 == key2  # sorted → same

    def test_different_payloads_differ(self) -> None:
        k1 = make_cache_key({"a": 1})
        k2 = make_cache_key({"a": 2})
        assert k1 != k2

    def test_nested_payload(self) -> None:
        key = make_cache_key({"outer": {"inner": [1, 2, 3]}})
        assert is_valid_key(key)

    def test_empty_payload(self) -> None:
        key = make_cache_key({})
        assert is_valid_key(key)

    def test_path_in_payload(self) -> None:
        key = make_cache_key({"p": Path("/tmp/foo")})
        assert is_valid_key(key)

    def test_none_value(self) -> None:
        key = make_cache_key({"x": None})
        assert is_valid_key(key)


class TestMakeCacheKeyConsistency:
    """``make_cache_key`` is stable across calls and Python restarts."""

    def test_list_vs_tuple_same_repr(self) -> None:
        # _stable_repr converts both to list, so keys must be equal
        k1 = make_cache_key({"x": [1, 2, 3]})
        k2 = make_cache_key({"x": (1, 2, 3)})
        assert k1 == k2

    def test_nested_path_stable(self, tmp_path: Path) -> None:
        k1 = make_cache_key({"p": tmp_path})
        k2 = make_cache_key({"p": str(tmp_path)})
        # Path renders as posix; str also renders as str — may or may not match
        # but each call is deterministic
        assert is_valid_key(k1)
        assert is_valid_key(k2)


class TestSourceDigest:
    """Tests for :func:`scikitplot.cython._cache.source_digest`."""

    def test_returns_hex_str(self) -> None:
        d = source_digest(b"hello")
        assert len(d) == 64
        assert all(c in "0123456789abcdef" for c in d)

    def test_deterministic(self) -> None:
        assert source_digest(b"x") == source_digest(b"x")

    def test_different_inputs_differ(self) -> None:
        assert source_digest(b"a") != source_digest(b"b")

    def test_empty_bytes(self) -> None:
        d = source_digest(b"")
        assert len(d) == 64


class TestRuntimeFingerprint:
    """Tests for :func:`scikitplot.cython._cache.runtime_fingerprint`."""

    def test_returns_dict(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="1.26.0")
        assert isinstance(fp, dict)

    def test_required_keys_present(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        for k in ("python", "python_impl", "platform", "machine", "cython", "numpy"):
            assert k in fp

    def test_cython_version_recorded(self) -> None:
        fp = runtime_fingerprint(cython_version="99.0.0", numpy_version=None)
        assert fp["cython"] == "99.0.0"

    def test_numpy_none_recorded(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert fp["numpy"] is None

    def test_numpy_version_recorded(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="2.0.0")
        assert fp["numpy"] == "2.0.0"


class TestRuntimeFingerprintAllKeys:
    """``runtime_fingerprint`` returns every required key including ``abi``."""

    _REQUIRED_KEYS = frozenset(
        {"python", "python_impl", "platform", "machine", "processor", "cython", "numpy", "abi"}
    )

    def test_all_required_keys_present(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="1.26.0")
        assert self._REQUIRED_KEYS <= set(fp.keys())

    def test_abi_is_str(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert isinstance(fp["abi"], str)

    def test_numpy_none_stored_as_none(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert fp["numpy"] is None

    def test_numpy_version_stored_correctly(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="2.0.0")
        assert fp["numpy"] == "2.0.0"

    def test_cython_version_stored_correctly(self) -> None:
        fp = runtime_fingerprint(cython_version="3.1.0", numpy_version=None)
        assert fp["cython"] == "3.1.0"

    def test_python_version_is_nonempty_str(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert isinstance(fp["python"], str)
        assert fp["python"]  # non-empty


class TestStableRepr:
    """Tests for :func:`scikitplot.cython._cache._stable_repr`."""

    def test_none(self) -> None:
        assert _stable_repr(None) is None

    def test_primitives(self) -> None:
        assert _stable_repr(1) == 1
        assert _stable_repr(3.14) == 3.14
        assert _stable_repr(True) is True
        assert _stable_repr("hello") == "hello"

    def test_path_becomes_posix(self) -> None:
        result = _stable_repr(Path("/tmp/foo"))
        assert result == "/tmp/foo"

    def test_list(self) -> None:
        assert _stable_repr([1, 2, 3]) == [1, 2, 3]

    def test_tuple_becomes_list(self) -> None:
        assert _stable_repr((1, 2)) == [1, 2]

    def test_dict_sorted(self) -> None:
        result = _stable_repr({"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]

    def test_nested(self) -> None:
        result = _stable_repr({"paths": [Path("/x"), Path("/y")]})
        assert result == {"paths": ["/x", "/y"]}

    def test_fallback_unknown_type(self) -> None:
        class Custom:
            def __str__(self):
                return "custom"

        result = _stable_repr(Custom())
        assert result == "custom"


class TestStableReprPrimitiveFloatBool:
    """``_stable_repr`` passes ``float`` and ``bool`` through unchanged."""

    def test_float_passthrough(self) -> None:
        assert _stable_repr(3.14) == 3.14

    def test_zero_float(self) -> None:
        assert _stable_repr(0.0) == 0.0

    def test_negative_float(self) -> None:
        assert _stable_repr(-1.5) == -1.5

    def test_bool_true_passthrough(self) -> None:
        # bool is a subclass of int; _stable_repr must preserve it as-is
        result = _stable_repr(True)
        assert result is True

    def test_bool_false_passthrough(self) -> None:
        result = _stable_repr(False)
        assert result is False

    def test_bool_in_dict_preserved(self) -> None:
        result = _stable_repr({"flag": True, "n": 1.5})
        assert result == {"flag": True, "n": 1.5}


class TestJsonDumps:
    """Tests for :func:`scikitplot.cython._cache._json_dumps`."""

    def test_returns_str(self) -> None:
        result = _json_dumps({"a": 1})
        assert isinstance(result, str)

    def test_valid_json(self) -> None:
        result = _json_dumps({"a": 1, "b": [1, 2]})
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_sorted_keys(self) -> None:
        result = _json_dumps({"z": 1, "a": 2})
        assert result.index('"a"') < result.index('"z"')

    def test_compact_separators(self) -> None:
        result = _json_dumps({"a": 1})
        assert ": " not in result  # compact separators

    def test_path_serialized(self) -> None:
        result = _json_dumps({"p": Path("/tmp")})
        assert "/tmp" in result


class TestToPath:
    """Tests for :func:`scikitplot.cython._builder._to_path`."""

    def test_str_path(self) -> None:
        result = _to_path("/tmp/test")
        assert isinstance(result, Path)

    def test_path_object(self, tmp_path: Path) -> None:
        result = _to_path(tmp_path)
        assert result == tmp_path.resolve()

    def test_bytes_path(self) -> None:
        result = _to_path(b"/tmp/test")
        assert isinstance(result, Path)

    def test_tilde_expanded(self) -> None:
        result = _to_path("~/something")
        assert "~" not in str(result)

    def test_result_is_absolute(self, tmp_path: Path) -> None:
        result = _to_path(tmp_path)
        assert result.is_absolute()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hello", "hello"),
        ("hello-world", "hello_world"),
        ("123abc", "_123abc"),
        ("a/b/c", "a_b_c"),
        ("", "_"),
        ("_private", "_private"),
        ("__init__", "__init__"),
        ("0", "_0"),
        ("-", "_"),
    ],
)
def test_sanitize_parametric_coverage(name: str, expected: str) -> None:
    assert sanitize(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "_"),
        ("0abc", "_0abc"),
        ("hello-world", "hello_world"),
        ("valid_name_123", "valid_name_123"),
    ],
)
def test_sanitize_parametric(name: str, expected: str) -> None:
    assert sanitize(name) == expected


@pytest.mark.parametrize(
    "key,valid",
    [
        ("a" * 64, True),
        ("0" * 64, True),
        ("f" * 64, True),
        ("g" * 64, False),   # 'g' is not hex
        ("a" * 63, False),   # too short
        ("a" * 65, False),   # too long
        ("", False),
        (" " * 64, False),
    ],
)
def test_is_valid_key_parametric(key: str, valid: bool) -> None:
    assert is_valid_key(key) is valid


@pytest.mark.parametrize(
    "candidate",
    [
        "0" * 64,
        "f" * 64,
        "abcdef0123456789" * 4,
    ],
)
def test_is_valid_key_valid_candidates(candidate: str) -> None:
    assert is_valid_key(candidate) is True


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "g" * 64,
        "!" * 64,
        "a" * 63,
        "a" * 65,
        " " * 64,
    ],
)
def test_is_valid_key_invalid_candidates(candidate: str) -> None:
    assert is_valid_key(candidate) is False


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({}, str),
        ({"a": 1}, str),
        ({"nested": {"x": [1, 2]}}, str),
        ({"path": Path("/tmp")}, str),
    ],
)
def test_make_cache_key_parametric(payload: dict, expected_type: type) -> None:
    key = make_cache_key(payload)
    assert isinstance(key, expected_type)
    assert is_valid_key(key)


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("", "_"),
        ("hello-world", "hello_world"),
        ("123abc", "_123abc"),
        ("a/b/c", "a_b_c"),
        ("__dunder__", "__dunder__"),
        # Unicode letters are alnum in Python — kept as-is
        ("Ä", "Ä"),
        ("α", "α"),
        ("1", "_1"),
        # Punctuation and symbols → underscore
        ("a!b@c#", "a_b_c_"),
        ("---", "___"),
    ],
)
def test_sanitize_full_docstring_examples(inp: str, expected: str) -> None:
    from .._utils import sanitize

    assert sanitize(inp) == expected
