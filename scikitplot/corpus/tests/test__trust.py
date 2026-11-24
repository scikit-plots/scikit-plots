"""Tests for the ``_trust`` contract (a byte-identical copy in each owner)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from .._trust import CODE_EXECUTING_FORMATS, require_trusted, verify_sha256


@pytest.mark.parametrize("fmt", sorted(CODE_EXECUTING_FORMATS))
@pytest.mark.parametrize("trusted", [False, None, 1, "true", "false"])
def test_code_executing_formats_need_true(fmt, trusted):
    with pytest.raises(ValueError, match=r"disabled by default.*trusted=True to f\(\)"):
        require_trusted(fmt, trusted, caller="f")


def test_true_opts_in_and_other_formats_pass():
    for fmt in CODE_EXECUTING_FORMATS:
        require_trusted(fmt, True, caller="f")
    require_trusted("json", False, caller="f")


@pytest.mark.parametrize("as_str", [False, True])
def test_matching_digest_passes_in_any_case(tmp_path, as_str):
    p = tmp_path / "a.bin"
    p.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    target = str(p) if as_str else Path(p)
    verify_sha256(target, f"  {digest.upper()}\n", caller="f")
    verify_sha256(target, None, caller="f")


def test_mismatch_and_malformed_digests_raise(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"payload")
    with pytest.raises(ValueError, match="does not match"):
        verify_sha256(p, "0" * 64, caller="f")
    for bad in ("abc", "g" * 64, "0" * 63):
        with pytest.raises(ValueError, match="64 hexadecimal digits"):
            verify_sha256(p, bad, caller="f")


def test_a_bad_digest_is_refused_before_the_file_is_opened(tmp_path):
    with pytest.raises(ValueError, match="64 hexadecimal"):
        verify_sha256(tmp_path / "missing", "x", caller="f")
