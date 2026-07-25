# scikitplot/cython/tests/test__cache_containment.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-CACHE-002.

``_artifact_from_meta_or_guess`` trusted the ``artifact`` field in ``meta.json``
verbatim — an absolute path was used as-is — so a tampered meta could redirect
the loader to an artifact **outside** the cache entry, with no integrity check.

The fix:
- the recorded artifact name is treated as a basename contained in build_dir;
  absolute paths, separators, and traversal are rejected;
- when ``meta`` records ``artifact_sha256``, the on-disk artifact's hash must
  match or a ``ValueError`` is raised.
"""
from __future__ import annotations

from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest

from .._cache import _artifact_from_meta_or_guess, find_entry_by_key, make_cache_key, write_meta


SUF = EXTENSION_SUFFIXES[0]


class TestContainment:
    def test_external_absolute_artifact_not_selected(self, tmp_path: Path) -> None:
        external = tmp_path / f"evil{SUF}"
        external.write_bytes(b"\x7fELF-EVIL")
        entry = tmp_path / "entry"
        entry.mkdir()
        result = _artifact_from_meta_or_guess(entry, {"artifact": str(external)})
        assert result is None  # external artifact refused

    def test_traversal_artifact_not_selected(self, tmp_path: Path) -> None:
        (tmp_path / f"evil{SUF}").write_bytes(b"ELF")
        entry = tmp_path / "entry"
        entry.mkdir()
        result = _artifact_from_meta_or_guess(entry, {"artifact": f"../evil{SUF}"})
        assert result is None

    def test_contained_basename_selected(self, tmp_path: Path) -> None:
        entry = tmp_path / "entry"
        entry.mkdir()
        art = entry / f"mod{SUF}"
        art.write_bytes(b"ELF")
        result = _artifact_from_meta_or_guess(entry, {"artifact": f"mod{SUF}"})
        assert result == art

    def test_find_entry_by_key_refuses_external(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        key = make_cache_key({"x": "containment"})
        entry = root / key
        entry.mkdir()
        external = tmp_path / f"evil{SUF}"
        external.write_bytes(b"ELF")
        write_meta(entry, {"kind": "module", "module_name": "victim",
                           "artifact": str(external)})
        with pytest.raises(FileNotFoundError):
            find_entry_by_key(root, key)


class TestIntegrity:
    def test_matching_hash_accepted(self, tmp_path: Path) -> None:
        entry = tmp_path / "entry"
        entry.mkdir()
        art = entry / f"mod{SUF}"
        data = b"ELF-CONTENT"
        art.write_bytes(data)
        meta = {"artifact": f"mod{SUF}", "artifact_sha256": sha256(data).hexdigest()}
        assert _artifact_from_meta_or_guess(entry, meta) == art

    def test_mismatched_hash_raises(self, tmp_path: Path) -> None:
        entry = tmp_path / "entry"
        entry.mkdir()
        art = entry / f"mod{SUF}"
        art.write_bytes(b"ELF-CONTENT")
        meta = {"artifact": f"mod{SUF}", "artifact_sha256": "0" * 64}  # wrong
        with pytest.raises(ValueError, match="integrity"):
            _artifact_from_meta_or_guess(entry, meta)

    def test_missing_hash_falls_back_to_containment(self, tmp_path: Path) -> None:
        """No recorded hash → containment still applies, no integrity error."""
        entry = tmp_path / "entry"
        entry.mkdir()
        art = entry / f"mod{SUF}"
        art.write_bytes(b"ELF")
        result = _artifact_from_meta_or_guess(entry, {"artifact": f"mod{SUF}"})
        assert result == art
