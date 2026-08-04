# corpus/_export/tests/test__export_security.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Pickle/joblib load-hardening gate (CORPUS-SEC-001)
==================================================

The default trust guard is preserved (pickle/joblib denied unless
``trusted=True``). On top of it: an optional ``expected_sha256`` is verified
*before* deserialization (tamper gate that runs before pickle can execute), and
the deserialized result is validated against the ``list[CorpusDocument]``
contract so a trusted-but-wrong artifact fails loudly.

Run with::

    pytest scikitplot/corpus/_export/tests/test__export_security.py -v
"""

from __future__ import annotations

import hashlib
import pathlib
import pickle

import pytest

from scikitplot.corpus._export._export import (
    _validate_loaded_documents,
    _verify_artifact_integrity,
    load_documents,
)
from scikitplot.corpus._schema import CorpusDocument, SourceType


def _docs(n=3):
    return [
        CorpusDocument.create("f.txt", i, f"document {i}", source_type=SourceType.BOOK)
        for i in range(n)
    ]


def _dump(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTrustGuardPreserved:
    def test_untrusted_pickle_denied(self, tmp_path):
        p = tmp_path / "good.pkl"
        _dump(_docs(), p)
        with pytest.raises(ValueError, match="disabled by default"):
            load_documents(p)  # trusted defaults to False (matches docstring example)

    def test_trusted_round_trip(self, tmp_path):
        p = tmp_path / "good.pkl"
        _dump(_docs(3), p)
        loaded = load_documents(p, trusted=True)
        assert isinstance(loaded, list) and len(loaded) == 3
        assert all(isinstance(x, CorpusDocument) for x in loaded)


class TestTypeValidation:
    def test_wrong_container_type(self, tmp_path):
        p = tmp_path / "dict.pkl"
        _dump({"not": "a list"}, p)
        with pytest.raises(TypeError):
            load_documents(p, trusted=True)

    def test_wrong_element_type(self, tmp_path):
        p = tmp_path / "strs.pkl"
        _dump(["not", "documents"], p)
        with pytest.raises(TypeError):
            load_documents(p, trusted=True)

    def test_validate_helper(self, tmp_path):
        assert _validate_loaded_documents(_docs(2), tmp_path / "x") is not None
        with pytest.raises(TypeError):
            _validate_loaded_documents("a string", tmp_path / "x")


class TestIntegrity:
    def test_correct_hash_loads(self, tmp_path):
        p = tmp_path / "good.pkl"
        _dump(_docs(2), p)
        assert len(load_documents(p, trusted=True, expected_sha256=_sha256(p))) == 2

    def test_hash_case_insensitive(self, tmp_path):
        p = tmp_path / "good.pkl"
        _dump(_docs(1), p)
        assert len(load_documents(p, trusted=True, expected_sha256=_sha256(p).upper())) == 1

    def test_wrong_hash_rejected(self, tmp_path):
        p = tmp_path / "good.pkl"
        _dump(_docs(), p)
        with pytest.raises(ValueError):
            load_documents(p, trusted=True, expected_sha256="0" * 64)

    def test_integrity_runs_before_deserialization(self, tmp_path):
        # A wrong-TYPE artifact with a wrong hash must raise ValueError
        # (integrity), not TypeError (type) — proving the hash gate runs first.
        p = tmp_path / "dict.pkl"
        _dump({"not": "a list"}, p)
        with pytest.raises(ValueError):
            load_documents(p, trusted=True, expected_sha256="0" * 64)

    def test_none_skips(self, tmp_path):
        p = tmp_path / "good.pkl"
        _dump(_docs(1), p)
        assert _verify_artifact_integrity(p, None) is None
