# corpus/_readers/tests/test__zip_depth.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Nested-archive depth cap gate (CORPUS-ARC-002)
==============================================

Nested archives are re-dispatched synchronously, so a *shared* depth cap
(carried in a contextvar, set once at the top of the chain) bounds recursion
and refuses a zip-quine / deeply nested-archive bomb with a typed
:class:`ArchiveNestingError` that aborts the whole archive.

Run with::

    pytest scikitplot/corpus/_readers/tests/test__zip_depth.py -v
"""

from __future__ import annotations

import io
import zipfile

import pytest

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._readers._zip import (
    DEFAULT_MAX_ARCHIVE_DEPTH,
    ArchiveNestingError,
    _archive_ctx,
)


def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_nested(path, levels):
    payload = _zip_bytes({"doc.txt": b"deepest"})
    name = "inner.zip"
    for i in range(levels):
        payload = _zip_bytes({name: payload})
        name = f"level{i}.zip"
    path.write_bytes(payload)
    return path


@pytest.fixture(autouse=True)
def _reset_ctx():
    token = _archive_ctx.set(None)
    yield
    _archive_ctx.reset(token)


class TestNestingDepthCap:
    def test_default_is_sane(self):
        assert DEFAULT_MAX_ARCHIVE_DEPTH >= 2

    def test_at_cap_refuses_before_processing(self, tmp_path):
        z = tmp_path / "simple.zip"
        z.write_bytes(_zip_bytes({"a.txt": b"text"}))
        reader = DocumentReader.create(z)
        _archive_ctx.set((reader.max_depth, reader.max_depth))
        with pytest.raises(ArchiveNestingError):
            list(reader.get_raw_chunks())

    def test_deep_nesting_beyond_cap_raises(self, tmp_path):
        nested = _make_nested(tmp_path / "outer.zip", levels=4)
        reader = DocumentReader.create(nested)
        reader.max_depth = 2  # shared cap for the whole chain
        with pytest.raises(ArchiveNestingError):
            list(reader.get_raw_chunks())
        assert _archive_ctx.get() is None  # context restored

    def test_shallow_nesting_within_cap_ok(self, tmp_path):
        nested = _make_nested(tmp_path / "outer.zip", levels=1)
        reader = DocumentReader.create(nested)
        reader.max_depth = 5
        list(reader.get_raw_chunks())  # must not raise
        assert _archive_ctx.get() is None

    def test_error_is_valueerror_subclass(self):
        assert issubclass(ArchiveNestingError, ValueError)
