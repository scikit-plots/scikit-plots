"""``BaseDecorator.deserialize`` refuses to unpickle unless trusted."""

from __future__ import annotations

import hashlib

import joblib
import pytest

from .._decorate import DummyDecorator


class _Marker:
    """Unpickling this writes a file: proof that code from the file ran."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (_touch, (self.path,))


def _touch(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ran")
    return None


@pytest.fixture
def hostile(tmp_path):
    marker = tmp_path / "marker"
    path = tmp_path / "state.joblib"
    joblib.dump(_Marker(str(marker)), path)
    return path, marker


@pytest.mark.parametrize("trusted", [False, None, "true"])
def test_untrusted_load_raises_before_unpickling(hostile, trusted):
    path, marker = hostile
    with pytest.raises(ValueError, match=r"DummyDecorator\.deserialize.*trusted=True"):
        DummyDecorator.deserialize(path, trusted=trusted)
    assert not marker.exists()


def test_digest_mismatch_raises_before_unpickling(hostile):
    path, marker = hostile
    with pytest.raises(ValueError, match="does not match"):
        DummyDecorator.deserialize(str(path), trusted=True, expected_sha256="0" * 64)
    assert not marker.exists()


def test_trusted_load_with_matching_digest_runs(hostile):
    path, marker = hostile
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    DummyDecorator.deserialize(path, trusted=True, expected_sha256=digest)
    assert marker.read_text(encoding="utf-8") == "ran"


def test_missing_file_still_returns_none(tmp_path):
    assert DummyDecorator.deserialize(tmp_path / "absent", trusted=True) is None
