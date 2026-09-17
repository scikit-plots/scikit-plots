"""
Durability-reporting regressions for the atomic publication helper (slice S-3).

``atomic_write_path`` syncs the staging file and then the containing directory.
Both syncs previously caught :class:`OSError` and returned, which gave one
behaviour to a platform that cannot perform the sync and to a disk that failed
the write. These checks pin the distinction: a platform limitation is a weaker
guarantee, a real I/O failure is a failure.

See Also
--------
scikitplot.corpus._atomic.atomic_write_path
scikitplot.corpus._atomic.atomic_write_bytes
"""

import errno
from unittest.mock import patch

import pytest

from .. import _atomic

UNSUPPORTED = [errno.EINVAL, errno.ENOTSUP, errno.ENOSYS]
REAL_FAILURES = [errno.EIO, errno.ENOSPC]


def _write(tmp_path):
    """Publish a small payload and return the target path."""
    target = tmp_path / "payload.bin"
    _atomic.atomic_write_path(target, lambda p: p.write_bytes(b"data"))
    return target


@pytest.mark.parametrize("code", REAL_FAILURES)
def test_real_sync_failure_propagates(tmp_path, code):
    """A disk that fails the sync is reported, not reported as success."""
    with patch.object(_atomic.os, "fsync", side_effect=OSError(code, "injected")):
        with pytest.raises(OSError) as excinfo:
            _write(tmp_path)
    assert excinfo.value.errno == code


@pytest.mark.parametrize("code", UNSUPPORTED)
def test_unsupported_sync_is_downgraded_not_failed(tmp_path, code):
    """A platform that cannot sync still publishes; the guarantee is weaker."""
    with patch.object(_atomic.os, "fsync", side_effect=OSError(code, "injected")):
        target = _write(tmp_path)
    assert target.is_file()
    assert target.read_bytes() == b"data"


@pytest.mark.parametrize("code", UNSUPPORTED)
def test_downgrade_is_reported(tmp_path, code, caplog):
    """The weaker guarantee is stated, naming the errno, not passed over silently."""
    with caplog.at_level("WARNING", logger=_atomic.__name__):
        with patch.object(_atomic.os, "fsync", side_effect=OSError(code, "injected")):
            _write(tmp_path)
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert errno.errorcode[code] in messages


@pytest.mark.parametrize("code", REAL_FAILURES)
def test_atomic_write_bytes_agrees_with_atomic_write_path(tmp_path, code):
    """The two helpers in this module no longer disagree about a failed sync."""
    with patch.object(_atomic.os, "fsync", side_effect=OSError(code, "injected")):
        with pytest.raises(OSError):
            _atomic.atomic_write_bytes(tmp_path / "a.bin", b"data")
        with pytest.raises(OSError):
            _atomic.atomic_write_path(tmp_path / "b.bin", lambda p: p.write_bytes(b"d"))


def test_no_staging_file_survives_a_failed_sync(tmp_path):
    """A refused publication leaves no temporary file behind."""
    with patch.object(_atomic.os, "fsync", side_effect=OSError(errno.EIO, "injected")):
        with pytest.raises(OSError):
            _write(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_successful_publication_is_unchanged(tmp_path):
    """The ordinary path still publishes exactly the bytes written."""
    target = _write(tmp_path)
    assert target.read_bytes() == b"data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["payload.bin"]
