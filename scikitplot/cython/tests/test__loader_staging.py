# scikitplot/cython/tests/test__loader_staging.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-LOAD-002.

Original behaviour staged raw extension bytes with a plain ``write_bytes`` into
a predictable, world-readable path and read-compared for collisions — following
symlinks and leaving a partial-write / TOCTOU window.  The corrected staging:

- creates the content-scoped directory with private (0700) permissions;
- refuses to load when the final path is a symlink (no-follow);
- writes to a unique temp file and atomically ``os.replace``-s into place;
- verifies the published content hash.

These tests exercise the helper directly (no compiler needed) plus the public
entrypoint's validation.
"""
from __future__ import annotations

import os
import stat
from hashlib import sha256
from pathlib import Path

import pytest

from .._loader import _stage_artifact_bytes_atomically, import_extension_from_bytes
from importlib.machinery import EXTENSION_SUFFIXES


DATA = b"ELF_FAKE\x00\x01\x02" + b"\x00" * 64


def _out(tmp_path: Path) -> tuple[Path, Path, str]:
    h = sha256(DATA).hexdigest()
    out_dir = tmp_path / "scikitplot_cython_import" / h[:16]
    out_dir.mkdir(parents=True)
    return out_dir, out_dir / f"mod{EXTENSION_SUFFIXES[0]}", h


class TestSecureStaging:
    def test_writes_and_verifies_new_artifact(self, tmp_path: Path) -> None:
        out_dir, out_path, h = _out(tmp_path)
        _stage_artifact_bytes_atomically(out_dir, out_path, DATA, expected_sha256=h)
        assert out_path.is_file()
        assert out_path.read_bytes() == DATA
        # No temp files left behind.
        assert list(out_dir.glob(".artifact-*")) == []

    def test_idempotent_when_correct_artifact_present(self, tmp_path: Path) -> None:
        out_dir, out_path, h = _out(tmp_path)
        out_path.write_bytes(DATA)  # already correct
        # Should return without error and without rewriting via temp file.
        _stage_artifact_bytes_atomically(out_dir, out_path, DATA, expected_sha256=h)
        assert out_path.read_bytes() == DATA
        assert list(out_dir.glob(".artifact-*")) == []

    def test_collision_on_different_bytes_raises(self, tmp_path: Path) -> None:
        out_dir, out_path, h = _out(tmp_path)
        out_path.write_bytes(b"DIFFERENT")
        with pytest.raises(OSError, match="collision"):
            _stage_artifact_bytes_atomically(out_dir, out_path, DATA, expected_sha256=h)

    @pytest.mark.skipif(
        not hasattr(os, "symlink"), reason="platform lacks symlink support"
    )
    def test_symlink_final_path_is_refused(self, tmp_path: Path) -> None:
        out_dir, out_path, h = _out(tmp_path)
        victim = tmp_path / "victim.bin"
        victim.write_bytes(b"secret-do-not-overwrite")
        try:
            os.symlink(victim, out_path)
        except (OSError, NotImplementedError):  # pragma: no cover
            pytest.skip("symlink not permitted in this environment")
        with pytest.raises(OSError):
            _stage_artifact_bytes_atomically(out_dir, out_path, DATA, expected_sha256=h)
        # Victim untouched.
        assert victim.read_bytes() == b"secret-do-not-overwrite"

    def test_content_scoped_dir_is_private(self, tmp_path: Path) -> None:
        """The content-scoped directory is created 0700 by the public API."""
        # Drive through the public entrypoint far enough to create the dir,
        # using bytes whose import will fail (fake ELF) but staging succeeds.
        filename = f"privmod{EXTENSION_SUFFIXES[0]}"
        with pytest.raises(Exception):  # noqa: B017 - import of fake artifact fails
            import_extension_from_bytes(
                DATA,
                module_name="privmod",
                artifact_filename=filename,
                temp_dir=tmp_path,
            )
        h = sha256(DATA).hexdigest()
        out_dir = tmp_path / "scikitplot_cython_import" / h[:16]
        assert out_dir.is_dir()
        if os.name == "posix":
            mode = stat.S_IMODE(os.lstat(out_dir).st_mode)
            # Owner has access; group/other must not (0700).
            assert mode & 0o077 == 0, f"dir not private: {oct(mode)}"


class TestPublicValidation:
    @pytest.mark.parametrize(
        "bad",
        ["", "has/slash.so", "has\\back.so", "noext"],
    )
    def test_invalid_artifact_filename_rejected(self, bad: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            import_extension_from_bytes(
                DATA, module_name="m", artifact_filename=bad, temp_dir=tmp_path
            )
