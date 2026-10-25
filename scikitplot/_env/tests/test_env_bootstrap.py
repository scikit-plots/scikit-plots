# scikitplot/_env/tests/test_env_bootstrap.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Bootstrap: pinned micromamba download/verify/extract, shell start-up block."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from unittest import mock

import pytest

from .. import _bootstrap
from .._bootstrap import _shell
from .._core._errors import ApprovalRequiredError, ConfigError, ExecutionError
from ._helpers import fake_host, write


def _archive(member="bin/micromamba", body=b"#!micromamba\n", kind="file"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo(member)
        if kind == "dir":
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        else:
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pins(data, member="bin/micromamba"):
    return {"linux-64": (hashlib.sha256(data).hexdigest(), member)}


class TestMicromamba:
    def test_pins_cover_the_bootstrap_platforms(self):
        assert set(_bootstrap.MICROMAMBA) == {
            "linux-64", "linux-aarch64", "osx-64", "osx-arm64", "win-64"}
        for sha, member in _bootstrap.MICROMAMBA.values():
            assert len(sha) == 64
            assert member.endswith(("micromamba", "micromamba.exe"))
        info = _bootstrap.micromamba_archive("win-64")
        assert info["url"] == ("https://conda.anaconda.org/conda-forge/win-64/"
                               "micromamba-2.9.0-0.tar.bz2")

    def test_download_verify_extract_and_reuse(self, tmp_path):
        data = _archive()
        opener = mock.Mock(side_effect=lambda url, timeout: _Response(data))
        with mock.patch.object(_bootstrap, "MICROMAMBA", _pins(data)):
            exe = _bootstrap.install_micromamba(tmp_path, "linux-64", opener=opener)
            assert exe == tmp_path / "bin" / "micromamba"
            assert exe.read_bytes() == b"#!micromamba\n"
            assert os.access(exe, os.X_OK)
            # second run: cached archive reused, also offline
            _bootstrap.install_micromamba(tmp_path, "linux-64", offline=True,
                                          opener=opener)
        assert opener.call_count == 1
        url, = opener.call_args.args
        assert url.startswith("https://conda.anaconda.org/")
        assert opener.call_args.kwargs["timeout"] == 60.0

    def test_hash_mismatch_installs_nothing(self, tmp_path):
        data = _archive()
        with mock.patch.object(_bootstrap, "MICROMAMBA", {"linux-64": ("0" * 64,
                                                                       "bin/micromamba")}):
            with pytest.raises(ExecutionError, match="SHA-256"):
                _bootstrap.install_micromamba(
                    tmp_path, "linux-64", opener=lambda url, timeout: _Response(data))
        assert not (tmp_path / "bin" / "micromamba").exists()
        assert list((tmp_path / "cache").iterdir()) == []

    def test_bad_cached_archive_is_downloaded_again(self, tmp_path):
        data = _archive()
        write(tmp_path / "cache" / "linux-64-micromamba-2.9.0-0.tar.bz2", "corrupt")
        with mock.patch.object(_bootstrap, "MICROMAMBA", _pins(data)):
            _bootstrap.install_micromamba(
                tmp_path, "linux-64", opener=lambda url, timeout: _Response(data))
        assert (tmp_path / "bin" / "micromamba").exists()

    def test_offline_without_cache(self, tmp_path):
        with pytest.raises(ExecutionError, match="not cached") as info:
            _bootstrap.install_micromamba(tmp_path, "linux-64", offline=True)
        assert "conda.anaconda.org" in info.value.hint

    def test_member_missing_or_not_a_file(self, tmp_path):
        for data in (_archive(member="bin/other"), _archive(kind="dir")):
            with mock.patch.object(_bootstrap, "MICROMAMBA", _pins(data)):
                with pytest.raises(ExecutionError):
                    _bootstrap.install_micromamba(
                        tmp_path / str(len(data)), "linux-64",
                        opener=lambda url, timeout, d=data: _Response(d))
            assert not (tmp_path / str(len(data)) / "bin" / "micromamba").exists()

    def test_http_error_and_unknown_platform(self, tmp_path):
        response = _Response(b"")
        response.status = 404
        with pytest.raises(ExecutionError, match="404"):
            _bootstrap.install_micromamba(tmp_path, "linux-64",
                                          opener=lambda url, timeout: response)
        with pytest.raises(ExecutionError, match="pinned"):
            _bootstrap.install_micromamba(tmp_path, "linux-ppc64le")

    def test_only_https(self, tmp_path):
        with pytest.raises(ExecutionError, match="not https"):
            _bootstrap._download("http://x/y", tmp_path / "f", timeout=1,
                                 opener=mock.Mock())


class TestShell:
    def test_blocks(self):
        bash = _shell.shell_block("bash", "/t", micromamba="/t/bin/micromamba",
                                  root_prefix="/r")
        assert "export PATH='/t/bin':\"$PATH\"" in bash
        assert "eval \"$('/t/bin/micromamba' shell hook --shell bash)\"" in bash
        assert "fish_add_path" in _shell.shell_block("fish", "/t")
        assert "$env:PATH" in _shell.shell_block("powershell", "/t")
        with pytest.raises(ConfigError, match="unknown shell"):
            _shell.shell_block("tcsh", "/t")
        with pytest.raises(ConfigError, match="contains"):
            _shell.shell_block("bash", "/it's/here")

    def test_write_is_idempotent_and_replaces(self, tmp_path):
        rc = write(tmp_path / ".bashrc", "alias ll='ls -l'")
        os.chmod(rc, 0o600)
        block = _shell.shell_block("bash", "/t")
        assert _shell.write_shell_block(rc, block)
        assert not _shell.write_shell_block(rc, block)
        text = rc.read_text()
        assert text.startswith("alias ll='ls -l'\n# >>> scikitplot env >>>")
        assert _shell.write_shell_block(rc, _shell.shell_block("bash", "/other"))
        text = rc.read_text()
        assert text.count(_shell.BEGIN) == 1
        assert "/other/bin" in text
        assert "/t/bin" not in text
        assert (rc.stat().st_mode & 0o777) == 0o600

    def test_unterminated_block_refused(self, tmp_path):
        rc = write(tmp_path / ".zshrc", f"{_shell.BEGIN}\nexport X=1\n")
        with pytest.raises(ConfigError, match="without"):
            _shell.write_shell_block(rc, _shell.shell_block("zsh", "/t"))
        assert rc.read_text() == f"{_shell.BEGIN}\nexport X=1\n"

    def test_rc_files(self, tmp_path):
        assert _shell.rc_file("bash", home=str(tmp_path)) == tmp_path / ".bashrc"
        assert _shell.rc_file("zsh", home="/h", environ={"ZDOTDIR": str(tmp_path)}) == \
            tmp_path / ".zshrc"
        with pytest.raises(ConfigError, match="powershell") as info:
            _shell.rc_file("powershell")
        assert "$PROFILE" in info.value.hint

    def test_api_needs_approval_to_write(self, tmp_path):
        from .._api import shell_init  # noqa: PLC0415

        host = fake_host(tmp_path, tools={"micromamba": "/m/micromamba"})
        rc = tmp_path / ".bashrc"
        block = shell_init("bash", host=host)
        assert "/m/micromamba" in block
        assert not rc.exists()
        with pytest.raises(ApprovalRequiredError):
            shell_init("bash", write=True, rc_file=rc, interaction="never", host=host)
        assert not rc.exists()
        shell_init("bash", write=True, rc_file=rc, approve=["shell_init"], host=host)
        assert rc.read_text() == block


class TestShellReviewFindings:
    def test_symlinked_rc_is_written_through(self, tmp_path):
        real = write(tmp_path / "dotfiles" / "bashrc", "# mine\n")
        link = tmp_path / ".bashrc"
        link.symlink_to(real)
        _shell.write_shell_block(link, _shell.shell_block("bash", "/t"))
        assert link.is_symlink()
        assert _shell.BEGIN in real.read_text()

    def test_non_utf8_rc_refused(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_bytes(b"\xff\xfe latin junk\n")
        with pytest.raises(ConfigError, match="not UTF-8"):
            _shell.write_shell_block(rc, _shell.shell_block("bash", "/t"))
        assert rc.read_bytes() == b"\xff\xfe latin junk\n"
