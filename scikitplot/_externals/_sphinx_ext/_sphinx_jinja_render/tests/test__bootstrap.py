"""
Tests for _url_helper._bootstrap.

Critical fix from last session
--------------------------------
The original test ``test_unreadable_file_raises_permission_error`` used
``chmod 0o000`` to trigger a PermissionError.  This silently *passes*
(i.e. the test does NOT raise) when running as root because root bypasses
all filesystem permission checks.

Root cause (5 whys):

1. Test fails to trigger PermissionError.
2. ``chmod 0o000`` has no effect for root user.
3. Linux ignores DAC (Discretionary Access Control) for UID 0.
4. CI and container environments frequently run as root.
5. The test never validated the production code path it claimed to test.

Fix: replace ``chmod`` with ``unittest.mock.patch`` so the PermissionError
is injected directly, regardless of the running user.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .._bootstrap import load_bootstrap_code
from .._constants import (
    BOOTSTRAP_CODE_FILENAME,
    FILE_ENCODING,
    WASM_BOOTSTRAP_CODE,
)


class TestLoadBootstrapCodeHappyPath:
    def test_returns_string(self) -> None:
        result = load_bootstrap_code(pkg_dir=Path(__file__).parent.parent)
        assert isinstance(result, str)

    def test_returns_non_empty_string(self) -> None:
        result = load_bootstrap_code(pkg_dir=Path(__file__).parent.parent)
        assert result.strip()

    def test_uses_override_file_when_present(self, tmp_path: Path) -> None:
        override = "# custom bootstrap\n"
        (tmp_path / BOOTSTRAP_CODE_FILENAME).write_text(
            override, encoding=FILE_ENCODING
        )
        result = load_bootstrap_code(pkg_dir=tmp_path)
        assert result == override

    def test_falls_back_to_embedded_when_file_absent(self, tmp_path: Path) -> None:
        result = load_bootstrap_code(pkg_dir=tmp_path)
        assert result == WASM_BOOTSTRAP_CODE

    def test_custom_filename_override(self, tmp_path: Path) -> None:
        custom_name = "my_custom_bootstrap.txt"
        content = "custom content\n"
        (tmp_path / custom_name).write_text(content, encoding=FILE_ENCODING)
        result = load_bootstrap_code(pkg_dir=tmp_path, filename=custom_name)
        assert result == content

    def test_custom_filename_absent_returns_embedded(self, tmp_path: Path) -> None:
        result = load_bootstrap_code(pkg_dir=tmp_path, filename="nonexistent.txt")
        assert result == WASM_BOOTSTRAP_CODE

    def test_default_pkg_dir_returns_non_empty(self) -> None:
        """No pkg_dir passed — uses the real package directory."""
        result = load_bootstrap_code()
        assert isinstance(result, str)
        assert result.strip()


class TestLoadBootstrapCodePermissionError:
    """
    Verify PermissionError from the on-disk file is propagated.

    The mock approach guarantees the test is root-safe: it injects the
    PermissionError at the ``Path.read_text`` call site regardless of
    filesystem permissions or the current user identity.
    """

    def test_permission_error_propagated(self, tmp_path: Path) -> None:
        # Create the override file so the ``exists()`` check returns True.
        (tmp_path / BOOTSTRAP_CODE_FILENAME).write_text(
            "code", encoding=FILE_ENCODING
        )
        exc = PermissionError(
            13, "Permission denied", str(tmp_path / BOOTSTRAP_CODE_FILENAME)
        )
        with patch.object(Path, "read_text", side_effect=exc):
            with pytest.raises(PermissionError):
                load_bootstrap_code(pkg_dir=tmp_path)

    def test_permission_error_not_swallowed_as_fallback(
        self, tmp_path: Path
    ) -> None:
        """A PermissionError must NOT silently fall through to the embedded code."""
        (tmp_path / BOOTSTRAP_CODE_FILENAME).write_text(
            "code", encoding=FILE_ENCODING
        )
        exc = PermissionError(13, "Permission denied")
        with patch.object(Path, "read_text", side_effect=exc):
            # Must re-raise, not return WASM_BOOTSTRAP_CODE.
            with pytest.raises(PermissionError):
                result = load_bootstrap_code(pkg_dir=tmp_path)
                # If we reach here the error was swallowed — explicit fail.
                pytest.fail(
                    f"Expected PermissionError but got result: {result!r}"
                )


class TestLoadBootstrapCodeInputValidation:
    def test_non_path_pkg_dir_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="pathlib.Path"):
            load_bootstrap_code(pkg_dir="/some/path")  # type: ignore[arg-type]

    def test_empty_filename_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="filename"):
            load_bootstrap_code(pkg_dir=tmp_path, filename="")

    def test_whitespace_filename_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="filename"):
            load_bootstrap_code(pkg_dir=tmp_path, filename="   ")

    def test_none_pkg_dir_uses_real_package_dir(self) -> None:
        """pkg_dir=None is explicitly allowed and falls back to _PKG_DIR."""
        result = load_bootstrap_code(pkg_dir=None)
        assert isinstance(result, str)
