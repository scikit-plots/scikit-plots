"""
Tests for _url_helper._validators.

Every public validator is tested for:
* happy-path (no exception),
* wrong-type input (TypeError),
* valid-type-but-bad-value (ValueError / FileNotFoundError / etc.).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._constants import MAX_URL_LENGTH, TEMPLATE_SUFFIX
from .._validators import (
    validate_directory,
    validate_kernel_name,
    validate_non_empty_string,
    validate_template_file,
    validate_url_length,
)


class TestValidateNonEmptyString:
    def test_valid_string_passes(self) -> None:
        validate_non_empty_string("hello", "param")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_non_empty_string("   ", "param")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_non_empty_string("", "param")

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="str"):
            validate_non_empty_string(42, "param")

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="str"):
            validate_non_empty_string(None, "param")

    def test_error_message_contains_param_name(self) -> None:
        with pytest.raises(ValueError, match="my_param"):
            validate_non_empty_string("", "my_param")


class TestValidateUrlLength:
    def test_short_url_passes(self) -> None:
        validate_url_length("https://example.com")

    def test_url_at_exact_limit_passes(self) -> None:
        url = "x" * MAX_URL_LENGTH
        validate_url_length(url)

    def test_url_exceeding_limit_raises_value_error(self) -> None:
        url = "x" * (MAX_URL_LENGTH + 1)
        with pytest.raises(ValueError, match=str(MAX_URL_LENGTH)):
            validate_url_length(url)

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            validate_url_length(123)  # type: ignore[arg-type]

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            validate_url_length("")


class TestValidateDirectory:
    def test_existing_directory_passes(self, tmp_path: Path) -> None:
        validate_directory(tmp_path, "src_dir")

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_directory(tmp_path / "nope", "src_dir")

    def test_file_path_raises_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            validate_directory(f, "src_dir")

    def test_non_path_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="pathlib.Path"):
            validate_directory("/some/path", "src_dir")  # type: ignore[arg-type]

    def test_error_contains_param_name(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_directory(tmp_path / "missing", "my_dir")


class TestValidateTemplateFile:
    def test_valid_template_passes(self, tmp_path: Path) -> None:
        p = tmp_path / f"page{TEMPLATE_SUFFIX}"
        p.write_text("{{ x }}", encoding="utf-8")
        validate_template_file(p)  # must not raise

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_template_file(tmp_path / f"missing{TEMPLATE_SUFFIX}")

    def test_wrong_suffix_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "page.rst"
        p.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match=TEMPLATE_SUFFIX):
            validate_template_file(p)

    def test_directory_raises_is_a_directory_error(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            validate_template_file(tmp_path)

    def test_non_path_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="pathlib.Path"):
            validate_template_file("not_a_path")  # type: ignore[arg-type]


class TestValidateKernelName:
    @pytest.mark.parametrize(
        "kernel",
        ["python", "python3", "ir", "julia-1.9", "xpython_3", "my.kernel"],
    )
    def test_valid_kernel_names_pass(self, kernel: str) -> None:
        validate_kernel_name(kernel)

    @pytest.mark.parametrize(
        "kernel",
        ["python 3", "my kernel", "kernel!", "ker nel"],
    )
    def test_invalid_characters_raise_value_error(self, kernel: str) -> None:
        with pytest.raises(ValueError, match="illegal"):
            validate_kernel_name(kernel)

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            validate_kernel_name("")

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            validate_kernel_name(None)  # type: ignore[arg-type]
