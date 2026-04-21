"""
Tests for _url_helper._constants.

Verifies that all named constants have the correct type, are non-empty
where applicable, and that WASM_FALLBACK_CODE is backward-compatible
with WASM_BOOTSTRAP_CODE.
"""
from __future__ import annotations

import pytest

from .._constants import (
    BOOTSTRAP_CODE_FILENAME,
    CODE_PARAM,
    DEFAULT_KERNEL_NAME,
    EXTENSION_NAME,
    FILE_ENCODING,
    JUPYTERLITE_BASE_URL,
    KERNEL_PARAM,
    MAX_URL_LENGTH,
    RST_SUFFIX,
    TEMPLATE_SUFFIX,
    WASM_BOOTSTRAP_CODE,
    WASM_FALLBACK_CODE,
)


class TestStringConstants:
    """All string constants must be non-empty str instances."""

    @pytest.mark.parametrize(
        "constant, name",
        [
            (EXTENSION_NAME, "EXTENSION_NAME"),
            (JUPYTERLITE_BASE_URL, "JUPYTERLITE_BASE_URL"),
            (DEFAULT_KERNEL_NAME, "DEFAULT_KERNEL_NAME"),
            (KERNEL_PARAM, "KERNEL_PARAM"),
            (CODE_PARAM, "CODE_PARAM"),
            (TEMPLATE_SUFFIX, "TEMPLATE_SUFFIX"),
            (RST_SUFFIX, "RST_SUFFIX"),
            (FILE_ENCODING, "FILE_ENCODING"),
            (BOOTSTRAP_CODE_FILENAME, "BOOTSTRAP_CODE_FILENAME"),
            (WASM_BOOTSTRAP_CODE, "WASM_BOOTSTRAP_CODE"),
            (WASM_FALLBACK_CODE, "WASM_FALLBACK_CODE"),
        ],
    )
    def test_is_non_empty_string(self, constant: object, name: str) -> None:
        assert isinstance(constant, str), f"{name} must be str"
        assert constant.strip(), f"{name} must not be empty / whitespace"


class TestUrlConstants:
    def test_base_url_starts_with_https(self) -> None:
        assert JUPYTERLITE_BASE_URL.startswith("https://")

    def test_max_url_length_is_positive_int(self) -> None:
        assert isinstance(MAX_URL_LENGTH, int)
        assert MAX_URL_LENGTH > 0

    def test_max_url_length_reasonable_range(self) -> None:
        # Must accommodate typical URLs but not exceed common browser limits.
        assert 2_048 <= MAX_URL_LENGTH <= 32_768


class TestSuffixConstants:
    def test_template_suffix_ends_with_template(self) -> None:
        assert TEMPLATE_SUFFIX.endswith(".template")

    def test_rst_suffix_is_dot_rst(self) -> None:
        assert RST_SUFFIX == ".rst"

    def test_template_suffix_starts_with_rst(self) -> None:
        assert TEMPLATE_SUFFIX.startswith(".rst")


class TestBootstrapCodeConstants:
    def test_wasm_bootstrap_code_non_empty(self) -> None:
        assert len(WASM_BOOTSTRAP_CODE) > 0

    def test_wasm_fallback_code_equals_bootstrap_code(self) -> None:
        """WASM_FALLBACK_CODE is an alias — must be identical."""
        assert WASM_FALLBACK_CODE == WASM_BOOTSTRAP_CODE

    def test_bootstrap_code_filename_is_txt_file(self) -> None:
        assert BOOTSTRAP_CODE_FILENAME.endswith(".txt")

    def test_bootstrap_code_filename_no_path_separator(self) -> None:
        import os

        assert os.sep not in BOOTSTRAP_CODE_FILENAME


class TestEncodingConstant:
    def test_file_encoding_is_valid(self) -> None:
        import codecs

        # Will raise LookupError if the codec is unknown.
        codecs.lookup(FILE_ENCODING)
