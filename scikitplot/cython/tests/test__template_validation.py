# scikitplot/cython/tests/test__template_validation.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-TPL-002.

Template metadata was permissively coerced: wrong-typed list entries were
silently discarded, ``schema_version`` was defaulted but never checked, and
support/extra-source references were not contained.  ``validate_template_info``
now strictly rejects unknown schema versions, wrong-typed entries, and
absolute/escaping references.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._templates_api import (
    TEMPLATE_SCHEMA_VERSION,
    TemplateInfo,
    TemplateValidationError,
    _contained_relpath,
    validate_template_info,
)


def _info(**kw) -> TemplateInfo:
    base = dict(template_id="cat/name", path=Path("name.pyx"), meta_path=None)
    base.update(kw)
    return TemplateInfo(**base)


class TestSchemaVersion:
    def test_unknown_version_rejected(self) -> None:
        with pytest.raises(TemplateValidationError, match="schema_version"):
            validate_template_info(_info(schema_version=TEMPLATE_SCHEMA_VERSION + 1))

    def test_zero_version_rejected(self) -> None:
        with pytest.raises(TemplateValidationError):
            validate_template_info(_info(schema_version=0))

    def test_current_version_ok(self) -> None:
        validate_template_info(_info(schema_version=TEMPLATE_SCHEMA_VERSION))


class TestWrongTypedEntries:
    def test_non_string_tag_rejected(self) -> None:
        with pytest.raises(TemplateValidationError, match="non-string"):
            validate_template_info(_info(), raw={"tags": ["ok", 42]})

    def test_non_string_support_path_rejected(self) -> None:
        with pytest.raises(TemplateValidationError, match="non-string"):
            validate_template_info(_info(), raw={"support_paths": ["a.pxi", None]})

    def test_list_field_must_be_list(self) -> None:
        with pytest.raises(TemplateValidationError, match="must be a list"):
            validate_template_info(_info(), raw={"tags": "notalist"})

    def test_demo_calls_must_be_objects(self) -> None:
        with pytest.raises(TemplateValidationError, match="demo_calls"):
            validate_template_info(_info(), raw={"demo_calls": [{"ok": 1}, "bad"]})

    def test_clean_raw_passes(self) -> None:
        validate_template_info(
            _info(tags=("fast",)),
            raw={"tags": ["fast"], "support_paths": [], "demo_calls": [{"call": "f"}]},
        )


class TestReferenceContainment:
    def test_parent_escape_rejected(self) -> None:
        with pytest.raises(TemplateValidationError, match="contained"):
            validate_template_info(_info(support_paths=("../evil.pxi",)))

    def test_absolute_posix_rejected(self) -> None:
        with pytest.raises(TemplateValidationError, match="contained"):
            validate_template_info(_info(extra_sources=("/etc/passwd",)))

    def test_nested_relative_ok(self) -> None:
        validate_template_info(_info(support_paths=("helpers/util.pxi",)))


class TestContainedRelpathHelper:
    @pytest.mark.parametrize(
        ("value", "ok"),
        [
            ("a/b.pxi", True),
            ("x.pxd", True),
            ("../x", False),
            ("a/../b", False),
            ("/abs/path", False),
            ("", False),
        ],
    )
    def test_containment(self, value: str, ok: bool) -> None:
        assert _contained_relpath(value) is ok


class TestReadTemplateInfoStrictParam:
    def test_strict_default_is_false(self) -> None:
        import inspect

        from .._templates_api import read_template_info

        sig = inspect.signature(read_template_info)
        assert sig.parameters["strict"].default is False
