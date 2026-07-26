# scikitplot/cython/tests/test__templates_containment.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-TPL-001.

The template / workflow / package-example resolvers built ``root / name`` from a
caller-supplied name without containment, so an absolute path or ``..`` traversal
could escape the packaged ``_templates`` tree.  All three resolvers now route
through ``_resolve_within`` and reject escapes with ``ValueError``.
"""
from __future__ import annotations

import pytest

from .._templates_api import (
    _PACKAGE_EXAMPLES_ROOT,
    _TEMPLATE_ROOT,
    _WORKFLOW_ROOT,
    _resolve_within,
    get_package_example_path,
    get_template_path,
    get_workflow_path,
    list_package_examples,
    list_workflows,
)


class TestResolveWithin:
    def test_relative_child_ok(self) -> None:
        p = _resolve_within(_TEMPLATE_ROOT, "module_cython")
        assert p == (_TEMPLATE_ROOT / "module_cython").resolve()

    def test_root_itself_ok(self) -> None:
        assert _resolve_within(_TEMPLATE_ROOT, ".") == _TEMPLATE_ROOT.resolve()

    def test_absolute_rejected(self) -> None:
        with pytest.raises(ValueError, match="relative"):
            _resolve_within(_TEMPLATE_ROOT, "/etc/passwd")

    @pytest.mark.parametrize("bad", ["../..", "../../_loader.py", "a/../../../x"])
    def test_traversal_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="escapes"):
            _resolve_within(_TEMPLATE_ROOT, bad)


class TestResolverContainment:
    def test_get_template_path_absolute_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_template_path("/etc/hostname.pyx")

    def test_get_template_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_template_path("../../_loader.py")

    def test_get_workflow_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_workflow_path("../../..")

    def test_get_package_example_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_package_example_path("../../_cache.py")


class TestValidLookupsStillWork:
    def test_valid_workflow(self) -> None:
        workflows = list_workflows()
        if workflows:
            p = get_workflow_path(workflows[0])
            assert p.is_dir()
            assert _WORKFLOW_ROOT.resolve() in p.parents or p == _WORKFLOW_ROOT.resolve()

    def test_valid_package_example(self) -> None:
        examples = list_package_examples()
        if examples:
            p = get_package_example_path(examples[0])
            assert p.is_dir()
            assert (
                _PACKAGE_EXAMPLES_ROOT.resolve() in p.parents
                or p == _PACKAGE_EXAMPLES_ROOT.resolve()
            )
