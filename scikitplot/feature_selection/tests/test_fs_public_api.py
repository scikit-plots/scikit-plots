# scikitplot/feature_selection/tests/test_fs_public_api.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The public surface: names resolve, are documented, and import nothing optional."""

from __future__ import annotations

import inspect
import subprocess
import sys

import scikitplot.feature_selection as fs


def test_every_public_name_resolves_once():
    assert sorted(fs.__all__) == sorted(set(fs.__all__))
    for name in fs.__all__:
        assert getattr(fs, name) is not None, name


def test_every_public_callable_has_a_numpydoc_docstring():
    for name in fs.__all__:
        obj = getattr(fs, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            doc = inspect.getdoc(obj) or ""
            assert doc.strip(), name
            if inspect.isfunction(obj):
                assert "Returns" in doc or "Yields" in doc, name


def test_importing_the_package_does_not_import_optional_libraries():
    code = (
        "import sys; import scikitplot.feature_selection; "
        "print(sorted(m for m in ('yaml', 'tomli_w', 'pydantic', 'matplotlib') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
