# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Package contract owned by :mod:`_hf_spaces_proxy._page_feedback.__init__`."""
from __future__ import annotations

import subprocess
import sys

import pytest

from ..._paths import RUNTIME_ROOT, STACK_ROOT
from ...._hf_spaces_proxy import _page_feedback as package

VENDORED = RUNTIME_ROOT / "_hf_spaces_proxy" / "_page_feedback"
SOURCE = STACK_ROOT / "_sphinx_feedback"

#: What the proxy carries. The Sphinx adapter, the aggregate builder and the
#: static assets stay in ``_sphinx_feedback``; the proxy is a server and has no
#: use for them.
EXPECTED_VENDORED = {
    "__init__.py",
    "_contracts.py",
    "_service/__init__.py",
    "_service/_config.py",
    "_service/_core.py",
    "_service/_github.py",
    "_service/_sqlite.py",
    "_service/app.py",
}


def _vendored_files() -> set[str]:
    return {
        path.relative_to(VENDORED).as_posix()
        for path in VENDORED.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def test_public_names_resolve() -> None:
    assert package.__all__ == sorted(package.__all__)
    for name in package.__all__:
        assert callable(getattr(package, name)) or isinstance(getattr(package, name), type)
    assert isinstance(package.__version__, str) and package.__version__


def test_vendored_file_set_is_exactly_the_server_subset() -> None:
    assert _vendored_files() == EXPECTED_VENDORED


def test_vendored_files_are_byte_identical_to_their_source() -> None:
    """
    The proxy's copy is a copy, not a fork.

    ``_page_feedback`` exists so the deployed proxy needs nothing outside its
    own folder. If a vendored file drifts from ``_sphinx_feedback``, the site
    and the server stop agreeing on what a feedback request is, and nothing
    else would notice.
    """
    if not SOURCE.is_dir():
        pytest.skip("the _sphinx_feedback source package is not in this checkout")
    drifted = [
        name
        for name in sorted(EXPECTED_VENDORED)
        if (VENDORED / name).read_bytes() != (SOURCE / name).read_bytes()
    ]
    assert drifted == [], "vendored page-feedback files differ from _sphinx_feedback: " + ", ".join(drifted)


def test_import_needs_neither_sphinx_nor_a_web_framework() -> None:
    """The helpers are dependency-free; importing them must stay that way."""
    code = (
        "import sys; sys.path.insert(0, {root!r}); import _page_feedback, _page_feedback._service; "
        "loaded = sorted(m for m in ('sphinx', 'fastapi', 'starlette', 'docutils') if m in sys.modules); "
        "print(loaded)"
    ).format(root=str(VENDORED.parent))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_setup_is_not_available_in_the_server_subset() -> None:
    """
    ``setup`` registers the Sphinx adapter, which the proxy does not carry.

    The name stays in ``__all__`` because the file is byte-identical to its
    source; calling it here fails loudly rather than half-registering.
    """
    assert not (VENDORED / "_sphinx.py").exists()
    with pytest.raises(ImportError):
        package.setup(object())
