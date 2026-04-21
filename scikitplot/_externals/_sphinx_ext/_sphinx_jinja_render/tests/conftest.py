"""
Shared pytest fixtures for the _url_helper test suite.

All fixtures are session- or function-scoped as appropriate.
Fixtures that touch the filesystem use ``tmp_path`` (function-scoped)
to guarantee isolation between tests.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from .._constants import (
    BOOTSTRAP_CODE_FILENAME,
    FILE_ENCODING,
    TEMPLATE_SUFFIX,
    WASM_BOOTSTRAP_CODE,
)


# ---------------------------------------------------------------------------
# Bootstrap-code fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pkg_dir_with_override(tmp_path: Path) -> Path:
    """Return a temp directory containing a valid bootstrap override file."""
    code = "import scikitplot  # override\n"
    (tmp_path / BOOTSTRAP_CODE_FILENAME).write_text(code, encoding=FILE_ENCODING)
    return tmp_path


@pytest.fixture()
def pkg_dir_without_override(tmp_path: Path) -> Path:
    """Return a temp directory with *no* bootstrap override file."""
    return tmp_path


@pytest.fixture()
def embedded_bootstrap_code() -> str:
    """Return the embedded WASM bootstrap code constant."""
    return WASM_BOOTSTRAP_CODE


# ---------------------------------------------------------------------------
# RST template fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def src_dir_with_templates(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Return (src_dir, rendered_content_map) for a directory with 2 templates."""
    templates = {
        "index.rst.template": "Hello {{ name }}!\n",
        "sub/page.rst.template": "Version: {{ version }}\n",
    }
    for rel, content in templates.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=FILE_ENCODING)

    context = {"name": "World", "version": "1.0"}
    expected = {
        "index.rst": "Hello World!\n",
        "sub/page.rst": "Version: 1.0\n",
    }
    return tmp_path, expected  # type: ignore[return-value]


@pytest.fixture()
def empty_src_dir(tmp_path: Path) -> Path:
    """Return a temp directory containing no RST templates."""
    return tmp_path


@pytest.fixture()
def single_template_file(tmp_path: Path) -> Path:
    """Return the path to a single valid RST template file."""
    p = tmp_path / f"example{TEMPLATE_SUFFIX}"
    p.write_text("Hello {{ name }}!\n", encoding=FILE_ENCODING)
    return p


# ---------------------------------------------------------------------------
# URL-builder fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_code_snippet() -> str:
    """Return a simple one-liner code snippet."""
    return "import scikitplot as skplt"


@pytest.fixture()
def multiline_code_snippet() -> str:
    """Return a multiline code snippet with special characters."""
    return textwrap.dedent(
        """\
        import micropip
        await micropip.install('scikit-plots')
        import scikitplot as skplt
        print('loaded')
        """
    )
