"""
Tests for ``_extension``: the build hook fails closed on a broken template.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("sphinx")

from .._extension import _on_builder_inited  # noqa: E402


def _app(srcdir, **kwargs):
    return types.SimpleNamespace(
        srcdir=str(srcdir),
        config=types.SimpleNamespace(index_template_kwargs=kwargs),
    )


def test_the_hook_renders_with_the_configured_context(tmp_path):
    (tmp_path / "index.rst.template").write_text(
        "See {{ development_link }}\n", encoding="utf-8"
    )
    _on_builder_inited(_app(tmp_path, development_link="devel/index"))
    assert (tmp_path / "index.rst").read_text(encoding="utf-8") == "See devel/index\n"


def test_a_broken_template_stops_the_build_even_with_an_old_output(tmp_path):
    (tmp_path / "index.rst.template").write_text(
        "See {{ not_configured }}\n", encoding="utf-8"
    )
    old = tmp_path / "index.rst"
    old.write_text("See the old link\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not regenerated"):
        _on_builder_inited(_app(tmp_path, development_link="devel/index"))
    assert old.read_text(encoding="utf-8") == "See the old link\n"
