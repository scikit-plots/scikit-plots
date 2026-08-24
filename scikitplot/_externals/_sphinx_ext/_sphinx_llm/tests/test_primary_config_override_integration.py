# SPDX-License-Identifier: BSD-3-Clause
"""Executable Sphinx regression for primary config override semantic parity."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("sphinx")
pytest.importorskip("sphinx_markdown_builder")

from sphinx.application import Sphinx

FIXTURE = Path(__file__).parent / "roots" / "test-primary-config-overrides"


def _assert_semantic_parity(outdir: Path):
    html = (outdir / "index.html").read_text(encoding="utf-8")
    markdown = (outdir / "index.html.md").read_text(encoding="utf-8")
    for rendered in (html, markdown):
        assert "OVERRIDE_CONTENT_SELECTED" in rendered
        assert "BASE_CONTENT_SELECTED" not in rendered


def test_programmatic_config_override_reaches_markdown_subbuild(tmp_path: Path):
    srcdir = tmp_path / "src"
    shutil.copytree(FIXTURE, srcdir)
    outdir = tmp_path / "html"
    doctreedir = tmp_path / "doctrees"

    # Use text; do not save a permanent conf.py.
    conf_content = (
        "project = 'sphinx-llm config parity fixture'\n"
        "extensions = [\n"
        "    'sphinx.ext.ifconfig',\n"
        "    'scikitplot._externals._sphinx_ext._sphinx_llm',\n"
        "]\n"
        "feature_mode = 'override'\n"
        "llms_txt_build_parallel = False\n"
        "llms_txt_full_build = False\n\n"
        "def setup(app):\n"
        "    app.add_config_value('feature_mode', 'base', 'env')\n"
    )
    (srcdir / "conf.py").write_text(conf_content, encoding="utf-8")

    app = Sphinx(
        srcdir=srcdir,
        confdir=srcdir,
        outdir=outdir,
        doctreedir=doctreedir,
        buildername="html",
        # confoverrides={
        #     "project": "sphinx-llm config parity fixture",
        #     "extensions": [
        #         "sphinx.ext.ifconfig",
        #         "scikitplot._externals._sphinx_ext._sphinx_llm",
        #     ],
        #     "feature_mode": "override",
        #     "llms_txt_build_parallel": False,
        #     "llms_txt_full_build": False,
        # },
        freshenv=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0
    _assert_semantic_parity(outdir)
