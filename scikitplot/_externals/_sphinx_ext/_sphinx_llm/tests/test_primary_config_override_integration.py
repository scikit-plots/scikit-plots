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

    # Absolute path to your custom configuration file
    custom_config_path = os.path.abspath(os.path.join(srcdir, "conf_overrides.py"))

    app = Sphinx(
        srcdir=srcdir,
        confdir=srcdir,
        outdir=outdir,
        doctreedir=doctreedir,
        buildername="html",
        confoverrides={
            "config_file": custom_config_path,  # Forces Sphinx to load conf_overrides.py
            "feature_mode": "override",
            "llms_txt_build_parallel": False,
        },
        freshenv=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0
    _assert_semantic_parity(outdir)
