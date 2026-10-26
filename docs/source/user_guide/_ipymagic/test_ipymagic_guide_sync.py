# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The notebook-magics guide: pages linked, every example executed (Rule 54)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAGES = ["index", "commands", "profiling", "automation"]
_BLOCK = re.compile(r"\.\. code-block:: (\w+)\n\n((?:   .*\n|\n)+)")


def _blocks(language: str) -> list[tuple[str, str]]:
    out = []
    for page in PAGES:
        text = (HERE / f"{page}.rst").read_text(encoding="utf-8")
        for lang, body in _BLOCK.findall(text):
            if lang == language:
                code = "\n".join(line[3:] for line in body.splitlines()).strip("\n")
                out.append((page, code + "\n"))
    return out


def test_index_links_every_page():
    index = (HERE / "index.rst").read_text(encoding="utf-8")
    assert ".. _ipymagic-index:" in index
    for page in PAGES[1:]:
        assert f"<{page}.rst>" in index, page
    root = (HERE.parent / "index.rst").read_text(encoding="utf-8")
    assert "./_ipymagic/index.rst" in root


def test_python_examples_run():
    pytest.importorskip("scikitplot._ipymagic")
    for page, code in _blocks("python"):
        exec(compile(code, f"{page}.rst", "exec"), {})  # noqa: S102


def test_ipython_examples_run(tmp_path, monkeypatch):
    pytest.importorskip("IPython")
    from IPython.testing import globalipapp  # noqa: PLC0415

    monkeypatch.setenv("SKPLT_ENV_HOME", str(tmp_path))
    globalipapp.start_ipython()
    shell = globalipapp.get_ipython()
    try:
        for page, code in _blocks("ipython"):
            outcome = shell.run_cell(code, store_history=False)
            if "evn plan" in code:  # the documented refusal
                assert type(outcome.error_in_exec).__name__ == "MagicUsageError"
                assert outcome.error_in_exec.hint == "did you mean 'env'?"
            else:
                assert outcome.success, (page, code, outcome.error_in_exec)
    finally:
        shell.run_line_magic("scikitplot_config", "raise_on_error=false timeout=none")
        shell.run_line_magic("unload_ext", "scikitplot")
