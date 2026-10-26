# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The environment-builder guide: pages linked, examples executed (Rule 54)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAGES = ["getting_started", "configuration", "targets", "automation", "how_it_works"]
_BLOCK = re.compile(r"\.\. code-block:: (\w+)\n\n((?:   .*\n|\n)+)")


def _blocks(page: str, language: str) -> list[str]:
    text = (HERE / f"{page}.rst").read_text(encoding="utf-8")
    return [
        "\n".join(line[3:] for line in body.splitlines()).strip("\n") + "\n"
        for lang, body in _BLOCK.findall(text)
        if lang == language
    ]


def test_index_links_every_page():
    index = (HERE / "index.rst").read_text(encoding="utf-8")
    assert ".. _env-index:" in index
    for page in PAGES:
        assert f"<{page}.rst>" in index, page
    root = (HERE.parent / "index.rst").read_text(encoding="utf-8")
    assert "./_env/index.rst" in root


def test_python_examples_run():
    pytest.importorskip("scikitplot._env")
    for page in PAGES:
        for i, code in enumerate(_blocks(page, "python")):
            exec(compile(code, f"{page}.rst[{i}]", "exec"), {})  # noqa: S102


def test_toml_example_is_a_valid_request(tmp_path):
    schema = pytest.importorskip("scikitplot._env._config._schema")
    (text,) = _blocks("configuration", "toml")
    path = tmp_path / "scikitplot-env.toml"
    path.write_text(text, encoding="utf-8")
    request = schema.load_request(path)
    # the file's own [policy] wins over its preset ("ci" would set "never")
    assert request.name == "analysis" and request.policy.interaction == "auto"
    assert dict(request.origins)["policy.interaction"].startswith("config:")
    assert [s.kind for s in request.sources] == ["requirements", "pyproject"]


def test_batch_example_lines_are_valid(tmp_path):
    schema = pytest.importorskip("scikitplot._env._config._schema")
    (text,) = _blocks("automation", "json")
    path = tmp_path / "requests.jsonl"
    path.write_text(text, encoding="utf-8")
    assert [rid for rid, _ in schema.load_batch(path)] == ["docs", "lite"]
    assert all(json.loads(line) for line in text.splitlines())


def test_error_codes_table_is_complete():
    errors = pytest.importorskip("scikitplot._env._core._errors")
    text = (HERE / "automation.rst").read_text(encoding="utf-8")
    codes = {
        cls.code
        for cls in vars(errors).values()
        if isinstance(cls, type)
        and issubclass(cls, errors.EnvError)
        and cls is not errors.EnvError
    }
    for code in codes | {"config_authority", "config_schema"}:
        assert f"``{code}``" in text, code
