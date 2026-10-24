# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The feature_selection guide: pages linked, names real, every example executed (Rule 54)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAGES = ["getting_started", "configuration", "pipelines_and_lineage", "drift"]
_BLOCK = re.compile(r"\.\. code-block:: (python|toml)\n\n((?:   .*\n|\n)+)")


def _repo_root() -> Path:
    for parent in HERE.parents:
        if (parent / "scikitplot").is_dir() and (parent / "docs").is_dir():
            return parent
    raise AssertionError("repository root not found")


def _blocks(page: str, language: str) -> list[str]:
    text = (HERE / f"{page}.rst").read_text(encoding="utf-8")
    return [
        "\n".join(line[3:] for line in body.splitlines())
        for lang, body in _BLOCK.findall(text)
        if lang == language
    ]


def test_index_links_every_page_and_the_gallery():
    index = (HERE / "index.rst").read_text(encoding="utf-8")
    assert ".. _feature_selection-index:" in index
    for page in PAGES:
        assert page in index, page
    readme = _repo_root() / "galleries/examples/feature_selection/README.txt"
    assert ".. _feature_selection_examples:" in readme.read_text(encoding="utf-8")
    root = (_repo_root() / "docs/source/user_guide/index.rst").read_text(
        encoding="utf-8"
    )
    assert "./feature_selection/index.rst" in root


def test_every_python_example_runs_in_order(tmp_path):
    pytest.importorskip("scikitplot.feature_selection")
    namespace: dict = {}
    cwd = os.getcwd()
    os.chdir(tmp_path)  # examples write schema.json, policy.toml, reference.json
    try:
        for page in PAGES:
            for i, code in enumerate(_blocks(page, "python")):
                if "policy.toml" in code:  # writing TOML needs the optional tomli-w
                    pytest.importorskip("tomli_w")
                exec(compile(code, f"{page}.rst[{i}]", "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)
        from scikitplot.feature_selection import default_registry

        default_registry.unregister("looks_like_percentage")  # the guide registers it
    assert namespace["lineage"].dropped() == {"customer_id": "filter"}
    assert set(namespace["report"]["status"]) <= {
        "stable",
        "warn",
        "drift",
        "not_applicable",
    }


def test_the_toml_example_is_a_valid_policy():
    from scikitplot.feature_selection import FeaturePolicy

    (toml,) = _blocks("configuration", "toml")
    policy = FeaturePolicy.loads(toml, "toml")
    assert policy.binning.rule == "knuth"
    assert policy.overrides_map == {"income": {"skewed": True}}
