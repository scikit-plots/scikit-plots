"""Tests for scikitplot.corpus._corpus_builder."""

from __future__ import annotations

import pathlib
import re

import pytest

#: The package README, next to the package; absent from some installs.
_README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _readme_example(heading: str) -> str:
    """Return the first python block after *heading* in the README."""
    text = _README.read_text(encoding="utf-8")
    section = text[text.index(heading) :]
    match = re.search(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert match is not None, heading
    return match.group(1)


class TestReadmeExample:
    """CX-09: the README's CorpusBuilder example runs as written."""

    @pytest.mark.skipif(not _README.is_file(), reason="README.md not installed")
    def test_end_to_end_example_finds_the_text(self, tmp_path, monkeypatch) -> None:
        data = tmp_path / "data"
        data.mkdir()
        (data / "report.txt").write_text(
            "Health spending gives financial protection.\n\nAnother paragraph.\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        namespace: dict = {}
        exec(_readme_example("## 2. End-to-end convenience"), namespace)  # noqa: S102
        assert namespace["hits"]
