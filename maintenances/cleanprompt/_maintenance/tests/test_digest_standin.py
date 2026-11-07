"""Tests for the output digest and the stand-in builder."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
output_digest = runpy.run_path(str(TOOLS / "output_digest.py"))
standin = runpy.run_path(str(TOOLS / "standin.py"))


class TestOutputDigest:
    def test_the_corpus_is_deterministic(self):
        first = list(output_digest["corpus"](40, 115))
        assert first == list(output_digest["corpus"](40, 115))
        assert first != list(output_digest["corpus"](40, 116))
        assert {fmt for _, fmt, _ in first} == {
            "text",
            "markdown",
            "env",
            "json",
            "csv",
        }

    def test_a_small_digest_is_stable(self):
        assert output_digest["digest"](12, 115) == output_digest["digest"](12, 115)

    def test_the_baseline_names_its_parameters(self):
        baseline = TOOLS.parent / output_digest["BASELINE"]
        record = json.loads(baseline.read_text(encoding="utf-8"))
        assert set(record) == {"documents", "seed", "sha256"}
        assert len(record["sha256"]) == len("0" * 64)  # hex SHA-256

    def test_a_check_with_other_parameters_is_refused(self, capsys):
        assert output_digest["main"](["--check", "--documents", "3"]) == 1
        assert "baseline was computed with" in capsys.readouterr().err


class TestStandin:
    def test_builds_and_rebuilds_its_own_directory(self, tmp_path):
        dest = tmp_path / "si"
        standin["build"](dest)
        assert (dest / "scikitplot/__init__.py").read_text(encoding="utf-8") == ""
        assert (dest / "scikitplot/cleanprompt/_engine.py").is_file()
        assert (dest / "scikitplot/_testing").is_dir()
        assert "filterwarnings = error" in (dest / "pytest.ini").read_text(
            encoding="utf-8"
        )
        (dest / "stale.txt").write_text("old", encoding="utf-8")
        standin["build"](dest)
        assert not (dest / "stale.txt").exists()
        assert not list(dest.rglob("__pycache__"))

    def test_refuses_a_directory_it_did_not_build(self, tmp_path):
        (tmp_path / "keep.txt").write_text("mine", encoding="utf-8")
        with pytest.raises(SystemExit, match="refusing"):
            standin["build"](tmp_path)
        assert (tmp_path / "keep.txt").is_file()
