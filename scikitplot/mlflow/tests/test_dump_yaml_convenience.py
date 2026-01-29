from __future__ import annotations

"""Tests for test_dump_yaml_convenience.py."""

from pathlib import Path
import pytest

from scikitplot.mlflow._project import dump_project_config_yaml


def test_dump_project_config_yaml_convenience(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()

    (cfg_dir / "mlflow.toml").write_text(
        '''
[profiles.local]
start_server = false

[profiles.local.session]
tracking_uri = "http://127.0.0.1:5000"
'''.lstrip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    pytest.importorskip("yaml")

    out = dump_project_config_yaml(profile="local")
    assert out == (tmp_path / "configs" / "mlflow.yaml")
    assert out.exists()
