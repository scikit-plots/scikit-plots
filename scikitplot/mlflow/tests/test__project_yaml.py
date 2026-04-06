from __future__ import annotations

"""Tests for test_project_yaml.py."""

from pathlib import Path
import pytest

yaml = pytest.importorskip("yaml")

from scikitplot.mlflow._project import load_project_config_yaml


def test_load_yaml_profile(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    yml = cfg_dir / "mlflow.yaml"
    yml.write_text(
        '''
profiles:
  local:
    start_server: false
    session:
      tracking_uri: "http://127.0.0.1:5000"
      experiment_name: "exp"
      create_experiment_if_missing: true
''',
        encoding="utf-8",
    )
    cfg = load_project_config_yaml(yml, profile="local")
    assert cfg.start_server is False
    assert cfg.session.tracking_uri == "http://127.0.0.1:5000"
    assert cfg.session.experiment_name == "exp"
