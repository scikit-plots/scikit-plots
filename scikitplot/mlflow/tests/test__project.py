from __future__ import annotations

"""Tests for project config discovery and config loading.

These tests also cover deterministic project marker customization (env + context manager).
"""

from pathlib import Path
import pytest

from scikitplot.mlflow._project import (
    find_project_root,
    load_project_config_toml,
    load_project_config,
)


def test_find_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_raises(tmp_path: Path) -> None:
    d = tmp_path / "x"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        find_project_root(d, markers=("does_not_exist.marker",))


def test_load_toml_normalizes_paths_and_empty_strings(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    toml_path = cfg_dir / "mlflow.toml"

    toml_path.write_text(
        '''
[profiles.local]
start_server = true

[profiles.local.session]
tracking_uri = "http://127.0.0.1:5001"
registry_uri = ""
startup_timeout_s = 1.0

[profiles.local.server]
host = "127.0.0.1"
port = 5001
backend_store_uri = "sqlite:///./.mlflow/mlflow.db"
default_artifact_root = "./.mlflow/artifacts"
serve_artifacts = true
strict_cli_compat = false
''',
        encoding="utf-8",
    )

    cfg = load_project_config_toml(toml_path, profile="local")
    assert cfg.start_server is True
    assert cfg.session.registry_uri is None

    assert cfg.server is not None
    assert cfg.server.backend_store_uri is not None
    assert cfg.server.backend_store_uri.startswith("sqlite:///")

    db_path = Path(cfg.server.backend_store_uri[len("sqlite:///"):])
    assert db_path.is_absolute()
    assert str(tmp_path.resolve()) in str(db_path)

    assert cfg.server.default_artifact_root is not None
    art_path = Path(cfg.server.default_artifact_root)
    assert art_path.is_absolute()
    assert art_path.exists()


def test_load_project_config_dispatch_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_project_config(p)


def test_project_markers_env_override(monkeypatch) -> None:
    """Environment variable override is strict JSON list."""
    import json as _json
    import scikitplot.mlflow._project as p

    monkeypatch.setenv("SCIKITPLOT_PROJECT_MARKERS", _json.dumps(["pyproject.toml", ".git", "configs/mlflow.toml"]))
    markers = p.get_project_markers()
    assert markers == ("pyproject.toml", ".git", "configs/mlflow.toml")


def test_project_markers_env_invalid_json_raises(monkeypatch) -> None:
    import pytest
    import scikitplot.mlflow._project as p

    monkeypatch.setenv("SCIKITPLOT_PROJECT_MARKERS", "not-json")
    with pytest.raises(ValueError):
        p.get_project_markers()


def test_project_markers_context_manager_restores(monkeypatch) -> None:
    import scikitplot.mlflow._project as p

    base = p.get_project_markers()
    with p.project_markers(["X.marker"]):
        assert p.get_project_markers() == ("X.marker",)
    assert p.get_project_markers() == base


def test_project_markers_setter_reset() -> None:
    import scikitplot.mlflow._project as p

    p.set_project_markers(["A", "B"])
    assert p.get_project_markers() == ("A", "B")
    p.set_project_markers(None)
    assert p.get_project_markers() == p.DEFAULT_PROJECT_MARKERS
