# scikitplot/mlflow/tests/test__project_session.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._project_session.

Naming convention: test__<module_name>.py

Covers
------
- session_from_toml    : delegates to load_project_config_toml + session (lines 39-43)
- session_from_file    : delegates to load_project_config + session (lines 67-71),
                         supports .yaml extension dispatch

Notes
-----
The real MLflow module and disk IO are mocked so tests are fast and hermetic.
"""

from __future__ import annotations

import contextlib
import importlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional
from dataclasses import dataclass

import pytest

from scikitplot.mlflow._config import SessionConfig, ServerConfig
from scikitplot.mlflow._project import ProjectConfig


# ---------------------------------------------------------------------------
# Shared stubs (mirrored from conftest but local so tests are self-contained)
# ---------------------------------------------------------------------------


@dataclass
class _RunInfo:
    run_id: str


@dataclass
class _Run:
    info: _RunInfo


class _DummyClient:
    def __init__(self, tracking_uri=None, registry_uri=None):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

    def set_tag(self, run_id, key, value):
        pass


class _DummyMlflow:
    def __init__(self):
        self.__version__ = "2.14.0"
        self._tracking_uri: Optional[str] = None
        self.tracking = SimpleNamespace(MlflowClient=self._make_client)
        self._experiments: Dict[str, Any] = {}

    def set_tracking_uri(self, uri):
        self._tracking_uri = uri

    def set_registry_uri(self, uri):
        pass

    def _make_client(self, tracking_uri=None, registry_uri=None):
        return _DummyClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    def get_experiment_by_name(self, name):
        return self._experiments.get(name)

    def set_experiment(self, name):
        self._experiments[name] = object()

    @contextlib.contextmanager
    def start_run(self, *args, **kwargs):
        yield _Run(info=_RunInfo(run_id="run-xyz"))

    def set_tags(self, tags):
        pass


# ---------------------------------------------------------------------------
# Minimal TOML and YAML config content
# ---------------------------------------------------------------------------

_MINIMAL_TOML = """\
[profiles.local]
start_server = false

[profiles.local.session]
tracking_uri = "http://127.0.0.1:5000"
"""

_MINIMAL_YAML = """\
profiles:
  local:
    start_server: false
    session:
      tracking_uri: "http://127.0.0.1:5000"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tmp(content: str, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="w", encoding="utf-8"
    )
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


# ===========================================================================
# session_from_toml
# ===========================================================================


class TestSessionFromToml:
    """Tests for session_from_toml (lines 39-43 of _project_session.py)."""

    def test_yields_mlflow_handle(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """session_from_toml must load config and yield a working MlflowHandle."""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(_MINIMAL_TOML, encoding="utf-8")
        # Create a marker so find_project_root succeeds
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        dummy = _DummyMlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        from scikitplot.mlflow._project_session import session_from_toml

        with session_from_toml(toml_file, profile="local") as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"

    def test_propagates_profile_arg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """profile parameter must be forwarded to load_project_config_toml."""
        toml = """\
[profiles.staging]
start_server = false

[profiles.staging.session]
tracking_uri = "http://staging:5000"
"""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(toml, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        dummy = _DummyMlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        from scikitplot.mlflow._project_session import session_from_toml

        with session_from_toml(toml_file, profile="staging") as h:
            assert h.tracking_uri == "http://staging:5000"

    def test_missing_profile_raises(self, tmp_path: Path) -> None:
        """Missing profile key in TOML must propagate KeyError."""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(_MINIMAL_TOML, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        from scikitplot.mlflow._project_session import session_from_toml

        with pytest.raises(KeyError):
            with session_from_toml(toml_file, profile="nonexistent"):
                pass


# ===========================================================================
# session_from_file
# ===========================================================================


class TestSessionFromFile:
    """Tests for session_from_file (lines 67-71 of _project_session.py)."""

    def test_toml_dispatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """session_from_file with .toml extension must work correctly."""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(_MINIMAL_TOML, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        dummy = _DummyMlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        from scikitplot.mlflow._project_session import session_from_file

        with session_from_file(toml_file, profile="local") as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"

    def test_yaml_dispatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """session_from_file with .yaml extension must load yaml config."""
        yaml_file = tmp_path / "mlflow.yaml"
        yaml_file.write_text(_MINIMAL_YAML, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        dummy = _DummyMlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        from scikitplot.mlflow._project_session import session_from_file

        with session_from_file(yaml_file, profile="local") as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported file extension (.ini) must raise ValueError."""
        bad_file = tmp_path / "config.ini"
        bad_file.write_text("[section]\nkey=val\n", encoding="utf-8")

        from scikitplot.mlflow._project_session import session_from_file

        with pytest.raises(ValueError, match=".ini"):
            with session_from_file(bad_file):
                pass

    def test_accepts_str_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """config_path can be a str (not only Path)."""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(_MINIMAL_TOML, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        dummy = _DummyMlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        from scikitplot.mlflow._project_session import session_from_file

        with session_from_file(str(toml_file), profile="local") as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"
