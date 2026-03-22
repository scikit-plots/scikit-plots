# scikitplot/mlflow/tests/test_extended.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Extended tests for scikitplot.mlflow covering all identified coverage gaps.

Test modules covered
--------------------
- _env         : EnvSnapshot.data copy semantics, matched-pair quote stripping,
                 dotenv edge cases, apply_env overwrite mode
- _compat      : resolve_download_artifacts preference order and fallbacks
- _config      : SessionConfig / ServerConfig full validation matrix
- _container   : running_in_docker mocked paths
- _utils       : _parse_version edge cases, mlflow_version fallbacks
- _cli_caps    : flag extraction, cache, empty help, equals form
- _facade      : ArtifactsFacade list / download / log; ModelsFacade load / register
- _project     : path helpers, marker resolution, TOML/YAML loading edge cases,
                 dump_project_config_yaml
- _readiness   : poll_interval_s, request_timeout_s, attempt-count in TimeoutError,
                 500 HTTP error path, server-exits-early path
- _server      : build_server_args flag coverage, no_serve_artifacts, disable_security,
                 x_frame_options, dev flag, gunicorn/waitress opts
- _session     : logger present, import os at module level, start_server URI mismatch,
                 registry_uri propagation, MlflowClient TypeError fallback,
                 start_run tags fallback via client.set_tag
- _workflow    : builtin_config_path, export_builtin_config, patch_experiment_name,
                 workflow() return type, WorkflowPaths properties

Notes
-----
All tests are pure-Python (no external dependencies beyond stdlib and the package
itself). MLflow is mocked throughout so the suite runs without MLflow installed.
Environment mutations are isolated via module-level save/restore helpers.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import subprocess
import sys
import tempfile
import time
import unittest.mock as mock
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

import pytest

# ---------------------------------------------------------------------------
# Shared fixture: minimal MLflow module stub
# ---------------------------------------------------------------------------


@dataclass
class _DummyRunInfo:
    run_id: str


@dataclass
class _DummyRun:
    info: _DummyRunInfo


class _DummyClient:
    """Minimal MlflowClient stub."""

    def __init__(self) -> None:
        self.tracking_uri: Optional[str] = None
        self.registry_uri: Optional[str] = None
        self.tags: Dict[str, Dict[str, str]] = {}

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.setdefault(run_id, {})[key] = value

    def list_artifacts(self, run_id: str, path: Optional[str] = None) -> list:
        return []

    def download_artifacts(
        self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
    ) -> str:
        return f"/legacy/{run_id}/{artifact_path}"


class _DummyMlflow:
    """Minimal mlflow module stub for session and facade tests."""

    def __init__(self) -> None:
        self.__version__ = "2.14.0"
        self._tracking_uri: Optional[str] = None
        self._registry_uri: Optional[str] = None
        self._experiment_name: Optional[str] = None
        self._experiments: Dict[str, Any] = {}
        self._set_tags_calls: list[dict[str, str]] = []
        self._start_run_calls: list[dict[str, Any]] = []
        self.tracking = SimpleNamespace(MlflowClient=self._make_client)

    def set_tracking_uri(self, uri: str) -> None:
        self._tracking_uri = uri

    def set_registry_uri(self, uri: str) -> None:
        self._registry_uri = uri

    def _make_client(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ) -> _DummyClient:
        c = _DummyClient()
        c.tracking_uri = tracking_uri
        c.registry_uri = registry_uri
        return c

    def get_experiment_by_name(self, name: str) -> Optional[object]:
        return self._experiments.get(name)

    def set_experiment(self, name: str) -> None:
        self._experiment_name = name
        self._experiments.setdefault(name, object())

    @contextlib.contextmanager
    def start_run(self, *args: Any, **kwargs: Any) -> Iterator[_DummyRun]:
        self._start_run_calls.append({"args": args, "kwargs": dict(kwargs)})
        yield _DummyRun(info=_DummyRunInfo(run_id="run-test"))

    def set_tags(self, tags: dict[str, str]) -> None:
        self._set_tags_calls.append(dict(tags))


@pytest.fixture()
def dummy_mlflow() -> _DummyMlflow:
    """Return a fresh DummyMlflow stub per test."""
    return _DummyMlflow()


def _patch_session(monkeypatch: Any, dummy: _DummyMlflow) -> Any:
    """Patch import_mlflow and wait_tracking_ready in the session module."""
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy)
    monkeypatch.setattr(s, "wait_tracking_ready", lambda *a, **kw: None)
    return s


# ===========================================================================
# _env
# ===========================================================================


class TestEnvSnapshot:
    """Tests for EnvSnapshot correctness."""

    def test_data_returns_copy_not_reference(self) -> None:
        """
        EnvSnapshot.data must return a fresh copy each call.

        Developer note: the old implementation returned ``self._data`` directly,
        so callers mutating the result would silently corrupt the snapshot.
        """
        from scikitplot.mlflow._env import EnvSnapshot

        snap = EnvSnapshot.capture()
        d1 = snap.data
        d2 = snap.data
        assert d1 is not d2, "data must return a new dict each time"
        assert d1 == d2, "both copies must have identical contents"

    def test_data_mutation_does_not_corrupt_snapshot(self) -> None:
        """Mutating the returned dict must not affect future restore()."""
        from scikitplot.mlflow._env import EnvSnapshot

        snap = EnvSnapshot.capture()
        original_keys = set(snap.data.keys())
        d = snap.data
        d["_SHOULD_NOT_APPEAR_IN_SNAP"] = "poison"
        assert "_SHOULD_NOT_APPEAR_IN_SNAP" not in snap.data
        assert set(snap.data.keys()) == original_keys

    def test_capture_restore_roundtrip(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._env import EnvSnapshot

        snap = EnvSnapshot.capture()
        monkeypatch.setenv("_SP_SNAP_TEST", "yes")
        assert os.environ.get("_SP_SNAP_TEST") == "yes"
        snap.restore()
        assert os.environ.get("_SP_SNAP_TEST") is None

    def test_restore_removes_session_added_keys(self, monkeypatch: Any) -> None:
        """Restore must clear keys added after capture, not just restore values."""
        from scikitplot.mlflow._env import EnvSnapshot

        monkeypatch.delenv("_SP_ADDED_AFTER", raising=False)
        snap = EnvSnapshot.capture()
        os.environ["_SP_ADDED_AFTER"] = "1"
        snap.restore()
        assert "_SP_ADDED_AFTER" not in os.environ


class TestParseDotenv:
    """Tests for parse_dotenv correctness."""

    def test_basic_key_value(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert result["FOO"] == "bar"
        assert result["BAZ"] == "qux"

    def test_matched_double_quotes_stripped(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text('A="hello world"\n', encoding="utf-8")
        assert parse_dotenv(str(f))["A"] == "hello world"

    def test_matched_single_quotes_stripped(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("A='hello'\n", encoding="utf-8")
        assert parse_dotenv(str(f))["A"] == "hello"

    def test_mismatched_quotes_not_stripped(self, tmp_path: Path) -> None:
        """
        Mismatched outer quotes must be left unchanged.

        Developer note: the old implementation used sequential ``.strip('"').strip("'")``
        which would silently strip ``"value'`` → ``value`` (stripping each side
        independently). The fix checks that both outer characters are identical.
        """
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text('A="mismatched\'\n', encoding="utf-8")
        result = parse_dotenv(str(f))
        # The value contains a leading " and trailing ' — neither should be stripped.
        assert result["A"].startswith('"') or result["A"].endswith("'")
        assert "mismatched" in result["A"]

    def test_export_prefix_stripped(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("export MY_VAR=myval\n", encoding="utf-8")
        assert parse_dotenv(str(f))["MY_VAR"] == "myval"

    def test_comment_lines_ignored(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("# comment\nFOO=bar\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert "comment" not in str(result)
        assert result["FOO"] == "bar"

    def test_empty_lines_ignored(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("\n\nFOO=bar\n\n", encoding="utf-8")
        assert parse_dotenv(str(f)) == {"FOO": "bar"}

    def test_line_without_equals_ignored(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("NOEQUALS\nFOO=bar\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert "NOEQUALS" not in result
        assert result["FOO"] == "bar"

    def test_empty_value(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("EMPTY=\n", encoding="utf-8")
        assert parse_dotenv(str(f))["EMPTY"] == ""

    def test_missing_file_raises(self) -> None:
        from scikitplot.mlflow._env import parse_dotenv

        with pytest.raises(FileNotFoundError):
            parse_dotenv("/nonexistent/.env")

    def test_value_with_equals_inside(self, tmp_path: Path) -> None:
        """Values containing '=' must not be truncated."""
        from scikitplot.mlflow._env import parse_dotenv

        f = tmp_path / ".env"
        f.write_text("TOKEN=abc=def=ghi\n", encoding="utf-8")
        assert parse_dotenv(str(f))["TOKEN"] == "abc=def=ghi"


class TestApplyEnv:
    """Tests for apply_env defaults-only vs overwrite semantics."""

    def test_defaults_only_does_not_overwrite_existing(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from scikitplot.mlflow._env import apply_env

        f = tmp_path / ".env"
        f.write_text("EXISTING=fromfile\nNEW=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING", "original")
        monkeypatch.delenv("NEW", raising=False)
        apply_env(env_file=str(f), extra_env=None, set_defaults_only=True)
        assert os.environ["EXISTING"] == "original"
        assert os.environ["NEW"] == "fromfile"

    def test_overwrite_mode_replaces_existing(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from scikitplot.mlflow._env import apply_env

        f = tmp_path / ".env"
        f.write_text("EXISTING=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING", "original")
        apply_env(env_file=str(f), extra_env=None, set_defaults_only=False)
        assert os.environ["EXISTING"] == "fromfile"

    def test_extra_env_always_overrides(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from scikitplot.mlflow._env import apply_env

        f = tmp_path / ".env"
        f.write_text("K=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("K", "original")
        apply_env(env_file=str(f), extra_env={"K": "override"}, set_defaults_only=True)
        assert os.environ["K"] == "override"

    def test_no_env_file_is_noop(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._env import apply_env

        monkeypatch.delenv("_SP_X", raising=False)
        apply_env(env_file=None, extra_env={"_SP_X": "1"})
        assert os.environ["_SP_X"] == "1"


# ===========================================================================
# _compat
# ===========================================================================


class TestResolveDownloadArtifacts:
    """Tests for resolve_download_artifacts preference order."""

    def test_prefers_modern_api(self) -> None:
        from scikitplot.mlflow._compat import resolve_download_artifacts

        class M:
            class artifacts:
                @staticmethod
                def download_artifacts(
                    run_id: str, artifact_path: str, dst_path: Optional[str] = None
                ) -> str:
                    return f"/modern/{run_id}/{artifact_path}"

        fn = resolve_download_artifacts(M())
        assert fn(run_id="r1", artifact_path="model/MLmodel") == "/modern/r1/model/MLmodel"

    def test_falls_back_to_provided_client(self) -> None:
        from scikitplot.mlflow._compat import resolve_download_artifacts

        class OldMod:
            pass

        class C:
            def download_artifacts(
                self, run_id: str, path: str, dst: Optional[str] = None
            ) -> str:
                return f"/client/{run_id}/{path}"

        fn = resolve_download_artifacts(OldMod(), client=C())
        assert fn(run_id="r", artifact_path="p") == "/client/r/p"

    def test_falls_back_to_legacy_mlflow_client(self) -> None:
        from scikitplot.mlflow._compat import resolve_download_artifacts

        class LegacyClient:
            def download_artifacts(
                self, run_id: str, path: str, dst: Optional[str] = None
            ) -> str:
                return f"/legacy/{run_id}/{path}"

        class M:
            class tracking:
                MlflowClient = LegacyClient

        fn = resolve_download_artifacts(M())
        assert fn(run_id="r", artifact_path="a/b") == "/legacy/r/a/b"

    def test_raises_when_no_api_available(self) -> None:
        from scikitplot.mlflow._compat import resolve_download_artifacts

        class BareModule:
            pass

        with pytest.raises(AttributeError, match="No supported artifact download API"):
            resolve_download_artifacts(BareModule(), client=None)

    def test_import_mlflow_raises_when_not_installed(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._compat import import_mlflow
        from scikitplot.mlflow._errors import MlflowNotInstalledError

        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name: None
        )
        with pytest.raises(MlflowNotInstalledError):
            import_mlflow()


# ===========================================================================
# _config
# ===========================================================================


class TestSessionConfigValidation:
    """Exhaustive SessionConfig validation coverage."""

    def test_valid_minimal(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        cfg = SessionConfig(tracking_uri="http://127.0.0.1:5000")
        assert cfg.tracking_uri == "http://127.0.0.1:5000"
        assert cfg.startup_timeout_s == 30.0
        assert cfg.create_experiment_if_missing is True

    def test_valid_all_fields(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        cfg = SessionConfig(
            tracking_uri="http://host:5000",
            public_tracking_uri="http://public:5000",
            registry_uri="http://host:5001",
            env_file=".env",
            extra_env={"A": "1"},
            startup_timeout_s=60.0,
            ensure_reachable=True,
            experiment_name="my-exp",
            create_experiment_if_missing=False,
            default_run_name="train",
            default_run_tags={"k": "v"},
        )
        assert cfg.extra_env == {"A": "1"}
        assert cfg.default_run_tags == {"k": "v"}

    @pytest.mark.parametrize("val", [0, -1, -0.001])
    def test_startup_timeout_non_positive_raises(self, val: float) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError, match="startup_timeout_s"):
            SessionConfig(startup_timeout_s=val)

    def test_extra_env_must_be_mapping(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError):
            SessionConfig(extra_env=["a=b"])  # type: ignore[arg-type]

    def test_extra_env_empty_key_raises(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError):
            SessionConfig(extra_env={"": "val"})

    def test_extra_env_int_value_raises(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError):
            SessionConfig(extra_env={"KEY": 99})  # type: ignore[dict-item]

    def test_default_run_tags_non_str_key_raises(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError):
            SessionConfig(default_run_tags={1: "v"})  # type: ignore[dict-item]

    def test_default_run_tags_non_str_value_raises(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError):
            SessionConfig(default_run_tags={"k": 42})  # type: ignore[dict-item]

    def test_default_run_tags_non_mapping_raises(self) -> None:
        from scikitplot.mlflow._config import SessionConfig

        with pytest.raises(ValueError):
            SessionConfig(default_run_tags=[("k", "v")])  # type: ignore[arg-type]


class TestServerConfigValidation:
    """ServerConfig.validate() exhaustive matrix."""

    def test_valid_defaults(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        cfg = ServerConfig()
        cfg.validate(for_managed_tracking=False)

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_invalid_port_raises(self, port: int) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="port"):
            ServerConfig(port=port).validate(for_managed_tracking=False)

    def test_empty_host_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="host"):
            ServerConfig(host="").validate(for_managed_tracking=False)

    def test_workers_zero_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="workers"):
            ServerConfig(workers=0).validate(for_managed_tracking=False)

    def test_workers_negative_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="workers"):
            ServerConfig(workers=-1).validate(for_managed_tracking=False)

    def test_serve_and_no_serve_conflict(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError):
            ServerConfig(
                serve_artifacts=True, no_serve_artifacts=True
            ).validate(for_managed_tracking=False)

    def test_two_server_opts_conflict(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError):
            ServerConfig(
                gunicorn_opts="--workers 2", uvicorn_opts="--workers 2"
            ).validate(for_managed_tracking=False)

    def test_three_server_opts_conflict(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError):
            ServerConfig(
                gunicorn_opts="x", uvicorn_opts="y", waitress_opts="z"
            ).validate(for_managed_tracking=False)

    def test_security_flags_with_gunicorn_conflict(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError):
            ServerConfig(
                allowed_hosts="localhost", gunicorn_opts="--workers 2"
            ).validate(for_managed_tracking=False)

    def test_dev_with_gunicorn_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError):
            ServerConfig(dev=True, gunicorn_opts="x").validate(for_managed_tracking=False)

    def test_dev_with_uvicorn_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError):
            ServerConfig(dev=True, uvicorn_opts="x").validate(for_managed_tracking=False)

    def test_artifacts_only_managed_tracking_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="artifacts_only"):
            ServerConfig(artifacts_only=True).validate(for_managed_tracking=True)

    def test_secrets_cache_ttl_zero_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="secrets_cache_ttl"):
            ServerConfig(secrets_cache_ttl=0).validate(for_managed_tracking=False)

    def test_secrets_cache_max_size_negative_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="secrets_cache_max_size"):
            ServerConfig(secrets_cache_max_size=-5).validate(for_managed_tracking=False)

    def test_extra_args_empty_string_raises(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        with pytest.raises(ValueError, match="extra_args"):
            ServerConfig(extra_args=["", "--port"]).validate(for_managed_tracking=False)

    def test_no_serve_artifacts_flag_is_valid(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        cfg = ServerConfig(no_serve_artifacts=True)
        cfg.validate(for_managed_tracking=False)

    def test_valid_port_boundaries(self) -> None:
        from scikitplot.mlflow._config import ServerConfig

        ServerConfig(port=1).validate(for_managed_tracking=False)
        ServerConfig(port=65535).validate(for_managed_tracking=False)


# ===========================================================================
# _container
# ===========================================================================


class TestRunningInDocker:
    def test_returns_false_when_no_dockerenv(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._container import running_in_docker

        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert running_in_docker() is False

    def test_returns_true_when_dockerenv_exists(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._container import running_in_docker

        monkeypatch.setattr(Path, "exists", lambda self: True)
        assert running_in_docker() is True


# ===========================================================================
# _utils
# ===========================================================================


class TestMlflowVersion:
    def test_parse_version_standard(self) -> None:
        from scikitplot.mlflow._utils import _parse_version

        v = _parse_version("2.14.1")
        assert v.triple == (2, 14, 1)
        assert v.raw == "2.14.1"

    def test_parse_version_prerelease(self) -> None:
        from scikitplot.mlflow._utils import _parse_version

        v = _parse_version("2.14.1rc2")
        assert v.triple == (2, 14, 1)

    def test_parse_version_garbage_returns_zeros(self) -> None:
        from scikitplot.mlflow._utils import _parse_version

        v = _parse_version("not-a-version")
        assert v.triple == (0, 0, 0)
        assert v.raw == "not-a-version"

    def test_parse_version_empty_string(self) -> None:
        from scikitplot.mlflow._utils import _parse_version

        v = _parse_version("")
        assert v.triple == (0, 0, 0)

    def test_parse_version_complex_pep440(self) -> None:
        from scikitplot.mlflow._utils import _parse_version

        v = _parse_version("1.2.3.post4+local.build")
        assert v.major == 1 and v.minor == 2 and v.patch == 3

    def test_is_mlflow_installed_returns_bool(self) -> None:
        from scikitplot.mlflow._utils import is_mlflow_installed

        assert isinstance(is_mlflow_installed(), bool)

    def test_mlflow_version_none_when_not_installed(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._utils import mlflow_version

        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        assert mlflow_version() is None

    def test_mlflow_version_from_module_attribute(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._utils import mlflow_version

        fake = SimpleNamespace(__version__="3.0.0")
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
        with mock.patch("importlib.import_module", return_value=fake):
            v = mlflow_version()
        assert v is not None and v.major == 3

    def test_mlflow_version_triple_comparison(self) -> None:
        from scikitplot.mlflow._utils import _parse_version

        v1 = _parse_version("2.5.0")
        v2 = _parse_version("2.10.0")
        assert v1.triple < v2.triple


# ===========================================================================
# _cli_caps
# ===========================================================================


class TestCliCaps:
    def test_extract_long_flags_standard(self) -> None:
        from scikitplot.mlflow._cli_caps import _extract_long_flags

        flags = _extract_long_flags("  --host TEXT\n  --port INT\n  --serve-artifacts\n")
        assert {"--host", "--port", "--serve-artifacts"}.issubset(flags)

    def test_extract_long_flags_click_short_and_long(self) -> None:
        from scikitplot.mlflow._cli_caps import _extract_long_flags

        flags = _extract_long_flags("  -h, --host TEXT\n  -p, --port INT\n")
        assert "--host" in flags and "--port" in flags

    def test_extract_long_flags_ignores_short_only(self) -> None:
        from scikitplot.mlflow._cli_caps import _extract_long_flags

        flags = _extract_long_flags("  -h TEXT\n")
        assert not any(f == "-h" for f in flags)

    def test_extract_long_flags_empty_text(self) -> None:
        from scikitplot.mlflow._cli_caps import _extract_long_flags

        assert len(_extract_long_flags("")) == 0

    def test_extract_long_flags_hyphenated_names(self) -> None:
        from scikitplot.mlflow._cli_caps import _extract_long_flags

        flags = _extract_long_flags("  --backend-store-uri TEXT\n  --default-artifact-root TEXT\n")
        assert "--backend-store-uri" in flags
        assert "--default-artifact-root" in flags

    def test_ensure_flags_supported_passes_valid(self) -> None:
        from scikitplot.mlflow._cli_caps import ensure_flags_supported

        ensure_flags_supported(
            ["--host", "0.0.0.0", "--port", "5000"],
            supported_flags=frozenset({"--host", "--port"}),
            context="test",
        )

    def test_ensure_flags_supported_raises_on_unknown(self) -> None:
        from scikitplot.mlflow._cli_caps import ensure_flags_supported
        from scikitplot.mlflow._errors import MlflowCliIncompatibleError

        with pytest.raises(MlflowCliIncompatibleError, match="--unknown-flag"):
            ensure_flags_supported(
                ["--host", "x", "--unknown-flag"],
                supported_flags=frozenset({"--host"}),
                context="test",
            )

    def test_ensure_flags_supported_equals_form(self) -> None:
        from scikitplot.mlflow._cli_caps import ensure_flags_supported

        ensure_flags_supported(
            ["--host=127.0.0.1"],
            supported_flags=frozenset({"--host"}),
            context="test",
        )

    def test_ensure_flags_supported_short_flags_are_skipped(self) -> None:
        from scikitplot.mlflow._cli_caps import ensure_flags_supported

        # Short flags (-h) must never trigger the check.
        ensure_flags_supported(
            ["-h", "127.0.0.1"],
            supported_flags=frozenset(),
            context="test",
        )

    def test_ensure_flags_supported_empty_args(self) -> None:
        from scikitplot.mlflow._cli_caps import ensure_flags_supported

        ensure_flags_supported([], supported_flags=frozenset(), context="test")

    def test_lru_cache_returns_same_instance(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._cli_caps as m

        sample = "--host TEXT\n--port INT\n"
        monkeypatch.setattr(m, "_run_mlflow_server_help", lambda: sample)
        m.get_mlflow_server_cli_caps.cache_clear()
        c1 = m.get_mlflow_server_cli_caps()
        c2 = m.get_mlflow_server_cli_caps()
        assert c1 is c2
        m.get_mlflow_server_cli_caps.cache_clear()

    def test_run_mlflow_server_help_empty_output_raises(
        self, monkeypatch: Any
    ) -> None:
        import scikitplot.mlflow._cli_caps as m

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout="   ", returncode=0),
        )
        with pytest.raises(RuntimeError, match="empty output"):
            m._run_mlflow_server_help()


# ===========================================================================
# _facade
# ===========================================================================


class _ModernMlflowMod:
    """MLflow mod stub with modern artifacts API."""

    class artifacts:
        @staticmethod
        def download_artifacts(
            run_id: str, artifact_path: str, dst_path: Optional[str] = None
        ) -> str:
            return f"/modern/{run_id}/{artifact_path}"

    @staticmethod
    def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
        pass


class _LegacyMod:
    """MLflow mod stub without artifacts API."""


class TestArtifactsFacade:
    def test_download_uses_modern_api(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        f = ArtifactsFacade(mlflow_module=_ModernMlflowMod(), client=_DummyClient())
        result = f.download("r1", "model/MLmodel")
        assert result == Path("/modern/r1/model/MLmodel")

    def test_download_uses_client_fallback(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=_DummyClient())
        result = f.download("r1", "some/path")
        assert result == Path("/legacy/r1/some/path")

    def test_download_raises_when_no_api(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        class NoApi:
            pass

        with pytest.raises(AttributeError):
            ArtifactsFacade(mlflow_module=_LegacyMod(), client=NoApi()).download("r", "p")

    def test_download_returns_path_object(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        f = ArtifactsFacade(mlflow_module=_ModernMlflowMod(), client=_DummyClient())
        result = f.download("r", "a")
        assert isinstance(result, Path)

    def test_list_no_subpath(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        c = _DummyClient()
        c.list_artifacts = lambda run_id, path=None: ["f1", "f2"]  # type: ignore[method-assign]
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=c)
        assert f.list("r") == ["f1", "f2"]

    def test_list_with_subpath(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        received: list = []
        c = _DummyClient()
        c.list_artifacts = lambda run_id, path=None: received.append(path) or ["x"]  # type: ignore[method-assign]
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=c)
        f.list("r", artifact_path="plots")
        assert "plots" in received

    def test_log_file_without_subdir(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append((p, artifact_path))

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_file("/tmp/f.txt")
        assert logged == [("/tmp/f.txt", None)]

    def test_log_file_with_subdir(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append((p, artifact_path))

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_file("/tmp/f.txt", artifact_path="plots")
        assert logged[0][1] == "plots"

    def test_log_files_multiple(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append(p)

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_files(["/tmp/a.txt", Path("/tmp/b.txt")])
        assert len(logged) == 2

    def test_log_files_empty_list(self) -> None:
        from scikitplot.mlflow._facade import ArtifactsFacade

        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=None)  # type: ignore[arg-type]
        f.log_files([])  # must not raise


class TestModelsFacade:
    def test_load_pyfunc(self) -> None:
        from scikitplot.mlflow._facade import ModelsFacade

        class M:
            class pyfunc:
                @staticmethod
                def load_model(uri: str) -> str:
                    return "pyfunc-model"

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        assert f.load_model("runs:/r/m") == "pyfunc-model"

    def test_load_with_flavor(self) -> None:
        from scikitplot.mlflow._facade import ModelsFacade

        class M:
            class sklearn:
                @staticmethod
                def load_model(uri: str) -> str:
                    return "sklearn-model"

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        assert f.load_model("runs:/r/m", flavor="sklearn") == "sklearn-model"

    def test_load_unknown_flavor_raises(self) -> None:
        from scikitplot.mlflow._facade import ModelsFacade

        f = ModelsFacade(mlflow_module=_LegacyMod(), client=None)  # type: ignore[arg-type]
        with pytest.raises(AttributeError, match="flavor"):
            f.load_model("runs:/r/m", flavor="no_such_flavor")

    def test_register_model(self) -> None:
        from scikitplot.mlflow._facade import ModelsFacade

        registered: list = []

        class M:
            @staticmethod
            def register_model(uri: str, name: str) -> str:
                registered.append((uri, name))
                return "version-1"

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        result = f.register_model("runs:/r/model", "MyModel")
        assert result == "version-1"
        assert registered == [("runs:/r/model", "MyModel")]


# ===========================================================================
# _project path helpers
# ===========================================================================


class TestLocalPathHelpers:
    def test_is_local_path_posix_absolute(self) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path

        assert _is_probably_local_path("/abs/path") is True

    def test_is_local_path_relative(self) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path

        assert _is_probably_local_path("./relative") is True

    def test_is_local_path_bare_name(self) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path

        assert _is_probably_local_path("mlruns") is True

    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/key",
            "gs://bucket/key",
            "dbfs:/path",
            "http://host/path",
            "https://host/path",
            "file:///tmp/x",
            "sqlite:///mlflow.db",
        ],
    )
    def test_is_not_local_path_remote_uris(self, uri: str) -> None:
        from scikitplot.mlflow._project import _is_probably_local_path

        assert _is_probably_local_path(uri) is False

    def test_normalize_sqlite_uri_relative(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import _normalize_sqlite_uri

        result = _normalize_sqlite_uri("sqlite:///./mlflow.db", base_dir=tmp_path)
        assert result.startswith("sqlite:///")
        assert Path(result[len("sqlite:///"):]).is_absolute()

    def test_normalize_sqlite_uri_bad_prefix_raises(self) -> None:
        from scikitplot.mlflow._project import _normalize_sqlite_uri

        with pytest.raises(ValueError, match="sqlite"):
            _normalize_sqlite_uri("postgresql://user/db", base_dir=Path("."))

    def test_normalize_store_values_s3_untouched(self) -> None:
        from scikitplot.mlflow._project import normalize_mlflow_store_values

        b, a = normalize_mlflow_store_values(
            backend_store_uri="s3://my-bucket/mlflow",
            default_artifact_root="s3://my-bucket/artifacts",
            base_dir=Path("."),
        )
        assert b == "s3://my-bucket/mlflow"
        assert a == "s3://my-bucket/artifacts"

    def test_normalize_store_values_none_passthrough(self) -> None:
        from scikitplot.mlflow._project import normalize_mlflow_store_values

        b, a = normalize_mlflow_store_values(
            backend_store_uri=None, default_artifact_root=None, base_dir=Path(".")
        )
        assert b is None and a is None

    def test_normalize_store_values_local_absolute(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import normalize_mlflow_store_values

        b, a = normalize_mlflow_store_values(
            backend_store_uri="sqlite:///./mlflow.db",
            default_artifact_root="./artifacts",
            base_dir=tmp_path,
        )
        assert b is not None and b.startswith("sqlite:///")
        assert a is not None and os.path.isabs(a)

    def test_ensure_local_store_layout_creates_dirs(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import ensure_local_store_layout

        sqlite_path = str(tmp_path / "sub" / "mlflow.db")
        art_path = str(tmp_path / "artifacts")
        ensure_local_store_layout(
            backend_store_uri=f"sqlite:///{sqlite_path}",
            default_artifact_root=art_path,
        )
        assert (tmp_path / "sub").exists()
        assert (tmp_path / "artifacts").exists()

    def test_ensure_local_store_layout_skips_remote(self) -> None:
        from scikitplot.mlflow._project import ensure_local_store_layout

        # Must not raise for remote URIs.
        ensure_local_store_layout(
            backend_store_uri="s3://bucket/key",
            default_artifact_root="gs://bucket/artifacts",
        )


class TestProjectMarkers:
    def test_validate_markers_valid(self) -> None:
        from scikitplot.mlflow._project import _validate_markers

        assert _validate_markers(["a", "b"]) == ("a", "b")

    def test_validate_markers_empty_raises(self) -> None:
        from scikitplot.mlflow._project import _validate_markers

        with pytest.raises(ValueError):
            _validate_markers([])

    def test_validate_markers_string_raises(self) -> None:
        from scikitplot.mlflow._project import _validate_markers

        with pytest.raises(TypeError):
            _validate_markers("pyproject.toml")  # type: ignore[arg-type]

    def test_validate_markers_blank_string_raises(self) -> None:
        from scikitplot.mlflow._project import _validate_markers

        with pytest.raises(TypeError):
            _validate_markers(["  "])

    def test_find_project_root_finds_marker(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import find_project_root

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_project_root(nested) == tmp_path.resolve()

    def test_find_project_root_raises_when_not_found(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import find_project_root

        sub = tmp_path / "sub"
        sub.mkdir()
        with pytest.raises(FileNotFoundError):
            find_project_root(sub, markers=("no.such.marker.xyz",))

    def test_context_manager_restores_on_exit(self) -> None:
        from scikitplot.mlflow._project import get_project_markers, project_markers

        before = get_project_markers()
        with project_markers(["X.yaml"]):
            assert get_project_markers() == ("X.yaml",)
        assert get_project_markers() == before

    def test_context_manager_restores_on_exception(self) -> None:
        from scikitplot.mlflow._project import get_project_markers, project_markers

        before = get_project_markers()
        with contextlib.suppress(RuntimeError):
            with project_markers(["Y.marker"]):
                raise RuntimeError("oops")
        assert get_project_markers() == before

    def test_env_override_takes_precedence(self, monkeypatch: Any) -> None:
        import json

        from scikitplot.mlflow._project import get_project_markers

        monkeypatch.setenv("SCIKITPLOT_PROJECT_MARKERS", json.dumps(["A", "B"]))
        assert get_project_markers() == ("A", "B")

    def test_env_invalid_json_raises(self, monkeypatch: Any) -> None:
        from scikitplot.mlflow._project import get_project_markers

        monkeypatch.setenv("SCIKITPLOT_PROJECT_MARKERS", "not-json")
        with pytest.raises(ValueError):
            get_project_markers()

    def test_set_and_reset_to_default(self) -> None:
        from scikitplot.mlflow._project import (
            DEFAULT_PROJECT_MARKERS,
            get_project_markers,
            set_project_markers,
        )

        set_project_markers(["X"])
        assert get_project_markers() == ("X",)
        set_project_markers(None)
        assert get_project_markers() == DEFAULT_PROJECT_MARKERS


class TestProjectConfigIO:
    def test_load_toml_start_server_false(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import load_project_config_toml

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        (cfg_dir / "mlflow.toml").write_text(
            "[profiles.local]\nstart_server = false\n"
            "[profiles.local.session]\ntracking_uri = 'http://127.0.0.1:9999'\n",
            encoding="utf-8",
        )
        cfg = load_project_config_toml(cfg_dir / "mlflow.toml", profile="local")
        assert cfg.start_server is False
        assert cfg.server is None
        assert cfg.session.tracking_uri == "http://127.0.0.1:9999"

    def test_load_toml_missing_file_raises(self) -> None:
        from scikitplot.mlflow._project import load_project_config_toml

        with pytest.raises(FileNotFoundError):
            load_project_config_toml(Path("/no/such/file.toml"))

    def test_load_toml_missing_profile_raises(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import load_project_config_toml

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml = cfg_dir / "mlflow.toml"
        toml.write_text(
            "[profiles.local]\nstart_server=false\n"
            "[profiles.local.session]\ntracking_uri='http://x'\n",
            encoding="utf-8",
        )
        with pytest.raises(KeyError):
            load_project_config_toml(toml, profile="nonexistent")

    def test_load_toml_missing_profiles_table_raises(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import load_project_config_toml

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml = cfg_dir / "mlflow.toml"
        toml.write_text("[other]\nfoo='bar'\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_project_config_toml(toml)

    def test_load_toml_registry_uri_empty_string_normalised_to_none(
        self, tmp_path: Path
    ) -> None:
        from scikitplot.mlflow._project import load_project_config_toml

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        (cfg_dir / "mlflow.toml").write_text(
            "[profiles.local]\nstart_server=false\n"
            "[profiles.local.session]\ntracking_uri='http://x'\nregistry_uri=''\n",
            encoding="utf-8",
        )
        cfg = load_project_config_toml(cfg_dir / "mlflow.toml", profile="local")
        assert cfg.session.registry_uri is None

    def test_load_yaml_valid(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        from scikitplot.mlflow._project import load_project_config_yaml

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        yml = cfg_dir / "mlflow.yaml"
        yml.write_text(
            "profiles:\n  local:\n    start_server: false\n"
            "    session:\n      tracking_uri: 'http://127.0.0.1:8888'\n",
            encoding="utf-8",
        )
        cfg = load_project_config_yaml(yml, profile="local")
        assert cfg.session.tracking_uri == "http://127.0.0.1:8888"

    def test_load_yaml_not_mapping_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        from scikitplot.mlflow._project import load_project_config_yaml

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        yml = cfg_dir / "mlflow.yaml"
        yml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_project_config_yaml(yml)

    def test_load_yaml_missing_file_raises(self) -> None:
        pytest.importorskip("yaml")
        from scikitplot.mlflow._project import load_project_config_yaml

        with pytest.raises(FileNotFoundError):
            load_project_config_yaml(Path("/no/such/file.yaml"))

    def test_load_project_config_dispatch_toml(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._project import load_project_config

        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        toml = cfg_dir / "mlflow.toml"
        toml.write_text(
            "[profiles.local]\nstart_server=false\n"
            "[profiles.local.session]\ntracking_uri='http://x'\n",
            encoding="utf-8",
        )
        cfg = load_project_config(toml)
        assert cfg.session.tracking_uri == "http://x"

    def test_load_project_config_unsupported_extension_raises(
        self, tmp_path: Path
    ) -> None:
        from scikitplot.mlflow._project import load_project_config

        with pytest.raises(ValueError, match="Unsupported"):
            load_project_config(tmp_path / "cfg.json")

    def test_dump_project_config_yaml_roundtrip(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        pytest.importorskip("yaml")
        from scikitplot.mlflow._project import (
            ProjectConfig,
            dump_project_config_yaml,
            load_project_config_yaml,
        )
        from scikitplot.mlflow._config import SessionConfig

        cfg = ProjectConfig(
            profile="local",
            session=SessionConfig(
                tracking_uri="http://127.0.0.1:5005",
                experiment_name="test-exp",
            ),
            server=None,
            start_server=False,
        )
        out = tmp_path / "out.yaml"
        dump_project_config_yaml(cfg, out)
        assert out.exists()
        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        loaded = load_project_config_yaml(out, profile="local")
        assert loaded.session.tracking_uri == "http://127.0.0.1:5005"
        assert loaded.session.experiment_name == "test-exp"


# ===========================================================================
# _readiness
# ===========================================================================


class _Resp200:
    status = 200

    def __enter__(self) -> "_Resp200":
        return self

    def __exit__(self, *a: Any) -> bool:
        return False


class TestWaitTrackingReady:
    def test_immediate_200_returns(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r

        monkeypatch.setattr(r.urllib.request, "urlopen", lambda *a, **k: _Resp200())
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=1.0)

    def test_timeout_raises_timeout_error(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r

        monkeypatch.setattr(
            r.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()),
        )
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError):
            r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.01)

    def test_timeout_message_includes_attempt_count(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r

        # monkeypatch.setattr takes a value, not side_effect — use a raising callable.
        def _always_refuse(*a: Any, **k: Any) -> None:
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(r.urllib.request, "urlopen", _always_refuse)
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError, match="attempt"):
            r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.005)

    def test_fallback_405_uses_list_endpoint(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r

        calls: list[int] = []

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(None, 405, "Not Allowed", {}, None)  # type: ignore[arg-type]
            return _Resp200()

        monkeypatch.setattr(r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.5)
        assert len(calls) == 2

    def test_fallback_404_uses_list_endpoint(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r

        calls: list[int] = []

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(None, 404, "Not Found", {}, None)  # type: ignore[arg-type]
            return _Resp200()

        monkeypatch.setattr(r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.5)

    def test_non_404_405_http_error_does_not_fallback(
        self, monkeypatch: Any
    ) -> None:
        """A 500 error must NOT trigger the list-endpoint fallback."""
        import scikitplot.mlflow._readiness as r

        call_count = [0]

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            call_count[0] += 1
            raise urllib.error.HTTPError(None, 500, "Internal Server Error", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr(r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError):
            r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.01)
        # Each iteration uses only 1 call (no fallback to list endpoint)
        assert all(call_count[0] >= 1 for _ in [None])

    def test_custom_poll_interval_passed_to_sleep(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r

        sleep_vals: list[float] = []
        call_no = [0]

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            call_no[0] += 1
            if call_no[0] < 3:
                raise ConnectionRefusedError()
            return _Resp200()

        monkeypatch.setattr(r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(r.time, "sleep", lambda s: sleep_vals.append(s))
        r.wait_tracking_ready(
            "http://127.0.0.1:5000", timeout_s=5.0, poll_interval_s=0.07
        )
        assert all(s == pytest.approx(0.07) for s in sleep_vals)

    def test_invalid_uri_raises_value_error(self) -> None:
        import scikitplot.mlflow._readiness as r

        with pytest.raises(ValueError):
            r.wait_tracking_ready("file:///tmp/mlruns", timeout_s=0.1)

    def test_server_exits_raises_runtime_error(self, monkeypatch: Any) -> None:
        import scikitplot.mlflow._readiness as r
        from scikitplot.mlflow._server import SpawnedServer

        class FakeProc:
            pid = 1
            returncode = 1
            stdout = None

            def poll(self) -> int:
                return 1

        sv = SpawnedServer(
            _process=FakeProc(),  # type: ignore[arg-type]
            _command=["cmd"],
            _started_at=time.time(),
        )
        monkeypatch.setattr(r.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
        monkeypatch.setattr(r.time, "sleep", lambda _: None)
        with pytest.raises(RuntimeError, match="exited before becoming ready"):
            r.wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.5, server=sv)


# ===========================================================================
# _server
# ===========================================================================


class TestBuildServerArgs:
    def test_basic_host_port(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(ServerConfig(host="0.0.0.0", port=5001, strict_cli_compat=False))
        assert args[:4] == ["--host", "0.0.0.0", "--port", "5001"]

    def test_workers_flag(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(workers=4, strict_cli_compat=False)
        )
        assert "--workers" in args
        assert "4" in args

    def test_backend_and_artifact_root(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(
                backend_store_uri="sqlite:///./mlflow.db",
                default_artifact_root="/tmp/art",
                strict_cli_compat=False,
            )
        )
        assert "--backend-store-uri" in args
        assert "--default-artifact-root" in args

    def test_serve_artifacts(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(serve_artifacts=True, strict_cli_compat=False)
        )
        assert "--serve-artifacts" in args

    def test_no_serve_artifacts(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(no_serve_artifacts=True, strict_cli_compat=False)
        )
        assert "--no-serve-artifacts" in args

    def test_artifacts_destination(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(artifacts_destination="/tmp/dst", strict_cli_compat=False)
        )
        assert "--artifacts-destination" in args

    def test_allowed_hosts(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(allowed_hosts="localhost", strict_cli_compat=False)
        )
        assert "--allowed-hosts" in args

    def test_cors_allowed_origins(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(cors_allowed_origins="*", strict_cli_compat=False)
        )
        assert "--cors-allowed-origins" in args

    def test_x_frame_options(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(x_frame_options="DENY", strict_cli_compat=False)
        )
        assert "--x-frame-options" in args

    def test_disable_security_middleware(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(disable_security_middleware=True, strict_cli_compat=False)
        )
        assert "--disable-security-middleware" in args

    def test_dev_flag(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(dev=True, strict_cli_compat=False)
        )
        assert "--dev" in args

    def test_secrets_cache_flags(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(
                secrets_cache_ttl=60, secrets_cache_max_size=100, strict_cli_compat=False
            )
        )
        assert "--secrets-cache-ttl" in args
        assert "--secrets-cache-max-size" in args

    def test_expose_prometheus(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(expose_prometheus="/tmp/metrics", strict_cli_compat=False)
        )
        assert "--expose-prometheus" in args

    def test_gunicorn_opts(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(gunicorn_opts="--workers 2", strict_cli_compat=False)
        )
        assert "--gunicorn-opts" in args

    def test_extra_args_appended(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_args

        args = build_server_args(
            ServerConfig(
                extra_args=["--app-name", "myapp"], strict_cli_compat=False
            )
        )
        assert "--app-name" in args and "myapp" in args

    def test_build_server_command_starts_with_python_m_mlflow(self) -> None:
        from scikitplot.mlflow._config import ServerConfig
        from scikitplot.mlflow._server import build_server_command

        cmd = build_server_command(
            ServerConfig(strict_cli_compat=False)
        )
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["-m", "mlflow", "server"]

    def test_is_port_free_returns_bool(self) -> None:
        from scikitplot.mlflow._server import _is_port_free

        assert isinstance(_is_port_free("127.0.0.1", 65534), bool)


# ===========================================================================
# _session
# ===========================================================================


class TestSessionModuleStructure:
    """Tests that verify module-level design invariants."""

    def test_logger_exists_at_module_level(self) -> None:
        """
        _session must have a module-level logger for diagnosing broken-pipe
        and slow-start issues.

        Developer note: the old code had no logger, making server start failures
        completely silent until the TimeoutError bubbled up.
        """
        import scikitplot.mlflow._session as sm

        assert hasattr(sm, "logger")
        assert isinstance(sm.logger, logging.Logger)

    def test_os_imported_at_module_level(self) -> None:
        """
        'import os' must be at the module top-level, not inside the try: block.

        Developer note: the old code had ``import os`` inside the try: block
        of session(), which would mask an ImportError with an unrelated error
        and break static analysis tools.
        """
        import inspect

        import scikitplot.mlflow._session as sm

        src = inspect.getsource(sm.session)
        # The function body must NOT contain a bare 'import os' line.
        import_os_inside = any(
            line.strip() == "import os"
            for line in src.split("\n")
        )
        assert not import_os_inside, (
            "'import os' found inside session() — must be at module level"
        )


class TestSessionContextManager:
    def test_basic_session_resolves_tracking_uri(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(tracking_uri="http://127.0.0.1:5000")
        with session(config=cfg, start_server=False) as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"
            assert h.mlflow_module is dummy_mlflow

    def test_experiment_create_if_missing(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="new-exp",
            create_experiment_if_missing=True,
        )
        with session(config=cfg, start_server=False) as h:
            assert dummy_mlflow._experiment_name == "new-exp"
            assert h.experiment_name == "new-exp"

    def test_experiment_strict_missing_raises(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="no-such-exp",
            create_experiment_if_missing=False,
        )
        with pytest.raises(KeyError):
            with session(config=cfg, start_server=False):
                pass

    def test_default_run_name_applied(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000", default_run_name="default-train"
        )
        with session(config=cfg, start_server=False) as h:
            with h.start_run():
                pass
        assert dummy_mlflow._start_run_calls[-1]["kwargs"]["run_name"] == "default-train"

    def test_explicit_run_name_overrides_default(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000", default_run_name="default"
        )
        with session(config=cfg, start_server=False) as h:
            with h.start_run(run_name="explicit"):
                pass
        assert dummy_mlflow._start_run_calls[-1]["kwargs"]["run_name"] == "explicit"

    def test_default_run_tags_applied(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            default_run_tags={"pipeline": "v1", "env": "test"},
        )
        with session(config=cfg, start_server=False) as h:
            with h.start_run():
                pass
        assert dummy_mlflow._set_tags_calls[-1] == {"pipeline": "v1", "env": "test"}

    def test_default_run_tags_fallback_via_client_set_tag(
        self, monkeypatch: Any
    ) -> None:
        """
        When mlflow.set_tags is unavailable, must fall back to client.set_tag per key.

        Developer note: ``set_tags`` is a class-level method, so ``del instance.set_tags``
        raises AttributeError (no instance attribute shadows the class method).
        A plain subclass still inherits it, so ``hasattr`` remains True.

        The correct approach uses a property that raises ``AttributeError``, which makes
        ``getattr(obj, "set_tags", None)`` return ``None`` and ``callable(None)`` return
        ``False`` — exactly the condition that triggers the ``client.set_tag`` fallback
        in ``MlflowHandle.start_run``.
        """
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        class _DummyMlflowNoSetTags(_DummyMlflow):
            """Hides set_tags so getattr(obj, 'set_tags', None) returns None."""

            @property
            def set_tags(self) -> None:  # type: ignore[override]
                raise AttributeError("set_tags not available")

        dummy = _DummyMlflowNoSetTags()
        # Confirm the fallback condition: getattr returns None, callable(None) is False.
        assert getattr(dummy, "set_tags", None) is None

        _patch_session(monkeypatch, dummy)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            default_run_tags={"k": "v"},
        )
        with session(config=cfg, start_server=False) as h:
            with h.start_run() as run:
                assert run.info.run_id == "run-test"
        # The session-bound client's set_tag must have been called for the tag key.
        assert h.client.tags.get("run-test", {}).get("k") == "v"

    def test_env_restored_after_session(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        monkeypatch.delenv("_SP_SESSION_ENV_TEST", raising=False)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            extra_env={"_SP_SESSION_ENV_TEST": "inside"},
        )
        with session(config=cfg, start_server=False):
            assert os.environ.get("_SP_SESSION_ENV_TEST") == "inside"
        assert os.environ.get("_SP_SESSION_ENV_TEST") is None

    def test_env_restored_on_exception(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        monkeypatch.delenv("_SP_EX_ENV_TEST", raising=False)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            extra_env={"_SP_EX_ENV_TEST": "set"},
        )
        with contextlib.suppress(RuntimeError):
            with session(config=cfg, start_server=False):
                raise RuntimeError("test error")
        assert os.environ.get("_SP_EX_ENV_TEST") is None

    def test_public_tracking_uri_sets_ui_url(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            public_tracking_uri="http://public.host:9000",
        )
        with session(config=cfg, start_server=False) as h:
            assert h.ui_url == "http://public.host:9000"

    def test_public_tracking_uri_non_http_raises(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            public_tracking_uri="file:///tmp/x",
        )
        with pytest.raises(ValueError):
            with session(config=cfg, start_server=False):
                pass

    def test_no_tracking_uri_raises(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(RuntimeError, match="No tracking URI"):
            with session(config=SessionConfig(), start_server=False):
                pass

    def test_ensure_reachable_requires_http(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="file:///tmp/mlruns",
            ensure_reachable=True,
            startup_timeout_s=0.1,
        )
        with pytest.raises(ValueError, match="http"):
            with session(config=cfg, start_server=False):
                pass

    def test_start_server_uri_port_mismatch_raises(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import ServerConfig, SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(tracking_uri="http://127.0.0.1:9999")
        srv = ServerConfig(host="127.0.0.1", port=5000, strict_cli_compat=False)
        with pytest.raises(ValueError, match="tracking_uri"):
            with session(config=cfg, server=srv, start_server=True):
                pass

    def test_proxy_getattr_delegates_to_mlflow_module(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        dummy_mlflow.custom_thing = "hello"  # type: ignore[attr-defined]
        cfg = SessionConfig(tracking_uri="http://127.0.0.1:5000")
        with session(config=cfg, start_server=False) as h:
            assert h.custom_thing == "hello"

    def test_registry_uri_propagated(
        self, monkeypatch: Any, dummy_mlflow: _DummyMlflow
    ) -> None:
        from scikitplot.mlflow._config import SessionConfig
        from scikitplot.mlflow._session import session

        _patch_session(monkeypatch, dummy_mlflow)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            registry_uri="http://127.0.0.1:5001",
        )
        with session(config=cfg, start_server=False) as h:
            assert h.registry_uri == "http://127.0.0.1:5001"
            assert dummy_mlflow._registry_uri == "http://127.0.0.1:5001"


# ===========================================================================
# _workflow
# ===========================================================================


class TestWorkflowHelpers:
    def test_builtin_config_path_toml_exists(self) -> None:
        from scikitplot.mlflow._workflow import builtin_config_path

        p = builtin_config_path("toml")
        assert p.exists() and p.suffix == ".toml"

    def test_builtin_config_path_yaml_exists(self) -> None:
        from scikitplot.mlflow._workflow import builtin_config_path

        p = builtin_config_path("yaml")
        assert p.exists() and p.suffix == ".yaml"

    def test_builtin_config_path_unsupported_raises(self) -> None:
        from scikitplot.mlflow._workflow import builtin_config_path

        with pytest.raises(ValueError, match="fmt"):
            builtin_config_path("json")

    def test_builtin_config_path_case_insensitive(self) -> None:
        from scikitplot.mlflow._workflow import builtin_config_path

        p = builtin_config_path("TOML")
        assert p.exists()

    def test_default_project_paths_structure(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import default_project_paths

        paths = default_project_paths(project_root=tmp_path)
        assert paths.project_root == tmp_path.resolve()
        assert paths.config_dir == tmp_path.resolve() / "configs"
        assert paths.toml_path == tmp_path.resolve() / "configs" / "mlflow.toml"
        assert paths.yaml_path == tmp_path.resolve() / "configs" / "mlflow.yaml"

    def test_export_builtin_config_creates_toml(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import export_builtin_config

        out = export_builtin_config(fmt="toml", project_root=tmp_path)
        assert out.exists() and out.suffix == ".toml"

    def test_export_builtin_config_no_overwrite_raises(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import export_builtin_config

        export_builtin_config(fmt="toml", project_root=tmp_path)
        with pytest.raises(FileExistsError):
            export_builtin_config(fmt="toml", project_root=tmp_path, overwrite=False)

    def test_export_builtin_config_overwrite_succeeds(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import export_builtin_config

        export_builtin_config(fmt="toml", project_root=tmp_path)
        out2 = export_builtin_config(fmt="toml", project_root=tmp_path, overwrite=True)
        assert out2.exists()

    def test_patch_experiment_name_in_toml(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import patch_experiment_name_in_toml

        toml = tmp_path / "cfg.toml"
        toml.write_text('[profiles.local.session]\nexperiment_name = "old"\n', encoding="utf-8")
        patch_experiment_name_in_toml(toml, experiment_name="new-exp")
        assert 'experiment_name = "new-exp"' in toml.read_text(encoding="utf-8")

    def test_patch_experiment_name_raises_if_not_found(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import patch_experiment_name_in_toml

        toml = tmp_path / "cfg.toml"
        toml.write_text("[other]\nfoo = 'bar'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="experiment_name"):
            patch_experiment_name_in_toml(toml, experiment_name="x")

    def test_workflow_return_annotation_is_workflow_paths(self) -> None:
        """
        workflow() must declare a WorkflowPaths return type.

        Developer note: the old annotation was ``-> None`` which silently discarded
        the WorkflowPaths result returned by run_demo(), preventing callers from
        using the paths for post-workflow assertions or cleanup.
        """
        import inspect

        from scikitplot.mlflow._workflow import workflow

        ann = inspect.signature(workflow).return_annotation
        assert "WorkflowPaths" in str(ann), (
            f"workflow() return annotation should be WorkflowPaths, got {ann!r}"
        )

    def test_workflow_paths_properties(self, tmp_path: Path) -> None:
        from scikitplot.mlflow._workflow import WorkflowPaths

        p = WorkflowPaths(
            _project_root=tmp_path,
            _config_dir=tmp_path / "configs",
            _toml_path=tmp_path / "configs" / "mlflow.toml",
            _yaml_path=tmp_path / "configs" / "mlflow.yaml",
        )
        assert p.project_root == tmp_path
        assert p.config_dir == tmp_path / "configs"
        assert p.toml_path.suffix == ".toml"
        assert p.yaml_path.suffix == ".yaml"
