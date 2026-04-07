# scikitplot/mlflow/tests/test__session.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._session.

Naming convention: test__<module_name>.py

Covers gaps (target: 90%+ from 82% baseline)
----------------------------------------------
- _parse_http_uri     : non-http scheme raises, missing hostname raises,
                        default ports (80/443), explicit port
- _hosts_equivalent   : all bind/tracking host combinations
- _default_tracking_host_for_bind : wildcard → 127.0.0.1, specific → pass through
- _set_experiment_strict : old API path (get_experiment_by_name absent), strict missing
- MlflowHandle        : proxy __getattr__, start_run client.set_tag fallback
- session             : no tracking URI raises, public_tracking_uri non-http raises,
                        start_server port mismatch raises, registry_uri propagated,
                        TypeError client fallback, server terminate on exit,
                        server terminate on exception
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
import os
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

import pytest

from scikitplot.mlflow._config import ServerConfig, SessionConfig
from scikitplot.mlflow._project import ProjectConfig
from scikitplot.mlflow._session import (
    MlflowHandle,
    _default_tracking_host_for_bind,
    _hosts_equivalent,
    _parse_http_uri,
    _set_experiment_strict,
    session,
)


# ---------------------------------------------------------------------------
# Stubs
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


class _Client:
    def __init__(self, tracking_uri=None, registry_uri=None):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.tags: Dict[str, Dict[str, str]] = {}

    def set_tag(self, run_id, key, value):
        self.tags.setdefault(run_id, {})[key] = value


class _Mlflow:
    def __init__(self):
        self.__version__ = "2.14.0"
        self._tracking_uri: Optional[str] = None
        self._registry_uri: Optional[str] = None
        self._experiments: Dict[str, Any] = {}
        self._set_tags_calls: list = []
        self._start_run_kwargs: list = []
        self.tracking = SimpleNamespace(MlflowClient=self._make_client)

    def set_tracking_uri(self, uri):
        self._tracking_uri = uri

    def set_registry_uri(self, uri):
        self._registry_uri = uri

    def _make_client(self, tracking_uri=None, registry_uri=None):
        return _Client(tracking_uri=tracking_uri, registry_uri=registry_uri)

    def get_experiment_by_name(self, name):
        return self._experiments.get(name)

    def set_experiment(self, name):
        self._experiments[name] = object()

    @contextlib.contextmanager
    def start_run(self, *args, **kwargs):
        self._start_run_kwargs.append(dict(kwargs))
        yield _Run(info=_RunInfo(run_id="run-abc"))

    def set_tags(self, tags):
        self._set_tags_calls.append(dict(tags))


def _patch_session(monkeypatch, dummy):
    """Patch _session module dependencies."""
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy)
    monkeypatch.setattr(s, "wait_tracking_ready", lambda *a, **kw: None)
    return s


# ===========================================================================
# _parse_http_uri
# ===========================================================================


class TestParseHttpUri:
    """Tests for _parse_http_uri."""

    def test_standard_http_uri(self) -> None:
        scheme, host, port = _parse_http_uri("http://127.0.0.1:5000")
        assert scheme == "http"
        assert host == "127.0.0.1"
        assert port == 5000

    def test_https_default_port_is_443(self) -> None:
        _, _, port = _parse_http_uri("https://mlflow.example.com")
        assert port == 443

    def test_http_default_port_is_80(self) -> None:
        _, _, port = _parse_http_uri("http://mlflow.example.com")
        assert port == 80

    def test_non_http_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="http"):
            _parse_http_uri("file:///tmp/mlruns")

    def test_explicit_port_preserved(self) -> None:
        _, _, port = _parse_http_uri("http://localhost:8080")
        assert port == 8080

    def test_trailing_slash_stripped(self) -> None:
        scheme, host, port = _parse_http_uri("http://localhost:5000/")
        assert scheme == "http"
        assert host == "localhost"


# ===========================================================================
# _hosts_equivalent
# ===========================================================================


class TestHostsEquivalent:
    """Tests for _hosts_equivalent."""

    def test_loopback_to_loopback(self) -> None:
        assert _hosts_equivalent("127.0.0.1", "127.0.0.1") is True
        assert _hosts_equivalent("127.0.0.1", "localhost") is True

    def test_wildcard_allows_loopback(self) -> None:
        assert _hosts_equivalent("0.0.0.0", "127.0.0.1") is True
        assert _hosts_equivalent("0.0.0.0", "localhost") is True
        assert _hosts_equivalent("::", "::1") is True

    def test_wildcard_rejects_external(self) -> None:
        assert _hosts_equivalent("0.0.0.0", "192.168.1.1") is False

    def test_specific_host_matches_itself(self) -> None:
        assert _hosts_equivalent("myhost", "myhost") is True

    def test_specific_host_rejects_different(self) -> None:
        assert _hosts_equivalent("myhost", "otherhost") is False

    def test_ipv6_loopback(self) -> None:
        assert _hosts_equivalent("::1", "::1") is True
        assert _hosts_equivalent("::1", "localhost") is True


# ===========================================================================
# _default_tracking_host_for_bind
# ===========================================================================


class TestDefaultTrackingHostForBind:
    """Tests for _default_tracking_host_for_bind."""

    def test_wildcard_ipv4_returns_loopback(self) -> None:
        assert _default_tracking_host_for_bind("0.0.0.0") == "127.0.0.1"

    def test_wildcard_ipv6_returns_loopback(self) -> None:
        assert _default_tracking_host_for_bind("::") == "127.0.0.1"

    def test_specific_host_returned_unchanged(self) -> None:
        assert _default_tracking_host_for_bind("myserver") == "myserver"

    def test_empty_host_returns_loopback(self) -> None:
        assert _default_tracking_host_for_bind("") == "127.0.0.1"

    def test_loopback_returns_loopback(self) -> None:
        assert _default_tracking_host_for_bind("127.0.0.1") == "127.0.0.1"


# ===========================================================================
# _set_experiment_strict
# ===========================================================================


class TestSetExperimentStrict:
    """Tests for _set_experiment_strict."""

    def test_creates_missing_experiment_when_allowed(self) -> None:
        dummy = _Mlflow()
        _set_experiment_strict(dummy, experiment_name="new-exp", create_if_missing=True)
        assert dummy._experiment_name_or_key() if hasattr(dummy, "_experiment_name_or_key") else "new-exp" in dummy._experiments

    def test_raises_when_strict_and_missing(self) -> None:
        dummy = _Mlflow()
        with pytest.raises(KeyError, match="new-exp"):
            _set_experiment_strict(dummy, experiment_name="new-exp", create_if_missing=False)

    def test_passes_when_strict_and_exists(self) -> None:
        dummy = _Mlflow()
        dummy._experiments["existing"] = object()
        _set_experiment_strict(dummy, experiment_name="existing", create_if_missing=False)

    def test_old_api_without_get_experiment_raises(self) -> None:
        """When get_experiment_by_name is absent, strict mode must raise KeyError."""
        # Use SimpleNamespace to model old MLflow without get_experiment_by_name.
        # Note: `del instance.class_method` cannot work for class-defined methods;
        # a stub without the attribute is the correct approach.
        old_api = SimpleNamespace(
            set_experiment=lambda name: None,
            # no get_experiment_by_name
        )
        with pytest.raises(KeyError, match="requires mlflow.get_experiment_by_name"):
            _set_experiment_strict(old_api, experiment_name="x", create_if_missing=False)


# ===========================================================================
# session - error paths and coverage gaps
# ===========================================================================


class TestSessionErrorPaths:
    """Tests for session() error and edge-case paths."""

    def test_no_tracking_uri_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no tracking URI is available, raise RuntimeError."""
        dummy = _Mlflow()
        _patch_session(monkeypatch, dummy)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        cfg = SessionConfig()  # no tracking_uri, no MLFLOW_TRACKING_URI
        with pytest.raises(RuntimeError, match="No tracking URI"):
            with session(config=cfg, start_server=False):
                pass

    def test_public_tracking_uri_non_http_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """public_tracking_uri with non-http scheme must raise ValueError."""
        dummy = _Mlflow()
        _patch_session(monkeypatch, dummy)

        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            public_tracking_uri="ftp://ui.example.com",
        )
        with pytest.raises(ValueError, match="http"):
            with session(config=cfg, start_server=False):
                pass

    def test_start_server_port_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracking URI port != server port → ValueError."""
        dummy = _Mlflow()
        _patch_session(monkeypatch, dummy)

        cfg = SessionConfig(tracking_uri="http://127.0.0.1:9999")
        srv_cfg = ServerConfig(host="127.0.0.1", port=5000, strict_cli_compat=False)
        with pytest.raises(ValueError, match="match the spawned server"):
            with session(config=cfg, server=srv_cfg, start_server=True):
                pass

    def test_registry_uri_propagated_to_mlflow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """registry_uri must call mlflow.set_registry_uri."""
        dummy = _Mlflow()
        _patch_session(monkeypatch, dummy)

        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            registry_uri="http://registry.example.com",
        )
        with session(config=cfg, start_server=False):
            pass
        assert dummy._registry_uri == "http://registry.example.com"

    def test_type_error_client_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        If MlflowClient(tracking_uri=..., registry_uri=...) raises TypeError,
        fall back to MlflowClient(tracking_uri=...) only.
        """
        dummy = _Mlflow()
        _patch_session(monkeypatch, dummy)

        call_log: list[dict] = []

        def _make_client_strict(**kwargs):
            call_log.append(kwargs)
            if "registry_uri" in kwargs:
                raise TypeError("registry_uri not supported")
            return _Client(tracking_uri=kwargs.get("tracking_uri"))

        dummy.tracking = SimpleNamespace(MlflowClient=_make_client_strict)

        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            registry_uri="http://registry.example.com",
        )
        with session(config=cfg, start_server=False) as h:
            assert h is not None
        # First call included registry_uri, second did not
        assert any("registry_uri" in c for c in call_log)

    def test_env_restored_after_exception_in_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment must be restored even when session body raises."""
        dummy = _Mlflow()
        _patch_session(monkeypatch, dummy)
        monkeypatch.delenv("_SP_SESS_TEST", raising=False)

        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            extra_env={"_SP_SESS_TEST": "1"},
        )
        with pytest.raises(RuntimeError, match="boom"):
            with session(config=cfg, start_server=False):
                assert os.environ.get("_SP_SESS_TEST") == "1"
                raise RuntimeError("boom")

        assert os.environ.get("_SP_SESS_TEST") is None


# ===========================================================================
# MlflowHandle - proxy __getattr__ and start_run fallback
# ===========================================================================


class TestMlflowHandle:
    """Tests for MlflowHandle attribute proxy and start_run tag fallback."""

    def _make_handle(self, dummy, *, default_run_tags=None, default_run_name=None):
        from scikitplot.mlflow._facade import ArtifactsFacade, ModelsFacade
        client = _Client()
        return MlflowHandle(
            _mlflow_module=dummy,
            _tracking_uri="http://127.0.0.1:5000",
            _registry_uri=None,
            _ui_url="http://127.0.0.1:5000",
            _client=client,
            _artifacts=ArtifactsFacade(mlflow_module=dummy, client=client),
            _models=ModelsFacade(mlflow_module=dummy, client=client),
            default_run_tags=default_run_tags,
            default_run_name=default_run_name,
        )

    def test_proxy_delegates_to_mlflow_module(self) -> None:
        dummy = _Mlflow()
        dummy.custom_attr = "hello"  # type: ignore[attr-defined]
        handle = self._make_handle(dummy)
        assert handle.custom_attr == "hello"

    def test_start_run_tag_fallback_via_client_set_tag(self) -> None:
        """When set_tags is absent, fall back to client.set_tag per key."""
        # Build a dummy mlflow module without set_tags using SimpleNamespace so
        # we never attempt to delete a class-defined method (which raises AttributeError).
        import contextlib

        run_kwargs: list = []

        @contextlib.contextmanager
        def _start_run(*args, **kwargs):
            run_kwargs.append(kwargs)
            yield _Run(info=_RunInfo(run_id="run-abc"))

        dummy_no_set_tags = SimpleNamespace(
            __version__="2.14.0",
            set_tracking_uri=lambda uri: None,
            set_registry_uri=lambda uri: None,
            get_experiment_by_name=lambda name: None,
            set_experiment=lambda name: None,
            start_run=_start_run,
            tracking=SimpleNamespace(MlflowClient=lambda **kw: _Client()),
            # intentionally omits set_tags to exercise the fallback branch
        )

        handle = self._make_handle(dummy_no_set_tags, default_run_tags={"env": "test"})
        with handle.start_run():
            pass

        assert handle._client.tags.get("run-abc", {}).get("env") == "test"

    def test_start_run_uses_set_tags_when_available(self) -> None:
        dummy = _Mlflow()
        handle = self._make_handle(dummy, default_run_tags={"env": "prod"})
        with handle.start_run():
            pass
        assert dummy._set_tags_calls == [{"env": "prod"}]

    def test_default_run_name_applied(self) -> None:
        dummy = _Mlflow()
        handle = self._make_handle(dummy, default_run_name="my-run")
        with handle.start_run():
            pass
        assert dummy._start_run_kwargs[-1].get("run_name") == "my-run"

    def test_explicit_run_name_overrides_default(self) -> None:
        dummy = _Mlflow()
        handle = self._make_handle(dummy, default_run_name="default")
        with handle.start_run(run_name="explicit"):
            pass
        assert dummy._start_run_kwargs[-1].get("run_name") == "explicit"

    def test_no_tags_no_set_tags_called(self) -> None:
        dummy = _Mlflow()
        handle = self._make_handle(dummy, default_run_tags=None)
        with handle.start_run():
            pass
        assert dummy._set_tags_calls == []


# ===========================================================================
# session_from_toml
# ===========================================================================


class TestSessionFromToml:
    """Tests for session_from_toml (lines 39-43 of _session.py)."""

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

        from scikitplot.mlflow._session import session_from_toml

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

        from scikitplot.mlflow._session import session_from_toml

        with session_from_toml(toml_file, profile="staging") as h:
            assert h.tracking_uri == "http://staging:5000"

    def test_missing_profile_raises(self, tmp_path: Path) -> None:
        """Missing profile key in TOML must propagate KeyError."""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(_MINIMAL_TOML, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        from scikitplot.mlflow._session import session_from_toml

        with pytest.raises(KeyError):
            with session_from_toml(toml_file, profile="nonexistent"):
                pass


# ===========================================================================
# session_from_file
# ===========================================================================


class TestSessionFromFile:
    """Tests for session_from_file (lines 67-71 of _session.py)."""

    def test_toml_dispatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """session_from_file with .toml extension must work correctly."""
        toml_file = tmp_path / "mlflow.toml"
        toml_file.write_text(_MINIMAL_TOML, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

        dummy = _DummyMlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        from scikitplot.mlflow._session import session_from_file

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

        from scikitplot.mlflow._session import session_from_file

        with session_from_file(yaml_file, profile="local") as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported file extension (.ini) must raise ValueError."""
        bad_file = tmp_path / "config.ini"
        bad_file.write_text("[section]\nkey=val\n", encoding="utf-8")

        from scikitplot.mlflow._session import session_from_file

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

        from scikitplot.mlflow._session import session_from_file

        with session_from_file(str(toml_file), profile="local") as h:
            assert h.tracking_uri == "http://127.0.0.1:5000"


# ===========================================================================
# Gap-fill: MlflowHandle properties (registry_uri, artifacts, models) lines 221,250,262
# ===========================================================================


class TestMlflowHandleProperties:
    """Tests for MlflowHandle property accessors not yet covered."""

    def _make_handle(self, dummy):
        from scikitplot.mlflow._facade import ArtifactsFacade, ModelsFacade
        client = _Client()
        return MlflowHandle(
            _mlflow_module=dummy,
            _tracking_uri="http://127.0.0.1:5000",
            _registry_uri="http://registry:5001",
            _ui_url="http://127.0.0.1:5000",
            _client=client,
            _artifacts=ArtifactsFacade(mlflow_module=dummy, client=client),
            _models=ModelsFacade(mlflow_module=dummy, client=client),
        )

    def test_registry_uri_property_returns_value(self) -> None:
        """MlflowHandle.registry_uri (line 221) must return the stored value."""
        from scikitplot.mlflow._facade import ArtifactsFacade, ModelsFacade
        dummy = _Mlflow()
        client = _Client()
        handle = MlflowHandle(
            _mlflow_module=dummy,
            _tracking_uri="http://127.0.0.1:5000",
            _registry_uri="http://reg:9000",
            _ui_url="http://127.0.0.1:5000",
            _client=client,
            _artifacts=ArtifactsFacade(mlflow_module=dummy, client=client),
            _models=ModelsFacade(mlflow_module=dummy, client=client),
        )
        assert handle.registry_uri == "http://reg:9000"

    def test_artifacts_property_returns_facade(self) -> None:
        """MlflowHandle.artifacts (line 250) must return an ArtifactsFacade."""
        from scikitplot.mlflow._facade import ArtifactsFacade
        dummy = _Mlflow()
        handle = self._make_handle(dummy)
        assert isinstance(handle.artifacts, ArtifactsFacade)

    def test_models_property_returns_facade(self) -> None:
        """MlflowHandle.models (line 262) must return a ModelsFacade."""
        from scikitplot.mlflow._facade import ModelsFacade
        dummy = _Mlflow()
        handle = self._make_handle(dummy)
        assert isinstance(handle.models, ModelsFacade)


# ===========================================================================
# Gap-fill: session() provider client path (line 454)
# ===========================================================================


class TestSessionProviderClientPath:
    """
    Tests for the provider.get_client() path inside session() (line 454).

    When a MlflowProvider is active, session() must use provider.get_client()
    instead of calling mlflow_mod.tracking.MlflowClient directly.
    """

    def test_provider_get_client_called_when_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scikitplot.mlflow._custom import MlflowProvider, set_provider, use_provider
        import scikitplot.mlflow._session as sess_mod

        dummy = _Mlflow()
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        factory_calls: list = []

        def custom_factory(tracking_uri, registry_uri):
            factory_calls.append((tracking_uri, registry_uri))
            return _Client(tracking_uri=tracking_uri)

        provider = MlflowProvider(module=dummy, client_factory=custom_factory)
        with use_provider(provider):
            cfg = SessionConfig(tracking_uri="http://127.0.0.1:5000")
            with session(config=cfg, start_server=False) as h:
                assert h is not None

        assert len(factory_calls) == 1
        assert factory_calls[0][0] == "http://127.0.0.1:5000"

    def test_no_provider_uses_mlflow_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a provider, must use mlflow_mod.tracking.MlflowClient."""
        from scikitplot.mlflow._custom import set_provider
        set_provider(None)

        dummy = _Mlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        cfg = SessionConfig(tracking_uri="http://127.0.0.1:5000")
        with session(config=cfg, start_server=False) as h:
            assert h.client.tracking_uri == "http://127.0.0.1:5000"


# ===========================================================================
# Gap-fill: session() start_server=True provides tracking_uri from server (line 380)
# ===========================================================================


class TestSessionStartServerDerivesUri:
    """
    When start_server=True and no tracking_uri is given, session() must
    construct the URI from the server host/port (line 380).
    """

    def test_tracking_uri_derived_from_server_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scikitplot.mlflow._session as sess_mod

        dummy = _Mlflow()
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)
        monkeypatch.setattr(sess_mod, "spawn_server", lambda cfg: _FakeSpawnedServer())
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        cfg = SessionConfig()  # no tracking_uri
        srv_cfg = ServerConfig(host="127.0.0.1", port=5001, strict_cli_compat=False)
        with session(config=cfg, server=srv_cfg, start_server=True) as h:
            assert "5001" in h.tracking_uri


# ---------------------------------------------------------------------------
# Stub needed for derived-URI test
# ---------------------------------------------------------------------------


class _FakeSpawnedServer:
    """Minimal SpawnedServer stub for start_server path tests."""

    class _FakeProc:
        returncode = None
        pid = 1
        stdout = None

        def poll(self):
            return None

    process = _FakeProc()
    command = ["mlflow", "server"]

    def terminate(self):
        pass

    def read_all_output(self):
        return ""


# ===========================================================================
# Gap-fill: _parse_http_uri empty hostname (line 69)
# ===========================================================================


class TestParseHttpUriEmptyHostname:
    """Tests for _parse_http_uri empty hostname validation."""

    def test_empty_hostname_raises_value_error(self) -> None:
        """URI with empty hostname must raise ValueError (line 69)."""
        # urlparse("http://") gives empty hostname
        with pytest.raises(ValueError, match="Invalid"):
            _parse_http_uri("http://")


# ===========================================================================
# Gap-fill: session() ensure_reachable path (lines 432-442)
# ===========================================================================


class TestSessionEnsureReachable:
    """Tests for the ensure_reachable=True path in session()."""

    def test_ensure_reachable_calls_wait_tracking_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_reachable=True must invoke wait_tracking_ready."""
        dummy = _Mlflow()
        wait_called: list = []
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(
            sess_mod,
            "wait_tracking_ready",
            lambda *a, **kw: wait_called.append(a),
        )
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            ensure_reachable=True,
            startup_timeout_s=0.1,
        )
        with session(config=cfg, start_server=False):
            pass
        assert len(wait_called) == 1

    def test_ensure_reachable_non_http_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_reachable=True with a file:// URI must raise ValueError (line 434)."""
        dummy = _Mlflow()
        import scikitplot.mlflow._session as sess_mod
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)
        cfg = SessionConfig(
            tracking_uri="file:///tmp/mlruns",
            ensure_reachable=True,
            startup_timeout_s=0.1,
        )
        with pytest.raises(ValueError, match="http"):
            with session(config=cfg, start_server=False):
                pass


# ===========================================================================
# Gap-fill: session() set_experiment strict path (line 466)
# ===========================================================================


class TestSessionSetExperimentStrict:
    """Tests for the experiment_name strict=False path in session()."""

    def test_create_if_missing_false_and_experiment_exists_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When experiment exists and create_if_missing=False, no error (line 466)."""
        dummy = _Mlflow()
        dummy._experiments["existing-exp"] = object()
        _patch_session(monkeypatch, dummy)
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="existing-exp",
            create_experiment_if_missing=False,
        )
        with session(config=cfg, start_server=False) as h:
            assert h.experiment_name == "existing-exp"


# ===========================================================================
# Gap-fill: session() spawned.terminate() exception path (lines 497-498)
# ===========================================================================


class TestSessionSpawnedTerminateException:
    """
    Tests for the finally-block terminate() exception swallowing (lines 497-498).

    When spawned.terminate() raises, session() must swallow it and still
    restore the environment.
    """

    def test_terminate_exception_does_not_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scikitplot.mlflow._session as sess_mod

        dummy = _Mlflow()
        monkeypatch.setattr(sess_mod, "import_mlflow", lambda: dummy)
        monkeypatch.setattr(sess_mod, "wait_tracking_ready", lambda *a, **kw: None)

        class _RaisingServer(_FakeSpawnedServer):
            def terminate(self) -> None:
                raise OSError("terminate failed")

        monkeypatch.setattr(sess_mod, "spawn_server", lambda cfg: _RaisingServer())
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        cfg = SessionConfig()
        srv_cfg = ServerConfig(host="127.0.0.1", port=5001, strict_cli_compat=False)
        with session(config=cfg, server=srv_cfg, start_server=True) as h:
            assert h is not None
        # Must not raise; environment must be restored
