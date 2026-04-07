# scikitplot/mlflow/tests/test__compat.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._compat.

Naming convention: test__<module_name>.py

Covers
------
- import_mlflow             : returns module via active provider, returns real mlflow
                               when installed, raises MlflowNotInstalledError when absent
- resolve_download_artifacts : prefers mlflow.artifacts.download_artifacts (modern),
                               falls back to session-bound client.download_artifacts,
                               falls back to legacy MlflowClient.download_artifacts,
                               raises AttributeError when no API is available

Notes
-----
All tests are pure-Python.  MLflow is mocked throughout so the suite runs
without a real MLflow installation.  Global provider state is reset per test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from scikitplot.mlflow._compat import import_mlflow, resolve_download_artifacts
from scikitplot.mlflow._custom import MlflowProvider, set_provider
from scikitplot.mlflow._errors import MlflowNotInstalledError


# ---------------------------------------------------------------------------
# Fixture: reset provider state before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    set_provider(None)
    yield
    set_provider(None)


# ---------------------------------------------------------------------------
# Module stubs
# ---------------------------------------------------------------------------


def _modern_mod() -> Any:
    """MLflow module stub with the modern artifacts.download_artifacts API."""

    class Artifacts:
        @staticmethod
        def download_artifacts(
            run_id: str, artifact_path: str, dst_path: Optional[str] = None
        ) -> str:
            return f"/modern/{run_id}/{artifact_path}"

    mod = types.ModuleType("mlflow")
    mod.artifacts = Artifacts()  # type: ignore[attr-defined]
    return mod


def _legacy_mod() -> Any:
    """MLflow module stub without artifacts namespace; has MlflowClient."""

    class LegacyClient:
        def download_artifacts(
            self, run_id: str, path: str, dst: Optional[str] = None
        ) -> str:
            return f"/legacy/{run_id}/{path}"

    mod = types.ModuleType("mlflow")
    mod.tracking = types.SimpleNamespace(MlflowClient=LegacyClient)  # type: ignore[attr-defined]
    return mod


def _bare_mod() -> Any:
    """MLflow module stub with no download API at all."""
    return types.ModuleType("mlflow")


class _BoundClient:
    """Session-bound client stub."""

    def download_artifacts(
        self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
    ) -> str:
        return f"/client/{run_id}/{artifact_path}"


# ===========================================================================
# import_mlflow
# ===========================================================================


class TestImportMlflow:
    """Tests for import_mlflow()."""

    def test_returns_provider_module_when_set(self) -> None:
        """When a provider is active, import_mlflow must return its module."""
        fake_mod = types.ModuleType("mlflow_fake")
        fake_mod.__version__ = "9.9.9"  # type: ignore[attr-defined]
        provider = MlflowProvider(module=fake_mod)
        set_provider(provider)
        result = import_mlflow()
        assert result is fake_mod

    def test_returns_real_mlflow_when_no_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a provider, import_mlflow must return the real mlflow module."""
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.__version__ = "2.0.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
        result = import_mlflow()
        assert result is fake_mlflow

    def test_raises_when_mlflow_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a provider and without mlflow installed, must raise."""
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        # Remove mlflow from sys.modules to prevent accidental success
        monkeypatch.setitem(sys.modules, "mlflow", None)  # type: ignore[assignment]
        with pytest.raises(MlflowNotInstalledError):
            import_mlflow()

    def test_provider_takes_precedence_over_installed_mlflow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider module must be returned even if real mlflow is also importable."""
        real_mlflow = types.ModuleType("mlflow")
        real_mlflow.__version__ = "2.0.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlflow", real_mlflow)
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

        custom_mod = types.ModuleType("mlflow_custom")
        provider = MlflowProvider(module=custom_mod)
        set_provider(provider)

        result = import_mlflow()
        assert result is custom_mod
        assert result is not real_mlflow

    def test_error_message_contains_pip_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error message must guide the user to install MLflow."""
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        monkeypatch.setitem(sys.modules, "mlflow", None)  # type: ignore[assignment]
        with pytest.raises(MlflowNotInstalledError, match="pip install mlflow"):
            import_mlflow()


# ===========================================================================
# resolve_download_artifacts
# ===========================================================================


class TestResolveDownloadArtifacts:
    """Tests for resolve_download_artifacts() preference order."""

    def test_prefers_modern_artifacts_api(self) -> None:
        """Priority 1: mlflow.artifacts.download_artifacts must be used when present."""
        fn = resolve_download_artifacts(_modern_mod())
        result = fn(run_id="r1", artifact_path="model/MLmodel")
        assert result == "/modern/r1/model/MLmodel"

    def test_modern_api_ignores_provided_client(self) -> None:
        """Even if a client is given, the modern API takes precedence."""
        client = _BoundClient()
        fn = resolve_download_artifacts(_modern_mod(), client=client)
        result = fn(run_id="r2", artifact_path="metrics.json")
        assert result.startswith("/modern/")

    def test_falls_back_to_provided_client(self) -> None:
        """Priority 2: session-bound client.download_artifacts when modern API absent."""
        fn = resolve_download_artifacts(_bare_mod(), client=_BoundClient())
        result = fn(run_id="r", artifact_path="p")
        assert result == "/client/r/p"

    def test_client_fallback_uses_keyword_args(self) -> None:
        """The client fallback wrapper must honour run_id= and artifact_path= kwargs."""
        received: dict = {}

        class C:
            def download_artifacts(
                self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
            ) -> str:
                received.update({"run_id": run_id, "artifact_path": artifact_path})
                return "/c/x"

        fn = resolve_download_artifacts(_bare_mod(), client=C())
        fn(run_id="abc", artifact_path="xyz")
        assert received == {"run_id": "abc", "artifact_path": "xyz"}

    def test_falls_back_to_legacy_mlflow_client(self) -> None:
        """Priority 3: legacy MlflowClient.download_artifacts when no modern API or client."""
        fn = resolve_download_artifacts(_legacy_mod())
        result = fn(run_id="r", artifact_path="a/b")
        assert result == "/legacy/r/a/b"

    def test_raises_when_no_api_available(self) -> None:
        """Must raise AttributeError when no download API can be resolved."""
        with pytest.raises(AttributeError, match="No supported artifact download API"):
            resolve_download_artifacts(_bare_mod(), client=None)

    def test_error_message_names_both_apis(self) -> None:
        """Error message must mention both modern and legacy API names."""
        with pytest.raises(AttributeError, match="download_artifacts"):
            resolve_download_artifacts(_bare_mod(), client=None)

    def test_returned_callable_is_callable(self) -> None:
        """resolve_download_artifacts must return a callable in all success paths."""
        assert callable(resolve_download_artifacts(_modern_mod()))
        assert callable(resolve_download_artifacts(_bare_mod(), client=_BoundClient()))
        assert callable(resolve_download_artifacts(_legacy_mod()))

    def test_dst_path_forwarded_by_client_wrapper(self) -> None:
        """Optional dst_path kwarg must be forwarded to the client."""
        received: dict = {}

        class C:
            def download_artifacts(
                self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
            ) -> str:
                received["dst_path"] = dst_path
                return "/x"

        fn = resolve_download_artifacts(_bare_mod(), client=C())
        fn(run_id="r", artifact_path="p", dst_path="/tmp/out")
        assert received["dst_path"] == "/tmp/out"

    def test_dst_path_defaults_to_none_in_client_wrapper(self) -> None:
        received: dict = {}

        class C:
            def download_artifacts(
                self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
            ) -> str:
                received["dst_path"] = dst_path
                return "/x"

        fn = resolve_download_artifacts(_bare_mod(), client=C())
        fn(run_id="r", artifact_path="p")
        assert received["dst_path"] is None


# ===========================================================================
# Module structure
# ===========================================================================


class TestModuleStructure:
    """Tests for public API surface of _compat."""

    def test_all_exports_present(self) -> None:
        import scikitplot.mlflow._compat as m
        assert set(m.__all__) == {"import_mlflow", "resolve_download_artifacts"}
