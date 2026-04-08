# scikitplot/mlflow/tests/test__facade.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._facade.

Naming convention: test__<module_name>.py

Covers
------
- ArtifactsFacade.download  : prefers modern mlflow.artifacts API, falls back to
                               session-bound client, raises AttributeError with no API,
                               always returns pathlib.Path
- ArtifactsFacade.list      : no subpath calls list_artifacts(run_id),
                               with subpath forwards artifact_path kwarg
- ArtifactsFacade.log_file  : without artifact_path, with artifact_path forwarded
- ArtifactsFacade.log_files : iterates over sequence, empty list is no-op
- ModelsFacade.load_model   : pyfunc default, explicit flavor, unknown flavor raises
- ModelsFacade.register_model : delegates to mlflow.register_model, returns result

Notes
-----
All MLflow interactions use anonymous stubs.
No external dependencies beyond stdlib and the package itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pytest

from scikitplot.mlflow._facade import ArtifactsFacade, ModelsFacade


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ModernMod:
    """MLflow-like module with modern artifacts.download_artifacts API."""

    class artifacts:
        @staticmethod
        def download_artifacts(
            run_id: str, artifact_path: str, dst_path: Optional[str] = None
        ) -> str:
            return f"/modern/{run_id}/{artifact_path}"


class _LegacyMod:
    """MLflow-like module without artifacts namespace; has MlflowClient only."""
    pass


class _DummyClient:
    """Minimal MlflowClient stub with download and list capabilities."""

    def __init__(self) -> None:
        self._listed: List[Any] = []
        self._downloaded: List[tuple] = []

    def download_artifacts(
        self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
    ) -> str:
        self._downloaded.append((run_id, artifact_path, dst_path))
        return f"/legacy/{run_id}/{artifact_path}"

    def list_artifacts(self, run_id: str, path: Optional[str] = None) -> List[str]:
        self._listed.append(path)
        return ["f1", "f2"] if path is None else [f"{path}/x"]


# ===========================================================================
# ArtifactsFacade.download
# ===========================================================================


class TestArtifactsFacadeDownload:
    """Tests for ArtifactsFacade.download()."""

    def test_prefers_modern_api(self) -> None:
        """When mlflow.artifacts.download_artifacts is present it must be used."""
        f = ArtifactsFacade(mlflow_module=_ModernMod(), client=_DummyClient())
        result = f.download("r1", "model/MLmodel")
        assert result == Path("/modern/r1/model/MLmodel")

    def test_falls_back_to_client(self) -> None:
        """Without modern API, must fall back to session-bound client."""
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=_DummyClient())
        result = f.download("r1", "some/path")
        assert result == Path("/legacy/r1/some/path")

    def test_raises_when_no_api_available(self) -> None:
        """With no API on either module or client, must raise AttributeError."""

        class NoApi:
            pass

        with pytest.raises(AttributeError):
            ArtifactsFacade(mlflow_module=_LegacyMod(), client=NoApi()).download(
                "r", "p"
            )

    def test_returns_path_object(self) -> None:
        """Return type must always be pathlib.Path, not a plain string."""
        f = ArtifactsFacade(mlflow_module=_ModernMod(), client=_DummyClient())
        result = f.download("r", "a")
        assert isinstance(result, Path)

    def test_client_fallback_returns_path(self) -> None:
        client = _DummyClient()
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=client)
        result = f.download("r2", "metrics.json")
        assert isinstance(result, Path)

    def test_dst_path_forwarded_to_modern_api(self) -> None:
        """Optional dst_path must be forwarded to the download callable."""
        received: dict = {}

        class M:
            class artifacts:
                @staticmethod
                def download_artifacts(
                    run_id: str, artifact_path: str, dst_path: Optional[str] = None
                ) -> str:
                    received["dst_path"] = dst_path
                    return "/x"

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.download("r", "p", dst_path="/tmp/out")
        assert received["dst_path"] == "/tmp/out"

    def test_modern_api_takes_precedence_over_client(self) -> None:
        """Even when a client with download_artifacts exists, modern API wins."""
        client = _DummyClient()
        f = ArtifactsFacade(mlflow_module=_ModernMod(), client=client)
        result = f.download("r", "a")
        # Modern path starts with /modern/
        assert str(result).startswith("/modern/")
        # Client must NOT have been called
        assert len(client._downloaded) == 0


# ===========================================================================
# ArtifactsFacade.list
# ===========================================================================


class TestArtifactsFacadeList:
    """Tests for ArtifactsFacade.list()."""

    def test_no_subpath_lists_all(self) -> None:
        client = _DummyClient()
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=client)
        result = f.list("run-abc")
        assert result == ["f1", "f2"]
        assert client._listed == [None]

    def test_with_subpath_forwards_path(self) -> None:
        client = _DummyClient()
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=client)
        result = f.list("run-abc", artifact_path="plots")
        # Client receives "plots" as path
        assert "plots" in client._listed
        assert result == ["plots/x"]

    def test_returns_list(self) -> None:
        client = _DummyClient()
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=client)
        result = f.list("r")
        assert isinstance(result, list)


# ===========================================================================
# ArtifactsFacade.log_file
# ===========================================================================


class TestArtifactsFacadeLogFile:
    """Tests for ArtifactsFacade.log_file()."""

    def test_without_artifact_path(self) -> None:
        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append((p, artifact_path))

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_file("/tmp/f.txt")
        assert logged == [("/tmp/f.txt", None)]

    def test_with_artifact_path(self) -> None:
        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append((p, artifact_path))

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_file("/tmp/f.txt", artifact_path="plots")
        assert logged[0] == ("/tmp/f.txt", "plots")

    def test_accepts_path_object(self) -> None:
        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append(p)

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_file(Path("/tmp/myfile.csv"))
        assert logged == ["/tmp/myfile.csv"]


# ===========================================================================
# ArtifactsFacade.log_files
# ===========================================================================


class TestArtifactsFacadeLogFiles:
    """Tests for ArtifactsFacade.log_files()."""

    def test_logs_multiple_files(self) -> None:
        logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                logged.append(p)

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_files(["/tmp/a.txt", Path("/tmp/b.txt")])
        assert len(logged) == 2
        assert "/tmp/a.txt" in logged

    def test_empty_list_is_noop(self) -> None:
        """log_files([]) must not raise."""
        f = ArtifactsFacade(mlflow_module=_LegacyMod(), client=None)  # type: ignore[arg-type]
        f.log_files([])  # must not raise

    def test_artifact_path_forwarded_to_each_file(self) -> None:
        paths_logged: list = []

        class M:
            @staticmethod
            def log_artifact(p: str, artifact_path: Optional[str] = None) -> None:
                paths_logged.append(artifact_path)

        f = ArtifactsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        f.log_files(["/a", "/b"], artifact_path="outputs")
        assert all(p == "outputs" for p in paths_logged)
        assert len(paths_logged) == 2


# ===========================================================================
# ModelsFacade
# ===========================================================================


class TestModelsFacade:
    """Tests for ModelsFacade methods."""

    def test_load_pyfunc_default(self) -> None:
        class M:
            class pyfunc:
                @staticmethod
                def load_model(uri: str) -> str:
                    return "pyfunc-model"

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        assert f.load_model("runs:/r/m") == "pyfunc-model"

    def test_load_with_explicit_flavor(self) -> None:
        class M:
            class sklearn:
                @staticmethod
                def load_model(uri: str) -> str:
                    return "sklearn-model"

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        assert f.load_model("runs:/r/m", flavor="sklearn") == "sklearn-model"

    def test_load_unknown_flavor_raises_attribute_error(self) -> None:
        f = ModelsFacade(mlflow_module=_LegacyMod(), client=None)  # type: ignore[arg-type]
        with pytest.raises(AttributeError, match="flavor"):
            f.load_model("runs:/r/m", flavor="no_such_flavor")

    def test_load_flavor_without_load_model_raises(self) -> None:
        """A flavor attribute that lacks load_model must raise AttributeError."""

        class M:
            class bad_flavor:
                pass  # no load_model

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        with pytest.raises(AttributeError):
            f.load_model("runs:/r/m", flavor="bad_flavor")

    def test_register_model_delegates_correctly(self) -> None:
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

    def test_register_model_returns_result(self) -> None:
        class M:
            @staticmethod
            def register_model(uri: str, name: str) -> dict:
                return {"name": name, "version": "2"}

        f = ModelsFacade(mlflow_module=M(), client=None)  # type: ignore[arg-type]
        result = f.register_model("runs:/r/m", "Model")
        assert result["version"] == "2"


# ===========================================================================
# Module structure
# ===========================================================================


class TestModuleStructure:
    """Tests for public API surface of _facade."""

    def test_all_exports_present(self) -> None:
        import scikitplot.mlflow._facade as m
        assert set(m.__all__) == {"ArtifactsFacade", "ModelsFacade"}

    def test_artifacts_facade_is_frozen_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(ArtifactsFacade)
        assert ArtifactsFacade.__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_models_facade_is_frozen_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(ModelsFacade)
        assert ModelsFacade.__dataclass_params__.frozen  # type: ignore[attr-defined]


# ===========================================================================
# Backward-compatible module-level tests retained from original test__facade.py
# ===========================================================================


def test_artifacts_download_uses_client_fallback() -> None:
    class MNoArtifacts:
        pass

    class C:
        def download_artifacts(
            self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
        ) -> str:
            return "/tmp/dl"

    f = ArtifactsFacade(mlflow_module=MNoArtifacts(), client=C())
    assert f.download("r", "a") == Path("/tmp/dl")


def test_artifacts_download_prefers_modern_api() -> None:
    class M:
        class artifacts:
            @staticmethod
            def download_artifacts(
                run_id: str, artifact_path: str, dst_path: Optional[str] = None
            ) -> str:
                return "/tmp/new"

    class C:
        def download_artifacts(
            self, run_id: str, artifact_path: str, dst_path: Optional[str] = None
        ) -> str:
            return "/tmp/old"

    f = ArtifactsFacade(mlflow_module=M(), client=C())
    assert f.download("r", "a") == Path("/tmp/new")


# ===========================================================================
# Gap-fill: ArtifactsFacade.download via active MlflowProvider (line 109)
# ===========================================================================


class TestArtifactsFacadeProviderPath:
    """
    Tests for ArtifactsFacade.download() when a MlflowProvider is active (line 109).

    When get_provider() returns a non-None provider, download() must use
    provider.get_artifact_downloader() rather than resolve_download_artifacts.
    """

    def test_download_uses_provider_downloader(self) -> None:
        from scikitplot.mlflow._custom import MlflowProvider, use_provider

        downloaded: list = []

        def custom_downloader(*, run_id: str, artifact_path: str, dst_path=None) -> str:
            downloaded.append((run_id, artifact_path))
            return f"/provider/{run_id}/{artifact_path}"

        stub_mod = type("M", (), {})()
        provider = MlflowProvider(
            module=stub_mod,
            artifact_downloader=custom_downloader,
        )

        with use_provider(provider):
            f = ArtifactsFacade(mlflow_module=stub_mod, client=None)  # type: ignore[arg-type]
            result = f.download("r1", "model/MLmodel")

        assert result == Path("/provider/r1/model/MLmodel")
        assert downloaded == [("r1", "model/MLmodel")]

    def test_download_uses_compat_when_no_provider(self) -> None:
        """Without an active provider, must fall through to resolve_download_artifacts."""
        from scikitplot.mlflow._custom import set_provider
        set_provider(None)

        f = ArtifactsFacade(mlflow_module=_ModernMod(), client=_DummyClient())
        result = f.download("r", "path")
        assert str(result).startswith("/modern/")
