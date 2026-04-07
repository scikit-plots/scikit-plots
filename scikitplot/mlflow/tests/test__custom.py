# scikitplot/mlflow/tests/test__custom.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._custom.

Naming convention: test__<module_name>.py

Covers
------
- MlflowProvider dataclass  : field defaults, get_client standard path,
                               get_client factory override, get_client TypeError fallback,
                               get_artifact_downloader with custom callable,
                               get_artifact_downloader falls through to _compat
- get_provider              : returns None by default, returns active provider
- set_provider              : sets provider, accepts None to clear
- use_provider              : context manager restores previous on normal exit,
                               context manager restores on exception,
                               nested use_provider scopes are independent

Notes
-----
All MLflow module interactions use lightweight anonymous stubs.
The global provider state is always reset in a fixture to prevent cross-test pollution.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from scikitplot.mlflow._custom import (
    MlflowProvider,
    get_provider,
    set_provider,
    use_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    """Reset the global provider to None before and after every test."""
    set_provider(None)
    yield
    set_provider(None)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_mlflow_stub(tracking_uri_recorded: list | None = None) -> Any:
    """Return a minimal MLflow-like module stub."""

    class FakeClient:
        def __init__(
            self,
            tracking_uri: str | None = None,
            registry_uri: str | None = None,
        ) -> None:
            self.tracking_uri = tracking_uri
            self.registry_uri = registry_uri

    class FakeTracking:
        MlflowClient = FakeClient

    stub = SimpleNamespace(tracking=FakeTracking())
    return stub


# ===========================================================================
# MlflowProvider dataclass
# ===========================================================================


class TestMlflowProviderDefaults:
    """Tests for MlflowProvider field defaults."""

    def test_version_default_is_none(self) -> None:
        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub)
        assert p.version is None

    def test_client_factory_default_is_none(self) -> None:
        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub)
        assert p.client_factory is None

    def test_artifact_downloader_default_is_none(self) -> None:
        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub)
        assert p.artifact_downloader is None

    def test_module_stored_correctly(self) -> None:
        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub)
        assert p.module is stub


class TestMlflowProviderGetClient:
    """Tests for MlflowProvider.get_client()."""

    def test_standard_path_uses_tracking_mlflow_client(self) -> None:
        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub)
        client = p.get_client("http://127.0.0.1:5000")
        assert client.tracking_uri == "http://127.0.0.1:5000"

    def test_registry_uri_forwarded(self) -> None:
        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub)
        client = p.get_client("http://127.0.0.1:5000", "http://registry:5001")
        assert client.registry_uri == "http://registry:5001"

    def test_client_factory_override_is_used(self) -> None:
        """When client_factory is set, it takes precedence over module.tracking.MlflowClient."""
        called_with: list = []

        def factory(tracking_uri: str, registry_uri: str | None) -> object:
            called_with.append((tracking_uri, registry_uri))
            return object()

        stub = _make_mlflow_stub()
        p = MlflowProvider(module=stub, client_factory=factory)
        p.get_client("http://x:5000", None)
        assert called_with == [("http://x:5000", None)]

    def test_type_error_fallback_omits_registry_uri(self) -> None:
        """
        When MlflowClient does not accept registry_uri (older MLflow), get_client
        must silently retry with tracking_uri only.
        """

        class OldClient:
            def __init__(self, tracking_uri: str | None = None) -> None:
                self.tracking_uri = tracking_uri

        class OldTracking:
            MlflowClient = OldClient

        old_stub = SimpleNamespace(tracking=OldTracking())
        p = MlflowProvider(module=old_stub)
        client = p.get_client("http://old:5000", "http://reg")
        assert client.tracking_uri == "http://old:5000"


class TestMlflowProviderGetArtifactDownloader:
    """Tests for MlflowProvider.get_artifact_downloader()."""

    def test_custom_artifact_downloader_returned_directly(self) -> None:
        """When artifact_downloader is set, it must be returned without calling _compat."""

        def my_downloader(*, run_id: str, artifact_path: str, **_: Any) -> str:
            return f"/custom/{run_id}/{artifact_path}"

        p = MlflowProvider(module=_make_mlflow_stub(), artifact_downloader=my_downloader)
        fn = p.get_artifact_downloader(client=None)
        assert fn is my_downloader

    def test_falls_through_to_compat_when_no_custom_downloader(self) -> None:
        """Without a custom downloader, must call resolve_download_artifacts."""

        class ModernMod:
            class artifacts:
                @staticmethod
                def download_artifacts(
                    run_id: str, artifact_path: str, dst_path: str | None = None
                ) -> str:
                    return f"/modern/{run_id}/{artifact_path}"

        p = MlflowProvider(module=ModernMod())
        fn = p.get_artifact_downloader(client=None)
        result = fn(run_id="r1", artifact_path="model")
        assert result == "/modern/r1/model"


# ===========================================================================
# get_provider / set_provider
# ===========================================================================


class TestGetSetProvider:
    """Tests for get_provider() and set_provider()."""

    def test_default_is_none(self) -> None:
        assert get_provider() is None

    def test_set_provider_makes_it_retrievable(self) -> None:
        p = MlflowProvider(module=_make_mlflow_stub())
        set_provider(p)
        assert get_provider() is p

    def test_set_provider_none_clears_active(self) -> None:
        p = MlflowProvider(module=_make_mlflow_stub())
        set_provider(p)
        set_provider(None)
        assert get_provider() is None

    def test_successive_set_provider_overwrites(self) -> None:
        p1 = MlflowProvider(module=_make_mlflow_stub(), version="1.0.0")
        p2 = MlflowProvider(module=_make_mlflow_stub(), version="2.0.0")
        set_provider(p1)
        set_provider(p2)
        assert get_provider() is p2

    def test_get_provider_returns_same_object(self) -> None:
        """get_provider() must return the exact object that was set."""
        p = MlflowProvider(module=_make_mlflow_stub())
        set_provider(p)
        assert get_provider() is p


# ===========================================================================
# use_provider context manager
# ===========================================================================


class TestUseProvider:
    """Tests for use_provider() context manager."""

    def test_activates_provider_inside_block(self) -> None:
        p = MlflowProvider(module=_make_mlflow_stub(), version="3.0.0")
        with use_provider(p):
            assert get_provider() is p

    def test_restores_none_after_block(self) -> None:
        p = MlflowProvider(module=_make_mlflow_stub())
        with use_provider(p):
            pass
        assert get_provider() is None

    def test_restores_previous_provider_after_block(self) -> None:
        p_outer = MlflowProvider(module=_make_mlflow_stub(), version="1.0.0")
        p_inner = MlflowProvider(module=_make_mlflow_stub(), version="2.0.0")
        set_provider(p_outer)
        with use_provider(p_inner):
            assert get_provider() is p_inner
        assert get_provider() is p_outer

    def test_restores_on_exception(self) -> None:
        p = MlflowProvider(module=_make_mlflow_stub())
        try:
            with use_provider(p):
                raise ValueError("oops")
        except ValueError:
            pass
        assert get_provider() is None

    def test_none_disables_provider_inside_block(self) -> None:
        p_outer = MlflowProvider(module=_make_mlflow_stub())
        set_provider(p_outer)
        with use_provider(None):
            assert get_provider() is None
        assert get_provider() is p_outer

    def test_nested_use_provider_is_independent(self) -> None:
        p1 = MlflowProvider(module=_make_mlflow_stub(), version="1.0.0")
        p2 = MlflowProvider(module=_make_mlflow_stub(), version="2.0.0")
        with use_provider(p1):
            assert get_provider() is p1
            with use_provider(p2):
                assert get_provider() is p2
            assert get_provider() is p1
        assert get_provider() is None

    def test_yields_none(self) -> None:
        """use_provider is a contextmanager that yields None (not the provider)."""
        p = MlflowProvider(module=_make_mlflow_stub())
        with use_provider(p) as result:
            assert result is None


# ===========================================================================
# Module structure
# ===========================================================================


class TestModuleStructure:
    """Tests for public API surface of _custom."""

    def test_all_exports_present(self) -> None:
        import scikitplot.mlflow._custom as m
        expected = {"MlflowProvider", "get_provider", "set_provider", "use_provider"}
        assert expected == set(m.__all__)

    def test_mlflow_provider_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(MlflowProvider)
