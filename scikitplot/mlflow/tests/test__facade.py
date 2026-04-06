from __future__ import annotations

"""Tests for test_facade.py."""

from pathlib import Path

from scikitplot.mlflow._facade import ArtifactsFacade


class MNoArtifacts:
    pass


def test_artifacts_download_uses_client_fallback() -> None:
    m = MNoArtifacts()
    class C:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            return "/tmp/dl"
    f = ArtifactsFacade(mlflow_module=m, client=C())
    assert f.download("r", "a") == Path("/tmp/dl")


def test_artifacts_download_prefers_modern_api() -> None:
    class M:
        class artifacts:
            @staticmethod
            def download_artifacts(run_id, artifact_path, dst_path=None):
                return "/tmp/new"
    class C:
        def download_artifacts(self, run_id, artifact_path, dst_path=None):
            return "/tmp/old"
    f = ArtifactsFacade(mlflow_module=M(), client=C())
    assert f.download("r", "a") == Path("/tmp/new")
