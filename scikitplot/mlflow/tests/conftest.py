from __future__ import annotations

"""Tests for conftest.py."""

import sys
from pathlib import Path as _Path

# Ensure repository root is importable regardless of how pytest is invoked.
# _ROOT = _Path(__file__).resolve().parents[1]
# if str(_ROOT) not in sys.path:
#     sys.path.insert(0, str(_ROOT))

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

import pytest


@dataclass
class DummyRunInfo:
    run_id: str


@dataclass
class DummyRun:
    info: DummyRunInfo


class DummyClient:
    def __init__(self) -> None:
        self.tags_set: Dict[str, Dict[str, str]] = {}

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags_set.setdefault(run_id, {})[key] = value

    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: Optional[str] = None) -> str:
        return (dst_path or "/tmp") + f"/{run_id}/{artifact_path}"


class DummyMlflowModule:
    """
    Minimal MLflow module stub for session tests.
    """

    def __init__(self) -> None:
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

    def _make_client(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None) -> DummyClient:
        c = DummyClient()
        c.tracking_uri = tracking_uri
        c.registry_uri = registry_uri
        return c

    def get_experiment_by_name(self, name: str) -> Optional[object]:
        return self._experiments.get(name)

    def set_experiment(self, name: str) -> None:
        self._experiment_name = name
        self._experiments.setdefault(name, object())

    @contextlib.contextmanager
    def start_run(self, *args: Any, **kwargs: Any) -> Iterator[DummyRun]:
        self._start_run_calls.append({"args": args, "kwargs": dict(kwargs)})
        run = DummyRun(info=DummyRunInfo(run_id="run-123"))
        yield run

    def set_tags(self, tags: dict[str, str]) -> None:
        self._set_tags_calls.append(dict(tags))


@pytest.fixture()
def dummy_mlflow() -> DummyMlflowModule:
    return DummyMlflowModule()
