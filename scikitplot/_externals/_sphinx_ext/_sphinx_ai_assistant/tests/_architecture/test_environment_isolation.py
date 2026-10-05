"""
The services under test never see the environment the session started in.

Notes
-----
**Developer notes.** The proxy and model applications take their
configuration from environment variables and read most of it when they are
imported, which under pytest is during collection. ``conftest.py`` removes
every variable the services read by name before any of them can be imported,
and puts the values back when the session ends. These tests pin that: the
names come from the service sources, nothing ambient reaches a service, and
the process gets its values back.

The failure this prevents was seen once: a push to ``main`` ran the suite
with the repository's real ``HF_TOKEN`` exported, and the Hugging Face
executor reported itself ``enabled`` where a pull request from a fork, which
has no secrets, saw ``plan-only``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Imported here, while pytest collects, as every other test module does: the
# service configures logging when it is imported, and conftest.py handles
# that at the end of collection.
from ..._hf_spaces_proxy import app
from .. import conftest
from .._paths import RUNTIME_ROOT

#: Deployment values that must never be inherited; each is read at import.
CREDENTIALS = ("HF_TOKEN", "HF_WRITE_TOKEN", "BACKEND_AUTH_TOKEN", "SHARE_WRITE_TOKEN")


def test_the_scan_reads_every_literal_form_and_nothing_else():
    source = textwrap.dedent(
        """
        import os
        A = os.environ.get("ALPHA", "")
        B = os.environ["BRAVO"]
        C = os.getenv("CHARLIE")
        os.environ.setdefault("DELTA", "1")
        os.environ.pop("ECHO", None)
        def later(name, settings, environ):
            return os.environ.get(name), settings.get("NOT_ENV"), environ.get("NOR_THIS")
        TEXT = 'os.environ.get("IN_A_STRING")'
        """
    )
    assert conftest._environment_names_read_by(source) == {
        "ALPHA",
        "BRAVO",
        "CHARLIE",
        "DELTA",
        "ECHO",
    }


def test_the_names_come_from_the_service_sources():
    names = conftest.SERVICE_ENVIRONMENT
    assert set(CREDENTIALS) <= names
    assert {"BACKEND_URL", "ALLOWED_MODELS", "STUB_ENABLED", "DEPLOYMENT_PROFILE"} <= names
    # Nothing a test process needs for itself is among them.
    assert not names & {"PATH", "HOME", "PYTHONPATH", "TMPDIR", "USER", "LANG", "CI"}
    assert names == conftest._service_environment_names(RUNTIME_ROOT)


def test_a_scan_that_finds_nothing_is_an_error_not_an_empty_list(tmp_path):
    service = tmp_path / conftest._SERVICE_PACKAGES[0]
    service.mkdir()
    (service / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="can no longer isolate"):
        conftest._service_environment_names(tmp_path)
    assert conftest._service_environment_names(tmp_path / "missing") == frozenset()


def test_withholding_removes_the_values_and_returns_them(monkeypatch):
    monkeypatch.setenv("SKPLT_ISOLATION_PROBE_A", "one")
    monkeypatch.setenv("SKPLT_ISOLATION_PROBE_B", "two")
    monkeypatch.setattr(conftest, "_SERVICE_PACKAGES", ("_no_such_service_package",))
    held = conftest._withhold_ambient_service_environment(
        frozenset({"SKPLT_ISOLATION_PROBE_A", "SKPLT_ISOLATION_PROBE_ABSENT"})
    )
    assert held == {"SKPLT_ISOLATION_PROBE_A": "one"}
    assert "SKPLT_ISOLATION_PROBE_A" not in os.environ
    assert os.environ["SKPLT_ISOLATION_PROBE_B"] == "two"


def test_a_service_imported_too_early_is_refused(monkeypatch):
    # The proxy is imported (at the top of this file). Had a variable still
    # been set when conftest.py loaded, the service would hold its value.
    monkeypatch.setenv("SKPLT_ISOLATION_PROBE_A", "one")
    with pytest.raises(RuntimeError, match="was imported before the test environment was isolated"):
        conftest._withhold_ambient_service_environment(frozenset({"SKPLT_ISOLATION_PROBE_A"}))
    assert os.environ["SKPLT_ISOLATION_PROBE_A"] == "one"
    # With nothing ambient there is nothing to refuse.
    monkeypatch.delenv("SKPLT_ISOLATION_PROBE_A")
    assert conftest._withhold_ambient_service_environment(frozenset({"SKPLT_ISOLATION_PROBE_A"})) == {}


def test_the_proxy_under_test_holds_no_inherited_credential():
    for name in CREDENTIALS:
        assert getattr(app, name) == "", name
    assert app.BACKEND_URL == ""
    assert app._huggingface_resource_executor_configured() is False


def test_values_exported_before_the_session_never_reach_the_service():
    """The whole path, in a fresh interpreter with a polluted environment."""
    package = conftest.__name__.rsplit(".", 2)[0]
    # The directory the package is imported from, read off its dotted name so
    # that the child resolves the same package wherever this checkout lives.
    parts = tuple(package.split("."))
    assert RUNTIME_ROOT.parts[-len(parts) :] == parts
    import_root = str(Path(*RUNTIME_ROOT.parts[: -len(parts)]))
    program = textwrap.dedent(
        f"""
        import importlib, json, os
        conftest = importlib.import_module("{package}.tests.conftest")
        seen = {{name: name in os.environ for name in ("HF_TOKEN", "BACKEND_URL", "UNRELATED")}}
        app = importlib.import_module("{package}._hf_spaces_proxy.app")
        result = {{
            "withheld": sorted(conftest._AMBIENT_SERVICE_ENVIRONMENT),
            "seen": seen,
            "token": app.HF_TOKEN,
            "backend": app.BACKEND_URL,
            "executor": app._huggingface_resource_executor_configured(),
        }}
        conftest.pytest_unconfigure(None)
        result["restored"] = [os.environ.get("HF_TOKEN"), os.environ.get("BACKEND_URL")]
        print("RESULT " + json.dumps(result))
        """
    )
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env.update(
        HF_TOKEN="hf_exported_before_the_session",
        BACKEND_URL="https://api.example.invalid/v1/chat/completions",
        UNRELATED="kept",
        PYTHONPATH=import_root + (os.pathsep + inherited if inherited else ""),
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    line = [row for row in done.stdout.splitlines() if row.startswith("RESULT ")][-1]
    result = json.loads(line[len("RESULT ") :])
    assert result["withheld"] == ["BACKEND_URL", "HF_TOKEN"]
    assert result["seen"] == {"HF_TOKEN": False, "BACKEND_URL": False, "UNRELATED": True}
    assert result["token"] == "" and result["backend"] == ""
    assert result["executor"] is False
    assert result["restored"] == [
        "hf_exported_before_the_session",
        "https://api.example.invalid/v1/chat/completions",
    ]


def test_a_value_set_during_the_session_is_not_overwritten_at_the_end(monkeypatch):
    # A stand-in for what was withheld: the real values must stay withheld
    # for the tests that run after this one.
    stand_in = {"SKPLT_ISOLATION_PROBE_A": "ambient", "SKPLT_ISOLATION_PROBE_B": "ambient"}
    monkeypatch.setattr(conftest, "_AMBIENT_SERVICE_ENVIRONMENT", stand_in)
    monkeypatch.setenv("SKPLT_ISOLATION_PROBE_A", "set by a test")
    monkeypatch.setenv("SKPLT_ISOLATION_PROBE_B", "to be removed")
    monkeypatch.delenv("SKPLT_ISOLATION_PROBE_B")
    conftest.pytest_unconfigure(None)
    assert os.environ["SKPLT_ISOLATION_PROBE_A"] == "set by a test"
    assert os.environ["SKPLT_ISOLATION_PROBE_B"] == "ambient"
    assert stand_in == {}
