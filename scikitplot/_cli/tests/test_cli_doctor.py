"""Doctor reports optional capabilities as explicit, separate read/write entries."""
import io
import json
import sys


def test_doctor_reports_readwrite_capabilities(monkeypatch):
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(["doctor", "--format", "json"])
    assert code == 0
    caps = json.loads(buf.getvalue())["capabilities"]
    # read and write are reported separately for serialization formats
    assert set(caps) == {
        "click", "rich", "yaml_read", "yaml_write", "toml_read", "toml_write",
    }
    for name, entry in caps.items():
        assert set(entry) == {"available", "provider"}, name
        assert isinstance(entry["available"], bool)
        if entry["available"]:
            assert isinstance(entry["provider"], str)
        else:
            assert entry["provider"] is None


def test_toml_read_available_via_stdlib(monkeypatch):
    # tomllib ships in the stdlib on Python >= 3.11; when present it satisfies
    # toml_read even if no toml *writer* is installed.
    import sys
    if sys.version_info < (3, 11):
        return
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    _argparse.run(["doctor", "--format", "json"])
    caps = json.loads(buf.getvalue())["capabilities"]
    assert caps["toml_read"]["available"] is True
    assert caps["toml_read"]["provider"] == "tomllib"


def test_env_collection_matches_multiple_prefixes(monkeypatch):
    import io
    import sys
    monkeypatch.setenv("SKPLT_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("SCIKITPLOT_CLI_FRONTEND", "argparse")
    monkeypatch.setenv("UNRELATED_VAR", "ignore-me")
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    _argparse.run(["doctor", "--mask-envs", "--format", "json"])
    envs = json.loads(buf.getvalue())["environment"]
    assert "SKPLT_LOGGING_LEVEL" in envs
    assert "SCIKITPLOT_CLI_FRONTEND" in envs   # second prefix now captured
    assert "UNRELATED_VAR" not in envs
    assert envs["SKPLT_LOGGING_LEVEL"] == "***"       # masked
    assert envs["SCIKITPLOT_CLI_FRONTEND"] == "***"


# ---------------------------------------------------------------------------
# Installation report (full and partial distributions)
# ---------------------------------------------------------------------------


def _doctor_json(monkeypatch):
    from .._frontends import _argparse

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(["doctor", "--format", "json"])
    return code, json.loads(buf.getvalue())


def test_doctor_reports_the_installation(monkeypatch):
    code, data = _doctor_json(monkeypatch)
    assert code == 0
    installation = data["installation"]
    assert set(installation) == {
        "flavor", "installed", "available", "core_api", "problems", "notes",
    }
    assert installation["flavor"] in {"full", "partial", "source"}
    assert isinstance(installation["installed"], dict)
    assert isinstance(installation["available"], dict)
    assert isinstance(installation["core_api"], int)
    assert isinstance(installation["problems"], list)
    assert isinstance(installation["notes"], list)


def test_doctor_status_follows_the_installation_report(monkeypatch):
    from ... import _distributions

    coherent = {"flavor": "partial", "installed": {"scikit-plots-skinny": "1"},
                "available": {}, "core_api": 1, "problems": [], "notes": []}
    monkeypatch.setattr(_distributions, "report", lambda: dict(coherent))
    code, data = _doctor_json(monkeypatch)
    assert (code, data["status"]) == (0, "ok")

    broken = dict(coherent, problems=["scikit-plots is installed together with ..."])
    monkeypatch.setattr(_distributions, "report", lambda: dict(broken))
    code, data = _doctor_json(monkeypatch)
    # The command reported what it found, so it still exits 0; the finding is
    # in the document.
    assert (code, data["status"]) == (0, "problems")
    assert data["installation"]["problems"] == broken["problems"]


def test_doctor_status_is_ok_when_there_are_only_notes(monkeypatch):
    """A note needs no action: a compatible mix of versions is not a problem."""
    from ... import _distributions

    noted = {"flavor": "partial", "installed": {"scikit-plots-skinny": "2"},
             "available": {}, "core_api": 1, "problems": [],
             "notes": ["Mixed versions: scikit-plots-mcp 1 with scikit-plots-skinny 2."]}
    monkeypatch.setattr(_distributions, "report", lambda: dict(noted))
    code, data = _doctor_json(monkeypatch)
    assert (code, data["status"]) == (0, "ok")
    assert data["installation"]["notes"] == noted["notes"]


def test_doctor_text_output_includes_the_flavor(monkeypatch):
    from .._frontends import _argparse

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    assert _argparse.run(["doctor"]) == 0
    assert "installation.flavor: " in buf.getvalue()
