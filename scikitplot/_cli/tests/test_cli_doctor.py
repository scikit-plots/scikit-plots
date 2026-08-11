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
