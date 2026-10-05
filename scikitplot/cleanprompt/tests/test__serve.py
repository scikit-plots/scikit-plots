"""Tests for :mod:`scikitplot.cleanprompt._serve`."""

from __future__ import annotations

import argparse

import pytest

from .. import CleanPromptError
from .. import _capabilities as caps
from .._serve import LOOPBACK, container_files, resolve_bind
from ._tiers import skip_reason

WEB = caps.probe("web").available


class TestResolveBind:
    """The one thing ``--docker`` changes, and the safety around it."""

    def test_local_default_is_loopback(self):
        address, exposure = resolve_bind(None, docker=False, allow_remote=False)
        assert address == "127.0.0.1"
        assert "only from this machine" in exposure

    def test_docker_default_is_all_interfaces(self):
        """Bound to loopback inside a container, the published port looks dead."""
        address, exposure = resolve_bind(None, docker=True, allow_remote=False)
        assert address == "0.0.0.0"  # noqa: S104 - the behaviour under test
        assert "container" in exposure

    def test_explicit_loopback_needs_no_acknowledgement(self):
        for address in sorted(LOOPBACK):
            assert resolve_bind(address, False, False)[0] == address

    def test_remote_bind_is_refused_without_acknowledgement(self):
        with pytest.raises(CleanPromptError) as caught:
            resolve_bind("0.0.0.0", docker=False, allow_remote=False)  # noqa: S104
        assert "no authentication" in str(caught.value)
        assert "--allow-remote" in str(caught.value)

    def test_remote_bind_is_allowed_once_acknowledged(self):
        address, exposure = resolve_bind("192.168.1.10", False, allow_remote=True)
        assert address == "192.168.1.10"
        assert "without authentication" in exposure

    def test_docker_is_its_own_acknowledgement(self):
        assert resolve_bind("0.0.0.0", docker=True, allow_remote=False)[0] == "0.0.0.0"  # noqa: S104

    def test_exposure_is_always_described(self):
        for args in ((None, False, False), (None, True, False), ("10.0.0.5", False, True)):
            assert resolve_bind(*args)[1].endswith((".", "you need it."))


class TestContainerFiles:
    """The emitted container files."""

    def test_returns_four_files(self):
        assert set(container_files()) == {
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore",
            "README.md",
        }

    def test_runs_as_a_non_root_user(self):
        """A container processing pasted personal data should not run as root."""
        assert "USER cleanprompt" in container_files()["Dockerfile"]

    def test_secret_key_has_no_default(self):
        """A default would be a published secret shared by every deployment."""
        compose = container_files()["docker-compose.yml"]
        assert "CLEANPROMPT_SECRET_KEY:?" in compose

    def test_port_is_published_on_loopback_by_default(self):
        assert '"127.0.0.1:5000:5000"' in container_files()["docker-compose.yml"]

    def test_port_is_honoured(self):
        files = container_files(port=8123)
        assert "EXPOSE 8123" in files["Dockerfile"]
        assert "127.0.0.1:8123:8123" in files["docker-compose.yml"]

    def test_ner_layer_is_opt_in(self):
        assert "spacy download" not in container_files()["Dockerfile"]
        assert "spacy download" in container_files(with_ner=True)["Dockerfile"]

    def test_entrypoint_uses_docker_mode(self):
        assert '"--docker"' in container_files()["Dockerfile"]

    def test_healthcheck_is_present(self):
        assert "HEALTHCHECK" in container_files()["Dockerfile"]

    def test_dockerignore_excludes_vaults(self):
        """A vault holds the values that were removed; never ship one."""
        ignore = container_files()[".dockerignore"]
        assert "vault*.json" in ignore
        assert "*.vault.json" in ignore

    def test_compose_drops_capabilities(self):
        compose = container_files()["docker-compose.yml"]
        assert "no-new-privileges:true" in compose
        assert "cap_drop" in compose

    def test_readme_explains_the_exposure(self):
        readme = container_files()["README.md"]
        assert "no authentication" in readme
        assert "0.0.0.0" in readme

    def test_readme_explains_the_missing_ner_tier(self):
        assert "--with-ner" in container_files()["README.md"]

    def test_files_are_non_empty_text(self):
        for name, body in container_files().items():
            assert isinstance(body, str) and body.strip(), name


class TestServe:
    """Guards that run before the server starts."""

    def _args(self, **overrides):
        base = {
            "host": None, "port": 5000, "docker": False, "allow_remote": False,
            "debug": False, "open": False, "profile": None, "config": None,
            "kinds": None, "hide": None, "allow": None, "word_boundary": False,
            "ignore_case": False, "ner": False, "ner_model": "en_core_web_lg",
            "overlap": None,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    @pytest.mark.skipif(not WEB, reason=skip_reason("web"))
    def test_docker_mode_requires_a_configured_secret_key(self, monkeypatch):
        import io

        monkeypatch.delenv("CLEANPROMPT_SECRET_KEY", raising=False)
        from .._serve import serve

        with pytest.raises(CleanPromptError) as caught:
            serve(self._args(docker=True), io.StringIO())
        assert "CLEANPROMPT_SECRET_KEY" in str(caught.value)
        assert "replica" in str(caught.value)

    @pytest.mark.skipif(not WEB, reason=skip_reason("web"))
    def test_remote_bind_is_refused_before_the_app_is_built(self, monkeypatch):
        import io

        monkeypatch.setenv("CLEANPROMPT_SECRET_KEY", "k" * 32)
        from .._serve import serve

        with pytest.raises(CleanPromptError, match="refusing to bind"):
            serve(self._args(host="0.0.0.0"), io.StringIO())  # noqa: S104

    def test_missing_web_tier_is_actionable(self, monkeypatch):
        import io

        from .. import CapabilityError
        from .._serve import serve

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            serve(self._args(), io.StringIO())
        assert "pip install" in str(caught.value)
