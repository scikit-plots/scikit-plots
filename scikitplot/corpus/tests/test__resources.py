# corpus/tests/test__resources.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline-safe resource gating gate (CORPUS-RES-001)
==================================================

``scikitplot.corpus._resources`` is the single boundary through which optional
NLTK data is accessed. The invariant under test: **local processing performs no
network access unless explicitly authorized**. A missing resource raises an
actionable :class:`ResourceUnavailableError` by default; downloads happen only
when ``allow_download=True`` or ``SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS`` is truthy.

NLTK itself is not required — a fake ``nltk`` module is injected so the suite is
deterministic on any machine.

Run with::

    pytest scikitplot/corpus/tests/test__resources.py -v
"""

from __future__ import annotations

import sys
import types

import pytest

from scikitplot.corpus import _resources as R


# --------------------------------------------------------------------------- #
# fake nltk
# --------------------------------------------------------------------------- #
class _FakeData:
    def __init__(self, present):
        self.present = set(present)

    def find(self, path):
        if path in self.present:
            return f"/fake/{path}"
        raise LookupError(f"Resource {path} not found.")


class _FakeNltk(types.ModuleType):
    def __init__(self, present):
        super().__init__("nltk")
        self.data = _FakeData(present)
        self.downloads: list[str] = []

    def download(self, name, quiet=False):
        self.downloads.append(name)
        return True


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(R.ENV_ALLOW_DOWNLOADS, raising=False)


def _install(monkeypatch, present):
    fake = _FakeNltk(present)
    monkeypatch.setitem(sys.modules, "nltk", fake)
    return fake


# --------------------------------------------------------------------------- #
class TestDownloadsAllowed:
    def test_default_false(self, no_env):
        assert R.downloads_allowed() is False

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(R.ENV_ALLOW_DOWNLOADS, "1")
        assert R.downloads_allowed(False) is False
        assert R.downloads_allowed(True) is True

    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("", False), ("off", False), ("nope", False),
    ])
    def test_env_values(self, monkeypatch, val, expected):
        monkeypatch.setenv(R.ENV_ALLOW_DOWNLOADS, val)
        assert R.downloads_allowed() is expected


class TestPreflight:
    def test_present_true_no_download(self, monkeypatch):
        fake = _install(monkeypatch, {"corpora/stopwords"})
        assert R.nltk_resource_available("corpora/stopwords") is True
        assert fake.downloads == []

    def test_missing_false_no_download(self, monkeypatch):
        fake = _install(monkeypatch, set())
        assert R.nltk_resource_available("corpora/wordnet") is False
        assert fake.downloads == []

    def test_no_nltk_returns_false(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nltk", None)  # force ImportError
        assert R.nltk_resource_available("corpora/x") is False


class TestEnsureResource:
    def test_present_returns_without_download(self, monkeypatch, no_env):
        fake = _install(monkeypatch, {"corpora/stopwords"})
        R.ensure_nltk_resource("corpora/stopwords", "stopwords")
        assert fake.downloads == []

    def test_missing_disabled_raises_zero_network(self, monkeypatch, no_env):
        fake = _install(monkeypatch, set())
        with pytest.raises(R.ResourceUnavailableError) as ei:
            R.ensure_nltk_resource("corpora/wordnet", "wordnet")
        assert fake.downloads == []  # the whole point: no implicit download
        msg = str(ei.value)
        assert "wordnet" in msg
        assert "python -m nltk.downloader wordnet" in msg
        assert R.ENV_ALLOW_DOWNLOADS in msg

    def test_missing_explicit_allow_downloads(self, monkeypatch, no_env):
        fake = _install(monkeypatch, set())
        R.ensure_nltk_resource("corpora/wordnet", "wordnet", allow_download=True)
        assert fake.downloads == ["wordnet"]

    def test_missing_env_authorized_downloads(self, monkeypatch):
        monkeypatch.setenv(R.ENV_ALLOW_DOWNLOADS, "1")
        fake = _install(monkeypatch, set())
        R.ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
        assert fake.downloads == ["punkt_tab"]

    def test_no_nltk_raises_actionable(self, monkeypatch, no_env):
        monkeypatch.setitem(sys.modules, "nltk", None)  # force ImportError
        with pytest.raises(R.ResourceUnavailableError) as ei:
            R.ensure_nltk_resource("corpora/stopwords", "stopwords")
        assert "pip install nltk" in str(ei.value)
