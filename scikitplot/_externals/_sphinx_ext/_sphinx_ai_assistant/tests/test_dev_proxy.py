# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/test_dev_proxy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of ``dev_proxy.py``: what importing it does, and what starting it needs.

``dev_proxy.py`` is a script (``python dev_proxy.py``) that ships inside the
``scikitplot`` package. Being in a package, it is also imported by anything
that walks the package. Its start-up check for ``HF_TOKEN`` used to be a
module-level statement, so importing it without the variable ended the
interpreter with ``SystemExit(1)``.

Notes
-----
**Developer.** The module is loaded from its file under a private name, with
``sys.path`` restored afterwards: at import it puts the sibling
``_hf_spaces_proxy`` directory on the path, which must not outlive the test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "dev_proxy.py"


def _load(monkeypatch, token):
    """Load ``dev_proxy.py`` with ``HF_TOKEN`` set to ``token`` (``None``: unset)."""
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("_dev_proxy_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_without_a_token_does_not_exit(monkeypatch):
    module = _load(monkeypatch, None)
    assert module.HF_TOKEN == ""
    assert callable(module.main)


def test_starting_without_a_token_exits_with_an_actionable_message(monkeypatch, caplog):
    module = _load(monkeypatch, None)

    def never(*args, **kwargs):  # the server must not be reached
        raise AssertionError("the server was started without a token")

    monkeypatch.setattr(module, "HTTPServer", never)
    with caplog.at_level("ERROR", logger="dev_proxy"), pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 1
    assert "HF_TOKEN environment variable is not set" in caplog.text
    assert "export HF_TOKEN=" in caplog.text


def test_starting_with_a_token_reaches_the_server(monkeypatch):
    module = _load(monkeypatch, "value-for-this-test-only")
    started = []

    class _Server:
        def __init__(self, address, handler):
            started.append(address)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            started.append("closed")

    monkeypatch.setattr(module, "HTTPServer", _Server)
    module.main()
    assert started == [("127.0.0.1", module.PORT), "closed"]
