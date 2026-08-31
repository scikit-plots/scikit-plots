from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("_endpoint_security_target", _ROOT / "__init__.py")
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)


class _TestLogger:
    def __init__(self):
        self.messages = []

    def warning(self, *args, **kwargs):
        self.messages.append((args, kwargs))


_MOD._logger = _TestLogger()


def test_buildtime_absolute_and_relative_endpoint_security():
    p = _MOD._validate_profile(
        {
            "label": "secure",
            "base": "  https://EXAMPLE.com:443/api/  ",
            "chat": "v1/chat/completions",
            "share": "/v1/share",
            "feedback": "https://feedback.example.com/hook?tenant=x",
            "training": None,
        },
        "secure",
    )
    assert p["base"] == "https://example.com/api"
    assert p["chat"] == "v1/chat/completions"
    assert p["share"] == "v1/share"
    assert p["feedback"] == "https://feedback.example.com/hook?tenant=x"
    assert p["training"] == ""


def test_buildtime_rejects_ambiguous_and_malicious_url_forms():
    p = _MOD._validate_profile(
        {
            "base": "https://user:secret@example.com/api",
            "chat": "../admin",
            "share": "%2e%2e/admin",
            "feedback": "https://good.example.com/%2e%2e/api",
            "training": "v1/%2Fadmin",
        },
        "bad",
    )
    assert p["base"] == ""
    assert p["chat"] == ""
    assert p["share"] == ""
    assert p["feedback"] == ""
    assert p["training"] == ""


def test_buildtime_rejects_overlong_values_and_base_query():
    p = _MOD._validate_profile(
        {
            "base": "https://good.example.com/base?tenant=x",
            "chat": "v1/" + "a" * 1100,
            "share": "https://good.example.com/" + "a" * 2100,
        },
        "long",
    )
    assert p["base"] == ""
    assert p["chat"] == ""
    assert p["share"] == ""


def test_buildtime_private_target_is_flagged_but_retained_for_trusted_local_dev():
    p = _MOD._validate_profile({"base": "http://127.0.0.1:8000"}, "dev")
    assert p["base"] == "http://127.0.0.1:8000"
    assert "base" in p["_warn"]
