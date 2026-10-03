from __future__ import annotations

from .._paths import RUNTIME_ROOT

import importlib.util
import json
from pathlib import Path

_ROOT = RUNTIME_ROOT
_SPEC = importlib.util.spec_from_file_location("_client_secret_boundary_target", _ROOT / "__init__.py")
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)


class _TestLogger:
    def __init__(self):
        self.messages: list[str] = []

    def warning(self, msg, *args, **kwargs):
        try:
            self.messages.append(str(msg) % args)
        except Exception:
            self.messages.append(str(msg))


class _ExplicitConfig:
    ai_assistant_endpoint_profiles = {
        "prod": {
            "label": "Prod",
            "base": "https://proxy.example.com",
            "shareToken": "BUILD-SHARE-SECRET-DO-NOT-LEAK",
        }
    }
    ai_assistant_endpoint_default_profile = "prod"



def _serialised_profiles(config) -> tuple[dict, str, _TestLogger]:
    logger = _TestLogger()
    _MOD._logger = logger
    profiles, default = _MOD._serialize_endpoint_profiles(config)
    return profiles, default, logger


def test_explicit_buildtime_endpoint_tokens_are_never_serialized():
    profiles, default, logger = _serialised_profiles(_ExplicitConfig())
    raw = json.dumps(profiles, sort_keys=True)
    assert default == "prod"
    assert profiles["prod"]["shareToken"] == ""
    assert "BUILD-SHARE-SECRET-DO-NOT-LEAK" not in raw
    joined = "\n".join(logger.messages)
    assert "BUILD-SHARE-SECRET-DO-NOT-LEAK" not in joined
    assert "build-time bearer credentials are never serialized" in joined


def test_validate_profile_never_echoes_secret_value_to_warning():
    logger = _TestLogger()
    _MOD._logger = logger
    secret = "<script>SUPER-SECRET-VALUE</script>"
    profile = _MOD._validate_profile(
        {"base": "https://proxy.example.com", "shareToken": secret},
        "malicious",
    )
    assert profile["shareToken"] == ""
    assert secret not in "\n".join(logger.messages)
