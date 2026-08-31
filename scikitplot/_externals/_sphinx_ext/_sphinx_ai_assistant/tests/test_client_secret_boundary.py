from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
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
            "feedbackToken": "BUILD-FEEDBACK-SECRET-DO-NOT-LEAK",
        }
    }
    ai_assistant_endpoint_default_profile = "prod"


class _LegacyConfig:
    ai_assistant_endpoint_profiles = {}
    ai_assistant_endpoint_default_profile = ""
    ai_assistant_panel_feedback_endpoint = "https://proxy.example.com/v1/feedback"
    ai_assistant_global_share_endpoint = "https://proxy.example.com/v1/share"
    ai_assistant_training_endpoint = ""
    ai_assistant_panel_feedback_token = "LEGACY-FEEDBACK-SECRET-DO-NOT-LEAK"
    ai_assistant_global_share_token = "LEGACY-SHARE-SECRET-DO-NOT-LEAK"
    ai_assistant_global_share_ttl_days = 30


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
    assert profiles["prod"]["feedbackToken"] == ""
    assert "BUILD-SHARE-SECRET-DO-NOT-LEAK" not in raw
    assert "BUILD-FEEDBACK-SECRET-DO-NOT-LEAK" not in raw
    joined = "\n".join(logger.messages)
    assert "BUILD-SHARE-SECRET-DO-NOT-LEAK" not in joined
    assert "BUILD-FEEDBACK-SECRET-DO-NOT-LEAK" not in joined
    assert "build-time bearer credentials are never serialized" in joined


def test_legacy_flat_buildtime_tokens_are_never_serialized():
    profiles, default, logger = _serialised_profiles(_LegacyConfig())
    raw = json.dumps(profiles, sort_keys=True)
    assert default == "default"
    assert profiles["default"]["shareToken"] == ""
    assert profiles["default"]["feedbackToken"] == ""
    assert "LEGACY-SHARE-SECRET-DO-NOT-LEAK" not in raw
    assert "LEGACY-FEEDBACK-SECRET-DO-NOT-LEAK" not in raw
    joined = "\n".join(logger.messages)
    assert "LEGACY-SHARE-SECRET-DO-NOT-LEAK" not in joined
    assert "LEGACY-FEEDBACK-SECRET-DO-NOT-LEAK" not in joined
    assert "will not be serialized into generated HTML" in joined


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
