from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT
from pathlib import Path

ROOT = RUNTIME_ROOT
INIT = (ROOT / "__init__.py").read_text()
EXAMPLE = (ROOT / "_example_conf.py").read_text()
JS = (ROOT / "_static" / "ai-assistant.js").read_text()


def test_run171_privacy_banner_config_is_registered_and_serialized():
    for key in (
        "ai_assistant_panel_chat_privacy_banner",
        "ai_assistant_panel_chat_privacy_text",
        "ai_assistant_panel_chat_privacy_more_text",
    ):
        assert key in INIT
        assert INIT.count(key) >= 2  # serialization + Sphinx config registration
    assert '"panelChatPrivacyBanner"' in INIT
    assert '"panelChatPrivacyText"' in INIT
    assert '"panelChatPrivacyMoreText"' in INIT


def test_run171_example_makes_stronger_claims_explicitly_operator_verified():
    assert "Use the text override ONLY" in EXAMPLE
    assert "actually verified stronger guarantees" in EXAMPLE
    assert "Zero data retention" in EXAMPLE
    assert "No AI training" in EXAMPLE


def test_run171_default_runtime_copy_does_not_claim_anonymity_or_zero_retention():
    start = JS.index("function _firstMessagePrivacyText(")
    end = JS.index("function _buildFirstMessagePrivacyBanner(", start)
    block = JS[start:end]
    assert "Anonymized by" not in block
    assert "Zero data retention" not in block
    assert "No AI training" not in block
    assert "Retention and AI-training policies depend on that provider" in block


def test_run171_internal_sheet_routes_are_not_public_event_projections():
    # Internal open events can carry DOM opener references for focus restoration;
    # they must never be projected through the public/integration event bridge.
    public_start = JS.index("function _publicAssistantEventDetail(")
    public_end = JS.index("function _dispatchAssistantEvent(", public_start)
    public_block = JS[public_start:public_end]
    assert "ai-assistant-open-privacy" not in public_block
    assert "ai-assistant-open-model-configuration" not in public_block
