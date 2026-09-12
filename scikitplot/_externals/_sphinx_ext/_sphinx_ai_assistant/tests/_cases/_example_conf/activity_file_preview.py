from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT
from pathlib import Path

ROOT = RUNTIME_ROOT
INIT = (ROOT / "__init__.py").read_text(encoding="utf-8")
EXAMPLE = (ROOT / "_example_conf.py").read_text(encoding="utf-8")
GUIDE = (ROOT / "ACTIVITY_AND_FILE_PREVIEW_GUIDE.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
JS = (ROOT / "_static" / "ai-assistant.js").read_text(encoding="utf-8")


def test_run172_config_registration_and_serialization():
    for cfg, wire in (
        ("ai_assistant_panel_activity_timeline", "panelActivityTimeline"),
        ("ai_assistant_panel_activity_auto_collapse", "panelActivityAutoCollapse"),
        ("ai_assistant_panel_generated_file_preview", "panelGeneratedFilePreview"),
    ):
        assert f'app.add_config_value("{cfg}"' in INIT
        assert f'"{wire}"' in INIT
        assert cfg in EXAMPLE


def test_run172_docs_separate_public_activity_from_hidden_reasoning():
    low = GUIDE.lower()
    assert "not a chain-of-thought viewer" in low
    assert "hidden model reasoning" in low
    assert "public summaries" in low
    assert "actual repository/storage mutation" in low
    assert "ACTIVITY_AND_FILE_PREVIEW_GUIDE.md" in README


def test_run172_latest_state_contract_is_documented():
    low = GUIDE.lower()
    assert "latest unavailable/removed state" in low
    assert "stale earlier bytes" in low
    assert "operation: \"remove\"" in GUIDE
    assert "download all latest files" in GUIDE.lower()


def test_run172_share_review_race_guard_was_not_weakened():
    needle = (
        "reviewed.action === 'cancel' || opConversationId !== boundConversationId "
        "|| opConversationId !== _getConversationId()"
    )
    assert needle in JS
