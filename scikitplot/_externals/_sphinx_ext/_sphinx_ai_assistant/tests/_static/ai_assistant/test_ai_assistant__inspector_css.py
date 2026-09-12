from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT
from pathlib import Path

ROOT = RUNTIME_ROOT
CSS = (ROOT / "_static" / "ai-assistant.css").read_text(encoding="utf-8")


def test_payload_code_wraps_long_content_without_changing_export_bytes():
    block = CSS[CSS.index("/* Run 52 — inspector readability"):]
    assert ".ai-assistant-panel-payload-code" in block
    assert "white-space: pre-wrap" in block
    assert "overflow-wrap: anywhere" in block


def test_payload_syntax_tokens_are_visually_distinct():
    for token in ("key", "string", "number", "boolean", "null", "punct"):
        assert f".ai-assistant-panel-json-token--{token}" in CSS


def test_format_tabs_have_visible_keyboard_focus():
    assert ".ai-assistant-panel-payload-format-btn:focus-visible" in CSS
