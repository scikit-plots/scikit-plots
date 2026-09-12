# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests owned by :mod:`_static.__init__`."""

from __future__ import annotations

import importlib

import pytest


class TestStaticSubpackage:
    """Tests for _static/__init__.py SVG constants and _PROVIDER_META."""

    @pytest.fixture(autouse=True)
    def _import_static(self):
        self._static = importlib.import_module(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._static"
        )

    # --- SVG constants ---

    def test_svg_copy_is_data_uri(self):
        assert self._static._SVG_COPY.startswith("data:image/svg+xml;base64,")

    def test_svg_markdown_is_data_uri(self):
        assert self._static._SVG_MARKDOWN.startswith("data:image/svg+xml;base64,")

    def test_svg_claude_is_data_uri(self):
        assert self._static._SVG_CLAUDE.startswith("data:image/svg+xml;base64,")

    def test_svg_chatgpt_is_data_uri(self):
        assert self._static._SVG_CHATGPT.startswith("data:image/svg+xml;base64,")

    def test_svg_gemini_is_data_uri(self):
        assert self._static._SVG_GEMINI.startswith("data:image/svg+xml;base64,")

    def test_svg_ollama_is_data_uri(self):
        assert self._static._SVG_OLLAMA.startswith("data:image/svg+xml;base64,")

    def test_svg_default_is_data_uri(self):
        assert self._static._SVG_DEFAULT.startswith("data:image/svg+xml;base64,")

    def test_all_svg_constants_non_empty(self):
        for name in ("_SVG_COPY", "_SVG_MARKDOWN", "_SVG_CLAUDE",
                     "_SVG_CHATGPT", "_SVG_GEMINI", "_SVG_OLLAMA", "_SVG_DEFAULT"):
            val = getattr(self._static, name)
            assert val and len(val) > 50, f"{name} too short or empty"

    def test_svg_constants_all_unique(self):
        vals = [
            self._static._SVG_CLAUDE, self._static._SVG_CHATGPT,
            self._static._SVG_GEMINI, self._static._SVG_OLLAMA,
        ]
        # Named provider icons should be distinct
        assert len(set(vals)) == len(vals), "Provider SVG constants are not all unique"

    # --- _PROVIDER_META ---

    def test_provider_meta_is_dict(self):
        assert isinstance(self._static._PROVIDER_META, dict)

    def test_provider_meta_non_empty(self):
        assert len(self._static._PROVIDER_META) >= 12

    def test_required_providers_present(self):
        required = {"claude", "chatgpt", "gemini", "ollama", "mistral",
                    "perplexity", "copilot", "groq", "you", "deepseek",
                    "huggingface", "custom"}
        for name in required:
            assert name in self._static._PROVIDER_META, f"Missing: {name!r}"

    def test_all_entries_have_icon_and_desc(self):
        for name, meta in self._static._PROVIDER_META.items():
            assert "icon" in meta and "desc" in meta, f"{name!r} incomplete"
            assert meta["icon"].startswith("data:image/svg+xml;base64,"), (
                f"{name!r} icon is not a data URI"
            )
            assert meta["desc"], f"{name!r} desc is empty"

    def test_mcp_tool_keys_present(self):
        for key in ("vscode", "claude_desktop", "cursor", "windsurf", "generic"):
            assert key in self._static._PROVIDER_META, f"MCP key {key!r} missing"

    def test_claude_icon_is_svg_claude(self):
        assert self._static._PROVIDER_META["claude"]["icon"] == self._static._SVG_CLAUDE

    def test_chatgpt_icon_is_svg_chatgpt(self):
        assert self._static._PROVIDER_META["chatgpt"]["icon"] == self._static._SVG_CHATGPT

    def test_all_in_all_list(self):
        for name in ("_SVG_COPY", "_SVG_DEFAULT", "_PROVIDER_META"):
            assert name in self._static.__all__
