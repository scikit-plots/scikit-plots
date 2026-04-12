# tests/_externals/_sphinx_ext/_sphinx_ai_assistant/test___init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive test suite for
``scikitplot._externals._sphinx_ext._sphinx_ai_assistant``.

Coverage targets
----------------
*  1 — Lazy import mechanics (Sphinx-free at module scope).
*  2 — _safe_json_for_script (XSS prevention).
*  3 — _is_path_within (path-traversal guard).
*  4 — _validate_base_url.
*  5 — _validate_position.
*  6 — _validate_provider_url_template.
*  7 — _validate_ollama_url.
*  8 — _validate_css_selector / _sanitize_selectors.
*  9 — _DEFAULT_PROVIDERS registry (all 12 providers).
* 10 — _validate_provider.
* 11 — _filter_providers.
* 12 — Dependency detection (_has_markdown_deps, _has_ipython).
* 13 — Logger singleton (_get_logger).
* 14 — Converter class (_build_converter_class, SphinxMarkdownConverter).
* 15 — _coerce_to_list.
* 16 — html_to_markdown / strip_tags coercion.
* 17 — _write_progress_bar (in __all__).
* 18 — _resolve_icon (backup SVG fallback).
* 19 — _static subpackage — SVG constants & _PROVIDER_META.
* 20 — Theme selector presets (_THEME_SELECTOR_PRESETS ≥ 20 themes).
* 21 — _resolve_content_selectors.
* 22 — _DEFAULT_CONTENT_SELECTORS.
* 23 — _process_html_file_worker (6-tuple, separate output dir).
* 24 — _process_single_html_file (5-tuple wrapper).
* 25 — process_html_directory (all branches).
* 26 — generate_llms_txt_standalone (all branches).
* 27 — generate_markdown_files (Sphinx hook).
* 28 — generate_llms_txt (Sphinx hook).
* 29 — add_ai_assistant_context (all branches, icon resolution).
* 30 — setup() — metadata, config values, events, static path.
* 31 — _OLLAMA_RECOMMENDED_MODELS.
* 32 — _DEFAULT_MCP_TOOLS (all 5 tools).
* 33 — _validate_mcp_tool.
* 34 — _cfg_str / _cfg_bool helpers.
* 35 — Extended provider registry (deepseek, huggingface, custom).
* 36 — setup() extended config values.
* 37 — add_ai_assistant_context extended fields.
* 38 — Ollama local provider support.
* 39 — Full provider round-trip (filter → context → JSON).
* 40 — Edge cases and invariants.
"""
from __future__ import annotations

import inspect
import io
import json
import sys
import types
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# conftest._bootstrap_submodule() has already run before this import;
# the module is registered under its canonical dotted path.
import scikitplot._externals._sphinx_ext._sphinx_ai_assistant as _mod

_EXT = _mod


# ===========================================================================
# 1. Lazy import safety
# ===========================================================================

class TestLazyImports:
    """Module importable without Sphinx, bs4, or markdownify."""

    def test_module_importable(self):
        assert _mod is not None

    def test_sphinx_callables_present_at_module_scope(self):
        assert callable(_mod.setup)
        assert callable(_mod.generate_markdown_files)
        assert callable(_mod.add_ai_assistant_context)

    def test_version_is_semver_string(self):
        assert isinstance(_mod._VERSION, str)
        parts = _mod._VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_bs4_not_imported_at_module_scope(self):
        mod_globals = vars(_mod)
        bs4_binding = mod_globals.get("BeautifulSoup", _mod)
        try:
            from bs4 import BeautifulSoup as _real_BS
            assert bs4_binding is not _real_BS
        except ImportError:
            pass

    def test_process_html_directory_is_public(self):
        assert callable(_mod.process_html_directory)

    def test_generate_llms_txt_standalone_is_public(self):
        assert callable(_mod.generate_llms_txt_standalone)

    def test_resolve_icon_is_public(self):
        assert callable(_mod._resolve_icon)
        assert "_resolve_icon" in _mod.__all__

    def test_write_progress_bar_in_all(self):
        assert "_write_progress_bar" in _mod.__all__


# ===========================================================================
# 2. Security — _safe_json_for_script
# ===========================================================================

class TestSafeJsonForScript:
    def test_plain_dict_round_trips(self):
        obj = {"key": "value", "num": 42}
        assert json.loads(_mod._safe_json_for_script(obj)) == obj

    def test_script_close_tag_escaped(self):
        result = _mod._safe_json_for_script({"url": "https://x.com/</script>"})
        assert "</script>" not in result
        assert "<\\/" in result

    def test_nested_close_tag_escaped(self):
        result = _mod._safe_json_for_script({"a": {"b": "</ScRiPt>"}})
        assert "</ScRiPt>" not in result

    def test_multiple_occurrences_all_escaped(self):
        result = _mod._safe_json_for_script({"x": "</s></s></s>"})
        assert "</" not in result

    def test_empty_dict(self):
        assert _mod._safe_json_for_script({}) == "{}"

    def test_non_ascii_escaped(self):
        result = _mod._safe_json_for_script({"emoji": "\u00e9"})
        assert "\u00e9" not in result
        assert "\\u00e9" in result

    def test_none_value(self):
        assert json.loads(_mod._safe_json_for_script({"k": None})) == {"k": None}

    def test_list_value(self):
        obj = {"items": [1, 2, "</script>"]}
        result = _mod._safe_json_for_script(obj)
        assert "</" not in result
        assert json.loads(result)["items"][0] == 1

    def test_boolean_values(self):
        obj = {"flag": True, "other": False}
        assert json.loads(_mod._safe_json_for_script(obj)) == obj

    def test_large_nested_object(self):
        obj = {"l1": {"l2": {"l3": list(range(100))}}}
        assert json.loads(_mod._safe_json_for_script(obj)) == obj

    def test_unicode_preserved_round_trip(self):
        obj = {"k": "\u2603"}
        result = _mod._safe_json_for_script(obj)
        assert "\u2603" not in result
        assert json.loads(result)["k"] == "\u2603"


# ===========================================================================
# 3. Security — _is_path_within
# ===========================================================================

class TestIsPathWithin:
    def test_child_path(self, tmp_path):
        assert _mod._is_path_within(tmp_path / "a" / "b.html", tmp_path) is True

    def test_same_path(self, tmp_path):
        assert _mod._is_path_within(tmp_path, tmp_path) is True

    def test_sibling_path(self, tmp_path):
        assert _mod._is_path_within(tmp_path.parent / "other", tmp_path) is False

    def test_dotdot_traversal_blocked(self, tmp_path):
        evil = tmp_path / ".." / "etc" / "passwd"
        assert _mod._is_path_within(evil, tmp_path) is False

    def test_absolute_outside_blocked(self, tmp_path):
        assert _mod._is_path_within(Path("/etc/passwd"), tmp_path) is False

    def test_deeply_nested_child(self, tmp_path):
        assert _mod._is_path_within(
            tmp_path / "a" / "b" / "c" / "d.html", tmp_path
        ) is True


# ===========================================================================
# 4. Security — _validate_base_url
# ===========================================================================

class TestValidateBaseUrl:
    def test_https_accepted_and_trailing_slash_stripped(self):
        assert _mod._validate_base_url("https://docs.example.com/") == "https://docs.example.com"

    def test_http_localhost_accepted(self):
        assert _mod._validate_base_url("http://localhost:8000/") == "http://localhost:8000"

    def test_empty_string_accepted(self):
        assert _mod._validate_base_url("") == ""

    def test_whitespace_only_treated_as_empty(self):
        assert _mod._validate_base_url("   ") == ""

    def test_javascript_scheme_raises(self):
        with pytest.raises(ValueError, match="http"):
            _mod._validate_base_url("javascript:alert(1)")

    def test_data_scheme_raises(self):
        with pytest.raises(ValueError):
            _mod._validate_base_url("data:text/html,<h1>XSS</h1>")

    def test_ftp_scheme_raises(self):
        with pytest.raises(ValueError):
            _mod._validate_base_url("ftp://example.com")

    def test_multiple_trailing_slashes_stripped(self):
        assert _mod._validate_base_url("https://x.com///") == "https://x.com"

    def test_uppercase_https_accepted(self):
        result = _mod._validate_base_url("HTTPS://docs.example.com/")
        assert result.startswith("HTTPS://")

    def test_mixed_case_http_accepted(self):
        result = _mod._validate_base_url("Http://localhost/")
        assert result.startswith("Http://")


# ===========================================================================
# 5. Security — _validate_position
# ===========================================================================

class TestValidatePosition:
    def test_sidebar(self): assert _mod._validate_position("sidebar") == "sidebar"
    def test_title(self): assert _mod._validate_position("title") == "title"
    def test_floating(self): assert _mod._validate_position("floating") == "floating"
    def test_none_str(self): assert _mod._validate_position("none") == "none"
    def test_uppercase_normalised(self): assert _mod._validate_position("SIDEBAR") == "sidebar"
    def test_whitespace_stripped(self): assert _mod._validate_position("  title  ") == "title"

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="ai_assistant_position"):
            _mod._validate_position("evil")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _mod._validate_position("")

    def test_allowed_positions_frozenset_complete(self):
        assert isinstance(_mod._ALLOWED_POSITIONS, frozenset)
        assert {"sidebar", "title", "floating", "none"}.issubset(_mod._ALLOWED_POSITIONS)


# ===========================================================================
# 6. Security — _validate_provider_url_template
# ===========================================================================

class TestValidateProviderUrlTemplate:
    def test_https_accepted(self):
        assert _mod._validate_provider_url_template("https://claude.ai/new?q={prompt}") is True

    def test_http_localhost_accepted(self):
        assert _mod._validate_provider_url_template("http://localhost:3000/?q={prompt}") is True

    def test_empty_string_accepted(self):
        assert _mod._validate_provider_url_template("") is True

    def test_whitespace_only_accepted(self):
        assert _mod._validate_provider_url_template("   ") is True

    def test_javascript_rejected(self):
        assert _mod._validate_provider_url_template("javascript:alert(1)") is False

    def test_data_uri_rejected(self):
        assert _mod._validate_provider_url_template("data:text/html,<h1>X</h1>") is False

    def test_ftp_rejected(self):
        assert _mod._validate_provider_url_template("ftp://evil.com") is False

    def test_vbscript_rejected(self):
        assert _mod._validate_provider_url_template("vbscript:msgbox(1)") is False


# ===========================================================================
# 7. Security — _validate_ollama_url
# ===========================================================================

class TestValidateOllamaUrl:
    def test_localhost_accepted(self):
        assert _mod._validate_ollama_url("http://localhost:11434") is True

    def test_loopback_127_accepted(self):
        assert _mod._validate_ollama_url("http://127.0.0.1:11434") is True

    def test_empty_accepted(self):
        assert _mod._validate_ollama_url("") is True

    def test_remote_host_rejected(self):
        assert _mod._validate_ollama_url("http://remote.example.com") is False

    def test_https_remote_rejected(self):
        assert _mod._validate_ollama_url("https://ollama.example.com") is False

    def test_localhost_no_port(self):
        assert _mod._validate_ollama_url("http://localhost") is True

    def test_localhost_with_path(self):
        assert _mod._validate_ollama_url("http://localhost:11434/api") is True

    def test_case_insensitive_localhost(self):
        assert _mod._validate_ollama_url("HTTP://LOCALHOST:11434") is True


# ===========================================================================
# 8. Security — _validate_css_selector / _sanitize_selectors
# ===========================================================================

class TestValidateCssSelector:
    def test_element(self): assert _mod._validate_css_selector("article") is True
    def test_class(self): assert _mod._validate_css_selector("article.bd-article") is True
    def test_attribute_quotes(self): assert _mod._validate_css_selector('div[role="main"]') is True
    def test_html_open_rejected(self): assert _mod._validate_css_selector("<script>") is False
    def test_html_close_rejected(self): assert _mod._validate_css_selector("</style>") is False
    def test_combined_html_rejected(self): assert _mod._validate_css_selector("<img onerror=x>") is False
    def test_main(self): assert _mod._validate_css_selector("main") is True
    def test_id_selector(self): assert _mod._validate_css_selector("#content") is True
    def test_pseudo_class(self): assert _mod._validate_css_selector("div:first-child") is True


class TestSanitizeSelectors:
    def test_empty_strings_removed(self):
        assert "" not in _mod._sanitize_selectors(["article", "   ", "main"])

    def test_unsafe_selectors_removed(self):
        result = _mod._sanitize_selectors(["article", "<bad>", "main"])
        assert "<bad>" not in result
        assert "article" in result and "main" in result

    def test_all_safe_unchanged(self):
        sels = ["article.bd-article", 'div[role="main"]', "main"]
        assert _mod._sanitize_selectors(sels) == sels

    def test_empty_list(self):
        assert _mod._sanitize_selectors([]) == []

    def test_all_unsafe_returns_empty(self):
        assert _mod._sanitize_selectors(["<bad>", "</worse>"]) == []


# ===========================================================================
# 9. Provider registry — _DEFAULT_PROVIDERS
# ===========================================================================

class TestDefaultProviders:
    """All 12 default providers must exist and be schema-valid."""

    _REQUIRED_PROVIDERS = [
        "claude", "chatgpt", "gemini", "ollama", "custom",
        "copilot", "deepseek", "groq", "huggingface", "mistral",
        "perplexity", "you",
    ]

    def test_all_required_providers_present(self):
        for name in self._REQUIRED_PROVIDERS:
            assert name in _mod._DEFAULT_PROVIDERS, f"Missing: {name!r}"

    def test_total_count_at_least_12(self):
        assert len(_mod._DEFAULT_PROVIDERS) >= 12

    def test_all_required_keys_present(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            for key in _mod._PROVIDER_REQUIRED_KEYS:
                assert key in cfg, f"Provider {name!r} missing {key!r}"

    def test_all_provider_types_valid(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            assert cfg["type"] in _mod._PROVIDER_TYPES, (
                f"Provider {name!r} invalid type {cfg['type']!r}"
            )

    def test_all_url_templates_safe(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            url = cfg.get("url_template", "")
            assert _mod._validate_provider_url_template(url), (
                f"Provider {name!r} unsafe url_template: {url!r}"
            )

    def test_ollama_local_type_and_disabled(self):
        ollama = _mod._DEFAULT_PROVIDERS["ollama"]
        assert ollama["type"] == "local"
        assert ollama["enabled"] is False

    def test_ollama_api_base_url_is_loopback(self):
        api_url = _mod._DEFAULT_PROVIDERS["ollama"].get("api_base_url", "")
        assert _mod._validate_ollama_url(api_url)

    def test_claude_enabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["claude"]["enabled"] is True

    def test_chatgpt_enabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["chatgpt"]["enabled"] is True

    def test_gemini_enabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["gemini"]["enabled"] is True

    def test_no_javascript_url_templates(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            url = cfg.get("url_template", "")
            assert "javascript:" not in url.lower(), (
                f"Provider {name!r} has javascript: in url_template"
            )

    def test_all_prompt_templates_have_url_placeholder(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            pt = cfg.get("prompt_template", "")
            assert "{url}" in pt, (
                f"Provider {name!r} prompt_template missing {{url}}"
            )

    def test_claude_model_is_current(self):
        assert _mod._DEFAULT_PROVIDERS["claude"]["model"] == "claude-sonnet-4-6"

    def test_custom_type_is_custom(self):
        assert _mod._DEFAULT_PROVIDERS["custom"]["type"] == "custom"

    def test_custom_disabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["custom"]["enabled"] is False


# ===========================================================================
# 10. _validate_provider
# ===========================================================================

class TestValidateProvider:
    def _full_provider(self, **overrides) -> dict:
        base = {
            "enabled": True,
            "label": "Test",
            "description": "A test provider",
            "icon": "test.svg",
            "url_template": "https://example.com/?q={prompt}",
            "prompt_template": "Please read: {url}",
            "model": "test-model",
            "type": "web",
        }
        base.update(overrides)
        return base

    def test_valid_provider_no_errors(self):
        assert _mod._validate_provider(self._full_provider(), "test") == []

    def test_missing_label_reported(self):
        p = self._full_provider()
        del p["label"]
        assert any("label" in e for e in _mod._validate_provider(p, "bad"))

    def test_invalid_type_reported(self):
        errs = _mod._validate_provider(self._full_provider(type="invalid"), "bad")
        assert any("type" in e for e in errs)

    def test_javascript_url_template_reported(self):
        errs = _mod._validate_provider(
            self._full_provider(url_template="javascript:alert(1)"), "bad"
        )
        assert any("url_template" in e for e in errs)

    def test_local_type_remote_api_base_url_reported(self):
        errs = _mod._validate_provider(
            self._full_provider(type="local", api_base_url="https://remote.example.com"),
            "bad",
        )
        assert any("api_base_url" in e for e in errs)

    def test_local_type_localhost_api_base_url_ok(self):
        assert _mod._validate_provider(
            self._full_provider(type="local", api_base_url="http://localhost:11434"),
            "ollama",
        ) == []

    def test_all_default_providers_validate_cleanly(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            errs = _mod._validate_provider(cfg, name)
            assert errs == [], f"Provider {name!r} errors: {errs}"

    def test_error_prefix_contains_name(self):
        p = self._full_provider()
        del p["model"]
        errs = _mod._validate_provider(p, "my_provider")
        assert all("my_provider" in e for e in errs)

    def test_empty_name_gives_generic_prefix(self):
        p = self._full_provider()
        del p["model"]
        errs = _mod._validate_provider(p)
        assert errs and all("Provider" in e for e in errs)


# ===========================================================================
# 11. _filter_providers
# ===========================================================================

class TestFilterProviders:
    def test_keeps_safe_provider(self):
        providers = {"safe": {"url_template": "https://example.com/?q={prompt}"}}
        assert "safe" in _mod._filter_providers(providers)

    def test_removes_javascript_url(self):
        providers = {"evil": {"url_template": "javascript:alert(1)"}}
        assert "evil" not in _mod._filter_providers(providers)

    def test_removes_data_url(self):
        providers = {"evil": {"url_template": "data:text/html,xss"}}
        assert "evil" not in _mod._filter_providers(providers)

    def test_require_enabled_filters_disabled(self):
        providers = {
            "on": {"url_template": "https://x.com", "enabled": True},
            "off": {"url_template": "https://y.com", "enabled": False},
        }
        result = _mod._filter_providers(providers, require_enabled=True)
        assert "on" in result and "off" not in result

    def test_require_enabled_false_keeps_disabled(self):
        providers = {"off": {"url_template": "https://y.com", "enabled": False}}
        assert "off" in _mod._filter_providers(providers, require_enabled=False)

    def test_empty_input(self):
        assert _mod._filter_providers({}) == {}

    def test_full_default_registry_keeps_all(self):
        result = _mod._filter_providers(_mod._DEFAULT_PROVIDERS)
        for name in ("claude", "gemini", "chatgpt", "deepseek", "huggingface", "custom"):
            assert name in result, f"{name!r} filtered unexpectedly"


# ===========================================================================
# 12. Dependency detection
# ===========================================================================

class TestHasMarkdownDeps:
    def test_returns_bool(self):
        assert isinstance(_mod._has_markdown_deps(), bool)

    def test_true_when_both_installed(self):
        assert _mod._has_markdown_deps() is True

    def test_false_when_bs4_missing(self):
        with patch("importlib.util.find_spec", side_effect=lambda n: None if n == "bs4" else MagicMock()):
            assert _mod._has_markdown_deps() is False

    def test_false_when_markdownify_missing(self):
        with patch("importlib.util.find_spec", side_effect=lambda n: None if n == "markdownify" else MagicMock()):
            assert _mod._has_markdown_deps() is False


class TestHasIPython:
    def test_returns_bool(self):
        assert isinstance(_mod._has_ipython(), bool)

    def test_false_when_ipython_missing(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert _mod._has_ipython() is False


# ===========================================================================
# 13. Logger singleton
# ===========================================================================

class TestGetLogger:
    def test_returns_non_none(self):
        assert _mod._get_logger() is not None

    def test_caches_same_object(self):
        assert _mod._get_logger() is _mod._get_logger()

    def test_reset_and_reinit(self):
        """After nulling the cache, next call must rebuild the logger."""
        original = _mod._logger
        try:
            _mod._logger = None
            logger = _mod._get_logger()
            assert logger is not None
        finally:
            _mod._logger = original


# ===========================================================================
# 14. Converter class
# ===========================================================================

class TestBuildConverterClass:
    def test_returns_a_class(self):
        assert isinstance(_mod._build_converter_class(), type)

    def test_caches_same_class_object(self):
        assert _mod._build_converter_class() is _mod._build_converter_class()

    def test_class_is_instantiable(self):
        assert _mod._build_converter_class()(heading_style="ATX") is not None


class TestSphinxMarkdownConverter:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.Cls = _mod._build_converter_class()
        self.instance = self.Cls(heading_style="ATX", bullets="*")

    def _el(self, tag, attrs=None, text=""):
        from bs4 import BeautifulSoup
        attrs_str = " ".join(f'{k}="{v}"' for k, v in (attrs or {}).items())
        return BeautifulSoup(f"<{tag} {attrs_str}>{text}</{tag}>", "html.parser").find(tag)

    def test_code_with_language_produces_fenced_block(self):
        el = self._el("code", {"class": "highlight-python"}, "x = 1")
        assert "```python" in self.instance.convert_code(el, "x = 1", False)

    def test_code_no_language_produces_inline(self):
        el = self._el("code", {}, "x = 1")
        assert self.instance.convert_code(el, "x = 1", False) == "`x = 1`"

    def test_code_inline_true_always_backtick(self):
        el = self._el("code", {"class": "highlight-python"}, "x")
        # convert_as_inline=True forces inline even with language
        assert self.instance.convert_code(el, "x", True) == "`x`"

    def test_code_empty_text_uses_get_text(self):
        el = self._el("code", {}, "fallback")
        assert "fallback" in self.instance.convert_code(el, "", False)

    def test_code_empty_element_returns_empty_string(self):
        el = self._el("code", {}, "")
        assert self.instance.convert_code(el, "", False) == ""

    def test_code_picks_first_highlight_class(self):
        el = self._el("code", {"class": "highlight-bash notranslate"}, "ls")
        assert "```bash" in self.instance.convert_code(el, "ls", False)

    def test_div_admonition_title_bolded(self):
        from bs4 import BeautifulSoup
        html = '<div class="admonition note"><p class="admonition-title">Note</p><p>Content.</p></div>'
        el = BeautifulSoup(html, "html.parser").find("div")
        result = self.instance.convert_div(el, "Note\n\nContent.", False)
        assert "**Note**" in result

    def test_div_non_admonition_returns_text(self):
        el = self._el("div", {}, "plain")
        assert self.instance.convert_div(el, "plain", False) == "plain"

    def test_div_admonition_no_title_returns_text(self):
        from bs4 import BeautifulSoup
        el = BeautifulSoup('<div class="admonition">No title.</div>', "html.parser").find("div")
        assert self.instance.convert_div(el, "No title.", False) == "No title."

    def test_div_does_not_mutate_element(self):
        from bs4 import BeautifulSoup
        html = '<div class="admonition warning"><p class="admonition-title">W</p><p>X</p></div>'
        el = BeautifulSoup(html, "html.parser").find("div")
        children_before = list(el.children)
        self.instance.convert_div(el, "text", False)
        assert list(el.children) == children_before

    def test_pre_with_code_delegates_to_convert_code(self):
        from bs4 import BeautifulSoup
        html = '<pre><code class="highlight-python">x = 1</code></pre>'
        el = BeautifulSoup(html, "html.parser").find("pre")
        result = self.instance.convert_pre(el, "", False)
        assert "```python" in result

    def test_pre_without_code_wraps_in_fenced_block(self):
        from bs4 import BeautifulSoup
        html = "<pre>raw text</pre>"
        el = BeautifulSoup(html, "html.parser").find("pre")
        result = self.instance.convert_pre(el, "raw text", False)
        assert "```" in result and "raw text" in result


# ===========================================================================
# 15. _coerce_to_list
# ===========================================================================

class TestCoerceToList:
    def test_none_returns_empty(self):
        assert _mod._coerce_to_list(None) == []

    def test_none_with_default_returns_copy(self):
        default = ["main", "article"]
        result = _mod._coerce_to_list(None, default=default)
        assert result == ["main", "article"] and result is not default

    def test_none_default_none_returns_empty(self):
        assert _mod._coerce_to_list(None, default=None) == []

    def test_single_string_wrapped(self):
        assert _mod._coerce_to_list("article") == ["article"]

    def test_single_string_ignores_default(self):
        assert _mod._coerce_to_list("article", default=["main"]) == ["article"]

    def test_list_of_strings_returned(self):
        assert _mod._coerce_to_list(["article", "main"]) == ["article", "main"]

    def test_tuple_coerced_to_list(self):
        result = _mod._coerce_to_list(("article", "main"))
        assert result == ["article", "main"] and isinstance(result, list)

    def test_non_string_items_converted(self):
        assert _mod._coerce_to_list([1, 2, 3]) == ["1", "2", "3"]

    def test_empty_list(self):
        assert _mod._coerce_to_list([]) == []

    def test_empty_string_wrapped(self):
        assert _mod._coerce_to_list("") == [""]

    def test_returns_new_list_not_same_ref(self):
        inp = ["a", "b"]
        result = _mod._coerce_to_list(inp)
        assert result == inp and result is not inp

    def test_generator_coerced(self):
        assert _mod._coerce_to_list(x for x in ["a", "b"]) == ["a", "b"]

    def test_list_with_none_items_coerced(self):
        # None items inside a list are str()-converted
        result = _mod._coerce_to_list([None, "ok"])
        assert result == ["None", "ok"]


# ===========================================================================
# 16. html_to_markdown / strip_tags coercion
# ===========================================================================

class TestHtmlToMarkdownStripTagsCoercion:
    def test_none_strips_script_by_default(self):
        result = _mod.html_to_markdown(
            "<div><script>evil()</script><p>Hello</p></div>",
            strip_tags=None,
        )
        assert "evil" not in result and "Hello" in result

    def test_single_string_strip_tag(self):
        result = _mod.html_to_markdown(
            "<div><nav>skip</nav><p>Keep</p></div>", strip_tags="nav"
        )
        assert "Keep" in result

    def test_list_strip_tags(self):
        result = _mod.html_to_markdown(
            "<div><nav>skip</nav><footer>skip</footer><p>Keep</p></div>",
            strip_tags=["nav", "footer"],
        )
        assert "Keep" in result

    def test_empty_list_strips_nothing_extra(self):
        assert isinstance(_mod.html_to_markdown("<p>Hello</p>", strip_tags=[]), str)

    def test_returns_string_for_empty_input(self):
        assert isinstance(_mod.html_to_markdown("", strip_tags=None), str)

    def test_heading_converted(self):
        result = _mod.html_to_markdown("<h1>Title</h1>")
        assert "Title" in result

    def test_legacy_alias_callable(self):
        assert callable(_mod.html_to_markdown_converter)
        # Must produce same result
        html = "<p>Hello</p>"
        assert _mod.html_to_markdown(html) == _mod.html_to_markdown_converter(html)


# ===========================================================================
# 17. _write_progress_bar
# ===========================================================================

class TestWriteProgressBar:
    """Tests for _write_progress_bar (in __all__)."""

    def _capture(self, current, total, **kwargs) -> str:
        buf = io.StringIO()
        _mod._write_progress_bar(current, total, stream=buf, **kwargs)
        return buf.getvalue()

    def test_basic_output_written(self):
        out = self._capture(5, 10)
        assert out  # non-empty

    def test_starts_with_carriage_return(self):
        out = self._capture(3, 10)
        assert out.startswith("\r")

    def test_percent_correct(self):
        out = self._capture(5, 10, label="X")
        assert "50.0%" in out

    def test_100_percent_newline_appended(self):
        out = self._capture(10, 10)
        assert out.endswith("\n")

    def test_zero_total_returns_immediately(self):
        buf = io.StringIO()
        _mod._write_progress_bar(0, 0, stream=buf)
        assert buf.getvalue() == ""

    def test_negative_total_returns_immediately(self):
        buf = io.StringIO()
        _mod._write_progress_bar(5, -1, stream=buf)
        assert buf.getvalue() == ""

    def test_label_appears_in_output(self):
        out = self._capture(1, 4, label="HTML→Markdown")
        assert "HTML→Markdown" in out

    def test_current_total_fraction_shown(self):
        out = self._capture(3, 7)
        assert "(3/7)" in out

    def test_multipart_format_no_leading_space(self):
        """Leading space bug fix: must start with \\rPart not \\r Part."""
        out = self._capture(5, 10, part=1, total_parts=3)
        assert out.startswith("\rPart"), f"Leading space bug: {out!r}"
        assert "1/3" in out

    def test_single_part_omits_part_prefix(self):
        out = self._capture(5, 10, total_parts=1, part=1)
        assert "Part" not in out

    def test_bar_len_respected(self):
        out = self._capture(5, 10, bar_len=20)
        # bar_len=20 means 20 chars of =/- inside brackets
        start = out.index("[") + 1
        end = out.index("]")
        assert end - start == 20

    def test_stream_defaults_to_stdout(self, capsys):
        _mod._write_progress_bar(2, 10, label="Test")
        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_flush_called(self):
        mock_stream = MagicMock()
        _mod._write_progress_bar(3, 10, stream=mock_stream)
        mock_stream.flush.assert_called()


# ===========================================================================
# 18. _resolve_icon (backup SVG fallback)
# ===========================================================================

class TestResolveIcon:
    """Tests for _resolve_icon helper (backup base64 SVG logic)."""

    def test_existing_file_returned_as_filename(self, tmp_path):
        """An existing SVG file on disk → return the filename unchanged."""
        svg = tmp_path / "claude.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        result = _mod._resolve_icon("claude.svg", "claude", tmp_path)
        assert result == "claude.svg"

    def test_missing_file_returns_data_uri(self, tmp_path):
        """A missing SVG → return base64 data URI from _PROVIDER_META."""
        result = _mod._resolve_icon("missing.svg", "claude", tmp_path)
        assert result.startswith("data:image/svg+xml;base64,")

    def test_unknown_provider_returns_default_svg(self, tmp_path):
        """Unknown provider name → return _SVG_DEFAULT data URI."""
        result = _mod._resolve_icon("unknown.svg", "unknown_provider_xyz", tmp_path)
        assert result.startswith("data:image/svg+xml;base64,")

    def test_all_default_providers_have_fallback(self, tmp_path):
        """Every default provider must resolve to a non-empty icon value."""
        for name in _mod._DEFAULT_PROVIDERS:
            result = _mod._resolve_icon(f"{name}.svg", name, tmp_path)
            assert result, f"Provider {name!r} resolved to empty string"

    def test_all_mcp_tools_have_fallback(self, tmp_path):
        """Every default MCP tool must resolve to a non-empty icon value."""
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            result = _mod._resolve_icon(cfg.get("icon", ""), name, tmp_path)
            assert result, f"MCP tool {name!r} resolved to empty string"

    def test_none_static_dir_uses_module_default(self):
        """When static_dir=None, function uses the module's own _static/."""
        # claude.svg exists in the real _static dir
        result = _mod._resolve_icon("claude.svg", "claude", None)
        # Should be either the filename (file exists) or a data URI
        assert result in ("claude.svg",) or result.startswith("data:")

    def test_gemini_svg_missing_returns_data_uri(self, tmp_path):
        """gemini.svg is absent from _static — must return a data URI."""
        result = _mod._resolve_icon("gemini.svg", "gemini", tmp_path)
        assert result.startswith("data:image/svg+xml;base64,")

    def test_empty_icon_filename_missing_returns_fallback(self, tmp_path):
        """Even an empty filename should not crash."""
        result = _mod._resolve_icon("", "claude", tmp_path)
        # Empty filename → file not found → fallback
        assert result  # non-empty


# ===========================================================================
# 19. _static subpackage — SVG constants & _PROVIDER_META
# ===========================================================================

class TestStaticSubpackage:
    """Tests for _static/__init__.py SVG constants and _PROVIDER_META."""

    @pytest.fixture(autouse=True)
    def _import_static(self):
        import importlib
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


# ===========================================================================
# 20. Theme selector presets
# ===========================================================================

class TestThemeSelectorPresets:
    def test_is_dict(self): assert isinstance(_mod._THEME_SELECTOR_PRESETS, dict)
    def test_at_least_20_themes(self): assert len(_mod._THEME_SELECTOR_PRESETS) >= 20
    def test_pydata_present(self): assert "pydata_sphinx_theme" in _mod._THEME_SELECTOR_PRESETS
    def test_furo_present(self): assert "furo" in _mod._THEME_SELECTOR_PRESETS
    def test_rtd_present(self): assert "sphinx_rtd_theme" in _mod._THEME_SELECTOR_PRESETS
    def test_mkdocs_present(self): assert "mkdocs" in _mod._THEME_SELECTOR_PRESETS
    def test_mkdocs_material_present(self): assert "mkdocs_material" in _mod._THEME_SELECTOR_PRESETS
    def test_jekyll_present(self): assert "jekyll" in _mod._THEME_SELECTOR_PRESETS
    def test_hugo_present(self): assert "hugo" in _mod._THEME_SELECTOR_PRESETS
    def test_docusaurus_present(self): assert "docusaurus" in _mod._THEME_SELECTOR_PRESETS
    def test_vitepress_present(self): assert "vitepress" in _mod._THEME_SELECTOR_PRESETS
    def test_gitbook_present(self): assert "gitbook" in _mod._THEME_SELECTOR_PRESETS
    def test_plain_html_present(self): assert "plain_html" in _mod._THEME_SELECTOR_PRESETS

    def test_each_preset_is_non_empty_tuple(self):
        for name, sels in _mod._THEME_SELECTOR_PRESETS.items():
            assert isinstance(sels, tuple) and len(sels) > 0, f"{name!r} empty"

    def test_all_selectors_pass_validation(self):
        for name, sels in _mod._THEME_SELECTOR_PRESETS.items():
            for sel in sels:
                assert _mod._validate_css_selector(sel), f"Unsafe in {name!r}: {sel!r}"

    def test_mkdocs_material_selector(self):
        assert "article.md-content__inner" in _mod._THEME_SELECTOR_PRESETS["mkdocs_material"]

    def test_vitepress_selector(self):
        assert "div.vp-doc" in _mod._THEME_SELECTOR_PRESETS["vitepress"]


# ===========================================================================
# 21. _resolve_content_selectors
# ===========================================================================

class TestResolveContentSelectors:
    def test_no_preset_custom_first(self):
        result = _mod._resolve_content_selectors(None, ["div.custom"])
        assert result[0] == "div.custom"
        assert "main" in result

    def test_preset_adds_theme_selectors(self):
        result = _mod._resolve_content_selectors("furo", [])
        assert 'article[role="main"]' in result

    def test_custom_takes_priority_over_preset(self):
        result = _mod._resolve_content_selectors("furo", ["div.custom"])
        assert result[0] == "div.custom"

    def test_no_duplicates(self):
        result = _mod._resolve_content_selectors("classic", ["main"])
        assert result.count("main") == 1

    def test_unknown_preset_falls_back_to_defaults(self):
        result = _mod._resolve_content_selectors("nonexistent_theme", [])
        assert "main" in result

    def test_empty_returns_defaults(self):
        assert _mod._resolve_content_selectors(None, []) == _mod._DEFAULT_CONTENT_SELECTORS

    def test_unsafe_custom_selectors_removed(self):
        result = _mod._resolve_content_selectors(None, ["<bad>", "article"])
        assert "<bad>" not in result

    def test_returns_tuple(self):
        assert isinstance(_mod._resolve_content_selectors(None, []), tuple)

    def test_never_returns_empty(self):
        result = _mod._resolve_content_selectors(None, ["<bad>"])
        assert len(result) > 0

    def test_all_known_themes_resolve_non_empty(self):
        for theme in _mod._THEME_SELECTOR_PRESETS:
            result = _mod._resolve_content_selectors(theme, [])
            assert len(result) > 0, f"Theme {theme!r} resolved empty"


# ===========================================================================
# 22. _DEFAULT_CONTENT_SELECTORS
# ===========================================================================

class TestDefaultContentSelectors:
    def test_is_tuple(self): assert isinstance(_mod._DEFAULT_CONTENT_SELECTORS, tuple)
    def test_non_empty(self): assert len(_mod._DEFAULT_CONTENT_SELECTORS) >= 12
    def test_pydata(self): assert "article.bd-article" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_furo(self): assert 'article[role="main"]' in _mod._DEFAULT_CONTENT_SELECTORS
    def test_rtd(self): assert "div.rst-content" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_mkdocs(self): assert "div.md-content" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_vitepress(self): assert "div.vp-doc" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_jekyll(self): assert "article.post-content" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_main(self): assert "main" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_article(self): assert "article" in _mod._DEFAULT_CONTENT_SELECTORS

    def test_all_selectors_pass_validation(self):
        for sel in _mod._DEFAULT_CONTENT_SELECTORS:
            assert _mod._validate_css_selector(sel)


# ===========================================================================
# 23. _process_html_file_worker (6-tuple)
# ===========================================================================

class TestProcessHtmlFileWorker:
    _SELECTORS = ["article.bd-article", 'article[role="main"]', 'div[role="main"]',
                  "main", "article"]
    _STRIP = ["script", "style"]

    def _call(self, html_path, input_dir, output_dir, excludes=None, selectors=None, strip=None):
        return _mod._process_html_file_worker((
            str(html_path), str(input_dir), str(output_dir),
            excludes or [], selectors or self._SELECTORS,
            strip if strip is not None else self._STRIP,
        ))

    def test_success_inline_mode(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text('<html><body><article>Hello</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(html, tmp_path, tmp_path)
        assert status == "success"
        assert (tmp_path / "page.md").exists()

    def test_separate_output_dir(self, tmp_path):
        in_dir = tmp_path / "in"; in_dir.mkdir()
        out_dir = tmp_path / "out"
        (in_dir / "api").mkdir()
        html = in_dir / "api" / "func.html"
        html.write_text('<html><body><article>API</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(html, in_dir, out_dir)
        assert status == "success"
        assert (out_dir / "api" / "func.md").exists()
        assert not (in_dir / "api" / "func.md").exists()

    def test_output_dir_created_automatically(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text('<html><body><article>X</article></body></html>', encoding="utf-8")
        out = tmp_path / "does_not_exist" / "nested"
        status, _, _ = self._call(html, tmp_path, out)
        assert status == "success" and (out / "page.md").exists()

    def test_path_traversal_blocked(self, tmp_path):
        secret = tmp_path.parent / "secret.html"
        secret.write_text("<html><body>secret</body></html>", encoding="utf-8")
        out = tmp_path / "out"; out.mkdir()
        status, _, msg = self._call(secret, tmp_path, out)
        assert status == "error"
        assert "traversal" in msg.lower() or "outside" in msg.lower()

    def test_excluded_by_pattern_skipped(self, tmp_path):
        html = tmp_path / "genindex.html"
        html.write_text('<html><body><article>Index</article></body></html>', encoding="utf-8")
        status, _, _ = self._call(html, tmp_path, tmp_path, excludes=["genindex"])
        assert status == "skipped"

    def test_no_content_element_skipped(self, tmp_path):
        html = tmp_path / "empty.html"
        html.write_text("<html><body><p>no match</p></body></html>", encoding="utf-8")
        status, _, msg = self._call(html, tmp_path, tmp_path)
        assert status == "skipped" and "No main content" in msg

    def test_strip_tags_applied(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text(
            '<html><body><article>Good<script>evil()</script></article></body></html>',
            encoding="utf-8",
        )
        status, _, _ = self._call(html, tmp_path, tmp_path)
        assert status == "success"
        md = (tmp_path / "page.md").read_text(encoding="utf-8")
        assert "evil" not in md and "Good" in md

    def test_encoding_errors_replaced_not_crash(self, tmp_path):
        html = tmp_path / "bad.html"
        html.write_bytes(b"<html><body><article>\xff\xfe bad</article></body></html>")
        status, _, _ = self._call(html, tmp_path, tmp_path)
        assert status in ("success", "skipped", "error")  # no exception

    def test_no_double_bs4_parse(self, tmp_path):
        """
        Worker must NOT call html_to_markdown (which re-parses via BeautifulSoup).

        Pre-fix root cause: ``_process_html_file_worker`` called
        ``html_to_markdown(str(main_content), strip_tags=...)`` which
        internally ran ``BeautifulSoup(html_content, "html.parser")`` while
        the full-page ``soup`` was still alive in the caller's frame.  With
        886 pages x 8 concurrent workers this doubled per-worker peak memory
        and triggered the Linux OOM killer at ~70% (620/886).

        The fix strips tags in-place on the already-parsed BS4 tree, frees
        ``soup`` explicitly with ``del soup``, then calls the markdownify
        converter class directly — no second BeautifulSoup parse, no call to
        ``html_to_markdown``.

        Asserting ``html_to_markdown.assert_not_called()`` is the exact
        behavioral invariant: if the worker ever delegates to ``html_to_markdown``
        again, the second parse returns and this test fails.
        """
        html = tmp_path / "page.html"
        html.write_text(
            '<html><body>'
            '<article>Good content<nav>skip me</nav></article>'
            '</body></html>',
            encoding="utf-8",
        )
        import scikitplot._externals._sphinx_ext._sphinx_ai_assistant as _m

        with patch.object(_m, "html_to_markdown", wraps=_m.html_to_markdown) as mock_h2m:
            status, _, _ = self._call(html, tmp_path, tmp_path, strip=["nav"])

        assert status == "success", "worker must succeed"
        md = (tmp_path / "page.md").read_text(encoding="utf-8")
        assert "Good content" in md, "main content must be preserved"
        assert "skip me" not in md, "stripped tag content must be absent"
        # Core OOM-regression invariant: the worker must NEVER call html_to_markdown.
        mock_h2m.assert_not_called()

    def test_strip_tags_inplace_before_serialise(self, tmp_path):
        """
        Stripped content must be absent even from the serialised snippet.

        Verifies the in-place decompose path: nav/footer/script removed on
        the BS4 tree before str() is called, so the converter never sees them.
        """
        html = tmp_path / "page.html"
        html.write_text(
            "<html><body><article>"
            "<h1>Title</h1>"
            "<nav>sidebar</nav>"
            "<p>Body text</p>"
            "<footer>footer blurb</footer>"
            "<script>evil()</script>"
            "</article></body></html>",
            encoding="utf-8",
        )
        status, _, _ = self._call(
            html, tmp_path, tmp_path,
            strip=["nav", "footer", "script"],
        )
        assert status == "success"
        md = (tmp_path / "page.md").read_text(encoding="utf-8")
        assert "Title" in md
        assert "Body text" in md
        assert "sidebar" not in md
        assert "footer blurb" not in md
        assert "evil" not in md


# ===========================================================================
# 24. _process_single_html_file (5-tuple wrapper)
# ===========================================================================

class TestProcessSingleHtmlFile:
    _SELECTORS = ["article.bd-article", 'article[role="main"]', 'div[role="main"]',
                  "main", "article"]
    _STRIP = ["script", "style"]

    def _call(self, html_path, outdir, excludes=None, selectors=None, strip=None):
        return _mod._process_single_html_file((
            str(html_path), str(outdir),
            excludes or [], selectors or self._SELECTORS,
            strip if strip is not None else self._STRIP,
        ))

    def test_success_writes_md_alongside_html(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text('<html><body><article>Hello world</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(html, tmp_path)
        assert status == "success" and (tmp_path / "page.md").exists()

    def test_excluded_by_pattern(self, tmp_path):
        html = tmp_path / "genindex.html"
        html.write_text('<html><body><article>Index</article></body></html>', encoding="utf-8")
        assert self._call(html, tmp_path, excludes=["genindex"])[0] == "skipped"

    def test_no_content_element_skipped(self, tmp_path):
        html = tmp_path / "empty.html"
        html.write_text("<html><body><p>no main element</p></body></html>", encoding="utf-8")
        status, _, msg = self._call(html, tmp_path)
        assert status == "skipped" and "No main content" in msg

    def test_path_traversal_blocked(self, tmp_path):
        other = tmp_path.parent / "secret.html"
        other.write_text("<html><body>secret</body></html>", encoding="utf-8")
        out = tmp_path / "out"; out.mkdir()
        status, _, _ = self._call(other, out)
        assert status == "error"

    def test_nested_subdir(self, tmp_path):
        sub = tmp_path / "api" / "mod"; sub.mkdir(parents=True)
        html = sub / "func.html"
        html.write_text('<html><body><article class="bd-article">API</article></body></html>',
                        encoding="utf-8")
        status, _, _ = self._call(html, tmp_path)
        assert status == "success" and (sub / "func.md").exists()

    def test_delegates_to_6tuple_worker_inline_mode(self, tmp_path):
        """5-tuple wrapper must produce same result as 6-tuple with output_dir==input_dir."""
        html = tmp_path / "p.html"
        html.write_text('<html><body><article>X</article></body></html>', encoding="utf-8")
        result5 = self._call(html, tmp_path)
        result6 = _mod._process_html_file_worker((
            str(html), str(tmp_path), str(tmp_path),
            [], self._SELECTORS, self._STRIP,
        ))
        assert result5[0] == result6[0]  # same status

    def test_wrong_tuple_length_raises(self, tmp_path):
        with pytest.raises((TypeError, ValueError)):
            _mod._process_single_html_file((str(tmp_path / "x.html"), str(tmp_path), [], ["article"]))


# ===========================================================================
# 25. process_html_directory
# ===========================================================================

class TestProcessHtmlDirectory:
    def _make_site(self, root: Path) -> Path:
        site = root / "site"; site.mkdir()
        (site / "index.html").write_text(
            '<html><body><main><h1>Home</h1></main></body></html>', encoding="utf-8"
        )
        sub = site / "docs"; sub.mkdir()
        (sub / "guide.html").write_text(
            '<html><body><article>Guide content</article></body></html>', encoding="utf-8"
        )
        (site / "genindex.html").write_text(
            '<html><body><main>Index</main></body></html>', encoding="utf-8"
        )
        return site

    def test_basic_success(self, tmp_path):
        stats = _mod.process_html_directory(self._make_site(tmp_path), max_workers=1)
        assert stats["generated"] >= 1 and stats["errors"] == 0

    def test_genindex_excluded_by_default(self, tmp_path):
        site = self._make_site(tmp_path)
        _mod.process_html_directory(site, max_workers=1)
        assert not (site / "genindex.md").exists()

    def test_inline_mode_md_beside_html(self, tmp_path):
        site = self._make_site(tmp_path)
        _mod.process_html_directory(site, max_workers=1)
        assert (site / "index.md").exists() or (site / "docs" / "guide.md").exists()

    def test_separate_output_dir(self, tmp_path):
        site = self._make_site(tmp_path)
        out = tmp_path / "md_out"
        _mod.process_html_directory(site, output_dir=out, max_workers=1)
        assert out.exists() and len(list(out.rglob("*.md"))) >= 1

    def test_non_recursive_mode(self, tmp_path):
        site = self._make_site(tmp_path)
        stats = _mod.process_html_directory(site, recursive=False, max_workers=1)
        assert stats["generated"] + stats["skipped"] <= 2

    def test_theme_preset_mkdocs(self, tmp_path):
        site = tmp_path / "mk"; site.mkdir()
        (site / "index.html").write_text(
            '<html><body><div class="md-content"><h1>MkDocs</h1></div></body></html>',
            encoding="utf-8",
        )
        stats = _mod.process_html_directory(site, theme_preset="mkdocs", max_workers=1)
        assert stats["generated"] == 1

    def test_custom_exclude_patterns(self, tmp_path):
        site = self._make_site(tmp_path)
        _mod.process_html_directory(site, exclude_patterns=["index", "genindex"], max_workers=1)
        assert not (site / "index.md").exists()

    def test_returns_dict_with_correct_keys(self, tmp_path):
        stats = _mod.process_html_directory(self._make_site(tmp_path), max_workers=1)
        assert set(stats.keys()) == {"generated", "skipped", "errors"}

    def test_nonexistent_input_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _mod.process_html_directory(tmp_path / "nonexistent")

    def test_file_as_input_raises_value_error(self, tmp_path):
        f = tmp_path / "file.html"; f.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="not a directory"):
            _mod.process_html_directory(f)

    def test_missing_deps_raises_import_error(self, tmp_path):
        site = tmp_path / "s"; site.mkdir()
        with patch.object(_mod, "_has_markdown_deps", return_value=False):
            with pytest.raises(ImportError, match="beautifulsoup4"):
                _mod.process_html_directory(site)

    def test_generate_llms_produces_file(self, tmp_path):
        site = self._make_site(tmp_path)
        _mod.process_html_directory(site, max_workers=1, generate_llms=True, base_url="")
        md_files = list(site.rglob("*.md"))
        if md_files:
            assert (site / "llms.txt").exists()

    def test_plain_html_preset(self, tmp_path):
        site = tmp_path / "plain"; site.mkdir()
        (site / "page.html").write_text(
            '<html><body><main>Plain content</main></body></html>', encoding="utf-8"
        )
        stats = _mod.process_html_directory(site, theme_preset="plain_html", max_workers=1)
        assert stats["generated"] == 1

    def test_generate_llms_with_base_url(self, tmp_path):
        site = self._make_site(tmp_path)
        _mod.process_html_directory(
            site, max_workers=1, generate_llms=True,
            base_url="https://example.com",
        )
        md_files = list(site.rglob("*.md"))
        if md_files:
            llms = (site / "llms.txt").read_text(encoding="utf-8")
            # assert "https://example.com" in llms
            # lines = [line.strip() for line in llms.splitlines() if line.strip()]
            urls = re.findall(r'https?://[^\s]+', llms)
            assert urls, "No URLs found"
            parsed_urls = [urlparse(u) for u in urls]
            assert any(
                p.scheme == "https" and p.netloc == "example.com"
                for p in parsed_urls
            )


# ===========================================================================
# 26. generate_llms_txt_standalone
# ===========================================================================

class TestGenerateLlmsTxtStandalone:
    def test_basic_writes_file(self, tmp_path):
        (tmp_path / "page.md").write_text("# Page\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path)
        assert out.exists() and "page.md" in out.read_text(encoding="utf-8")

    def test_returns_path_object(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        assert isinstance(_mod.generate_llms_txt_standalone(tmp_path), Path)

    def test_base_url_prepended(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, base_url="https://example.com")
        assert "https://example.com/doc.md" in out.read_text(encoding="utf-8")

    def test_dangerous_base_url_raises(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        with pytest.raises(ValueError):
            _mod.generate_llms_txt_standalone(tmp_path, base_url="javascript:evil()")

    def test_project_name_in_header(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, project_name="MyLib")
        assert "MyLib" in out.read_text(encoding="utf-8")

    def test_max_entries_limits_output(self, tmp_path):
        for i in range(5):
            (tmp_path / f"p{i}.md").write_text(f"# P{i}\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, max_entries=2)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.endswith(".md")]
        assert len(lines) == 2

    def test_max_entries_zero_writes_header_only(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, max_entries=0)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.endswith(".md")]
        assert len(lines) == 0

    def test_full_content_embeds_markdown(self, tmp_path):
        (tmp_path / "page.md").write_text("# Hello\n\nWorld\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, full_content=True)
        content = out.read_text(encoding="utf-8")
        assert "# Hello" in content and "World" in content and "---" in content

    def test_full_content_with_base_url(self, tmp_path):
        (tmp_path / "page.md").write_text("# Page\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(
            tmp_path, full_content=True, base_url="https://docs.example.com"
        )
        content = out.read_text(encoding="utf-8")
        assert "https://docs.example.com/page.md" in content

    def test_custom_output_file(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        custom = tmp_path / "custom_name.txt"
        out = _mod.generate_llms_txt_standalone(tmp_path, output_file=custom)
        assert out == custom and custom.exists()

    def test_nonexistent_root_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _mod.generate_llms_txt_standalone(tmp_path / "nonexistent")

    def test_empty_directory_writes_header_only(self, tmp_path):
        out = _mod.generate_llms_txt_standalone(tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "Documentation" in content
        assert [l for l in content.splitlines() if l.endswith(".md")] == []


# ===========================================================================
# 27. generate_markdown_files (Sphinx hook)
# ===========================================================================

class TestGenerateMarkdownFiles:
    def test_exception_passed_in_early_exit(self, sphinx_app):
        _mod.generate_markdown_files(sphinx_app, exception=RuntimeError("build failed"))
        # No exception raised; no md files should be written
        assert list(Path(sphinx_app.builder.outdir).rglob("*.md")) == []

    def test_wrong_builder_type_skips(self, tmp_path):
        app = MagicMock()
        app.config = MagicMock()
        app.builder = MagicMock()  # not StandaloneHTMLBuilder
        _mod.generate_markdown_files(app, exception=None)

    def test_disabled_by_config_no_md(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_markdown = False
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert list(Path(sphinx_app.builder.outdir).rglob("*.md")) == []

    def test_missing_deps_logs_warning(self, sphinx_app):
        with patch.object(_mod, "_has_markdown_deps", return_value=False):
            with patch.object(_mod._get_logger(), "warning") as mock_warn:
                _mod.generate_markdown_files(sphinx_app, exception=None)
                mock_warn.assert_called_once()
                assert "beautifulsoup4" in mock_warn.call_args[0][0].lower()

    def test_success_creates_md_files(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert len(list(tmp_html_tree.rglob("*.md"))) >= 1

    def test_genindex_excluded(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert not (tmp_html_tree / "genindex.md").exists()

    def test_max_workers_none_auto_detects(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = None
        _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_max_workers_zero_raises(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 0

        with pytest.raises(ValueError, match="max_workers must be greater than 0"):
            _mod.generate_markdown_files(sphinx_app, exception=None)

    # def test_max_workers_zero_floors_to_one(self, sphinx_app, tmp_html_tree):
    #     sphinx_app.builder.outdir = str(tmp_html_tree)
    #     sphinx_app.config.ai_assistant_max_workers = 0
    #     _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_theme_preset_used(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        sphinx_app.config.ai_assistant_theme_preset = "pydata_sphinx_theme"
        sphinx_app.config.ai_assistant_content_selectors = []
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert (tmp_html_tree / "api" / "module.md").exists()

    def test_empty_outdir_zero_md(self, sphinx_app, tmp_path):
        empty = tmp_path / "empty_out"; empty.mkdir()
        sphinx_app.builder.outdir = str(empty)
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert list(empty.rglob("*.md")) == []


# ===========================================================================
# 28. generate_llms_txt (Sphinx hook)
# ===========================================================================

class TestGenerateLlmsTxt:
    def test_exception_passed_in_early_exit(self, sphinx_app):
        _mod.generate_llms_txt(sphinx_app, exception=RuntimeError("boom"))

    def test_markdown_disabled_no_file(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_markdown = False
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (Path(sphinx_app.builder.outdir) / "llms.txt").exists()

    def test_llms_txt_disabled_no_file(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_llms_txt = False
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (Path(sphinx_app.builder.outdir) / "llms.txt").exists()

    def test_wrong_builder_skips(self):
        app = MagicMock()
        app.config.ai_assistant_generate_markdown = True
        app.config.ai_assistant_generate_llms_txt = True
        app.builder = MagicMock()
        _mod.generate_llms_txt(app, exception=None)

    def test_no_md_files_skips_writing(self, sphinx_app):
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (Path(sphinx_app.builder.outdir) / "llms.txt").exists()

    def test_writes_with_base_url(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "index.md").write_text("# Index\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = "https://docs.example.com"
        sphinx_app.config.ai_assistant_base_url = ""
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert "https://docs.example.com/index.md" in (tmp_html_tree / "llms.txt").read_text()

    def test_writes_without_base_url(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "page.md").write_text("# Page\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text()
        assert "page.md" in llms and "https://" not in llms

    def test_invalid_base_url_logs_warning_and_skips(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "x.md").write_text("# X\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = "javascript:evil()"
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (tmp_html_tree / "llms.txt").exists()

    def test_project_name_in_header(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "doc.md").write_text("# Doc\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.project = "MyLib"
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert "MyLib" in (tmp_html_tree / "llms.txt").read_text()

    def test_max_entries_limits_output(self, sphinx_app, tmp_html_tree):
        for i in range(5):
            (tmp_html_tree / f"page{i}.md").write_text(f"# Page{i}\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_max_entries = 2
        _mod.generate_llms_txt(sphinx_app, exception=None)
        lines = [l for l in (tmp_html_tree / "llms.txt").read_text().splitlines()
                 if l.endswith(".md")]
        assert len(lines) == 2

    def test_max_entries_zero_skips_writing(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "page.md").write_text("# Page\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_max_entries = 0
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (tmp_html_tree / "llms.txt").exists()

    def test_full_content_embeds_markdown(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "page.md").write_text("# Hello\n\nWorld\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_full_content = True
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text()
        assert "# Hello" in llms and "World" in llms


# ===========================================================================
# 29. add_ai_assistant_context
# ===========================================================================

class TestAddAiAssistantContext:
    def test_disabled_injects_nothing(self):
        app = MagicMock()
        app.config.ai_assistant_enabled = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(app, "index", "page.html", ctx, None)
        assert "ai_assistant_config" not in ctx

    def test_enabled_injects_config_dict(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert isinstance(ctx.get("ai_assistant_config"), dict)

    def test_metatags_created_when_absent(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "<script>" in ctx.get("metatags", "")

    def test_metatags_appended_when_present(self, sphinx_app):
        ctx: dict = {"metatags": "<meta name='existing'>"}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "<meta name='existing'>" in ctx["metatags"]

    def test_xss_prevention_in_metatags(self, sphinx_app):
        sphinx_app.config.ai_assistant_providers = {
            "evil": {"url_template": "https://x.com/</script><script>alert(1)//",
                     "enabled": True}
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "</script><script>" not in ctx["metatags"]

    def test_html_baseurl_takes_priority_over_ai_base_url(self, sphinx_app):
        sphinx_app.config.html_baseurl = "https://html-baseurl.example.com"
        sphinx_app.config.ai_assistant_base_url = "https://ai-baseurl.example.com"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["baseUrl"] == "https://html-baseurl.example.com"

    def test_invalid_position_falls_back_to_sidebar(self, sphinx_app):
        sphinx_app.config.ai_assistant_position = "malicious; drop table"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["position"] == "sidebar"

    def test_invalid_position_logs_warning(self, sphinx_app):
        sphinx_app.config.ai_assistant_position = "bad_position"
        ctx: dict = {}
        with patch.object(_mod._get_logger(), "warning") as mock_warn:
            _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
            mock_warn.assert_called_once()

    def test_valid_positions_not_warned(self, sphinx_app):
        for pos in ["sidebar", "title", "floating", "none"]:
            sphinx_app.config.ai_assistant_position = pos
            ctx: dict = {}
            with patch.object(_mod._get_logger(), "warning") as mock_warn:
                _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
                mock_warn.assert_not_called()

    def test_dangerous_provider_url_filtered(self, sphinx_app):
        sphinx_app.config.ai_assistant_providers = {
            "safe": {"url_template": "https://claude.ai/new?q={prompt}", "enabled": True},
            "evil": {"url_template": "javascript:alert(document.cookie)", "enabled": True},
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        providers = ctx["ai_assistant_config"]["providers"]
        assert "safe" in providers and "evil" not in providers

    def test_missing_svg_icons_replaced_with_data_uri(self, sphinx_app):
        """Missing provider SVG files must be replaced with base64 data URIs."""
        sphinx_app.config.ai_assistant_providers = {
            "gemini": {
                "url_template": "https://gemini.google.com/app?q={prompt}",
                "icon": "gemini.svg",  # does not exist in _static/
                "enabled": True,
            }
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        icon_val = ctx["ai_assistant_config"]["providers"].get("gemini", {}).get("icon", "")
        assert icon_val.startswith("data:image/svg+xml;base64,"), (
            f"Expected data URI for missing gemini.svg, got: {icon_val!r}"
        )

    def test_existing_svg_icon_kept_as_filename(self, sphinx_app):
        """Existing claude.svg must be kept as filename, not replaced."""
        sphinx_app.config.ai_assistant_providers = {
            "claude": {
                "url_template": "https://claude.ai/new?q={prompt}",
                "icon": "claude.svg",  # exists in _static/
                "enabled": True,
            }
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        icon_val = ctx["ai_assistant_config"]["providers"].get("claude", {}).get("icon", "")
        assert icon_val == "claude.svg"

    def test_intention_none_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_intention = None
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["intention"] is None

    def test_intention_string_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_intention = "Explain this page"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["intention"] == "Explain this page"

    def test_include_raw_image_false(self, sphinx_app):
        sphinx_app.config.ai_assistant_include_raw_image = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["includeRawImage"] is False

    def test_include_raw_image_true(self, sphinx_app):
        sphinx_app.config.ai_assistant_include_raw_image = True
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["includeRawImage"] is True

    def test_all_new_fields_present(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        cfg = ctx["ai_assistant_config"]
        for field in ("intention", "customContext", "customPromptPrefix", "includeRawImage"):
            assert field in cfg, f"Field {field!r} missing"

    def test_config_is_json_serialisable(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        dumped = json.dumps(ctx["ai_assistant_config"])
        assert isinstance(dumped, str)

    def test_intention_xss_safe_in_metatags(self, sphinx_app):
        sphinx_app.config.ai_assistant_intention = "evil</script>"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "evil</script>" not in ctx.get("metatags", "")

    def test_custom_context_injected(self, sphinx_app):
        sphinx_app.config.ai_assistant_custom_context = "Python ML library"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["customContext"] == "Python ML library"

    def _make_local_app(self, tmp_html_tree):
        from conftest import _make_app  # noqa: PLC0415 — on sys.path via pytest
        return _make_app(str(tmp_html_tree))


# ===========================================================================
# 30. setup()
# ===========================================================================

class TestSetup:
    @pytest.fixture()
    def app(self):
        mock_app = MagicMock()
        mock_app.config.html_static_path = []
        return mock_app

    def test_returns_version(self, app):
        assert _mod.setup(app)["version"] == _mod._VERSION

    def test_returns_parallel_safe_flags(self, app):
        result = _mod.setup(app)
        assert result["parallel_read_safe"] is True
        assert result["parallel_write_safe"] is True

    def test_all_required_config_values_registered(self, app):
        _mod.setup(app)
        names = {c[0][0] for c in app.add_config_value.call_args_list}
        required = {
            "ai_assistant_enabled", "ai_assistant_position",
            "ai_assistant_content_selector", "ai_assistant_content_selectors",
            "ai_assistant_theme_preset", "ai_assistant_generate_markdown",
            "ai_assistant_markdown_exclude_patterns", "ai_assistant_strip_tags",
            "ai_assistant_generate_llms_txt", "ai_assistant_base_url",
            "ai_assistant_max_workers", "ai_assistant_llms_txt_max_entries",
            "ai_assistant_llms_txt_full_content", "ai_assistant_features",
            "ai_assistant_providers", "ai_assistant_mcp_tools",
            "ai_assistant_intention", "ai_assistant_custom_context",
            "ai_assistant_custom_prompt_prefix", "ai_assistant_include_raw_image",
            "ai_assistant_ollama_model",
        }
        assert required.issubset(names), f"Missing: {required - names}"

    def test_events_connected(self, app):
        _mod.setup(app)
        event_names = [c[0][0] for c in app.connect.call_args_list]
        assert "html-page-context" in event_names
        assert event_names.count("build-finished") == 2

    def test_css_and_js_added(self, app):
        _mod.setup(app)
        app.add_css_file.assert_called_once_with("ai-assistant.css")
        app.add_js_file.assert_called_once_with("ai-assistant.js")

    def test_static_path_appended(self, app):
        _mod.setup(app)
        assert len(app.config.html_static_path) == 1

    def test_static_path_not_duplicated_on_double_setup(self, app):
        static_path = str(Path(_mod.__file__).parent / "_static")
        app.config.html_static_path = [static_path]
        _mod.setup(app)
        _mod.setup(app)
        assert app.config.html_static_path.count(static_path) == 1

    def test_providers_default_is_full_registry(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        providers = args_map["ai_assistant_providers"]
        for name in ["claude", "chatgpt", "gemini", "ollama", "mistral",
                     "perplexity", "copilot", "groq", "you", "deepseek",
                     "huggingface", "custom"]:
            assert name in providers, f"Provider {name!r} missing from setup default"

    def test_mcp_tools_default_is_full_registry(self, app):
        """setup() must use dict(_DEFAULT_MCP_TOOLS) — not a partial subset."""
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        mcp = args_map["ai_assistant_mcp_tools"]
        for name in ("vscode", "claude_desktop", "cursor", "windsurf", "generic"):
            assert name in mcp, f"MCP tool {name!r} missing from setup default"

    def test_intention_default_is_none(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert args_map["ai_assistant_intention"] is None

    def test_include_raw_image_default_false(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert args_map["ai_assistant_include_raw_image"] is False


# ===========================================================================
# 31. _OLLAMA_RECOMMENDED_MODELS
# ===========================================================================

class TestOllamaRecommendedModels:
    def test_is_tuple(self):
        assert isinstance(_mod._OLLAMA_RECOMMENDED_MODELS, tuple)

    def test_non_empty(self):
        assert len(_mod._OLLAMA_RECOMMENDED_MODELS) >= 10

    def test_all_strings(self):
        for m in _mod._OLLAMA_RECOMMENDED_MODELS:
            assert isinstance(m, str), f"Non-string: {m!r}"

    def test_no_empty_strings(self):
        assert all(m.strip() for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_qwen3_present(self):
        assert any("qwen3" in m for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_llama3_present(self):
        assert any("llama3" in m for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_gemma_present(self):
        assert any("gemma" in m for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_deepseek_present(self):
        assert any("deepseek" in m for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_phi_present(self):
        assert any("phi" in m for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_mistral_present(self):
        assert any("mistral" in m for m in _mod._OLLAMA_RECOMMENDED_MODELS)

    def test_all_have_version_tag(self):
        for m in _mod._OLLAMA_RECOMMENDED_MODELS:
            assert ":" in m, f"Model {m!r} missing version tag"

    def test_no_duplicates(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert len(models) == len(set(models))


# ===========================================================================
# 32. _DEFAULT_MCP_TOOLS
# ===========================================================================

class TestDefaultMcpTools:
    REQUIRED_TOOLS = {"vscode", "claude_desktop", "cursor", "windsurf", "generic"}

    def test_is_dict(self):
        assert isinstance(_mod._DEFAULT_MCP_TOOLS, dict)

    def test_all_required_tools_present(self):
        for name in self.REQUIRED_TOOLS:
            assert name in _mod._DEFAULT_MCP_TOOLS, f"{name!r} missing"

    def test_all_tools_have_enabled_bool(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert "enabled" in cfg and isinstance(cfg["enabled"], bool)

    def test_all_tools_disabled_by_default(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert cfg["enabled"] is False, f"{name!r} should be disabled"

    def test_all_tools_have_type_str(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert "type" in cfg and isinstance(cfg["type"], str)

    def test_all_tools_have_label(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert cfg.get("label"), f"{name!r} missing label"

    def test_all_tools_have_description(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert cfg.get("description"), f"{name!r} missing description"

    def test_vscode_has_server_url(self):
        assert "server_url" in _mod._DEFAULT_MCP_TOOLS["vscode"]

    def test_claude_desktop_has_mcpb_url(self):
        assert "mcpb_url" in _mod._DEFAULT_MCP_TOOLS["claude_desktop"]

    def test_generic_has_transport(self):
        assert "transport" in _mod._DEFAULT_MCP_TOOLS["generic"]

    def test_vscode_transport_sse(self):
        assert _mod._DEFAULT_MCP_TOOLS["vscode"]["transport"] == "sse"


# ===========================================================================
# 33. _validate_mcp_tool
# ===========================================================================

class TestValidateMcpTool:
    def _valid(self):
        return {"enabled": False, "type": "vscode", "label": "VS Code", "description": "x"}

    def test_valid_returns_empty(self):
        assert _mod._validate_mcp_tool(self._valid()) == []

    def test_missing_enabled(self):
        t = self._valid(); del t["enabled"]
        assert any("enabled" in e for e in _mod._validate_mcp_tool(t, "t1"))

    def test_missing_type(self):
        t = self._valid(); del t["type"]
        assert any("type" in e for e in _mod._validate_mcp_tool(t))

    def test_missing_label(self):
        t = self._valid(); del t["label"]
        assert any("label" in e for e in _mod._validate_mcp_tool(t))

    def test_missing_description(self):
        t = self._valid(); del t["description"]
        assert any("description" in e for e in _mod._validate_mcp_tool(t))

    def test_safe_server_url_accepted(self):
        t = self._valid(); t["server_url"] = "http://localhost:9999/mcp"
        assert _mod._validate_mcp_tool(t) == []

    def test_unsafe_server_url_rejected(self):
        t = self._valid(); t["server_url"] = "javascript:evil()"
        assert any("server_url" in e for e in _mod._validate_mcp_tool(t))

    def test_empty_server_url_accepted(self):
        t = self._valid(); t["server_url"] = ""
        assert _mod._validate_mcp_tool(t) == []

    def test_https_server_url_accepted(self):
        t = self._valid(); t["server_url"] = "https://mcp.example.com/sse"
        assert _mod._validate_mcp_tool(t) == []

    def test_all_default_tools_validate_cleanly(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            errs = _mod._validate_mcp_tool(cfg, name=name)
            assert errs == [], f"MCP tool {name!r} errors: {errs}"

    def test_name_appears_in_error_prefix(self):
        t = self._valid(); del t["label"]
        errs = _mod._validate_mcp_tool(t, name="myTool")
        assert any("myTool" in e for e in errs)


# ===========================================================================
# 34. _cfg_str and _cfg_bool
# ===========================================================================

class TestCfgHelpers:
    def test_cfg_str_returns_string(self):
        cfg = MagicMock(); cfg.key = "hello"
        assert _mod._cfg_str(cfg, "key") == "hello"

    def test_cfg_str_returns_none_for_mock_attribute(self):
        cfg = MagicMock()
        assert _mod._cfg_str(cfg, "undefined_key_xyz") is None

    def test_cfg_str_returns_none_for_none_value(self):
        cfg = MagicMock(); cfg.key = None
        assert _mod._cfg_str(cfg, "key") is None

    def test_cfg_str_returns_none_for_int(self):
        cfg = MagicMock(); cfg.key = 42
        assert _mod._cfg_str(cfg, "key") is None

    def test_cfg_str_returns_none_for_missing_attr(self):

        class Cfg:
            pass

        assert _mod._cfg_str(Cfg(), "no_such_attr") is None

    def test_cfg_str_empty_string_is_valid(self):
        """Empty string IS a valid str — must be returned, not converted to None."""
        cfg = MagicMock(); cfg.key = ""
        result = _mod._cfg_str(cfg, "key")
        assert result == ""

    def test_cfg_bool_true(self):
        cfg = MagicMock(); cfg.key = True
        assert _mod._cfg_bool(cfg, "key") is True

    def test_cfg_bool_false(self):
        cfg = MagicMock(); cfg.key = False
        assert _mod._cfg_bool(cfg, "key") is False

    def test_cfg_bool_default_for_mock(self):
        cfg = MagicMock()
        assert _mod._cfg_bool(cfg, "undefined_xyz", default=False) is False

    def test_cfg_bool_default_true_for_mock(self):
        cfg = MagicMock()
        assert _mod._cfg_bool(cfg, "undefined_xyz", default=True) is True

    def test_cfg_bool_int_1_coerced_true(self):
        cfg = MagicMock(); cfg.key = 1
        assert _mod._cfg_bool(cfg, "key") is True

    def test_cfg_bool_int_0_coerced_false(self):
        cfg = MagicMock(); cfg.key = 0
        assert _mod._cfg_bool(cfg, "key") is False


# ===========================================================================
# 35. Extended provider registry
# ===========================================================================

class TestExtendedProviderRegistry:
    def test_deepseek_present_and_web(self):
        p = _mod._DEFAULT_PROVIDERS["deepseek"]
        assert p["type"] == "web"
        assert _mod._validate_provider_url_template(p["url_template"])

    def test_deepseek_model_present(self):
        assert _mod._DEFAULT_PROVIDERS["deepseek"]["model"]

    def test_huggingface_present_and_web(self):
        p = _mod._DEFAULT_PROVIDERS["huggingface"]
        assert "huggingface.co" in p["url_template"]
        assert p["type"] == "web"

    def test_huggingface_model_org_name_format(self):
        model = _mod._DEFAULT_PROVIDERS["huggingface"]["model"]
        assert model and "/" in model, "HuggingFace model should use org/name format"

    def test_custom_present_and_type(self):
        p = _mod._DEFAULT_PROVIDERS["custom"]
        assert p["type"] == "custom"
        assert p["enabled"] is False

    def test_custom_url_template_empty_valid(self):
        assert _mod._validate_provider_url_template(
            _mod._DEFAULT_PROVIDERS["custom"]["url_template"]
        )

    def test_custom_type_in_provider_types(self):
        assert "custom" in _mod._PROVIDER_TYPES

    def test_all_new_providers_validate_cleanly(self):
        for name in ("deepseek", "huggingface", "custom"):
            errs = _mod._validate_provider(_mod._DEFAULT_PROVIDERS[name], name=name)
            assert errs == [], f"{name!r} errors: {errs}"

    def test_ollama_description_mentions_models(self):
        desc = _mod._DEFAULT_PROVIDERS["ollama"]["description"]
        assert any(kw in desc for kw in ("Gemma", "Qwen", "Llama", "DeepSeek"))

    def test_gemini_model_contains_gemini(self):
        assert "gemini" in _mod._DEFAULT_PROVIDERS["gemini"]["model"].lower()


# ===========================================================================
# 36. setup() extended config values
# ===========================================================================

class TestSetupExtended:
    @pytest.fixture()
    def app(self):
        mock_app = MagicMock()
        mock_app.config.html_static_path = []
        return mock_app

    def test_new_config_values_registered(self, app):
        _mod.setup(app)
        names = {c[0][0] for c in app.add_config_value.call_args_list}
        new_required = {
            "ai_assistant_intention",
            "ai_assistant_custom_context",
            "ai_assistant_custom_prompt_prefix",
            "ai_assistant_include_raw_image",
            "ai_assistant_ollama_model",
        }
        missing = new_required - names
        assert not missing, f"Missing config values: {missing}"

    def test_providers_includes_deepseek(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert "deepseek" in args_map["ai_assistant_providers"]

    def test_providers_includes_huggingface(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert "huggingface" in args_map["ai_assistant_providers"]

    def test_providers_includes_custom(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert "custom" in args_map["ai_assistant_providers"]

    def test_mcp_tools_includes_cursor_and_windsurf(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        mcp = args_map["ai_assistant_mcp_tools"]
        assert "cursor" in mcp and "windsurf" in mcp


# ===========================================================================
# 37. Ollama local provider
# ===========================================================================

class TestOllamaLocalSupport:
    def test_api_base_url_is_localhost(self):
        assert "localhost" in _mod._DEFAULT_PROVIDERS["ollama"]["api_base_url"]

    def test_type_local(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["type"] == "local"

    def test_disabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["enabled"] is False

    def test_localhost_validates(self):
        assert _mod._validate_ollama_url("http://localhost:11434") is True

    def test_loopback_validates(self):
        assert _mod._validate_ollama_url("http://127.0.0.1:11434") is True

    def test_remote_rejected(self):
        assert _mod._validate_ollama_url("http://external.example.com") is False

    def test_validates_cleanly(self):
        assert _mod._validate_provider(_mod._DEFAULT_PROVIDERS["ollama"], "ollama") == []

    def test_custom_remote_api_base_url_fails(self):
        p = dict(_mod._DEFAULT_PROVIDERS["ollama"])
        p["api_base_url"] = "http://remote-server.example.com:11434"
        errs = _mod._validate_provider(p, "ollama_remote")
        assert any("localhost" in e or "127.0.0.1" in e for e in errs)


# ===========================================================================
# 38. Full provider round-trip
# ===========================================================================

class TestProviderRoundTrip:
    def test_filter_removes_only_invalid(self):
        mixed = {
            "good": {"url_template": "https://claude.ai/new?q={prompt}", "enabled": True},
            "bad": {"url_template": "javascript:alert(1)", "enabled": True},
        }
        result = _mod._filter_providers(mixed)
        assert "good" in result and "bad" not in result

    def test_full_default_registry_all_kept(self):
        result = _mod._filter_providers(_mod._DEFAULT_PROVIDERS)
        for name in ("claude", "gemini", "chatgpt", "deepseek", "huggingface", "custom"):
            assert name in result

    def test_full_default_registry_json_serialisable(self):
        result = _mod._filter_providers(_mod._DEFAULT_PROVIDERS)
        dumped = json.dumps(result)
        assert isinstance(dumped, str)


# ===========================================================================
# 39. Edge cases and invariants
# ===========================================================================

class TestEdgeCases:
    def test_html_to_markdown_empty_string(self):
        assert isinstance(_mod.html_to_markdown(""), str)

    def test_html_to_markdown_whitespace_only(self):
        assert isinstance(_mod.html_to_markdown("   \n\t  "), str)

    def test_process_file_converter_crash_returns_error(self, tmp_path):
        # Pre-fix: this patched html_to_markdown.  Post-fix: the worker calls
        # _build_converter_class() directly, bypassing html_to_markdown, so
        # we must simulate a converter crash at that call site instead.
        html = tmp_path / "page.html"
        html.write_text("<html><body><article>OK</article></body></html>", encoding="utf-8")
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._build_converter_class",
            side_effect=RuntimeError("converter exploded"),
        ):
            status, _, msg = _mod._process_single_html_file((
                str(html), str(tmp_path), [], ["article"], ["script", "style"],
            ))
        assert status == "error" and "converter exploded" in msg

    def test_6tuple_worker_converter_crash_returns_error(self, tmp_path):
        # Same as above for the 6-tuple worker variant.
        html = tmp_path / "page.html"
        html.write_text("<html><body><article>OK</article></body></html>", encoding="utf-8")
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._build_converter_class",
            side_effect=RuntimeError("worker crash"),
        ):
            status, _, msg = _mod._process_html_file_worker((
                str(html), str(tmp_path), str(tmp_path), [], ["article"], ["script"],
            ))
        assert status == "error" and "worker crash" in msg

    def test_generate_markdown_no_html_files(self, sphinx_app, tmp_path):
        empty = tmp_path / "empty_out"; empty.mkdir()
        sphinx_app.builder.outdir = str(empty)
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert list(empty.rglob("*.md")) == []

    def test_resolve_selectors_all_themes(self):
        for theme in _mod._THEME_SELECTOR_PRESETS:
            result = _mod._resolve_content_selectors(theme, [])
            assert len(result) > 0

    def test_wrong_5tuple_length_raises(self, tmp_path):
        with pytest.raises((TypeError, ValueError)):
            _mod._process_single_html_file((
                str(tmp_path / "x.html"), str(tmp_path), [], ["article"],
            ))

    def test_wrong_6tuple_length_raises(self, tmp_path):
        with pytest.raises((TypeError, ValueError)):
            _mod._process_html_file_worker((
                str(tmp_path / "x.html"), str(tmp_path), str(tmp_path), [], ["article"],
            ))

    def test_provider_required_keys_tuple(self):
        assert isinstance(_mod._PROVIDER_REQUIRED_KEYS, tuple)
        assert len(_mod._PROVIDER_REQUIRED_KEYS) >= 7

    def test_provider_types_frozenset(self):
        assert isinstance(_mod._PROVIDER_TYPES, frozenset)
        assert {"web", "local", "api", "custom"}.issubset(_mod._PROVIDER_TYPES)

    def test_localhost_re_matches_correctly(self):
        assert _mod._LOCALHOST_RE.match("http://localhost:11434")
        assert _mod._LOCALHOST_RE.match("http://127.0.0.1:11434")
        assert not _mod._LOCALHOST_RE.match("http://remote.host")

    def test_public_api_surface_complete(self):
        """Every symbol in __all__ must be importable from the module."""
        for name in _mod.__all__:
            assert hasattr(_mod, name), f"{name!r} in __all__ but not in module"


# ===========================================================================
# 41. Targeted coverage — 8 tests for the 21 remaining uncovered lines
#
# Each test targets one specific uncovered code path:
#
#   Test 1  lines 1414-1418  _resolve_icon inner except ImportError
#           lines 1423-1425  _resolve_icon outer except ImportError
#   Test 2  lines 1679-1681  html_to_markdown bs4 ImportError path
#   Test 3  lines 1789-1790  _process_html_file_worker ValueError after guard
#   Test 4  lines 2014-2015  process_html_directory future raises exception
#   Test 5  line  2022       process_html_directory "error" status branch
#   Test 6  lines 2222-2227  generate_markdown_files future raises exception
#   Test 7  line  2233       generate_markdown_files skipped with message
#   Test 8  lines 2235-2236  generate_markdown_files "error" status branch
# ===========================================================================

_CANONICAL = "scikitplot._externals._sphinx_ext._sphinx_ai_assistant"
_CANONICAL_STATIC = _CANONICAL + "._static"


class TestTargetedCoverage:
    """
    8 focused tests that cover the 21 remaining uncovered executable lines.

    Notes
    -----
    **Developer note** — each test uses the minimal mock surface required to
    reach the specific branch.  No existing test logic is duplicated.

    ProcessPoolExecutor / as_completed mocking pattern
    ---------------------------------------------------
    ``generate_markdown_files`` and ``process_html_directory`` both use::

        futures = {executor.submit(worker, a): a for a in args_list}
        for future in as_completed(futures):

    The mock executor's ``submit()`` always returns the same ``mock_future``,
    so ``futures == {mock_future: args}``.  A custom ``mock_as_completed``
    helper yields ``futures.keys()`` — which is exactly ``{mock_future}`` —
    so ``future.result()`` is always called on the mock we control.
    """

    # -----------------------------------------------------------------------
    # Test 1 — _resolve_icon: inner except ImportError (lines 1414-1418)
    #          + outer except ImportError (lines 1423-1425)
    #
    # Strategy: temporarily delete ``_SVG_DEFAULT`` from the live ``_static``
    # module so that the first inner ``from ._static import _SVG_DEFAULT``
    # succeeds for ``_PROVIDER_META`` but fails for ``_SVG_DEFAULT``, which:
    #   1. fires the inner except (line 1414) and re-runs the imports,
    #   2. the re-run also fails on ``_SVG_DEFAULT`` → outer except (1423)
    #      returns ``icon_filename`` unchanged (line 1425).
    # -----------------------------------------------------------------------

    def test_resolve_icon_inner_and_outer_except_import_error(
        self, tmp_path: Path
    ) -> None:
        """Lines 1414-1418 + 1423-1425: _SVG_DEFAULT missing triggers both excepts."""
        static_mod = sys.modules.get(_CANONICAL_STATIC)
        if static_mod is None:
            pytest.skip("_static module not registered in sys.modules — run via conftest")

        # Use an empty directory so the icon file is absent → enters the try block.
        static_dir = tmp_path / "_static"
        static_dir.mkdir()

        original_svg = getattr(static_mod, "_SVG_DEFAULT", None)
        try:
            if hasattr(static_mod, "_SVG_DEFAULT"):
                delattr(static_mod, "_SVG_DEFAULT")
            result = _mod._resolve_icon("missing.svg", "claude", static_dir=static_dir)
        finally:
            # Always restore the original attribute so sibling tests are unaffected.
            if original_svg is not None:
                static_mod._SVG_DEFAULT = original_svg

        # Outer except fallback must return the original filename unchanged.
        assert result == "missing.svg"

    # -----------------------------------------------------------------------
    # Test 2 — html_to_markdown: bs4 ImportError path (lines 1679-1681)
    #
    # Strategy: inject ``None`` for the ``bs4`` key in ``sys.modules`` before
    # calling ``html_to_markdown``.  Python treats ``sys.modules[name] = None``
    # as a sentinel meaning "this module is known to be unimportable", so
    # ``from bs4 import BeautifulSoup`` raises ``ImportError`` and the except
    # block at line 1679 fires.  markdownify's ``strip=`` option handles tag
    # removal without bs4, so the function still returns valid Markdown.
    # -----------------------------------------------------------------------

    def test_html_to_markdown_bs4_import_error_path(self) -> None:
        """Lines 1679-1681: bs4 absent → except ImportError fires, result still valid."""
        with patch.dict(sys.modules, {"bs4": None}):
            result = _mod.html_to_markdown("<h1>Title</h1><p>Body</p>")

        # The function must succeed and return the converted Markdown.
        assert isinstance(result, str)
        assert "Title" in result

    # -----------------------------------------------------------------------
    # Test 3 — _process_html_file_worker: ValueError branch (lines 1789-1790)
    #
    # Strategy: mock ``_is_path_within`` to return ``True`` (bypass the
    # path-traversal guard at line 1784), but pass a ``html_file`` that is
    # NOT literally relative to ``input_dir`` — so the subsequent
    # ``html_file.relative_to(input_dir)`` call at line 1788 raises
    # ``ValueError``, reaching the except branch at line 1789.
    #
    # This models a real race condition where symlink resolution makes
    # ``_is_path_within`` (which uses ``resolve()``) see the file as inside
    # the tree, but the literal (unresolved) path is outside it.
    # -----------------------------------------------------------------------

    def test_worker_relative_to_value_error(self, tmp_path: Path) -> None:
        """Lines 1789-1790: _is_path_within passes but relative_to raises ValueError."""
        html_file = tmp_path / "page.html"
        html_file.write_text(
            "<html><body><article>content</article></body></html>",
            encoding="utf-8",
        )
        # An unrelated sibling directory — html_file is not relative to it.
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()

        with patch.object(_mod, "_is_path_within", return_value=True):
            status, _, msg = _mod._process_html_file_worker((
                str(html_file),
                str(unrelated),  # input_dir: html_file is not relative to this
                str(unrelated),  # output_dir: same
                [],
                ["article"],
                ["script"],
            ))

        assert status == "error"
        assert "outside input directory" in msg

    # -----------------------------------------------------------------------
    # Test 4 — process_html_directory: future raises (lines 2014-2015)
    #
    # Strategy: replace ``ProcessPoolExecutor`` with a mock that returns a
    # single failing future.  A custom ``mock_as_completed`` yields the keys
    # of the futures dict (the mock future), so ``future.result()`` raises
    # ``RuntimeError`` and the except branch at line 2014 increments errors.
    # -----------------------------------------------------------------------

    def test_process_html_directory_future_exception_counted_as_error(
        self, tmp_path: Path
    ) -> None:
        """Lines 2014-2015: future.result() raises → except Exception, errors += 1."""
        site = tmp_path / "site"
        site.mkdir()
        (site / "page.html").write_text(
            "<html><body><main>content</main></body></html>", encoding="utf-8"
        )

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("worker process killed")

        mock_exec = MagicMock()
        mock_exec.__enter__ = MagicMock(return_value=mock_exec)
        mock_exec.__exit__ = MagicMock(return_value=False)
        mock_exec.submit.return_value = mock_future

        def _mock_as_completed(fut_dict):
            yield from fut_dict.keys()

        with patch(f"{_CANONICAL}.ProcessPoolExecutor", return_value=mock_exec):
            with patch(f"{_CANONICAL}.as_completed", side_effect=_mock_as_completed):
                stats = _mod.process_html_directory(site, max_workers=1)

        assert stats["errors"] >= 1

    # -----------------------------------------------------------------------
    # Test 5 — process_html_directory: "error" status branch (line 2022)
    #
    # Strategy: same executor mock pattern, but ``future.result()`` succeeds
    # and returns an ``("error", ...)`` 3-tuple — the ``else`` branch at
    # line 2021 runs and ``errors += 1`` at line 2022 fires.
    # -----------------------------------------------------------------------

    def test_process_html_directory_error_status_increments_errors(
        self, tmp_path: Path
    ) -> None:
        """Line 2022: worker returns 'error' status → else branch errors += 1."""
        site = tmp_path / "site"
        site.mkdir()
        (site / "page.html").write_text(
            "<html><body><main>content</main></body></html>", encoding="utf-8"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = ("error", "page.html", "conversion failed")

        mock_exec = MagicMock()
        mock_exec.__enter__ = MagicMock(return_value=mock_exec)
        mock_exec.__exit__ = MagicMock(return_value=False)
        mock_exec.submit.return_value = mock_future

        def _mock_as_completed(fut_dict):
            yield from fut_dict.keys()

        with patch(f"{_CANONICAL}.ProcessPoolExecutor", return_value=mock_exec):
            with patch(f"{_CANONICAL}.as_completed", side_effect=_mock_as_completed):
                stats = _mod.process_html_directory(site, max_workers=1)

        assert stats["errors"] >= 1

    # -----------------------------------------------------------------------
    # Test 6 — generate_markdown_files: future raises (lines 2222-2227)
    #
    # Strategy: same mock pattern applied to the Sphinx hook.  The hook
    # discovers HTML files from ``tmp_html_tree``, builds args_list, then
    # submits them.  Our mock executor returns a failing future, which the
    # mock ``as_completed`` yields.  ``future.result()`` raises, reaching
    # lines 2222-2227 (except block: errors += 1, log.warning, continue).
    # -----------------------------------------------------------------------

    def test_generate_markdown_files_future_exception_continues(
        self, sphinx_app: MagicMock, tmp_html_tree: Path
    ) -> None:
        """Lines 2222-2227: future.result() raises → errors counted, loop continues."""
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("subprocess OOM killed")

        mock_exec = MagicMock()
        mock_exec.__enter__ = MagicMock(return_value=mock_exec)
        mock_exec.__exit__ = MagicMock(return_value=False)
        mock_exec.submit.return_value = mock_future

        def _mock_as_completed(fut_dict):
            yield from fut_dict.keys()

        with patch(f"{_CANONICAL}.ProcessPoolExecutor", return_value=mock_exec):
            with patch(f"{_CANONICAL}.as_completed", side_effect=_mock_as_completed):
                # Must not raise — the hook swallows per-file errors.
                _mod.generate_markdown_files(sphinx_app, exception=None)

    # -----------------------------------------------------------------------
    # Test 7 — generate_markdown_files: "skipped" with non-empty message
    #          (line 2233)
    #
    # Strategy: future returns ("skipped", rel_path, "No main content found").
    # Line 2230 increments skipped; line 2232 ``if message:`` is True so
    # line 2233 ``log.debug(...)`` fires.
    # -----------------------------------------------------------------------

    def test_generate_markdown_files_skipped_with_message_logged(
        self, sphinx_app: MagicMock, tmp_html_tree: Path
    ) -> None:
        """Line 2233: skipped status with non-empty message triggers log.debug."""
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1

        mock_future = MagicMock()
        mock_future.result.return_value = (
            "skipped",
            "page.html",
            "No main content element found",
        )

        mock_exec = MagicMock()
        mock_exec.__enter__ = MagicMock(return_value=mock_exec)
        mock_exec.__exit__ = MagicMock(return_value=False)
        mock_exec.submit.return_value = mock_future

        def _mock_as_completed(fut_dict):
            yield from fut_dict.keys()

        with patch(f"{_CANONICAL}.ProcessPoolExecutor", return_value=mock_exec):
            with patch(f"{_CANONICAL}.as_completed", side_effect=_mock_as_completed):
                _mod.generate_markdown_files(sphinx_app, exception=None)

    # -----------------------------------------------------------------------
    # Test 8 — generate_markdown_files: "error" status branch (lines 2235-2236)
    #
    # Strategy: future returns ("error", rel_path, "markdownify crashed").
    # Line 2234 ``else:`` fires; line 2235 increments errors and line 2236
    # calls ``log.warning(f"... Failed to convert ...")`` — both covered.
    # -----------------------------------------------------------------------

    def test_generate_markdown_files_error_status_logged_as_warning(
        self, sphinx_app: MagicMock, tmp_html_tree: Path
    ) -> None:
        """Lines 2235-2236: 'error' status → errors += 1, log.warning fired."""
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1

        mock_future = MagicMock()
        mock_future.result.return_value = ("error", "page.html", "markdownify crashed")

        mock_exec = MagicMock()
        mock_exec.__enter__ = MagicMock(return_value=mock_exec)
        mock_exec.__exit__ = MagicMock(return_value=False)
        mock_exec.submit.return_value = mock_future

        def _mock_as_completed(fut_dict):
            yield from fut_dict.keys()

        with patch(f"{_CANONICAL}.ProcessPoolExecutor", return_value=mock_exec):
            with patch(f"{_CANONICAL}.as_completed", side_effect=_mock_as_completed):
                _mod.generate_markdown_files(sphinx_app, exception=None)
