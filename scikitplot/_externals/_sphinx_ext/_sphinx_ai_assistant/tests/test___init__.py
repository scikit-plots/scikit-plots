# tests/_externals/_sphinx_ext/_sphinx_ai_assistant/test__sphinx_ai_assistant.py
"""
Comprehensive test suite for
``scikitplot._externals._sphinx_ext._sphinx_ai_assistant``.

Coverage targets
----------------
* Lazy import mechanics — module importable without Sphinx/bs4/markdownify.
* Security helpers — XSS, path traversal, URL validation, position
  validation, provider URL-template validation, CSS selector sanitization,
  Ollama local-only validation, provider schema validation.
* AI provider registry — all 9 default providers present and schema-valid.
* Provider filtering — _filter_providers removes dangerous URL templates.
* Theme selector presets — 24+ themes, _resolve_content_selectors merge.
* Extended worker — _process_html_file_worker (6-tuple), separate output dir.
* Per-file worker — _process_single_html_file (5-tuple wrapper).
* Standalone HTML walker — process_html_directory (all branches).
* Standalone llms.txt — generate_llms_txt_standalone (all branches).
* Jupyter widget — see test__jupyter.py for full coverage.
* Markdown conversion — converter class, html_to_markdown, strip_tags.
* Build hooks — generate_markdown_files, generate_llms_txt (all branches).
* Template context — add_ai_assistant_context (all branches).
* Extension setup — setup() metadata, all config values, event hooks.
* Edge cases — empty HTML, no workers, encoding errors.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

import scikitplot._externals._sphinx_ext._sphinx_ai_assistant as _mod

_EXT = _mod


# ===========================================================================
# 1. Lazy import safety
# ===========================================================================

class TestLazyImports:
    def test_module_importable(self):
        assert _mod is not None

    def test_no_sphinx_at_module_scope(self):
        assert callable(_mod.setup)
        assert callable(_mod.generate_markdown_files)
        assert callable(_mod.add_ai_assistant_context)

    def test_version_string(self):
        assert isinstance(_mod._VERSION, str)
        parts = _mod._VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_no_bs4_at_module_scope(self):
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

    def test_display_jupyter_ai_button_is_public(self):
        assert callable(_mod.display_jupyter_ai_button)


# ===========================================================================
# 2. Security — _safe_json_for_script
# ===========================================================================

class TestSafeJsonForScript:
    def test_plain_dict_unchanged_semantics(self):
        obj = {"key": "value", "num": 42}
        assert json.loads(_mod._safe_json_for_script(obj)) == obj

    def test_script_close_tag_escaped(self):
        result = _mod._safe_json_for_script({"url": "https://x.com/</script>"})
        assert "</script>" not in result
        assert "<\\/" in result

    def test_nested_close_tag(self):
        result = _mod._safe_json_for_script({"a": {"b": "</ScRiPt>"}})
        assert "</ScRiPt>" not in result

    def test_multiple_occurrences(self):
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

    def test_dotdot_traversal(self, tmp_path):
        evil = tmp_path / ".." / "etc" / "passwd"
        assert _mod._is_path_within(evil, tmp_path) is False

    def test_absolute_outside(self, tmp_path):
        assert _mod._is_path_within(Path("/etc/passwd"), tmp_path) is False

    def test_nested_deep(self, tmp_path):
        assert _mod._is_path_within(tmp_path / "a" / "b" / "c" / "d.html", tmp_path) is True


# ===========================================================================
# 4. Security — _validate_base_url
# ===========================================================================

class TestValidateBaseUrl:
    def test_https_accepted(self):
        assert _mod._validate_base_url("https://docs.example.com/") == "https://docs.example.com"

    def test_http_accepted(self):
        assert _mod._validate_base_url("http://localhost:8000/") == "http://localhost:8000"

    def test_empty_string_accepted(self):
        assert _mod._validate_base_url("") == ""

    def test_whitespace_only_accepted(self):
        assert _mod._validate_base_url("   ") == ""

    def test_javascript_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _mod._validate_base_url("javascript:alert(1)")

    def test_data_scheme_rejected(self):
        with pytest.raises(ValueError):
            _mod._validate_base_url("data:text/html,<h1>XSS</h1>")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError):
            _mod._validate_base_url("ftp://example.com")

    def test_trailing_slashes_stripped(self):
        assert _mod._validate_base_url("https://x.com///") == "https://x.com"

    def test_uppercase_https_accepted(self):
        assert _mod._validate_base_url("HTTPS://docs.example.com/").startswith("HTTPS://")

    def test_mixed_case_http_accepted(self):
        assert _mod._validate_base_url("Http://localhost/").startswith("Http://")


# ===========================================================================
# 5. Security — _validate_position
# ===========================================================================

class TestValidatePosition:
    def test_sidebar_accepted(self): assert _mod._validate_position("sidebar") == "sidebar"
    def test_title_accepted(self): assert _mod._validate_position("title") == "title"
    def test_floating_accepted(self): assert _mod._validate_position("floating") == "floating"
    def test_none_str_accepted(self): assert _mod._validate_position("none") == "none"
    def test_uppercase_normalised(self): assert _mod._validate_position("SIDEBAR") == "sidebar"
    def test_whitespace_stripped(self): assert _mod._validate_position("  title  ") == "title"

    def test_unknown_rejected(self):
        with pytest.raises(ValueError, match="ai_assistant_position"):
            _mod._validate_position("evil")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            _mod._validate_position("")

    def test_allowed_positions_frozenset(self):
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

    def test_empty_accepted(self):
        assert _mod._validate_provider_url_template("") is True

    def test_whitespace_accepted(self):
        assert _mod._validate_provider_url_template("   ") is True

    def test_javascript_rejected(self):
        assert _mod._validate_provider_url_template("javascript:alert(1)") is False

    def test_data_rejected(self):
        assert _mod._validate_provider_url_template("data:text/html,<h1>X</h1>") is False

    def test_ftp_rejected(self):
        assert _mod._validate_provider_url_template("ftp://evil.com") is False

    def test_vbscript_rejected(self):
        assert _mod._validate_provider_url_template("vbscript:msgbox(1)") is False


# ===========================================================================
# 7. Security — _validate_ollama_url (new)
# ===========================================================================

class TestValidateOllamaUrl:
    def test_localhost_accepted(self):
        assert _mod._validate_ollama_url("http://localhost:11434") is True

    def test_127_0_0_1_accepted(self):
        assert _mod._validate_ollama_url("http://127.0.0.1:11434") is True

    def test_empty_accepted(self):
        assert _mod._validate_ollama_url("") is True

    def test_remote_rejected(self):
        assert _mod._validate_ollama_url("http://remote.example.com") is False

    def test_https_remote_rejected(self):
        assert _mod._validate_ollama_url("https://ollama.example.com") is False

    def test_localhost_no_port(self):
        assert _mod._validate_ollama_url("http://localhost") is True

    def test_localhost_with_path(self):
        assert _mod._validate_ollama_url("http://localhost:11434/api") is True


# ===========================================================================
# 8. Security — _validate_css_selector
# ===========================================================================

class TestValidateCssSelector:
    def test_simple_element(self): assert _mod._validate_css_selector("article") is True
    def test_class_selector(self): assert _mod._validate_css_selector("article.bd-article") is True
    def test_attribute_with_quotes(self): assert _mod._validate_css_selector('div[role="main"]') is True
    def test_html_open_rejected(self): assert _mod._validate_css_selector("<script>") is False
    def test_html_close_rejected(self): assert _mod._validate_css_selector("</style>") is False
    def test_combined_html_rejected(self): assert _mod._validate_css_selector("<img onerror=x>") is False
    def test_generic_main(self): assert _mod._validate_css_selector("main") is True


class TestSanitizeSelectors:
    def test_empty_strings_removed(self):
        assert "" not in _mod._sanitize_selectors(["article", "   ", "main"])

    def test_unsafe_removed(self):
        result = _mod._sanitize_selectors(["article", "<bad>", "main"])
        assert "<bad>" not in result

    def test_all_safe(self):
        sels = ["article.bd-article", 'div[role="main"]', "main"]
        assert _mod._sanitize_selectors(sels) == sels

    def test_empty_list(self):
        assert _mod._sanitize_selectors([]) == []


# ===========================================================================
# 9. Provider registry — _DEFAULT_PROVIDERS
# ===========================================================================

class TestDefaultProviders:
    """All 9 default providers must exist and have valid schema."""

    _REQUIRED_PROVIDERS = [
        "claude", "chatgpt", "gemini", "ollama",
        "mistral", "perplexity", "copilot", "groq", "you",
    ]

    def test_all_providers_present(self):
        for name in self._REQUIRED_PROVIDERS:
            assert name in _mod._DEFAULT_PROVIDERS, f"Missing provider: {name!r}"

    def test_all_required_keys_present(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            for key in _mod._PROVIDER_REQUIRED_KEYS:
                assert key in cfg, f"Provider {name!r} missing key {key!r}"

    def test_all_types_valid(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            assert cfg["type"] in _mod._PROVIDER_TYPES, (
                f"Provider {name!r} has invalid type {cfg['type']!r}"
            )

    def test_all_url_templates_safe(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            url = cfg.get("url_template", "")
            assert _mod._validate_provider_url_template(url), (
                f"Provider {name!r} has unsafe url_template {url!r}"
            )

    def test_ollama_is_local_type(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["type"] == "local"

    def test_ollama_disabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["enabled"] is False

    def test_ollama_api_base_url_is_local(self):
        api_url = _mod._DEFAULT_PROVIDERS["ollama"].get("api_base_url", "")
        assert _mod._validate_ollama_url(api_url), (
            f"Ollama api_base_url {api_url!r} is not loopback"
        )

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
                f"Provider {name!r} contains javascript: in url_template"
            )

    def test_prompt_templates_have_url_placeholder(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            pt = cfg.get("prompt_template", "")
            assert "{url}" in pt, (
                f"Provider {name!r} prompt_template missing {{url}} placeholder"
            )


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
        errors = _mod._validate_provider(self._full_provider(), "test")
        assert errors == []

    def test_missing_key_reported(self):
        prov = self._full_provider()
        del prov["label"]
        errors = _mod._validate_provider(prov, "bad")
        assert any("label" in e for e in errors)

    def test_invalid_type_reported(self):
        prov = self._full_provider(type="invalid_type")
        errors = _mod._validate_provider(prov, "bad")
        assert any("type" in e for e in errors)

    def test_javascript_url_template_reported(self):
        prov = self._full_provider(url_template="javascript:alert(1)")
        errors = _mod._validate_provider(prov, "bad")
        assert any("url_template" in e for e in errors)

    def test_local_type_remote_api_base_url_reported(self):
        prov = self._full_provider(
            type="local",
            api_base_url="https://remote.example.com",
        )
        errors = _mod._validate_provider(prov, "bad")
        assert any("api_base_url" in e for e in errors)

    def test_local_type_localhost_api_base_url_ok(self):
        prov = self._full_provider(
            type="local",
            api_base_url="http://localhost:11434",
        )
        errors = _mod._validate_provider(prov, "ollama")
        assert errors == []

    def test_all_default_providers_validate_cleanly(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            errors = _mod._validate_provider(cfg, name)
            assert errors == [], (
                f"Provider {name!r} has validation errors: {errors}"
            )


# ===========================================================================
# 11. _filter_providers
# ===========================================================================

class TestFilterProviders:
    def test_keeps_safe_providers(self):
        providers = {
            "safe": {"url_template": "https://example.com/?q={prompt}"},
        }
        result = _mod._filter_providers(providers)
        assert "safe" in result

    def test_removes_javascript_url(self):
        providers = {
            "evil": {"url_template": "javascript:alert(1)"},
        }
        result = _mod._filter_providers(providers)
        assert "evil" not in result

    def test_removes_data_url(self):
        providers = {
            "evil": {"url_template": "data:text/html,xss"},
        }
        assert "evil" not in _mod._filter_providers(providers)

    def test_require_enabled_filters_disabled(self):
        providers = {
            "on": {"url_template": "https://x.com", "enabled": True},
            "off": {"url_template": "https://y.com", "enabled": False},
        }
        result = _mod._filter_providers(providers, require_enabled=True)
        assert "on" in result
        assert "off" not in result

    def test_require_enabled_false_keeps_disabled(self):
        providers = {
            "off": {"url_template": "https://y.com", "enabled": False},
        }
        result = _mod._filter_providers(providers, require_enabled=False)
        assert "off" in result

    def test_empty_providers(self):
        assert _mod._filter_providers({}) == {}


# ===========================================================================
# 12. Dependency detection
# ===========================================================================

class TestHasMarkdownDeps:
    def test_returns_bool(self):
        assert isinstance(_mod._has_markdown_deps(), bool)

    def test_true_when_both_installed(self):
        assert _mod._has_markdown_deps() is True

    def test_false_when_bs4_missing(self, monkeypatch):
        with patch("importlib.util.find_spec", side_effect=lambda n: None if n == "bs4" else MagicMock()):
            assert _mod._has_markdown_deps() is False

    def test_false_when_markdownify_missing(self, monkeypatch):
        with patch("importlib.util.find_spec", side_effect=lambda n: None if n == "markdownify" else MagicMock()):
            assert _mod._has_markdown_deps() is False


class TestHasIPython:
    def test_returns_bool(self):
        assert isinstance(_mod._has_ipython(), bool)


# ===========================================================================
# 13. Logger singleton
# ===========================================================================

class TestGetLogger:
    def test_returns_logger(self):
        assert _mod._get_logger() is not None

    def test_caching(self):
        assert _mod._get_logger() is _mod._get_logger()


# ===========================================================================
# 14. Converter class
# ===========================================================================

class TestBuildConverterClass:
    def test_returns_class(self):
        assert isinstance(_mod._build_converter_class(), type)

    def test_caching_same_object(self):
        assert _mod._build_converter_class() is _mod._build_converter_class()

    def test_instantiable_with_defaults(self):
        assert _mod._build_converter_class()(heading_style="ATX") is not None


class TestSphinxMarkdownConverter:
    @pytest.fixture(autouse=True)
    def _cls(self):
        self.Cls = _mod._build_converter_class()
        self.instance = self.Cls(heading_style="ATX", bullets="*")

    def _el(self, tag, attrs=None, text=""):
        from bs4 import BeautifulSoup
        attrs_str = " ".join(f'{k}="{v}"' for k, v in (attrs or {}).items())
        return BeautifulSoup(f"<{tag} {attrs_str}>{text}</{tag}>", "html.parser").find(tag)

    def test_convert_code_with_language(self):
        el = self._el("code", {"class": "highlight-python"}, "x = 1")
        assert "```python" in self.instance.convert_code(el, "x = 1", False)

    def test_convert_code_no_language(self):
        el = self._el("code", {}, "x = 1")
        assert self.instance.convert_code(el, "x = 1", False) == "`x = 1`"

    def test_convert_code_inline(self):
        el = self._el("code", {"class": "highlight-python"}, "x")
        assert self.instance.convert_code(el, "x", True) == "`x`"

    def test_convert_code_empty_text_fallback(self):
        el = self._el("code", {}, "fallback")
        assert "fallback" in self.instance.convert_code(el, "", False)

    def test_convert_code_empty_element(self):
        el = self._el("code", {}, "")
        assert self.instance.convert_code(el, "", False) == ""

    def test_convert_code_multiple_classes_picks_highlight(self):
        el = self._el("code", {"class": "highlight-bash notranslate"}, "ls")
        assert "```bash" in self.instance.convert_code(el, "ls", False)

    def test_convert_div_admonition(self):
        from bs4 import BeautifulSoup
        html = '<div class="admonition note"><p class="admonition-title">Note</p><p>Content.</p></div>'
        el = BeautifulSoup(html, "html.parser").find("div")
        result = self.instance.convert_div(el, "Note\n\nContent.", False)
        assert "**Note**" in result

    def test_convert_div_non_admonition(self):
        el = self._el("div", {}, "plain")
        assert self.instance.convert_div(el, "plain", False) == "plain"

    def test_convert_div_admonition_no_title(self):
        from bs4 import BeautifulSoup
        el = BeautifulSoup('<div class="admonition">No title.</div>', "html.parser").find("div")
        assert self.instance.convert_div(el, "No title.", False) == "No title."

    def test_convert_div_does_not_mutate(self):
        from bs4 import BeautifulSoup
        html = '<div class="admonition warning"><p class="admonition-title">W</p><p>X</p></div>'
        el = BeautifulSoup(html, "html.parser").find("div")
        original = list(el.children)
        self.instance.convert_div(el, "W\n\nX.", False)
        assert list(el.children) == original

    def test_convert_pre_with_code_child(self):
        from bs4 import BeautifulSoup
        el = BeautifulSoup('<pre><code class="highlight-bash">ls -la</code></pre>', "html.parser").find("pre")
        assert "ls -la" in self.instance.convert_pre(el, "", False)

    def test_convert_pre_no_code_child(self):
        el = self._el("pre", {}, "raw text")
        result = self.instance.convert_pre(el, "raw text", False)
        assert "```" in result and "raw text" in result

    def test_convert_pre_empty(self):
        el = self._el("pre", {}, "")
        assert self.instance.convert_pre(el, "", False) == ""


# ===========================================================================
# 15. html_to_markdown
# ===========================================================================

class TestHtmlToMarkdown:
    def test_heading(self):
        assert "# Title" in _mod.html_to_markdown("<h1>Title</h1>")

    def test_paragraph(self):
        assert "Hello world" in _mod.html_to_markdown("<p>Hello world</p>")

    def test_code_block(self):
        assert "x=1" in _mod.html_to_markdown('<pre><code class="highlight-python">x=1</code></pre>')

    def test_strips_script(self):
        result = _mod.html_to_markdown("<script>alert(1)</script><p>safe</p>")
        assert "alert" not in result and "safe" in result

    def test_legacy_alias(self):
        assert _mod.html_to_markdown_converter is _mod.html_to_markdown


class TestHtmlToMarkdownExtended:
    def test_default_strips_script_and_style(self):
        result = _mod.html_to_markdown("<style>body{color:red}</style><script>evil()</script><p>ok</p>")
        assert "evil" not in result and "ok" in result

    def test_custom_strip_tags_removes_nav(self):
        result = _mod.html_to_markdown(
            "<nav><a>Home</a></nav><article>Content</article>",
            strip_tags=["nav", "script", "style"],
        )
        assert "Home" not in result and "Content" in result

    def test_strip_tags_none_uses_defaults(self):
        result = _mod.html_to_markdown("<script>bad()</script><p>good</p>", strip_tags=None)
        assert "bad" not in result and "good" in result

    def test_multiple_custom_tags_stripped(self):
        html = "<header>TOP</header><footer>BOTTOM</footer><article>BODY</article>"
        result = _mod.html_to_markdown(html, strip_tags=["header", "footer", "script", "style"])
        assert "TOP" not in result and "BOTTOM" not in result and "BODY" in result

    def test_empty_html_returns_string(self):
        assert isinstance(_mod.html_to_markdown(""), str)

    def test_whitespace_html_returns_string(self):
        assert isinstance(_mod.html_to_markdown("   \n\t  "), str)

    def test_no_bs4_fallback_does_not_crash(self):
        import builtins
        real_import = builtins.__import__

        def import_blocker(name, *args, **kwargs):
            if name == "bs4":
                raise ImportError("bs4 not installed (test)")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            try:
                result = _mod.html_to_markdown("<p>hello</p>")
                assert isinstance(result, str)
            except ImportError:
                pass
            except TypeError as exc:
                pytest.fail(f"html_to_markdown raised TypeError when bs4 absent: {exc}")


# ===========================================================================
# 16. _process_html_file_worker (6-tuple, new)
# ===========================================================================

class TestProcessHtmlFileWorker:
    """Tests for the extended 6-tuple worker with separate input/output dirs."""

    _SELECTORS = ["article", "main", 'div[role="main"]']
    _STRIP = ["script", "style"]

    def _call(self, html_path, input_dir, output_dir, excludes=None):
        return _mod._process_html_file_worker((
            str(html_path),
            str(input_dir),
            str(output_dir),
            excludes or [],
            self._SELECTORS,
            self._STRIP,
        ))

    def test_inline_mode_writes_alongside(self, tmp_path):
        """output_dir == input_dir → .md file next to .html file."""
        html = tmp_path / "page.html"
        html.write_text('<html><body><article>Hello</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(html, tmp_path, tmp_path)
        assert status == "success"
        assert (tmp_path / "page.md").exists()

    def test_separate_output_dir_mirrors_structure(self, tmp_path):
        """output_dir != input_dir → directory structure is mirrored."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        sub = input_dir / "api"
        sub.mkdir()
        html = sub / "func.html"
        html.write_text('<html><body><article>API</article></body></html>', encoding="utf-8")

        status, rel, msg = self._call(html, input_dir, output_dir)
        assert status == "success"
        assert (output_dir / "api" / "func.md").exists()
        assert not (sub / "func.md").exists()  # not written in input dir

    def test_output_dir_created_automatically(self, tmp_path):
        """output_dir that doesn't exist yet is created on demand."""
        html = tmp_path / "page.html"
        html.write_text('<html><body><article>X</article></body></html>', encoding="utf-8")
        out = tmp_path / "does_not_exist" / "nested"
        status, rel, msg = self._call(html, tmp_path, out)
        assert status == "success"
        assert (out / "page.md").exists()

    def test_path_traversal_blocked(self, tmp_path):
        secret = tmp_path.parent / "secret.html"
        secret.write_text("<html><body>secret</body></html>", encoding="utf-8")
        outdir = tmp_path / "out"
        outdir.mkdir()
        status, rel, msg = self._call(secret, tmp_path, outdir)
        assert status == "error"
        assert "traversal" in msg.lower() or "outside" in msg.lower()

    def test_excluded_by_pattern(self, tmp_path):
        html = tmp_path / "genindex.html"
        html.write_text('<html><body><article>Index</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(html, tmp_path, tmp_path, excludes=["genindex"])
        assert status == "skipped"

    def test_no_content_element_skipped(self, tmp_path):
        html = tmp_path / "empty.html"
        html.write_text("<html><body><p>no match</p></body></html>", encoding="utf-8")
        status, rel, msg = self._call(html, tmp_path, tmp_path)
        assert status == "skipped"
        assert "No main content" in msg

    def test_encoding_errors_replaced(self, tmp_path):
        html = tmp_path / "bad.html"
        html.write_bytes(b"<html><body><article>\xff\xfe bad</article></body></html>")
        status, rel, msg = self._call(html, tmp_path, tmp_path)
        assert status in ("success", "skipped", "error")

    def test_strip_tags_applied(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text(
            '<html><body><article>Good<script>evil()</script></article></body></html>',
            encoding="utf-8",
        )
        status, _, _ = self._call(html, tmp_path, tmp_path)
        assert status == "success"
        md = (tmp_path / "page.md").read_text(encoding="utf-8")
        assert "evil" not in md
        assert "Good" in md


# ===========================================================================
# 17. _process_single_html_file (5-tuple wrapper)
# ===========================================================================

class TestProcessSingleHtmlFile:
    _SELECTORS = ["article.bd-article", 'article[role="main"]', 'div[role="main"]', "main", "article"]
    _STRIP_TAGS = ["script", "style"]

    def _call(self, html_path, outdir, excludes=None, selectors=None, strip_tags=None):
        return _mod._process_single_html_file((
            html_path, outdir, excludes or [], selectors or self._SELECTORS,
            strip_tags if strip_tags is not None else self._STRIP_TAGS,
        ))

    def test_success(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text('<html><body><article>Hello world</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status == "success"
        assert (tmp_path / "page.md").exists()

    def test_excluded_by_pattern(self, tmp_path):
        html = tmp_path / "genindex.html"
        html.write_text('<html><body><article>Index</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(str(html), str(tmp_path), excludes=["genindex"])
        assert status == "skipped"

    def test_no_content_element(self, tmp_path):
        html = tmp_path / "empty.html"
        html.write_text("<html><body><p>no main element</p></body></html>", encoding="utf-8")
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status == "skipped"
        assert "No main content" in msg

    def test_path_traversal_blocked(self, tmp_path):
        other = tmp_path.parent / "secret.html"
        other.write_text("<html><body>secret</body></html>", encoding="utf-8")
        outdir = tmp_path / "out"
        outdir.mkdir()
        status, rel, msg = self._call(str(other), str(outdir))
        assert status == "error"

    def test_encoding_error_replaced(self, tmp_path):
        html = tmp_path / "bad.html"
        html.write_bytes(b"<html><body><article>\xff\xfe bad bytes</article></body></html>")
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status in ("success", "skipped", "error")

    def test_nested_subdir(self, tmp_path):
        sub = tmp_path / "api" / "mod"
        sub.mkdir(parents=True)
        html = sub / "func.html"
        html.write_text('<html><body><article class="bd-article">API</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status == "success"
        assert (sub / "func.md").exists()

    def test_wrong_tuple_length_raises(self, tmp_path):
        with pytest.raises((TypeError, ValueError)):
            _mod._process_single_html_file((str(tmp_path / "x.html"), str(tmp_path), [], ["article"]))


# ===========================================================================
# 18. process_html_directory (standalone walker, new)
# ===========================================================================

class TestProcessHtmlDirectory:
    """Integration tests for the non-Sphinx standalone walker."""

    def _make_site(self, root: Path, theme: str = "generic") -> Path:
        """Create a minimal HTML site tree."""
        site = root / "site"
        site.mkdir()
        (site / "index.html").write_text(
            '<html><body><main><h1>Home</h1></main></body></html>', encoding="utf-8"
        )
        sub = site / "docs"
        sub.mkdir()
        (sub / "guide.html").write_text(
            '<html><body><article>Guide content</article></body></html>', encoding="utf-8"
        )
        (site / "genindex.html").write_text(
            '<html><body><main>Index</main></body></html>', encoding="utf-8"
        )
        return site

    def test_basic_success(self, tmp_path):
        site = self._make_site(tmp_path)
        stats = _mod.process_html_directory(site, max_workers=1)
        assert stats["generated"] >= 1
        assert stats["errors"] == 0

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
        out = tmp_path / "markdown_output"
        _mod.process_html_directory(site, output_dir=out, max_workers=1)
        assert out.exists()
        md_files = list(out.rglob("*.md"))
        assert len(md_files) >= 1

    def test_non_recursive_mode(self, tmp_path):
        site = self._make_site(tmp_path)
        stats = _mod.process_html_directory(site, recursive=False, max_workers=1)
        # Only top-level HTML files (index.html, genindex.html)
        # genindex is excluded so at most 1 generated
        assert stats["generated"] + stats["skipped"] <= 2

    def test_theme_preset_mkdocs(self, tmp_path):
        site = tmp_path / "mkdocs_site"
        site.mkdir()
        (site / "index.html").write_text(
            '<html><body><div class="md-content"><h1>MkDocs</h1></div></body></html>',
            encoding="utf-8",
        )
        stats = _mod.process_html_directory(
            site, theme_preset="mkdocs", max_workers=1
        )
        assert stats["generated"] == 1

    def test_custom_exclude_patterns(self, tmp_path):
        site = self._make_site(tmp_path)
        stats = _mod.process_html_directory(
            site, exclude_patterns=["index", "genindex"], max_workers=1
        )
        assert not (site / "index.md").exists()

    def test_returns_dict_with_correct_keys(self, tmp_path):
        site = self._make_site(tmp_path)
        stats = _mod.process_html_directory(site, max_workers=1)
        assert set(stats.keys()) == {"generated", "skipped", "errors"}

    def test_nonexistent_input_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _mod.process_html_directory(tmp_path / "nonexistent")

    def test_file_as_input_raises(self, tmp_path):
        f = tmp_path / "file.html"
        f.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="not a directory"):
            _mod.process_html_directory(f)

    def test_missing_deps_raises_import_error(self, tmp_path):
        site = tmp_path / "s"
        site.mkdir()
        with patch.object(_mod, "_has_markdown_deps", return_value=False):
            with pytest.raises(ImportError, match="beautifulsoup4"):
                _mod.process_html_directory(site)

    def test_generate_llms_produces_file(self, tmp_path):
        site = self._make_site(tmp_path)
        _mod.process_html_directory(
            site, max_workers=1, generate_llms=True, base_url=""
        )
        # Only check if any md files were generated first
        md_files = list(site.rglob("*.md"))
        if md_files:
            assert (site / "llms.txt").exists()

    def test_plain_html_preset(self, tmp_path):
        site = tmp_path / "plain"
        site.mkdir()
        (site / "page.html").write_text(
            '<html><body><main>Plain content</main></body></html>', encoding="utf-8"
        )
        stats = _mod.process_html_directory(
            site, theme_preset="plain_html", max_workers=1
        )
        assert stats["generated"] == 1


# ===========================================================================
# 19. generate_llms_txt_standalone (new)
# ===========================================================================

class TestGenerateLlmsTxtStandalone:
    def test_basic_writes_file(self, tmp_path):
        (tmp_path / "page.md").write_text("# Page\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path)
        assert out.exists()
        assert "page.md" in out.read_text(encoding="utf-8")

    def test_returns_path(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        result = _mod.generate_llms_txt_standalone(tmp_path)
        assert isinstance(result, Path)

    def test_base_url_prepended(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Doc\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, base_url="https://example.com")
        content = out.read_text(encoding="utf-8")
        assert "https://example.com/doc.md" in content

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

    def test_max_entries_zero_writes_empty_file(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, max_entries=0)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.endswith(".md")]
        assert len(lines) == 0

    def test_full_content_embeds_markdown(self, tmp_path):
        (tmp_path / "page.md").write_text("# Hello\n\nWorld\n", encoding="utf-8")
        out = _mod.generate_llms_txt_standalone(tmp_path, full_content=True)
        content = out.read_text(encoding="utf-8")
        assert "# Hello" in content and "World" in content and "---" in content

    def test_custom_output_file(self, tmp_path):
        (tmp_path / "x.md").write_text("# X\n", encoding="utf-8")
        custom = tmp_path / "custom_name.txt"
        out = _mod.generate_llms_txt_standalone(tmp_path, output_file=custom)
        assert out == custom
        assert custom.exists()

    def test_nonexistent_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _mod.generate_llms_txt_standalone(tmp_path / "nonexistent")

    def test_empty_directory_writes_header_only(self, tmp_path):
        out = _mod.generate_llms_txt_standalone(tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "Documentation" in content
        md_lines = [l for l in content.splitlines() if l.endswith(".md")]
        assert md_lines == []


class TestThemeSelectorPresets:
    def test_presets_is_dict(self): assert isinstance(_mod._THEME_SELECTOR_PRESETS, dict)
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

    def test_each_preset_non_empty_tuple(self):
        for name, sels in _mod._THEME_SELECTOR_PRESETS.items():
            assert isinstance(sels, tuple) and len(sels) > 0, f"{name!r} is empty"

    def test_all_selectors_safe(self):
        for name, sels in _mod._THEME_SELECTOR_PRESETS.items():
            for sel in sels:
                assert _mod._validate_css_selector(sel), f"Unsafe in {name!r}: {sel!r}"

    def test_at_least_20_themes(self):
        assert len(_mod._THEME_SELECTOR_PRESETS) >= 20

    def test_mkdocs_material_selector(self):
        assert "article.md-content__inner" in _mod._THEME_SELECTOR_PRESETS["mkdocs_material"]

    def test_vitepress_selector(self):
        assert "div.vp-doc" in _mod._THEME_SELECTOR_PRESETS["vitepress"]


class TestResolveContentSelectors:
    def test_no_preset_returns_custom_then_defaults(self):
        result = _mod._resolve_content_selectors(None, ["div.custom"])
        assert result[0] == "div.custom"
        assert "main" in result

    def test_preset_adds_theme_selectors(self):
        result = _mod._resolve_content_selectors("furo", [])
        assert 'article[role="main"]' in result

    def test_custom_takes_priority(self):
        result = _mod._resolve_content_selectors("furo", ["div.custom"])
        assert result[0] == "div.custom"

    def test_no_duplicates(self):
        result = _mod._resolve_content_selectors("classic", ["main"])
        assert result.count("main") == 1

    def test_unknown_preset_falls_back(self):
        result = _mod._resolve_content_selectors("nonexistent_theme", [])
        assert "main" in result

    def test_empty_returns_defaults(self):
        assert _mod._resolve_content_selectors(None, []) == _mod._DEFAULT_CONTENT_SELECTORS

    def test_unsafe_custom_removed(self):
        result = _mod._resolve_content_selectors(None, ["<bad>", "article"])
        assert "<bad>" not in result

    def test_returns_tuple(self):
        assert isinstance(_mod._resolve_content_selectors(None, []), tuple)

    def test_never_empty(self):
        result = _mod._resolve_content_selectors(None, ["<bad>"])
        assert len(result) > 0

    def test_all_themes_resolve(self):
        for theme in _mod._THEME_SELECTOR_PRESETS:
            result = _mod._resolve_content_selectors(theme, [])
            assert len(result) > 0


# ===========================================================================
# 23. generate_markdown_files (Sphinx hook)
# ===========================================================================

class TestGenerateMarkdownFiles:
    def test_exception_passed_in(self, sphinx_app):
        _mod.generate_markdown_files(sphinx_app, exception=RuntimeError("build failed"))

    def test_wrong_builder_type(self, tmp_path):
        app = MagicMock()
        app.config = MagicMock()
        app.builder = MagicMock()
        _mod.generate_markdown_files(app, exception=None)

    def test_disabled_by_config(self, sphinx_app):
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

    def test_max_workers_auto_detect(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = None
        _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_max_workers_floor_one(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 0
        _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_theme_preset_used(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        sphinx_app.config.ai_assistant_theme_preset = "pydata_sphinx_theme"
        sphinx_app.config.ai_assistant_content_selectors = []
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert (tmp_html_tree / "api" / "module.md").exists()

    def test_no_html_files_zero_md(self, sphinx_app, tmp_path):
        empty = tmp_path / "empty_out"
        empty.mkdir()
        sphinx_app.builder.outdir = str(empty)
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert list(empty.rglob("*.md")) == []


# ===========================================================================
# 24. generate_llms_txt (Sphinx hook)
# ===========================================================================

class TestGenerateLlmsTxt:
    def test_exception_passed_in(self, sphinx_app):
        _mod.generate_llms_txt(sphinx_app, exception=RuntimeError("boom"))

    def test_markdown_disabled(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_markdown = False
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (Path(sphinx_app.builder.outdir) / "llms.txt").exists()

    def test_llms_txt_disabled(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_llms_txt = False
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (Path(sphinx_app.builder.outdir) / "llms.txt").exists()

    def test_wrong_builder(self):
        app = MagicMock()
        app.config = MagicMock()
        app.config.ai_assistant_generate_markdown = True
        app.config.ai_assistant_generate_llms_txt = True
        app.builder = MagicMock()
        _mod.generate_llms_txt(app, exception=None)

    def test_no_md_files(self, sphinx_app):
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

    def test_invalid_base_url_skipped(self, sphinx_app, tmp_html_tree):
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


class TestGenerateLlmsTxtExtended:
    def test_max_entries_limits_output(self, sphinx_app, tmp_html_tree):
        for i in range(5):
            (tmp_html_tree / f"page{i}.md").write_text(f"# Page{i}\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_max_entries = 2
        _mod.generate_llms_txt(sphinx_app, exception=None)
        lines = [l for l in (tmp_html_tree / "llms.txt").read_text().splitlines() if l.endswith(".md")]
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
# 25. add_ai_assistant_context
# ===========================================================================

class TestAddAiAssistantContext:
    def test_disabled(self):
        app = MagicMock()
        app.config.ai_assistant_enabled = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(app, "index", "page.html", ctx, None)
        assert "ai_assistant_config" not in ctx

    def test_enabled_injects_config(self, sphinx_app):
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
            "evil": {"url_template": "https://x.com/</script><script>alert(1)//", "enabled": True}
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "</script><script>" not in ctx["metatags"]

    def test_html_baseurl_takes_priority(self, sphinx_app):
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

    def test_dangerous_provider_url_filtered(self, sphinx_app):
        sphinx_app.config.ai_assistant_providers = {
            "safe": {"url_template": "https://claude.ai/new?q={prompt}", "enabled": True},
            "evil": {"url_template": "javascript:alert(document.cookie)", "enabled": True},
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        providers = ctx["ai_assistant_config"]["providers"]
        assert "safe" in providers and "evil" not in providers

    def test_valid_positions_not_warned(self, sphinx_app):
        for pos in ["sidebar", "title", "floating", "none"]:
            sphinx_app.config.ai_assistant_position = pos
            ctx: dict = {}
            with patch.object(_mod._get_logger(), "warning") as mock_warn:
                _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
                mock_warn.assert_not_called()


# ===========================================================================
# 26. setup()
# ===========================================================================

class TestSetup:
    @pytest.fixture()
    def app(self):
        mock_app = MagicMock()
        mock_app.config.html_static_path = []
        return mock_app

    def test_returns_version(self, app):
        assert _mod.setup(app)["version"] == _mod._VERSION

    def test_returns_parallel_safe(self, app):
        result = _mod.setup(app)
        assert result["parallel_read_safe"] is True
        assert result["parallel_write_safe"] is True

    def test_required_config_values_registered(self, app):
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
        }
        assert required.issubset(names)

    def test_events_connected(self, app):
        _mod.setup(app)
        event_names = [c[0][0] for c in app.connect.call_args_list]
        assert "html-page-context" in event_names
        assert event_names.count("build-finished") == 2

    def test_css_js_added(self, app):
        _mod.setup(app)
        app.add_css_file.assert_called_once_with("ai-assistant.css")
        app.add_js_file.assert_called_once_with("ai-assistant.js")

    def test_static_path_appended(self, app):
        _mod.setup(app)
        assert len(app.config.html_static_path) == 1

    def test_static_path_not_duplicated(self, app):
        static_path = str(Path(_mod.__file__).parent / "_static")
        app.config.html_static_path = [static_path]
        _mod.setup(app)
        _mod.setup(app)
        assert app.config.html_static_path.count(static_path) == 1

    def test_providers_default_is_full_registry(self, app):
        """Default providers must include all 9 providers."""
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        providers = args_map["ai_assistant_providers"]
        for name in ["claude", "chatgpt", "gemini", "ollama", "mistral",
                     "perplexity", "copilot", "groq", "you"]:
            assert name in providers


# ===========================================================================
# 27. Default content selectors
# ===========================================================================

class TestDefaultContentSelectors:
    def test_is_tuple(self): assert isinstance(_mod._DEFAULT_CONTENT_SELECTORS, tuple)
    def test_non_empty(self): assert len(_mod._DEFAULT_CONTENT_SELECTORS) >= 12
    def test_pydata_present(self): assert "article.bd-article" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_furo_present(self): assert 'article[role="main"]' in _mod._DEFAULT_CONTENT_SELECTORS
    def test_rtd_present(self): assert "div.rst-content" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_mkdocs_present(self): assert "div.md-content" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_vitepress_present(self): assert "div.vp-doc" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_jekyll_present(self): assert "article.post-content" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_main_present(self): assert "main" in _mod._DEFAULT_CONTENT_SELECTORS
    def test_article_present(self): assert "article" in _mod._DEFAULT_CONTENT_SELECTORS

    def test_all_selectors_safe(self):
        for sel in _mod._DEFAULT_CONTENT_SELECTORS:
            assert _mod._validate_css_selector(sel)


# ===========================================================================
# 28. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_html_to_markdown_empty_string(self):
        assert isinstance(_mod.html_to_markdown(""), str)

    def test_html_to_markdown_only_whitespace(self):
        assert isinstance(_mod.html_to_markdown("   \n\t  "), str)

    def test_process_file_converter_crash_returns_error(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text("<html><body><article>OK</article></body></html>", encoding="utf-8")
        with patch("scikitplot._externals._sphinx_ext._sphinx_ai_assistant.html_to_markdown",
                   side_effect=RuntimeError("converter exploded")):
            status, rel, msg = _mod._process_single_html_file((
                str(html), str(tmp_path), [], ["article"], ["script", "style"],
            ))
        assert status == "error" and "converter exploded" in msg

    def test_6tuple_worker_converter_crash_returns_error(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text("<html><body><article>OK</article></body></html>", encoding="utf-8")
        with patch("scikitplot._externals._sphinx_ext._sphinx_ai_assistant.html_to_markdown",
                   side_effect=RuntimeError("worker crash")):
            status, rel, msg = _mod._process_html_file_worker((
                str(html), str(tmp_path), str(tmp_path), [], ["article"], ["script"],
            ))
        assert status == "error" and "worker crash" in msg

    def test_generate_markdown_files_no_html_files(self, sphinx_app, tmp_path):
        empty = tmp_path / "empty_out"
        empty.mkdir()
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
        assert {"web", "local", "api"}.issubset(_mod._PROVIDER_TYPES)

    def test_jupyter_content_selectors_tuple(self):
        assert isinstance(_mod._JUPYTER_CONTENT_SELECTORS, tuple)
        assert len(_mod._JUPYTER_CONTENT_SELECTORS) >= 5

    def test_localhost_re_matches_localhost(self):
        assert _mod._LOCALHOST_RE.match("http://localhost:11434")
        assert _mod._LOCALHOST_RE.match("http://127.0.0.1:11434")
        assert not _mod._LOCALHOST_RE.match("http://remote.host")


# ===========================================================================
# 29. _coerce_to_list
# ===========================================================================

class TestCoerceToList:
    """Tests for the _coerce_to_list type-coercion helper."""

    def test_none_returns_empty_list(self):
        assert _mod._coerce_to_list(None) == []

    def test_none_with_default_returns_copy(self):
        default = ["main", "article"]
        result = _mod._coerce_to_list(None, default=default)
        assert result == ["main", "article"]
        # Must be a copy, not the same object
        assert result is not default

    def test_none_default_is_none_returns_empty(self):
        assert _mod._coerce_to_list(None, default=None) == []

    def test_single_string_wrapped(self):
        assert _mod._coerce_to_list("article") == ["article"]

    def test_single_string_ignores_default(self):
        assert _mod._coerce_to_list("article", default=["main"]) == ["article"]

    def test_list_of_strings_returned(self):
        assert _mod._coerce_to_list(["article", "main"]) == ["article", "main"]

    def test_tuple_coerced(self):
        result = _mod._coerce_to_list(("article", "main"))
        assert result == ["article", "main"]
        assert isinstance(result, list)

    def test_non_string_items_converted(self):
        result = _mod._coerce_to_list([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_empty_list_returned_as_is(self):
        assert _mod._coerce_to_list([]) == []

    def test_empty_string_wrapped(self):
        assert _mod._coerce_to_list("") == [""]

    def test_returns_new_list_from_list_input(self):
        inp = ["a", "b"]
        result = _mod._coerce_to_list(inp)
        assert result == inp
        assert result is not inp

    def test_generator_coerced(self):
        result = _mod._coerce_to_list(x for x in ["a", "b"])
        assert result == ["a", "b"]


class TestHtmlToMarkdownStripTagsCoercion:
    """Verify html_to_markdown accepts str | list[str] | None for strip_tags."""

    def test_none_uses_default_strips_script(self):
        result = _mod.html_to_markdown(
            "<div><script>evil()</script><p>Hello</p></div>",
            strip_tags=None,
        )
        assert "evil" not in result
        assert "Hello" in result

    def test_single_string_strip_tag(self):
        result = _mod.html_to_markdown(
            "<div><nav>skip</nav><p>Keep</p></div>",
            strip_tags="nav",
        )
        assert "Keep" in result

    def test_list_strip_tags_unchanged(self):
        result = _mod.html_to_markdown(
            "<div><nav>skip</nav><footer>skip</footer><p>Keep</p></div>",
            strip_tags=["nav", "footer"],
        )
        assert "Keep" in result

    def test_empty_list_strips_nothing_extra(self):
        result = _mod.html_to_markdown(
            "<p>Hello</p>",
            strip_tags=[],
        )
        assert isinstance(result, str)

    def test_returns_string_always(self):
        assert isinstance(_mod.html_to_markdown("", strip_tags=None), str)
        assert isinstance(_mod.html_to_markdown("", strip_tags="nav"), str)
        assert isinstance(_mod.html_to_markdown("", strip_tags=[]), str)


# ===========================================================================
# 34. Extended provider registry
# ===========================================================================

class TestProviderRegistryExtended:
    """Verify the full default provider registry is correct and complete."""

    EXPECTED_PROVIDERS = [
        "claude", "chatgpt", "gemini", "ollama",
        "mistral", "perplexity", "copilot", "groq", "you",
    ]

    def test_claude_model_is_current(self):
        """Claude model should be claude-sonnet-4-6 (current production)."""
        assert _mod._DEFAULT_PROVIDERS["claude"]["model"] == "claude-sonnet-4-6"

    def test_chatgpt_model_present(self):
        assert "model" in _mod._DEFAULT_PROVIDERS["chatgpt"]

    def test_gemini_enabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["gemini"]["enabled"] is True

    def test_ollama_type_is_local(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["type"] == "local"

    def test_all_expected_providers_present(self):
        for name in self.EXPECTED_PROVIDERS:
            assert name in _mod._DEFAULT_PROVIDERS, f"{name!r} missing"

    def test_all_providers_have_prompt_template_with_url(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            pt = cfg.get("prompt_template", "")
            assert "{url}" in pt, f"Provider {name!r} prompt_template missing {{url}}"

    def test_all_enabled_providers_have_safe_url(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            if cfg.get("enabled"):
                assert _mod._validate_provider_url_template(
                    cfg["url_template"]
                ), f"Provider {name!r} has unsafe url_template"

    def test_no_provider_url_uses_javascript_scheme(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            url = cfg.get("url_template", "")
            assert not url.lower().startswith("javascript:"), (
                f"Provider {name!r} has javascript: url_template"
            )

    def test_all_providers_pass_validate_provider(self):
        for name, cfg in _mod._DEFAULT_PROVIDERS.items():
            errors = _mod._validate_provider(cfg, name=name)
            assert errors == [], f"Provider {name!r} validation errors: {errors}"


class TestPublicApiSurface:
    """Verify all new public symbols are importable at module level."""

    def test_coerce_to_list_importable(self):
        assert callable(_mod._coerce_to_list)

    def test_display_jupyter_notebook_ai_button_importable(self):
        assert callable(_mod.display_jupyter_notebook_ai_button)

    def test_build_jupyter_widget_html_accepts_intention(self):
        import inspect
        sig = inspect.signature(_mod._build_jupyter_widget_html)
        assert "intention" in sig.parameters

    def test_build_jupyter_widget_html_accepts_custom_context(self):
        import inspect
        sig = inspect.signature(_mod._build_jupyter_widget_html)
        assert "custom_context" in sig.parameters

    def test_build_jupyter_widget_html_accepts_notebook_mode(self):
        import inspect
        sig = inspect.signature(_mod._build_jupyter_widget_html)
        assert "notebook_mode" in sig.parameters

    def test_build_jupyter_widget_html_accepts_include_outputs(self):
        import inspect
        sig = inspect.signature(_mod._build_jupyter_widget_html)
        assert "include_outputs" in sig.parameters

    def test_display_jupyter_ai_button_accepts_intention(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_ai_button)
        assert "intention" in sig.parameters

    def test_display_jupyter_ai_button_accepts_notebook_mode(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_ai_button)
        assert "notebook_mode" in sig.parameters

    def test_html_to_markdown_accepts_str_strip_tags(self):
        import inspect
        sig = inspect.signature(_mod.html_to_markdown)
        p = sig.parameters["strip_tags"]
        # Should default to None (accepts None, str, list)
        assert p.default is None

    def test_notebook_ai_button_default_include_outputs_false(self):
        """include_outputs defaults to False since v0.4.0 — opt-in for cell outputs."""
        import inspect
        sig = inspect.signature(_mod.display_jupyter_notebook_ai_button)
        assert sig.parameters["include_outputs"].default is False

    def test_notebook_ai_button_default_intention_none(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_notebook_ai_button)
        assert sig.parameters["intention"].default is None


# ===========================================================================
# 37. _OLLAMA_RECOMMENDED_MODELS
# ===========================================================================

class TestOllamaRecommendedModels:
    """Tests for the _OLLAMA_RECOMMENDED_MODELS constant."""

    def test_is_tuple(self):
        assert isinstance(_mod._OLLAMA_RECOMMENDED_MODELS, tuple)

    def test_non_empty(self):
        assert len(_mod._OLLAMA_RECOMMENDED_MODELS) >= 10

    def test_all_strings(self):
        for m in _mod._OLLAMA_RECOMMENDED_MODELS:
            assert isinstance(m, str), f"Non-string model: {m!r}"

    def test_no_empty_strings(self):
        for m in _mod._OLLAMA_RECOMMENDED_MODELS:
            assert m.strip(), "Empty model name in list"

    def test_qwen3_present(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert any("qwen3" in m for m in models)

    def test_llama3_present(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert any("llama3" in m for m in models)

    def test_gemma_present(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert any("gemma" in m for m in models)

    def test_deepseek_present(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert any("deepseek" in m for m in models)

    def test_phi_present(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert any("phi" in m for m in models)

    def test_mistral_present(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert any("mistral" in m for m in models)

    def test_all_have_tag_or_latest(self):
        """All model names must include ':' (tag separator) for clarity."""
        for m in _mod._OLLAMA_RECOMMENDED_MODELS:
            assert ":" in m, f"Model {m!r} missing version tag (e.g. ':latest')"

    def test_no_duplicates(self):
        models = _mod._OLLAMA_RECOMMENDED_MODELS
        assert len(models) == len(set(models))


# ===========================================================================
# 38. _DEFAULT_MCP_TOOLS
# ===========================================================================

class TestDefaultMcpTools:
    """Tests for the _DEFAULT_MCP_TOOLS constant."""

    REQUIRED_TOOLS = {"vscode", "claude_desktop", "cursor", "windsurf", "generic"}

    def test_is_dict(self):
        assert isinstance(_mod._DEFAULT_MCP_TOOLS, dict)

    def test_all_required_tools_present(self):
        for name in self.REQUIRED_TOOLS:
            assert name in _mod._DEFAULT_MCP_TOOLS, f"{name!r} missing"

    def test_all_tools_have_enabled(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert "enabled" in cfg, f"{name!r} missing 'enabled'"
            assert isinstance(cfg["enabled"], bool)

    def test_all_tools_disabled_by_default(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert cfg["enabled"] is False, f"{name!r} should be disabled by default"

    def test_all_tools_have_type(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert "type" in cfg, f"{name!r} missing 'type'"
            assert isinstance(cfg["type"], str)

    def test_all_tools_have_label(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert "label" in cfg and cfg["label"], f"{name!r} missing label"

    def test_all_tools_have_description(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            assert "description" in cfg and cfg["description"]

    def test_vscode_has_server_url(self):
        assert "server_url" in _mod._DEFAULT_MCP_TOOLS["vscode"]

    def test_claude_desktop_has_mcpb_url(self):
        assert "mcpb_url" in _mod._DEFAULT_MCP_TOOLS["claude_desktop"]

    def test_generic_has_transport(self):
        assert "transport" in _mod._DEFAULT_MCP_TOOLS["generic"]

    def test_vscode_transport_sse(self):
        assert _mod._DEFAULT_MCP_TOOLS["vscode"]["transport"] == "sse"


# ===========================================================================
# 39. _validate_mcp_tool
# ===========================================================================

class TestValidateMcpTool:
    """Tests for _validate_mcp_tool()."""

    def _valid_tool(self):
        return {
            "enabled": False,
            "type": "vscode",
            "label": "VS Code",
            "description": "Connect MCP",
        }

    def test_valid_tool_returns_empty(self):
        assert _mod._validate_mcp_tool(self._valid_tool()) == []

    def test_missing_enabled_reported(self):
        t = self._valid_tool()
        del t["enabled"]
        errs = _mod._validate_mcp_tool(t, "tool1")
        assert any("enabled" in e for e in errs)

    def test_missing_type_reported(self):
        t = self._valid_tool()
        del t["type"]
        errs = _mod._validate_mcp_tool(t)
        assert any("type" in e for e in errs)

    def test_missing_label_reported(self):
        t = self._valid_tool()
        del t["label"]
        errs = _mod._validate_mcp_tool(t)
        assert any("label" in e for e in errs)

    def test_missing_description_reported(self):
        t = self._valid_tool()
        del t["description"]
        errs = _mod._validate_mcp_tool(t)
        assert any("description" in e for e in errs)

    def test_safe_server_url_accepted(self):
        t = self._valid_tool()
        t["server_url"] = "http://localhost:9999/mcp"
        assert _mod._validate_mcp_tool(t) == []

    def test_unsafe_server_url_rejected(self):
        t = self._valid_tool()
        t["server_url"] = "javascript:evil()"
        errs = _mod._validate_mcp_tool(t)
        assert any("server_url" in e for e in errs)

    def test_empty_server_url_accepted(self):
        t = self._valid_tool()
        t["server_url"] = ""
        assert _mod._validate_mcp_tool(t) == []

    def test_all_default_mcp_tools_validate_cleanly(self):
        for name, cfg in _mod._DEFAULT_MCP_TOOLS.items():
            errs = _mod._validate_mcp_tool(cfg, name=name)
            assert errs == [], f"MCP tool {name!r} errors: {errs}"

    def test_name_appears_in_error_prefix(self):
        t = self._valid_tool()
        del t["label"]
        errs = _mod._validate_mcp_tool(t, name="myTool")
        assert any("myTool" in e for e in errs)


# ===========================================================================
# 40. _cfg_str and _cfg_bool helpers
# ===========================================================================

class TestCfgHelpers:
    """Tests for _cfg_str() and _cfg_bool() defensive config accessors."""

    def test_cfg_str_returns_string(self):
        cfg = MagicMock()
        cfg.my_key = "hello"
        assert _mod._cfg_str(cfg, "my_key") == "hello"

    def test_cfg_str_returns_none_for_mock(self):
        cfg = MagicMock()
        # Accessing any undefined attribute on MagicMock returns a MagicMock
        result = _mod._cfg_str(cfg, "undefined_key_xyz")
        assert result is None

    def test_cfg_str_returns_none_for_none_value(self):
        cfg = MagicMock()
        cfg.my_key = None
        assert _mod._cfg_str(cfg, "my_key") is None

    def test_cfg_str_returns_none_for_int(self):
        cfg = MagicMock()
        cfg.my_key = 42
        assert _mod._cfg_str(cfg, "my_key") is None

    def test_cfg_str_missing_key_returns_none(self):
        class Cfg:
            pass
        assert _mod._cfg_str(Cfg(), "no_such_attr") is None

    def test_cfg_bool_returns_true(self):
        cfg = MagicMock()
        cfg.my_key = True
        assert _mod._cfg_bool(cfg, "my_key") is True

    def test_cfg_bool_returns_false(self):
        cfg = MagicMock()
        cfg.my_key = False
        assert _mod._cfg_bool(cfg, "my_key") is False

    def test_cfg_bool_returns_default_for_mock(self):
        cfg = MagicMock()
        # MagicMock attribute → fallback to default
        result = _mod._cfg_bool(cfg, "undefined_xyz", default=False)
        assert result is False

    def test_cfg_bool_default_true_for_mock(self):
        cfg = MagicMock()
        result = _mod._cfg_bool(cfg, "undefined_xyz", default=True)
        assert result is True

    def test_cfg_bool_int_coerced(self):
        cfg = MagicMock()
        cfg.my_key = 1
        assert _mod._cfg_bool(cfg, "my_key") is True

    def test_cfg_bool_zero_is_false(self):
        cfg = MagicMock()
        cfg.my_key = 0
        assert _mod._cfg_bool(cfg, "my_key") is False


# ===========================================================================
# 41. Extended provider registry (deepseek, huggingface, custom)
# ===========================================================================

class TestExtendedProviderRegistry:
    """Tests for newly added providers and updated registry."""

    def test_deepseek_present(self):
        assert "deepseek" in _mod._DEFAULT_PROVIDERS

    def test_deepseek_is_web_type(self):
        assert _mod._DEFAULT_PROVIDERS["deepseek"]["type"] == "web"

    def test_deepseek_url_safe(self):
        url = _mod._DEFAULT_PROVIDERS["deepseek"]["url_template"]
        assert _mod._validate_provider_url_template(url)

    def test_deepseek_model_present(self):
        assert _mod._DEFAULT_PROVIDERS["deepseek"]["model"]

    def test_huggingface_present(self):
        assert "huggingface" in _mod._DEFAULT_PROVIDERS

    def test_huggingface_url_huggingface_co(self):
        url = _mod._DEFAULT_PROVIDERS["huggingface"]["url_template"]
        assert "huggingface.co" in url

    def test_huggingface_is_web_type(self):
        assert _mod._DEFAULT_PROVIDERS["huggingface"]["type"] == "web"

    def test_huggingface_model_open_source(self):
        model = _mod._DEFAULT_PROVIDERS["huggingface"]["model"]
        # Should be an open model (Llama, Qwen, Gemma, etc.)
        assert model and "/" in model  # HF model IDs use org/name format

    def test_custom_present(self):
        assert "custom" in _mod._DEFAULT_PROVIDERS

    def test_custom_type_is_custom(self):
        assert _mod._DEFAULT_PROVIDERS["custom"]["type"] == "custom"

    def test_custom_disabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["custom"]["enabled"] is False

    def test_custom_url_template_empty_is_valid(self):
        url = _mod._DEFAULT_PROVIDERS["custom"]["url_template"]
        # Empty URL template is valid (API-only endpoint)
        assert _mod._validate_provider_url_template(url)

    def test_custom_type_in_provider_types(self):
        assert "custom" in _mod._PROVIDER_TYPES

    def test_total_providers_count(self):
        # claude, gemini, chatgpt, ollama, custom,
        # copilot, deepseek, groq, huggingface, mistral, perplexity, you = 12
        assert len(_mod._DEFAULT_PROVIDERS) >= 12

    def test_all_new_providers_validate_cleanly(self):
        for name in ("deepseek", "huggingface", "custom"):
            errs = _mod._validate_provider(
                _mod._DEFAULT_PROVIDERS[name], name=name
            )
            assert errs == [], f"{name!r} errors: {errs}"

    def test_ollama_description_mentions_models(self):
        desc = _mod._DEFAULT_PROVIDERS["ollama"]["description"]
        assert any(kw in desc for kw in ("Gemma", "Qwen", "Llama", "DeepSeek"))

    def test_ollama_model_updated(self):
        # Should use llama3.2:latest with tag
        model = _mod._DEFAULT_PROVIDERS["ollama"]["model"]
        assert ":" in model  # must have version tag

    def test_gemini_model_updated(self):
        model = _mod._DEFAULT_PROVIDERS["gemini"]["model"]
        assert "gemini" in model.lower()

    def test_perplexity_model_updated(self):
        model = _mod._DEFAULT_PROVIDERS["perplexity"]["model"]
        assert model  # non-empty


class TestAIAssistantDirective:
    """Tests for the AIAssistantDirective Sphinx RST directive."""

    def test_class_exists(self):
        assert hasattr(_mod, "AIAssistantDirective")

    def test_class_is_callable(self):
        assert callable(_mod.AIAssistantDirective)

    def test_run_method_exists(self):
        assert callable(getattr(_mod.AIAssistantDirective, "run", None))

    def test_option_spec_attr_exists(self):
        assert hasattr(_mod.AIAssistantDirective, "option_spec")
        assert isinstance(_mod.AIAssistantDirective.option_spec, dict)

    def test_required_arguments_zero(self):
        assert _mod.AIAssistantDirective.required_arguments == 0

    def test_has_content_false(self):
        assert _mod.AIAssistantDirective.has_content is False

    def test_run_returns_raw_html_node(self):
        """run() must produce a docutils raw node with HTML content."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        directive = _mod.AIAssistantDirective.__new__(_mod.AIAssistantDirective)
        directive.options = {
            "providers": "claude,chatgpt",
            "position": "inline",
            "intention": "Help me understand",
        }
        result = directive.run()
        assert len(result) == 1
        node = result[0]
        assert isinstance(node, nodes.raw)
        assert "claude" in node.astext().lower() or "ai-btn" in node.astext()

    def test_run_with_empty_options(self):
        """run() with no options must not raise."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        directive = _mod.AIAssistantDirective.__new__(_mod.AIAssistantDirective)
        directive.options = {}
        result = directive.run()
        assert len(result) == 1

    def test_run_include_raw_image_option(self):
        """include_raw_image option is parsed as boolean string."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        directive = _mod.AIAssistantDirective.__new__(_mod.AIAssistantDirective)
        directive.options = {"include_raw_image": "true"}
        result = directive.run()
        html = result[0].astext()
        assert "INCLUDE_RAW_IMAGE = true" in html

    def test_run_position_floating(self):
        """Floating position propagated to widget HTML."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        directive = _mod.AIAssistantDirective.__new__(_mod.AIAssistantDirective)
        directive.options = {"position": "floating"}
        result = directive.run()
        html = result[0].astext()
        assert "fixed" in html or "floating" in html.lower()

    def test_build_directive_node_function(self):
        """_build_ai_assistant_directive_node returns non-empty HTML."""
        html = _mod._build_ai_assistant_directive_node(
            providers=["claude"],
            position="inline",
            intention="Test",
            custom_context=None,
            page_url="",
            mcp_tools=None,
            include_raw_image=False,
        )
        assert isinstance(html, str)
        assert "ai-btn" in html or "claude" in html.lower()

    def test_directive_node_xss_safe_intention(self):
        """Malicious intention in directive is XSS-escaped."""
        html = _mod._build_ai_assistant_directive_node(
            providers=["claude"],
            position="inline",
            intention="</script><img onerror=hack>",
            custom_context=None,
            page_url="",
            mcp_tools=None,
            include_raw_image=False,
        )
        assert "</script><img" not in html


# ===========================================================================
# 46. Sphinx role — _ai_ask_role
# ===========================================================================

class TestAiAskRole:
    """Tests for the _ai_ask_role inline RST role."""

    def test_function_exists(self):
        assert callable(_mod._ai_ask_role)

    def test_returns_list_pair(self):
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        inliner = MagicMock()
        nodes_out, msgs = _mod._ai_ask_role(
            "ai_ask", ":ai_ask:`How does this work?`",
            "How does this work?", 1, inliner,
        )
        assert isinstance(nodes_out, list)
        assert isinstance(msgs, list)

    def test_produces_raw_node(self):
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        inliner = MagicMock()
        nodes_out, _ = _mod._ai_ask_role(
            "ai_ask", ":ai_ask:`Help?`", "Help?", 1, inliner,
        )
        assert len(nodes_out) == 1
        assert isinstance(nodes_out[0], nodes.raw)

    def test_role_embeds_intention(self):
        """The role text becomes the intention in the widget HTML."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        inliner = MagicMock()
        nodes_out, _ = _mod._ai_ask_role(
            "ai_ask", ":ai_ask:`Explain this function`",
            "Explain this function", 1, inliner,
        )
        html = nodes_out[0].astext()
        assert "Explain this function" in html

    def test_role_empty_text_safe(self):
        """Empty role text does not crash."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        inliner = MagicMock()
        nodes_out, _ = _mod._ai_ask_role(
            "ai_ask", ":ai_ask:``", "", 1, inliner,
        )
        assert isinstance(nodes_out, list)

    def test_role_xss_safe(self):
        """Malicious role text is JSON-escaped."""
        try:
            from docutils import nodes
        except ImportError:
            pytest.skip("docutils not available")

        inliner = MagicMock()
        nodes_out, _ = _mod._ai_ask_role(
            "ai_ask",
            ":ai_ask:`</script><script>alert(1)</script>`",
            "</script><script>alert(1)</script>",
            1, inliner,
        )
        html = nodes_out[0].astext()
        assert "</script><script>" not in html


# ===========================================================================
# 47. setup() — new config values and directive/role registration
# ===========================================================================

class TestSetupExtended:
    """Tests for new config values and directive/role added in setup()."""

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
            "ai_assistant_notebook_mode",
            "ai_assistant_include_outputs",
            "ai_assistant_ollama_model",
        }
        assert new_required.issubset(names), (
            f"Missing: {new_required - names}"
        )

    def test_intention_default_is_none(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert args_map["ai_assistant_intention"] is None

    def test_include_raw_image_default_false(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert args_map["ai_assistant_include_raw_image"] is False

    def test_include_outputs_default_true(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert args_map["ai_assistant_include_outputs"] is True

    def test_notebook_mode_default_false(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        assert args_map["ai_assistant_notebook_mode"] is False

    def test_providers_now_includes_deepseek(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        providers = args_map["ai_assistant_providers"]
        assert "deepseek" in providers

    def test_providers_now_includes_huggingface(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        providers = args_map["ai_assistant_providers"]
        assert "huggingface" in providers

    def test_providers_now_includes_custom(self, app):
        _mod.setup(app)
        args_map = {c[0][0]: c[0][1] for c in app.add_config_value.call_args_list}
        providers = args_map["ai_assistant_providers"]
        assert "custom" in providers

    def test_directive_registered_when_docutils_available(self, app):
        """add_directive must be called with 'ai-assistant'."""
        try:
            import docutils  # noqa: F401
        except ImportError:
            pytest.skip("docutils not available")
        _mod.setup(app)
        directive_calls = [
            c[0][0] for c in app.add_directive.call_args_list
        ]
        assert "ai-assistant" in directive_calls

    def test_role_registered_when_docutils_available(self, app):
        """add_role must be called with 'ai_ask'."""
        try:
            import docutils  # noqa: F401
        except ImportError:
            pytest.skip("docutils not available")
        _mod.setup(app)
        role_calls = [c[0][0] for c in app.add_role.call_args_list]
        assert "ai_ask" in role_calls


# ===========================================================================
# 48. add_ai_assistant_context — new config fields
# ===========================================================================

class TestAddAiAssistantContextExtended:
    """Tests for new fields injected by add_ai_assistant_context()."""

    @pytest.fixture()
    def sphinx_app(self, tmp_html_tree):
        # _make_app is available from conftest imported at module scope
        from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests.conftest import _make_app
        return _make_app(str(tmp_html_tree))

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

    def test_custom_context_string_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_custom_context = "Python ML library"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["customContext"] == "Python ML library"

    def test_include_raw_image_false_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_include_raw_image = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["includeRawImage"] is False

    def test_include_raw_image_true_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_include_raw_image = True
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["includeRawImage"] is True

    def test_notebook_mode_false_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_notebook_mode = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["notebookMode"] is False

    def test_include_outputs_true_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_include_outputs = True
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["includeOutputs"] is True

    def test_include_outputs_false_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_include_outputs = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["includeOutputs"] is False

    def test_all_new_fields_present_in_config(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        cfg = ctx["ai_assistant_config"]
        for field in ("intention", "customContext", "customPromptPrefix",
                      "includeRawImage", "notebookMode", "includeOutputs"):
            assert field in cfg, f"Field {field!r} missing from config"

    def test_config_is_json_serialisable(self, sphinx_app):
        """The full config dict must be JSON-serialisable (no MagicMock values)."""
        import json
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        # Should not raise
        dumped = json.dumps(ctx["ai_assistant_config"])
        assert isinstance(dumped, str)

    def test_intention_xss_safe_in_metatags(self, sphinx_app):
        sphinx_app.config.ai_assistant_intention = "evil</script>"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "evil</script>" not in ctx.get("metatags", "")


# ===========================================================================
# 49. Ollama provider — offline / local model support
# ===========================================================================

class TestOllamaLocalSupport:
    """End-to-end tests for Ollama offline provider configuration."""

    def test_ollama_api_base_url_default_localhost(self):
        assert "localhost" in _mod._DEFAULT_PROVIDERS["ollama"]["api_base_url"]

    def test_ollama_type_local(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["type"] == "local"

    def test_ollama_disabled_by_default(self):
        assert _mod._DEFAULT_PROVIDERS["ollama"]["enabled"] is False

    def test_ollama_validate_api_base_url_localhost(self):
        assert _mod._validate_ollama_url("http://localhost:11434") is True

    def test_ollama_validate_api_base_url_loopback(self):
        assert _mod._validate_ollama_url("http://127.0.0.1:11434") is True

    def test_ollama_validate_rejects_remote(self):
        assert _mod._validate_ollama_url("http://external-llm.example.com") is False

    def test_ollama_enabled_via_provider_configs_in_widget(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="oll1",
            providers=["ollama"],
            provider_configs={
                "ollama": {"enabled": True, "model": "qwen3:latest"}
            },
        )
        assert "ollama" in html.lower()

    def test_ollama_model_override_qwen3(self):
        """Ollama can be enabled with qwen3 model; widget renders the button."""
        html = _mod._build_jupyter_widget_html(
            widget_id="oll2",
            providers=["ollama"],
            provider_configs={
                "ollama": {
                    **_mod._DEFAULT_PROVIDERS["ollama"],
                    "enabled": True,
                    "model": "qwen3:latest",
                }
            },
        )
        # The button is rendered (Ollama label appears)
        assert "Ollama" in html or "ollama" in html.lower()

    def test_ollama_model_override_gemma3(self):
        """Ollama can be enabled with gemma3 model; widget renders the button."""
        html = _mod._build_jupyter_widget_html(
            widget_id="oll3",
            providers=["ollama"],
            provider_configs={
                "ollama": {
                    **_mod._DEFAULT_PROVIDERS["ollama"],
                    "enabled": True,
                    "model": "gemma3:latest",
                }
            },
        )
        assert "Ollama" in html or "ollama" in html.lower()

    def test_ollama_validate_provider_clean(self):
        errs = _mod._validate_provider(
            _mod._DEFAULT_PROVIDERS["ollama"], name="ollama"
        )
        assert errs == []

    def test_custom_api_base_url_remote_rejected_by_validate(self):
        """A custom Ollama provider pointing at a remote host must fail validation."""
        p = dict(_mod._DEFAULT_PROVIDERS["ollama"])
        p["api_base_url"] = "http://remote-server.example.com:11434"
        errs = _mod._validate_provider(p, name="ollama_remote")
        assert any("localhost" in e or "127.0.0.1" in e for e in errs)


# ===========================================================================
# 50. Full-provider round-trip — filter + widget
# ===========================================================================

class TestProviderRoundTrip:
    """Integration tests: filter → widget → JSON-clean."""

    def test_all_enabled_providers_render_in_widget(self):
        """Every enabled provider should produce at least its label in the widget."""
        enabled = {
            k: v for k, v in _mod._DEFAULT_PROVIDERS.items()
            if v.get("enabled")
        }
        for name, cfg in enabled.items():
            html = _mod._build_jupyter_widget_html(
                widget_id=f"rtrip_{name}",
                providers=[name],
            )
            assert cfg["label"] in html, (
                f"Provider {name!r} label {cfg['label']!r} not in widget"
            )

    def test_filter_providers_removes_only_invalid(self):
        mixed = {
            "good": {
                "url_template": "https://claude.ai/new?q={prompt}",
                "enabled": True,
            },
            "bad": {
                "url_template": "javascript:alert(1)",
                "enabled": True,
            },
        }
        result = _mod._filter_providers(mixed)
        assert "good" in result
        assert "bad" not in result

    def test_full_default_registry_filter_keeps_all_valid(self):
        result = _mod._filter_providers(_mod._DEFAULT_PROVIDERS)
        # All default providers have valid URL templates
        # (custom has empty URL which is also valid)
        for name in ("claude", "gemini", "chatgpt", "deepseek", "huggingface"):
            assert name in result

    def test_widget_html_with_all_providers_is_json_safe(self):
        import json
        all_providers = list(_mod._DEFAULT_PROVIDERS.keys())
        # Override to make all enabled for testing
        overrides = {
            name: {**cfg, "enabled": True}
            for name, cfg in _mod._DEFAULT_PROVIDERS.items()
        }
        html = _mod._build_jupyter_widget_html(
            widget_id="all",
            providers=all_providers,
            provider_configs=overrides,
        )
        # Extract the BUTTONS JSON from the HTML
        start = html.index("var BUTTONS          = ") + len("var BUTTONS          = ")
        end   = html.index(";\n  var EXPLICIT_CONTENT")
        buttons_json = html[start:end]
        parsed = json.loads(buttons_json)
        assert isinstance(parsed, list)
        # custom has empty url_template → filtered out
        valid_names = [b["name"] for b in parsed]
        assert "claude" in valid_names
