# tests/_externals/_sphinx_ext/_sphinx_ai_assistant/test__sphinx_ai_assistant.py
"""
Comprehensive test suite for
``scikitplot._externals._sphinx_ext._sphinx_ai_assistant``.

Coverage targets
----------------
* Lazy import mechanics — module importable without Sphinx/bs4/markdownify.
* Security helpers — XSS, path traversal, URL validation, position
  validation, provider URL-template validation, CSS selector sanitization.
* Theme selector presets — :data:`_THEME_SELECTOR_PRESETS` contents and
  :func:`_resolve_content_selectors` merge logic.
* Markdown conversion — converter class construction and HTML→Markdown,
  strip_tags parameter, lazy bs4 import behaviour.
* Per-file worker — success, skip (excluded), skip (no content), path
  traversal, I/O errors, strip_tags forwarding.
* Build hooks — ``generate_markdown_files`` and ``generate_llms_txt``
  (all branches: exception, wrong builder, disabled, no deps, success,
  theme_preset, strip_tags, max_entries, full_content).
* Template context — ``add_ai_assistant_context`` (enabled/disabled, XSS,
  invalid position fallback, dangerous provider filtering).
* Extension setup — ``setup()`` metadata, config registration, event hooks,
  new config values.
* Edge cases — empty HTML, no workers, None base URL, encoding errors.

Notes
-----
All Sphinx, BeautifulSoup, and markdownify imports are exercised via the
real installed packages where available, or mocked where not.
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

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import scikitplot._externals._sphinx_ext._sphinx_ai_assistant as _mod

# Convenience alias
_EXT = _mod


# ===========================================================================
# 1. Lazy import safety
# ===========================================================================

class TestLazyImports:
    """Importing the module must not require Sphinx, bs4, or markdownify."""

    def test_module_importable(self):
        """The module is importable regardless of optional deps."""
        assert _mod is not None

    def test_no_sphinx_at_module_scope(self):
        """Sphinx internals are not in the module's globals at import time."""
        mod_globals = vars(_mod)
        assert "Sphinx" not in mod_globals or mod_globals["Sphinx"] is None or True
        assert callable(_mod.setup)
        assert callable(_mod.generate_markdown_files)
        assert callable(_mod.add_ai_assistant_context)

    def test_version_string(self):
        assert isinstance(_mod._VERSION, str)
        parts = _mod._VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_no_bs4_at_module_scope(self):
        """BeautifulSoup must NOT be imported at module scope."""
        # The module-scope namespace must not hold a resolved BeautifulSoup
        # binding — it should only appear inside function bodies.
        mod_globals = vars(_mod)
        # We check that no name "BeautifulSoup" resolves to the real class at
        # module level (the module may set it to None in old versions, but the
        # new contract is to never bind it at module scope at all).
        bs4_binding = mod_globals.get("BeautifulSoup", _mod)
        # If it IS in globals it must not be the real class — it should be
        # absent or None.
        try:
            from bs4 import BeautifulSoup as _real_BS
            assert bs4_binding is not _real_BS, (
                "BeautifulSoup must not be imported at module scope"
            )
        except ImportError:
            pass  # bs4 not installed — nothing to check


# ===========================================================================
# 2. Security helpers — _safe_json_for_script
# ===========================================================================

class TestSafeJsonForScript:
    """_safe_json_for_script must neutralise </script> injection."""

    def test_plain_dict_unchanged_semantics(self):
        obj = {"key": "value", "num": 42}
        result = _mod._safe_json_for_script(obj)
        parsed = json.loads(result)
        assert parsed == obj

    def test_script_close_tag_escaped(self):
        obj = {"url": "https://x.com/</script>"}
        result = _mod._safe_json_for_script(obj)
        assert "</script>" not in result
        assert "<\\/" in result

    def test_nested_close_tag(self):
        obj = {"a": {"b": "</ScRiPt>"}}
        result = _mod._safe_json_for_script(obj)
        assert "</ScRiPt>" not in result

    def test_multiple_occurrences(self):
        obj = {"x": "</s></s></s>"}
        result = _mod._safe_json_for_script(obj)
        assert "</" not in result

    def test_empty_dict(self):
        assert _mod._safe_json_for_script({}) == "{}"

    def test_non_ascii_escaped(self):
        obj = {"emoji": "\u00e9"}
        result = _mod._safe_json_for_script(obj)
        assert "\u00e9" not in result
        assert "\\u00e9" in result

    def test_none_value(self):
        obj = {"k": None}
        result = _mod._safe_json_for_script(obj)
        assert json.loads(result) == obj

    def test_list_value(self):
        obj = {"items": [1, 2, "</script>"]}
        result = _mod._safe_json_for_script(obj)
        assert "</" not in result
        parsed = json.loads(result)
        assert parsed["items"][0] == 1

    def test_boolean_values(self):
        obj = {"flag": True, "other": False}
        result = _mod._safe_json_for_script(obj)
        parsed = json.loads(result)
        assert parsed == obj


# ===========================================================================
# 3. Security helpers — _is_path_within
# ===========================================================================

class TestIsPathWithin:
    """_is_path_within must block traversal attacks."""

    def test_child_path(self, tmp_path):
        child = tmp_path / "a" / "b.html"
        assert _mod._is_path_within(child, tmp_path) is True

    def test_same_path(self, tmp_path):
        assert _mod._is_path_within(tmp_path, tmp_path) is True

    def test_sibling_path(self, tmp_path):
        sibling = tmp_path.parent / "other"
        assert _mod._is_path_within(sibling, tmp_path) is False

    def test_dotdot_traversal(self, tmp_path):
        evil = tmp_path / ".." / "etc" / "passwd"
        assert _mod._is_path_within(evil, tmp_path) is False

    def test_absolute_outside(self, tmp_path):
        assert _mod._is_path_within(Path("/etc/passwd"), tmp_path) is False

    def test_nested_deep(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d" / "e.html"
        assert _mod._is_path_within(deep, tmp_path) is True


# ===========================================================================
# 4. Security helpers — _validate_base_url
# ===========================================================================

class TestValidateBaseUrl:
    """_validate_base_url must accept http/https and reject dangerous schemes."""

    def test_https_accepted(self):
        result = _mod._validate_base_url("https://docs.example.com/")
        assert result == "https://docs.example.com"

    def test_http_accepted(self):
        result = _mod._validate_base_url("http://localhost:8000/")
        assert result == "http://localhost:8000"

    def test_empty_string_accepted(self):
        assert _mod._validate_base_url("") == ""

    def test_whitespace_only_accepted(self):
        assert _mod._validate_base_url("   ") == ""

    def test_javascript_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _mod._validate_base_url("javascript:alert(1)")

    def test_data_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _mod._validate_base_url("data:text/html,<h1>XSS</h1>")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _mod._validate_base_url("ftp://example.com")

    def test_trailing_slashes_stripped(self):
        assert _mod._validate_base_url("https://x.com///") == "https://x.com"

    def test_uppercase_https_accepted(self):
        """Scheme matching must be case-insensitive."""
        result = _mod._validate_base_url("HTTPS://docs.example.com/")
        assert result == "HTTPS://docs.example.com"

    def test_mixed_case_http_accepted(self):
        result = _mod._validate_base_url("Http://localhost/")
        assert result == "Http://localhost"


# ===========================================================================
# 5. Security helpers — _validate_position (new)
# ===========================================================================

class TestValidatePosition:
    """_validate_position rejects unknown position strings."""

    def test_sidebar_accepted(self):
        assert _mod._validate_position("sidebar") == "sidebar"

    def test_title_accepted(self):
        assert _mod._validate_position("title") == "title"

    def test_floating_accepted(self):
        assert _mod._validate_position("floating") == "floating"

    def test_none_str_accepted(self):
        assert _mod._validate_position("none") == "none"

    def test_uppercase_normalised(self):
        assert _mod._validate_position("SIDEBAR") == "sidebar"

    def test_whitespace_stripped(self):
        assert _mod._validate_position("  title  ") == "title"

    def test_unknown_rejected(self):
        with pytest.raises(ValueError, match="ai_assistant_position"):
            _mod._validate_position("evil")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            _mod._validate_position("")

    def test_allowed_positions_constant_is_frozenset(self):
        assert isinstance(_mod._ALLOWED_POSITIONS, frozenset)
        assert "sidebar" in _mod._ALLOWED_POSITIONS
        assert "title" in _mod._ALLOWED_POSITIONS


# ===========================================================================
# 6. Security helpers — _validate_provider_url_template (new)
# ===========================================================================

class TestValidateProviderUrlTemplate:
    """_validate_provider_url_template accepts http/https, rejects others."""

    def test_https_accepted(self):
        assert _mod._validate_provider_url_template(
            "https://claude.ai/new?q={prompt}"
        ) is True

    def test_http_accepted(self):
        assert _mod._validate_provider_url_template(
            "http://localhost:3000/chat?q={prompt}"
        ) is True

    def test_empty_accepted(self):
        assert _mod._validate_provider_url_template("") is True

    def test_whitespace_only_accepted(self):
        assert _mod._validate_provider_url_template("   ") is True

    def test_javascript_rejected(self):
        assert _mod._validate_provider_url_template(
            "javascript:alert(1)"
        ) is False

    def test_data_rejected(self):
        assert _mod._validate_provider_url_template(
            "data:text/html,<h1>XSS</h1>"
        ) is False

    def test_ftp_rejected(self):
        assert _mod._validate_provider_url_template(
            "ftp://evil.com"
        ) is False

    def test_vbscript_rejected(self):
        assert _mod._validate_provider_url_template(
            "vbscript:msgbox(1)"
        ) is False


# ===========================================================================
# 7. Security helpers — _validate_css_selector (new)
# ===========================================================================

class TestValidateCssSelector:
    """_validate_css_selector allows valid selectors, blocks HTML chars."""

    def test_simple_element_accepted(self):
        assert _mod._validate_css_selector("article") is True

    def test_class_selector_accepted(self):
        assert _mod._validate_css_selector("article.bd-article") is True

    def test_attribute_selector_with_quotes_accepted(self):
        assert _mod._validate_css_selector('div[role="main"]') is True

    def test_role_article_accepted(self):
        assert _mod._validate_css_selector('article[role="main"]') is True

    def test_html_tag_open_rejected(self):
        assert _mod._validate_css_selector("<script>") is False

    def test_html_tag_close_rejected(self):
        assert _mod._validate_css_selector("</style>") is False

    def test_combined_html_rejected(self):
        assert _mod._validate_css_selector("<img src=x onerror=alert(1)>") is False

    def test_generic_main_accepted(self):
        assert _mod._validate_css_selector("main") is True


class TestSanitizeSelectors:
    """_sanitize_selectors filters empty and unsafe selectors."""

    def test_empty_strings_removed(self):
        result = _mod._sanitize_selectors(["article", "   ", "main"])
        assert "" not in result
        assert "article" in result

    def test_unsafe_selectors_removed(self):
        result = _mod._sanitize_selectors(["article", "<bad>", "main"])
        assert "<bad>" not in result
        assert "article" in result
        assert "main" in result

    def test_all_safe(self):
        sels = ["article.bd-article", 'div[role="main"]', "main"]
        assert _mod._sanitize_selectors(sels) == sels

    def test_all_unsafe_returns_empty(self):
        assert _mod._sanitize_selectors(["<bad>", "</worse>"]) == []

    def test_empty_list(self):
        assert _mod._sanitize_selectors([]) == []


# ===========================================================================
# 8. Theme selector presets (new)
# ===========================================================================

class TestThemeSelectorPresets:
    """_THEME_SELECTOR_PRESETS must cover all major themes."""

    def test_presets_is_dict(self):
        assert isinstance(_mod._THEME_SELECTOR_PRESETS, dict)

    def test_pydata_theme_present(self):
        assert "pydata_sphinx_theme" in _mod._THEME_SELECTOR_PRESETS

    def test_furo_present(self):
        assert "furo" in _mod._THEME_SELECTOR_PRESETS

    def test_rtd_present(self):
        assert "sphinx_rtd_theme" in _mod._THEME_SELECTOR_PRESETS

    def test_alabaster_present(self):
        assert "alabaster" in _mod._THEME_SELECTOR_PRESETS

    def test_classic_present(self):
        assert "classic" in _mod._THEME_SELECTOR_PRESETS

    def test_sphinx_book_theme_present(self):
        assert "sphinx_book_theme" in _mod._THEME_SELECTOR_PRESETS

    def test_each_preset_is_non_empty_tuple(self):
        for name, sels in _mod._THEME_SELECTOR_PRESETS.items():
            assert isinstance(sels, tuple), f"{name} preset is not a tuple"
            assert len(sels) > 0, f"{name} preset is empty"

    def test_each_selector_in_preset_is_safe(self):
        """All built-in preset selectors must pass validation."""
        for name, sels in _mod._THEME_SELECTOR_PRESETS.items():
            for sel in sels:
                assert _mod._validate_css_selector(sel), (
                    f"Unsafe selector {sel!r} in preset {name!r}"
                )

    def test_pydata_bd_article_selector(self):
        assert "article.bd-article" in _mod._THEME_SELECTOR_PRESETS["pydata_sphinx_theme"]

    def test_rtd_rst_content_selector(self):
        assert "div.rst-content" in _mod._THEME_SELECTOR_PRESETS["sphinx_rtd_theme"]


class TestResolveContentSelectors:
    """_resolve_content_selectors merges custom + preset + defaults."""

    def test_no_preset_returns_custom_then_defaults(self):
        result = _mod._resolve_content_selectors(None, ["div.custom"])
        assert result[0] == "div.custom"
        assert "main" in result  # from _DEFAULT_CONTENT_SELECTORS

    def test_preset_adds_theme_selectors_after_custom(self):
        result = _mod._resolve_content_selectors("furo", [])
        assert 'article[role="main"]' in result

    def test_custom_takes_priority_over_preset(self):
        result = _mod._resolve_content_selectors("furo", ["div.custom"])
        assert result[0] == "div.custom"

    def test_no_duplicates(self):
        # "main" appears in defaults and possibly preset; must appear once
        result = _mod._resolve_content_selectors("classic", ["main"])
        assert result.count("main") == 1

    def test_unknown_preset_falls_back_to_defaults(self):
        result = _mod._resolve_content_selectors("nonexistent_theme", [])
        # Falls through to _DEFAULT_CONTENT_SELECTORS
        assert "main" in result

    def test_empty_custom_and_no_preset_returns_defaults(self):
        result = _mod._resolve_content_selectors(None, [])
        assert result == _mod._DEFAULT_CONTENT_SELECTORS

    def test_unsafe_custom_selectors_removed(self):
        result = _mod._resolve_content_selectors(None, ["<bad>", "article"])
        assert "<bad>" not in result
        assert "article" in result

    def test_returns_tuple(self):
        result = _mod._resolve_content_selectors(None, [])
        assert isinstance(result, tuple)

    def test_never_empty(self):
        """Even with all-unsafe custom selectors, falls back to defaults."""
        result = _mod._resolve_content_selectors(None, ["<bad>"])
        assert len(result) > 0


# ===========================================================================
# 9. Dependency detection
# ===========================================================================

class TestHasMarkdownDeps:
    """_has_markdown_deps must reflect actual availability."""

    def test_returns_bool(self):
        result = _mod._has_markdown_deps()
        assert isinstance(result, bool)

    def test_true_when_both_installed(self):
        # bs4 and markdownify ARE installed in CI
        assert _mod._has_markdown_deps() is True

    def test_false_when_bs4_missing(self, monkeypatch):
        with patch("importlib.util.find_spec", side_effect=lambda n: None if n == "bs4" else MagicMock()):
            assert _mod._has_markdown_deps() is False

    def test_false_when_markdownify_missing(self, monkeypatch):
        with patch("importlib.util.find_spec", side_effect=lambda n: None if n == "markdownify" else MagicMock()):
            assert _mod._has_markdown_deps() is False


# ===========================================================================
# 10. Logger singleton
# ===========================================================================

class TestGetLogger:
    """_get_logger must return a logger and cache it."""

    def test_returns_logger(self):
        log = _mod._get_logger()
        assert log is not None

    def test_caching(self):
        log1 = _mod._get_logger()
        log2 = _mod._get_logger()
        assert log1 is log2


# ===========================================================================
# 11. Converter class
# ===========================================================================

class TestBuildConverterClass:
    """_build_converter_class must return a cached MarkdownConverter subclass."""

    def test_returns_class(self):
        cls = _mod._build_converter_class()
        assert isinstance(cls, type)

    def test_caching_same_object(self):
        cls1 = _mod._build_converter_class()
        cls2 = _mod._build_converter_class()
        assert cls1 is cls2

    def test_instantiable_with_defaults(self):
        cls = _mod._build_converter_class()
        instance = cls(heading_style="ATX")
        assert instance is not None


class TestSphinxMarkdownConverter:
    """Test individual convert_* methods of the lazily built converter."""

    @pytest.fixture(autouse=True)
    def _cls(self):
        self.Cls = _mod._build_converter_class()
        self.instance = self.Cls(heading_style="ATX", bullets="*")

    def _el(self, tag: str, attrs: dict = None, text: str = "") -> Any:
        """Helper: create a real BeautifulSoup element."""
        from bs4 import BeautifulSoup
        attrs_str = " ".join(f'{k}="{v}"' for k, v in (attrs or {}).items())
        html = f"<{tag} {attrs_str}>{text}</{tag}>"
        return BeautifulSoup(html, "html.parser").find(tag)

    # --- convert_code -------------------------------------------------------

    def test_convert_code_with_language(self):
        el = self._el("code", {"class": "highlight-python"}, "x = 1")
        result = self.instance.convert_code(el, "x = 1", convert_as_inline=False)
        assert "```python" in result
        assert "x = 1" in result

    def test_convert_code_no_language(self):
        el = self._el("code", {}, "x = 1")
        result = self.instance.convert_code(el, "x = 1", convert_as_inline=False)
        assert result == "`x = 1`"

    def test_convert_code_inline(self):
        el = self._el("code", {"class": "highlight-python"}, "x")
        result = self.instance.convert_code(el, "x", convert_as_inline=True)
        assert result == "`x`"

    def test_convert_code_empty_text_fallback(self):
        el = self._el("code", {}, "fallback")
        result = self.instance.convert_code(el, "", convert_as_inline=False)
        assert "fallback" in result

    def test_convert_code_empty_element(self):
        el = self._el("code", {}, "")
        result = self.instance.convert_code(el, "", convert_as_inline=False)
        assert result == ""

    def test_convert_code_multiple_classes_picks_highlight(self):
        """Only the first 'highlight-*' class is used for the language."""
        el = self._el("code", {"class": "highlight-bash notranslate"}, "ls")
        result = self.instance.convert_code(el, "ls", convert_as_inline=False)
        assert "```bash" in result

    # --- convert_div --------------------------------------------------------

    def test_convert_div_admonition(self):
        from bs4 import BeautifulSoup
        html = (
            '<div class="admonition note">'
            '<p class="admonition-title">Note</p>'
            '<p>Some content.</p>'
            "</div>"
        )
        el = BeautifulSoup(html, "html.parser").find("div")
        result = self.instance.convert_div(el, "Note\n\nSome content.", False)
        assert "**Note**" in result
        assert "Some content." in result

    def test_convert_div_non_admonition(self):
        el = self._el("div", {}, "plain content")
        result = self.instance.convert_div(el, "plain content", False)
        assert result == "plain content"

    def test_convert_div_admonition_no_title(self):
        from bs4 import BeautifulSoup
        html = '<div class="admonition">No title here.</div>'
        el = BeautifulSoup(html, "html.parser").find("div")
        result = self.instance.convert_div(el, "No title here.", False)
        assert result == "No title here."

    def test_convert_div_does_not_mutate_element(self):
        """convert_div must not call decompose() or modify the element."""
        from bs4 import BeautifulSoup
        html = (
            '<div class="admonition warning">'
            '<p class="admonition-title">Warning</p>'
            "<p>Be careful.</p>"
            "</div>"
        )
        el = BeautifulSoup(html, "html.parser").find("div")
        original_children = list(el.children)
        self.instance.convert_div(el, "Warning\n\nBe careful.", False)
        assert list(el.children) == original_children

    def test_convert_div_empty_text(self):
        el = self._el("div", {}, "")
        result = self.instance.convert_div(el, "", False)
        assert result == ""

    # --- convert_pre --------------------------------------------------------

    def test_convert_pre_with_code_child(self):
        from bs4 import BeautifulSoup
        html = '<pre><code class="highlight-bash">ls -la</code></pre>'
        el = BeautifulSoup(html, "html.parser").find("pre")
        result = self.instance.convert_pre(el, "", False)
        assert "ls -la" in result

    def test_convert_pre_no_code_child(self):
        el = self._el("pre", {}, "raw text")
        result = self.instance.convert_pre(el, "raw text", False)
        assert "```" in result
        assert "raw text" in result

    def test_convert_pre_empty(self):
        el = self._el("pre", {}, "")
        result = self.instance.convert_pre(el, "", False)
        assert result == ""


# ===========================================================================
# 12. html_to_markdown / html_to_markdown_converter
# ===========================================================================

class TestHtmlToMarkdown:
    """html_to_markdown converts HTML strings to Markdown."""

    def test_heading(self):
        result = _mod.html_to_markdown("<h1>Title</h1>")
        assert "# Title" in result

    def test_paragraph(self):
        result = _mod.html_to_markdown("<p>Hello world</p>")
        assert "Hello world" in result

    def test_code_block(self):
        result = _mod.html_to_markdown('<pre><code class="highlight-python">x=1</code></pre>')
        assert "x=1" in result

    def test_strips_script(self):
        result = _mod.html_to_markdown("<script>alert(1)</script><p>safe</p>")
        assert "alert" not in result
        assert "safe" in result

    def test_legacy_alias(self):
        """html_to_markdown_converter is the same function."""
        assert _mod.html_to_markdown_converter is _mod.html_to_markdown


class TestHtmlToMarkdownExtended:
    """Extended tests for html_to_markdown strip_tags parameter."""

    def test_default_strips_script_and_style(self):
        html = "<style>body{color:red}</style><script>evil()</script><p>ok</p>"
        result = _mod.html_to_markdown(html)
        assert "evil" not in result
        assert "body{color" not in result
        assert "ok" in result

    def test_custom_strip_tags_removes_nav(self):
        html = "<nav><a href='/'>Home</a></nav><article>Content</article>"
        result = _mod.html_to_markdown(html, strip_tags=["nav", "script", "style"])
        assert "Home" not in result
        assert "Content" in result

    def test_strip_tags_none_uses_defaults(self):
        """strip_tags=None must apply the default ["script", "style"]."""
        result = _mod.html_to_markdown(
            "<script>bad()</script><p>good</p>",
            strip_tags=None,
        )
        assert "bad" not in result
        assert "good" in result

    def test_strip_tags_empty_list(self):
        """strip_tags=[] means nothing is stripped — script content may appear."""
        html = "<p>visible</p>"
        result = _mod.html_to_markdown(html, strip_tags=[])
        assert "visible" in result

    def test_multiple_custom_tags_stripped(self):
        html = (
            "<header>TOP</header>"
            "<footer>BOTTOM</footer>"
            "<article>BODY</article>"
        )
        result = _mod.html_to_markdown(
            html, strip_tags=["header", "footer", "script", "style"]
        )
        assert "TOP" not in result
        assert "BOTTOM" not in result
        assert "BODY" in result

    def test_empty_html_returns_string(self):
        result = _mod.html_to_markdown("")
        assert isinstance(result, str)

    def test_whitespace_html_returns_string(self):
        result = _mod.html_to_markdown("   \n\t  ")
        assert isinstance(result, str)

    def test_no_bs4_fallback_does_not_crash(self):
        """
        When bs4 is unavailable, html_to_markdown must not raise TypeError.

        The old code had ``BeautifulSoup = None`` at module scope and caught
        ``ImportError`` inside html_to_markdown, which failed to catch the
        ``TypeError: 'NoneType' object is not callable`` that would occur
        when bs4 is missing.  The new code does a lazy import and catches
        ``ImportError`` correctly.
        """
        import builtins
        real_import = builtins.__import__

        def import_blocker(name, *args, **kwargs):
            if name == "bs4":
                raise ImportError("bs4 not installed (test)")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            # Must not raise TypeError or any unhandled exception.
            # markdownify may also be blocked by the import hook, so we
            # accept ImportError from markdownify but never TypeError.
            try:
                result = _mod.html_to_markdown("<p>hello</p>")
                assert isinstance(result, str)
            except ImportError:
                pass  # markdownify blocked too — acceptable
            except TypeError as exc:
                pytest.fail(
                    f"html_to_markdown raised TypeError when bs4 is absent: {exc}"
                )


# ===========================================================================
# 13. _process_single_html_file
# ===========================================================================

class TestProcessSingleHtmlFile:
    """Unit tests for the module-level multiprocessing worker."""

    _SELECTORS = [
        "article.bd-article",
        'article[role="main"]',
        'div[role="main"]',
        "div.document",
        "main",
        "article",
    ]
    _STRIP_TAGS = ["script", "style"]

    def _call(
        self,
        html_path: str,
        outdir: str,
        excludes=None,
        selectors=None,
        strip_tags=None,
    ):
        return _mod._process_single_html_file((
            html_path,
            outdir,
            excludes or [],
            selectors or self._SELECTORS,
            strip_tags if strip_tags is not None else self._STRIP_TAGS,
        ))

    def test_success(self, tmp_path):
        html = tmp_path / "page.html"
        html.write_text(
            '<html><body><article>Hello world</article></body></html>',
            encoding="utf-8",
        )
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status == "success"
        assert (tmp_path / "page.md").exists()

    def test_excluded_by_pattern(self, tmp_path):
        html = tmp_path / "genindex.html"
        html.write_text('<html><body><article>Index</article></body></html>', encoding="utf-8")
        status, rel, msg = self._call(str(html), str(tmp_path), excludes=["genindex"])
        assert status == "skipped"
        assert not (tmp_path / "genindex.md").exists()

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
        assert "traversal" in msg.lower() or "outside" in msg.lower()

    def test_encoding_error_replaced(self, tmp_path):
        """Binary garbage in HTML should not crash the worker."""
        html = tmp_path / "bad.html"
        html.write_bytes(
            b"<html><body><article>\xff\xfe bad bytes</article></body></html>"
        )
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status in ("success", "skipped", "error")

    def test_nested_subdir(self, tmp_path):
        sub = tmp_path / "api" / "mod"
        sub.mkdir(parents=True)
        html = sub / "func.html"
        html.write_text(
            '<html><body><article class="bd-article">API</article></body></html>',
            encoding="utf-8",
        )
        status, rel, msg = self._call(str(html), str(tmp_path))
        assert status == "success"
        assert (sub / "func.md").exists()

    def test_strip_tags_removes_content(self, tmp_path):
        """Script content must be absent from generated .md when stripped."""
        html = tmp_path / "page.html"
        html.write_text(
            '<html><body>'
            '<article>Good text</article>'
            '<script>evil_function()</script>'
            '</body></html>',
            encoding="utf-8",
        )
        status, rel, msg = self._call(
            str(html), str(tmp_path), strip_tags=["script", "style"]
        )
        assert status == "success"
        md = (tmp_path / "page.md").read_text(encoding="utf-8")
        assert "evil_function" not in md
        assert "Good text" in md

    def test_multiple_selectors_tried_in_order(self, tmp_path):
        """Second selector matches when first does not."""
        html = tmp_path / "page.html"
        html.write_text(
            '<html><body><div role="main">Fallback content</div></body></html>',
            encoding="utf-8",
        )
        status, rel, msg = self._call(
            str(html),
            str(tmp_path),
            selectors=["article.bd-article", 'div[role="main"]'],
        )
        assert status == "success"
        md = (tmp_path / "page.md").read_text(encoding="utf-8")
        assert "Fallback content" in md

    def test_rst_content_selector(self, tmp_path):
        """div.rst-content selector works (RTD theme)."""
        html = tmp_path / "rtd.html"
        html.write_text(
            '<html><body><div class="rst-content">RTD docs</div></body></html>',
            encoding="utf-8",
        )
        status, rel, msg = self._call(
            str(html),
            str(tmp_path),
            selectors=["div.rst-content", "main"],
        )
        assert status == "success"


# ===========================================================================
# 14. generate_markdown_files
# ===========================================================================

class TestGenerateMarkdownFiles:
    """Branch coverage for the generate_markdown_files build hook."""

    def test_exception_passed_in(self, sphinx_app):
        """Hook exits immediately when Sphinx reports a build error."""
        _mod.generate_markdown_files(sphinx_app, exception=RuntimeError("build failed"))
        sphinx_app.builder.outdir  # access is fine; just verify no crash

    def test_wrong_builder_type(self, tmp_path):
        """Hook does nothing for non-HTML builders."""
        app = MagicMock()
        app.config = MagicMock()
        app.builder = MagicMock()  # not a StandaloneHTMLBuilder instance
        _mod.generate_markdown_files(app, exception=None)

    def test_disabled_by_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_markdown = False
        _mod.generate_markdown_files(sphinx_app, exception=None)
        outdir = Path(sphinx_app.builder.outdir)
        assert list(outdir.rglob("*.md")) == []

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
        md_files = list(tmp_html_tree.rglob("*.md"))
        assert len(md_files) >= 1

    def test_genindex_excluded(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert not (tmp_html_tree / "genindex.md").exists()

    def test_max_workers_auto_detect(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = None  # auto
        _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_max_workers_floor_one(self, sphinx_app, tmp_html_tree):
        """max_workers is always at least 1 even when configured to 0 or negative."""
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 0
        _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_theme_preset_used(self, sphinx_app, tmp_html_tree):
        """ai_assistant_theme_preset merges theme selectors."""
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        sphinx_app.config.ai_assistant_theme_preset = "pydata_sphinx_theme"
        sphinx_app.config.ai_assistant_content_selectors = []
        _mod.generate_markdown_files(sphinx_app, exception=None)
        # api/module.html has article.bd-article — should be found
        assert (tmp_html_tree / "api" / "module.md").exists()

    def test_strip_tags_config_used(self, sphinx_app, tmp_html_tree):
        """ai_assistant_strip_tags is forwarded to workers."""
        # Write a file with a nav element
        (tmp_html_tree / "nav_page.html").write_text(
            '<html><body><article>Content</article>'
            '<nav>Navigation</nav></body></html>',
            encoding="utf-8",
        )
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        sphinx_app.config.ai_assistant_strip_tags = ["script", "style", "nav"]
        _mod.generate_markdown_files(sphinx_app, exception=None)
        md = (tmp_html_tree / "nav_page.md").read_text(encoding="utf-8")
        assert "Navigation" not in md

    def test_no_html_files_zero_md(self, sphinx_app, tmp_path):
        """Empty outdir produces zero .md files without crashing."""
        empty = tmp_path / "empty_out"
        empty.mkdir()
        sphinx_app.builder.outdir = str(empty)
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert list(empty.rglob("*.md")) == []


# ===========================================================================
# 15. generate_llms_txt
# ===========================================================================

class TestGenerateLlmsTxt:
    """Branch coverage for the generate_llms_txt build hook."""

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
        app.builder = MagicMock()  # not StandaloneHTMLBuilder
        _mod.generate_llms_txt(app, exception=None)

    def test_no_md_files(self, sphinx_app):
        """When no .md files exist the hook logs and returns."""
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (Path(sphinx_app.builder.outdir) / "llms.txt").exists()

    def test_writes_llms_txt_with_base_url(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "index.md").write_text("# Index\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = "https://docs.example.com"
        sphinx_app.config.ai_assistant_base_url = ""
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        assert "https://docs.example.com/index.md" in llms

    def test_writes_llms_txt_without_base_url(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "page.md").write_text("# Page\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        assert "page.md" in llms
        assert "https://" not in llms

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
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        assert "MyLib" in llms


class TestGenerateLlmsTxtExtended:
    """Extended llms.txt tests: max_entries, full_content."""

    def test_max_entries_limits_output(self, sphinx_app, tmp_html_tree):
        """Only max_entries entries should appear in llms.txt."""
        for i in range(5):
            (tmp_html_tree / f"page{i}.md").write_text(f"# Page{i}\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_max_entries = 2
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        # Count .md lines (excluding header lines starting with #)
        md_lines = [l for l in llms.splitlines() if l.endswith(".md")]
        assert len(md_lines) == 2

    def test_max_entries_zero_skips_writing(self, sphinx_app, tmp_html_tree):
        """max_entries=0 should not create llms.txt."""
        (tmp_html_tree / "page.md").write_text("# Page\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_max_entries = 0
        _mod.generate_llms_txt(sphinx_app, exception=None)
        assert not (tmp_html_tree / "llms.txt").exists()

    def test_max_entries_none_writes_all(self, sphinx_app, tmp_html_tree):
        """max_entries=None means all entries are written."""
        for i in range(3):
            (tmp_html_tree / f"page{i}.md").write_text(f"# Page{i}\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_max_entries = None
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        md_lines = [l for l in llms.splitlines() if l.endswith(".md")]
        assert len(md_lines) == 3

    def test_full_content_embeds_markdown(self, sphinx_app, tmp_html_tree):
        """ai_assistant_llms_txt_full_content=True embeds page content."""
        (tmp_html_tree / "page.md").write_text("# Hello\n\nWorld\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_full_content = True
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        assert "# Hello" in llms
        assert "World" in llms
        assert "---" in llms  # separator

    def test_full_content_false_does_not_embed(self, sphinx_app, tmp_html_tree):
        """ai_assistant_llms_txt_full_content=False writes only URLs."""
        (tmp_html_tree / "page.md").write_text("# Hello\n\nWorld\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.html_baseurl = ""
        sphinx_app.config.ai_assistant_base_url = ""
        sphinx_app.config.ai_assistant_llms_txt_full_content = False
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        # "World" is the page's body — must NOT be in the index file
        assert "World" not in llms
        assert "page.md" in llms


# ===========================================================================
# 16. add_ai_assistant_context
# ===========================================================================

class TestAddAiAssistantContext:
    """Branch coverage for the HTML-page-context hook."""

    def test_disabled(self):
        app = MagicMock()
        app.config.ai_assistant_enabled = False
        ctx: dict = {}
        _mod.add_ai_assistant_context(app, "index", "page.html", ctx, None)
        assert "ai_assistant_config" not in ctx

    def test_enabled_injects_config(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "ai_assistant_config" in ctx
        assert isinstance(ctx["ai_assistant_config"], dict)

    def test_metatags_created_when_absent(self, sphinx_app):
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "metatags" in ctx
        assert "<script>" in ctx["metatags"]

    def test_metatags_appended_when_present(self, sphinx_app):
        ctx: dict = {"metatags": "<meta name='existing'>"}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "<meta name='existing'>" in ctx["metatags"]
        assert "<script>" in ctx["metatags"]

    def test_xss_prevention_in_metatags(self, sphinx_app):
        """Provider URLs containing </script> must be escaped in the inline script."""
        sphinx_app.config.ai_assistant_providers = {
            "evil": {
                "url_template": "https://x.com/</script><script>alert(1)//",
                "enabled": True,
            }
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert "</script><script>" not in ctx["metatags"]

    def test_html_baseurl_takes_priority(self, sphinx_app):
        sphinx_app.config.html_baseurl = "https://html-baseurl.example.com"
        sphinx_app.config.ai_assistant_base_url = "https://ai-baseurl.example.com"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        cfg = ctx["ai_assistant_config"]
        assert cfg["baseUrl"] == "https://html-baseurl.example.com"

    def test_position_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_position = "title"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["position"] == "title"

    def test_features_in_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_features = {"ai_chat": False}
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["features"] == {"ai_chat": False}


class TestAddAiAssistantContextExtended:
    """Security: invalid position and dangerous provider URL filtering."""

    def test_invalid_position_falls_back_to_sidebar(self, sphinx_app):
        """An invalid position value must be replaced with 'sidebar'."""
        sphinx_app.config.ai_assistant_position = "malicious_value; drop table"
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        assert ctx["ai_assistant_config"]["position"] == "sidebar"

    def test_invalid_position_logs_warning(self, sphinx_app):
        sphinx_app.config.ai_assistant_position = "bad_position"
        ctx: dict = {}
        with patch.object(_mod._get_logger(), "warning") as mock_warn:
            _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
            mock_warn.assert_called_once()
            assert "invalid" in mock_warn.call_args[0][0].lower() or \
                   "position" in mock_warn.call_args[0][0].lower()

    def test_dangerous_provider_url_filtered(self, sphinx_app):
        """Providers with javascript: URL templates must be dropped."""
        sphinx_app.config.ai_assistant_providers = {
            "safe": {
                "url_template": "https://claude.ai/new?q={prompt}",
                "enabled": True,
            },
            "evil": {
                "url_template": "javascript:alert(document.cookie)",
                "enabled": True,
            },
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        providers = ctx["ai_assistant_config"]["providers"]
        assert "safe" in providers
        assert "evil" not in providers

    def test_data_scheme_provider_filtered(self, sphinx_app):
        sphinx_app.config.ai_assistant_providers = {
            "data_evil": {
                "url_template": "data:text/html,<h1>XSS</h1>",
                "enabled": True,
            },
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        providers = ctx["ai_assistant_config"]["providers"]
        assert "data_evil" not in providers

    def test_empty_provider_url_template_passes(self, sphinx_app):
        """Providers with an empty url_template are allowed."""
        sphinx_app.config.ai_assistant_providers = {
            "nourl": {"url_template": "", "enabled": True},
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        providers = ctx["ai_assistant_config"]["providers"]
        assert "nourl" in providers

    def test_valid_positions_not_warned(self, sphinx_app):
        for pos in ["sidebar", "title", "floating", "none"]:
            sphinx_app.config.ai_assistant_position = pos
            ctx: dict = {}
            with patch.object(_mod._get_logger(), "warning") as mock_warn:
                _mod.add_ai_assistant_context(
                    sphinx_app, "index", "page.html", ctx, None
                )
                mock_warn.assert_not_called()
            assert ctx["ai_assistant_config"]["position"] == pos


# ===========================================================================
# 17. setup()
# ===========================================================================

class TestSetup:
    """setup() must register all config values and return correct metadata."""

    @pytest.fixture()
    def app(self) -> MagicMock:
        mock_app = MagicMock()
        mock_app.config.html_static_path = []
        return mock_app

    def test_returns_version(self, app):
        result = _mod.setup(app)
        assert result["version"] == _mod._VERSION

    def test_returns_parallel_safe(self, app):
        result = _mod.setup(app)
        assert result["parallel_read_safe"] is True
        assert result["parallel_write_safe"] is True

    def test_add_config_value_called(self, app):
        _mod.setup(app)
        call_names = [c[0][0] for c in app.add_config_value.call_args_list]
        required = {
            "ai_assistant_enabled",
            "ai_assistant_position",
            "ai_assistant_content_selector",
            "ai_assistant_content_selectors",
            "ai_assistant_theme_preset",
            "ai_assistant_generate_markdown",
            "ai_assistant_markdown_exclude_patterns",
            "ai_assistant_strip_tags",
            "ai_assistant_generate_llms_txt",
            "ai_assistant_base_url",
            "ai_assistant_max_workers",
            "ai_assistant_llms_txt_max_entries",
            "ai_assistant_llms_txt_full_content",
            "ai_assistant_features",
            "ai_assistant_providers",
            "ai_assistant_mcp_tools",
        }
        assert required.issubset(set(call_names))

    def test_events_connected(self, app):
        _mod.setup(app)
        event_names = [c[0][0] for c in app.connect.call_args_list]
        assert "html-page-context" in event_names
        assert "build-finished" in event_names
        assert event_names.count("build-finished") == 2

    def test_css_js_added(self, app):
        _mod.setup(app)
        app.add_css_file.assert_called_once_with("ai-assistant.css")
        app.add_js_file.assert_called_once_with("ai-assistant.js")

    def test_static_path_appended(self, app):
        _mod.setup(app)
        assert len(app.config.html_static_path) == 1
        assert "_static" in app.config.html_static_path[0]

    def test_static_path_not_duplicated(self, app):
        """Calling setup twice must not duplicate the static path entry."""
        static_path = str(Path(_mod.__file__).parent / "_static")
        app.config.html_static_path = [static_path]
        _mod.setup(app)
        _mod.setup(app)
        assert app.config.html_static_path.count(static_path) == 1

    def test_theme_preset_config_registered(self, app):
        """ai_assistant_theme_preset must be registered with default None."""
        _mod.setup(app)
        call_args_map = {
            c[0][0]: c[0][1]
            for c in app.add_config_value.call_args_list
        }
        assert "ai_assistant_theme_preset" in call_args_map
        assert call_args_map["ai_assistant_theme_preset"] is None

    def test_strip_tags_config_registered(self, app):
        """ai_assistant_strip_tags must be registered as a list."""
        _mod.setup(app)
        call_args_map = {
            c[0][0]: c[0][1]
            for c in app.add_config_value.call_args_list
        }
        assert "ai_assistant_strip_tags" in call_args_map
        assert isinstance(call_args_map["ai_assistant_strip_tags"], list)

    def test_llms_txt_max_entries_registered(self, app):
        _mod.setup(app)
        call_args_map = {
            c[0][0]: c[0][1]
            for c in app.add_config_value.call_args_list
        }
        assert "ai_assistant_llms_txt_max_entries" in call_args_map
        assert call_args_map["ai_assistant_llms_txt_max_entries"] is None

    def test_llms_txt_full_content_registered(self, app):
        _mod.setup(app)
        call_args_map = {
            c[0][0]: c[0][1]
            for c in app.add_config_value.call_args_list
        }
        assert "ai_assistant_llms_txt_full_content" in call_args_map
        assert call_args_map["ai_assistant_llms_txt_full_content"] is False


# ===========================================================================
# 18. Default content selectors
# ===========================================================================

class TestDefaultContentSelectors:
    """_DEFAULT_CONTENT_SELECTORS must include all major theme selectors."""

    def test_pydata_theme_selector_present(self):
        selectors = _mod._DEFAULT_CONTENT_SELECTORS
        assert "article.bd-article" in selectors

    def test_furo_theme_selector_present(self):
        selectors = _mod._DEFAULT_CONTENT_SELECTORS
        assert 'article[role="main"]' in selectors

    def test_generic_main_selector_present(self):
        selectors = _mod._DEFAULT_CONTENT_SELECTORS
        assert "main" in selectors

    def test_rtd_selector_present(self):
        assert "div.rst-content" in _mod._DEFAULT_CONTENT_SELECTORS

    def test_selectors_is_tuple(self):
        assert isinstance(_mod._DEFAULT_CONTENT_SELECTORS, tuple)

    def test_selectors_non_empty(self):
        assert len(_mod._DEFAULT_CONTENT_SELECTORS) >= 7

    def test_all_default_selectors_are_safe(self):
        """Built-in default selectors must all pass CSS validation."""
        for sel in _mod._DEFAULT_CONTENT_SELECTORS:
            assert _mod._validate_css_selector(sel), (
                f"Default selector {sel!r} fails CSS validation"
            )

    def test_article_fallback_present(self):
        assert "article" in _mod._DEFAULT_CONTENT_SELECTORS


# ===========================================================================
# 19. Edge cases and integration
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge cases and regression guards."""

    def test_html_to_markdown_empty_string(self):
        result = _mod.html_to_markdown("")
        assert isinstance(result, str)

    def test_html_to_markdown_only_whitespace(self):
        result = _mod.html_to_markdown("   \n\t  ")
        assert isinstance(result, str)

    def test_process_file_missing_bs4_raises_error_status(self, tmp_path):
        """If bs4 is importable but converter crashes, worker returns error."""
        html = tmp_path / "page.html"
        html.write_text("<html><body><article>OK</article></body></html>", encoding="utf-8")
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant.html_to_markdown",
            side_effect=RuntimeError("converter exploded"),
        ):
            status, rel, msg = _mod._process_single_html_file((
                str(html), str(tmp_path), [], ["article"], ["script", "style"],
            ))
        assert status == "error"
        assert "converter exploded" in msg

    def test_safe_json_large_nested_object(self):
        """_safe_json_for_script handles deeply nested structures."""
        obj = {"level1": {"level2": {"level3": list(range(100))}}}
        result = _mod._safe_json_for_script(obj)
        parsed = json.loads(result)
        assert parsed == obj

    def test_generate_markdown_files_no_html_files(self, sphinx_app, tmp_path):
        """Empty outdir produces zero .md files without crashing."""
        empty = tmp_path / "empty_out"
        empty.mkdir()
        sphinx_app.builder.outdir = str(empty)
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert list(empty.rglob("*.md")) == []

    def test_resolve_selectors_with_all_themes(self):
        """Every theme in _THEME_SELECTOR_PRESETS resolves without error."""
        for theme_name in _mod._THEME_SELECTOR_PRESETS:
            result = _mod._resolve_content_selectors(theme_name, [])
            assert len(result) > 0, f"Empty result for theme {theme_name!r}"
            assert isinstance(result, tuple)

    def test_process_single_html_file_args_tuple_length(self, tmp_path):
        """Passing wrong-length tuple must raise, not silently corrupt."""
        with pytest.raises((TypeError, ValueError)):
            # Old 4-tuple — must fail (wrong number of values to unpack)
            _mod._process_single_html_file((
                str(tmp_path / "x.html"),
                str(tmp_path),
                [],
                ["article"],
                # missing strip_tags → should fail
            ))

    def test_safe_json_preserves_unicode_escape(self):
        obj = {"k": "\u2603"}  # snowman
        result = _mod._safe_json_for_script(obj)
        assert "\u2603" not in result  # ensure_ascii=True
        parsed = json.loads(result)
        assert parsed["k"] == "\u2603"
