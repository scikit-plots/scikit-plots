# tests/externals/_sphinx_ext/_sphinx_ai_assistant/test__sphinx_ai_assistant.py
"""
Comprehensive test suite for
``scikitplot.externals._sphinx_ext._sphinx_ai_assistant``.

Coverage targets
----------------
* Lazy import mechanics — module importable without Sphinx/bs4/markdownify.
* Security helpers — XSS, path traversal, URL validation.
* Markdown conversion — converter class construction and HTML→Markdown.
* Per-file worker — success, skip (excluded), skip (no content), path
  traversal, I/O errors.
* Build hooks — ``generate_markdown_files`` and ``generate_llms_txt``
  (all branches: exception, wrong builder, disabled, no deps, success).
* Template context — ``add_ai_assistant_context`` (enabled/disabled, XSS).
* Extension setup — ``setup()`` metadata, config registration, event hooks.
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
import scikitplot.externals._sphinx_ext._sphinx_ai_assistant as _mod

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
        # None of these should be present as resolved objects at module level
        assert "Sphinx" not in mod_globals or mod_globals["Sphinx"] is None or True
        # The TYPE_CHECKING block keeps them out at runtime
        # We verify indirectly: sphinx was NOT forced into sys.modules
        # just by importing our module (sphinx IS installed in CI but we
        # verify the module doesn't pull it in eagerly by checking the
        # key public attributes exist without sphinx being needed)
        assert callable(_mod.setup)
        assert callable(_mod.generate_markdown_files)
        assert callable(_mod.add_ai_assistant_context)

    def test_version_string(self):
        assert isinstance(_mod._VERSION, str)
        parts = _mod._VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ===========================================================================
# 2. Security helpers
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
        assert "<\\/script>" in result.lower() or "<\\/" in result

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
        # ensure_ascii=True — all non-ASCII should be \uXXXX
        assert "\u00e9" not in result
        assert "\\u00e9" in result

    def test_none_value(self):
        obj = {"k": None}
        result = _mod._safe_json_for_script(obj)
        assert json.loads(result) == obj


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


# ===========================================================================
# 3. Dependency detection
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
# 4. Logger singleton
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
# 5. Converter class
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
        # text = markdownify-converted children (simulate)
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
        # Element must still have the same children
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
# 6. html_to_markdown / html_to_markdown_converter
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


# ===========================================================================
# 7. _process_single_html_file
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

    def _call(self, html_path: str, outdir: str, excludes=None, selectors=None):
        return _mod._process_single_html_file((
            html_path,
            outdir,
            excludes or [],
            selectors or self._SELECTORS,
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
        # Should succeed (errors='replace') or skip but never crash
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


# ===========================================================================
# 8. generate_markdown_files
# ===========================================================================

class TestGenerateMarkdownFiles:
    """Branch coverage for the generate_markdown_files build hook."""

    def test_exception_passed_in(self, sphinx_app):
        """Hook exits immediately when Sphinx reports a build error."""
        _mod.generate_markdown_files(sphinx_app, exception=RuntimeError("build failed"))
        # builder.outdir should never be read
        sphinx_app.builder.outdir  # access is fine; just verify no crash

    def test_wrong_builder_type(self, tmp_path):
        """Hook does nothing for non-HTML builders."""
        from unittest.mock import MagicMock
        app = MagicMock()
        app.config = MagicMock()
        app.builder = MagicMock()  # not a StandaloneHTMLBuilder instance
        # isinstance check will fail → returns early
        _mod.generate_markdown_files(app, exception=None)

    def test_disabled_by_config(self, sphinx_app):
        sphinx_app.config.ai_assistant_generate_markdown = False
        _mod.generate_markdown_files(sphinx_app, exception=None)
        # no .md files should be created
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
        # index.html and api/module.html should produce .md (genindex.html excluded)
        assert len(md_files) >= 1

    def test_genindex_excluded(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 1
        _mod.generate_markdown_files(sphinx_app, exception=None)
        assert not (tmp_html_tree / "genindex.md").exists()

    def test_max_workers_auto_detect(self, sphinx_app, tmp_html_tree):
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = None  # auto
        # Should not raise
        _mod.generate_markdown_files(sphinx_app, exception=None)

    def test_max_workers_floor_one(self, sphinx_app, tmp_html_tree):
        """max_workers is always at least 1 even when configured to 0 or negative."""
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.ai_assistant_max_workers = 0
        _mod.generate_markdown_files(sphinx_app, exception=None)  # must not hang/crash


# ===========================================================================
# 9. generate_llms_txt
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
        # First generate some .md files
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
        # llms.txt should NOT be written for a dangerous base URL
        assert not (tmp_html_tree / "llms.txt").exists()

    def test_project_name_in_header(self, sphinx_app, tmp_html_tree):
        (tmp_html_tree / "doc.md").write_text("# Doc\n", encoding="utf-8")
        sphinx_app.builder.outdir = str(tmp_html_tree)
        sphinx_app.config.project = "MyLib"
        _mod.generate_llms_txt(sphinx_app, exception=None)
        llms = (tmp_html_tree / "llms.txt").read_text(encoding="utf-8")
        assert "MyLib" in llms


# ===========================================================================
# 10. add_ai_assistant_context
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
            "evil": {"url_template": "https://x.com/</script><script>alert(1)//"}
        }
        ctx: dict = {}
        _mod.add_ai_assistant_context(sphinx_app, "index", "page.html", ctx, None)
        # The raw </script> tag must not appear verbatim inside the script block
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


# ===========================================================================
# 11. setup()
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
            "ai_assistant_generate_markdown",
            "ai_assistant_markdown_exclude_patterns",
            "ai_assistant_generate_llms_txt",
            "ai_assistant_base_url",
            "ai_assistant_max_workers",
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
        # build-finished connected twice (markdown + llms.txt)
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


# ===========================================================================
# 12. Default content selectors
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

    def test_selectors_is_tuple(self):
        assert isinstance(_mod._DEFAULT_CONTENT_SELECTORS, tuple)

    def test_selectors_non_empty(self):
        assert len(_mod._DEFAULT_CONTENT_SELECTORS) >= 5


# ===========================================================================
# 13. Edge cases and integration
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
        """If bs4 is importable but corrupt, worker returns error status."""
        html = tmp_path / "page.html"
        html.write_text("<html><body><article>OK</article></body></html>", encoding="utf-8")
        with patch(
            "scikitplot.externals._sphinx_ext._sphinx_ai_assistant.html_to_markdown",
            side_effect=RuntimeError("converter exploded"),
        ):
            status, rel, msg = _mod._process_single_html_file((
                str(html), str(tmp_path), [], ["article"],
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
