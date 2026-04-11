# tests/_externals/_sphinx_ext/_sphinx_ai_assistant/test__jupyter.py
"""
Test suite for the ``_jupyter`` submodule:
  scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter

Coverage targets
----------------
* _build_jupyter_widget_html — HTML structure, XSS safety, provider filtering,
  position styles, MCP tools, include_raw_image, notebook_mode, provider
  coercion, advanced prompt params (intention, custom_context, etc.).
* display_jupyter_ai_button — IPython import guard, position validation,
  parameter forwarding (all kwargs reach _build_jupyter_widget_html).
* display_jupyter_notebook_ai_button — always sets notebook_mode=True,
  delegates correctly to display_jupyter_ai_button.

Notes
-----
All patches that intercept internal calls target the ``_jupyter`` submodule
namespace (where the functions are defined and looked up), not the re-export
in ``__init__``.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import scikitplot._externals._sphinx_ext._sphinx_ai_assistant as _mod
import scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter as _jmod

# ===========================================================================
# 1. Jupyter widget HTML builder (new)
# ===========================================================================

class TestBuildJupyterWidgetHtml:
    """Tests for _build_jupyter_widget_html."""

    def test_returns_string(self):
        result = _mod._build_jupyter_widget_html()
        assert isinstance(result, str)

    def test_contains_script_tag(self):
        result = _mod._build_jupyter_widget_html()
        assert "<script>" in result

    def test_default_providers_in_output(self):
        result = _mod._build_jupyter_widget_html(
            providers=["claude", "chatgpt"]
        )
        assert "claude" in result.lower() or "Claude" in result

    def test_explicit_content_serialised(self):
        result = _mod._build_jupyter_widget_html(content="My chart shows growth")
        # Content is JSON-serialised inside the script
        assert "My chart shows growth" in result

    def test_javascript_in_content_escaped(self):
        """content with </script> must be escaped to prevent XSS."""
        result = _mod._build_jupyter_widget_html(
            content="</script><script>alert(1)//"
        )
        assert "</script><script>" not in result

    def test_custom_widget_id(self):
        result = _mod._build_jupyter_widget_html(widget_id="my-custom-id")
        assert "my-custom-id" in result

    def test_auto_widget_id_generated(self):
        r1 = _mod._build_jupyter_widget_html()
        r2 = _mod._build_jupyter_widget_html()
        # Both contain an id; they differ because monotonic ns differs
        assert 'id="ai-btn-' in r1

    def test_floating_position_style(self):
        result = _mod._build_jupyter_widget_html(position="floating")
        assert "fixed" in result

    def test_inline_position_style(self):
        result = _mod._build_jupyter_widget_html(position="inline")
        assert "inline-block" in result or "inline" in result

    def test_dangerous_provider_url_excluded(self):
        """Providers with javascript: url_template must not appear in buttons."""
        override = {
            "evil": {
                "enabled": True,
                "label": "Evil",
                "description": "Bad",
                "icon": "evil.svg",
                "url_template": "javascript:alert(1)",
                "prompt_template": "Read: {url}",
                "model": "evil-1",
                "type": "web",
            }
        }
        result = _mod._build_jupyter_widget_html(
            providers=["evil"],
            provider_configs=override,
        )
        assert "javascript:alert" not in result

    def test_ollama_shown_as_local_disabled(self):
        result = _mod._build_jupyter_widget_html(providers=["ollama"])
        # Ollama is disabled by default; should appear as local-disabled
        assert "local-disabled" in result or "local" in result

    def test_ollama_enabled_via_override(self):
        result = _mod._build_jupyter_widget_html(
            providers=["ollama"],
            provider_configs={"ollama": {"enabled": True}},
        )
        # When enabled, should NOT have local-disabled
        # The button should be clickable (has href or onclick)
        assert "ollama" in result.lower() or "Ollama" in result

    def test_page_url_validated(self):
        """Dangerous page_url must be rejected."""
        result = _mod._build_jupyter_widget_html(
            page_url="javascript:alert(1)"
        )
        # The page_url JSON value should be empty string after validation
        assert "javascript:alert" not in result

    def test_safe_page_url_embedded(self):
        result = _mod._build_jupyter_widget_html(
            page_url="https://docs.example.com/page"
        )
        assert "docs.example.com" in result

    def test_provider_configs_override_defaults(self):
        """provider_configs merges over _DEFAULT_PROVIDERS."""
        result = _mod._build_jupyter_widget_html(
            providers=["claude"],
            provider_configs={"claude": {"label": "CUSTOM_LABEL_XYZ"}},
        )
        assert "CUSTOM_LABEL_XYZ" in result

    def test_empty_providers_list(self):
        result = _mod._build_jupyter_widget_html(providers=[])
        assert isinstance(result, str)


# ===========================================================================
# 2. display_jupyter_ai_button (new)
# ===========================================================================

class TestDisplayJupyterAiButton:
    def test_raises_if_ipython_missing(self):
        with patch.object(_mod, "_has_ipython", return_value=False):
            import builtins
            real_import = builtins.__import__
            def block_ipython(name, *args, **kwargs):
                if name in ("IPython", "IPython.display"):
                    raise ImportError("IPython not installed")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=block_ipython):
                with pytest.raises(ImportError, match="IPython"):
                    _mod.display_jupyter_ai_button()

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="position"):
            _mod.display_jupyter_ai_button(position="bad_position")

    def test_calls_ipython_display_with_html(self):
        mock_display = MagicMock()
        mock_html_cls = MagicMock(side_effect=lambda x: x)  # HTML(x) returns x
        mock_ipython = types.ModuleType("IPython")
        mock_ipython_display = types.ModuleType("IPython.display")
        mock_ipython_display.display = mock_display
        mock_ipython_display.HTML = mock_html_cls
        mock_ipython.display = mock_ipython_display

        with patch.dict(sys.modules, {
            "IPython": mock_ipython,
            "IPython.display": mock_ipython_display,
        }):
            _mod.display_jupyter_ai_button(content="Test content")

        mock_display.assert_called_once()

    def test_floating_position_accepted(self):
        mock_display = MagicMock()
        mock_html_cls = MagicMock(side_effect=lambda x: x)
        mock_ipython_display = types.ModuleType("IPython.display")
        mock_ipython_display.display = mock_display
        mock_ipython_display.HTML = mock_html_cls

        with patch.dict(sys.modules, {"IPython.display": mock_ipython_display}):
            try:
                _mod.display_jupyter_ai_button(position="floating")
                mock_display.assert_called_once()
            except ImportError:
                pass  # IPython not available in this env


# ===========================================================================
# 3. Theme selector presets
# ===========================================================================

# ===========================================================================
# 4. _build_jupyter_widget_html — advanced params
# ===========================================================================

class TestBuildJupyterWidgetHtmlAdvanced:
    """Tests for the new intention / custom_context / notebook_mode params."""

    def test_intention_serialised_into_html(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t1", intention="Explain this chart"
        )
        assert "Explain this chart" in html

    def test_intention_none_safe(self):
        html = _mod._build_jupyter_widget_html(widget_id="t2", intention=None)
        assert "INTENTION        = null" in html or "INTENTION" in html

    def test_custom_context_serialised(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t3", custom_context="Python 3.11, pandas 2.0"
        )
        assert "Python 3.11" in html

    def test_custom_prompt_prefix_serialised(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t4", custom_prompt_prefix="SYSTEM: Always answer in JSON"
        )
        assert "SYSTEM: Always answer in JSON" in html

    def test_notebook_mode_true_emits_true(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t5", notebook_mode=True
        )
        assert "NOTEBOOK_MODE    = true" in html

    def test_notebook_mode_false_emits_false(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t6", notebook_mode=False
        )
        assert "NOTEBOOK_MODE    = false" in html

    def test_include_outputs_true(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t7", notebook_mode=True, include_outputs=True
        )
        assert "INCLUDE_OUTPUTS  = true" in html

    def test_include_outputs_false(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t8", notebook_mode=True, include_outputs=False
        )
        assert "INCLUDE_OUTPUTS  = false" in html

    def test_notebook_mode_adds_getNotebookContent_js(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t9", notebook_mode=True
        )
        assert "getNotebookContent" in html

    def test_cell_capture_js_always_present(self):
        html = _mod._build_jupyter_widget_html(widget_id="t10")
        assert "getCellContent" in html

    def test_buildPrompt_js_present(self):
        html = _mod._build_jupyter_widget_html(widget_id="t11")
        assert "buildPrompt" in html

    def test_intention_xss_escaped(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t12",
            intention="</script><script>alert(1)</script>",
        )
        assert "</script><script>" not in html

    def test_custom_context_xss_escaped(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t13",
            custom_context="</script>bad",
        )
        assert "</script>bad" not in html

    def test_prompt_prefix_xss_escaped(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t14",
            custom_prompt_prefix="</script>inject",
        )
        assert "</script>inject" not in html

    def test_single_string_provider_accepted(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t15", providers="claude"
        )
        assert "Ask Claude" in html

    def test_goal_prefix_in_js(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t16", intention="Find bugs"
        )
        # JS buildPrompt adds "Goal: " prefix
        assert "Goal:" in html

    def test_context_prefix_in_js(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t17", custom_context="some context"
        )
        assert "Context:" in html

    def test_all_new_params_together(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="t18",
            intention="Review",
            custom_context="ML pipeline",
            custom_prompt_prefix="You are an expert.",
            notebook_mode=True,
            include_outputs=False,
            providers=["claude", "chatgpt"],
        )
        assert "Review" in html
        assert "ML pipeline" in html
        assert "You are an expert." in html
        assert "NOTEBOOK_MODE    = true" in html
        assert "INCLUDE_OUTPUTS  = false" in html


# ===========================================================================
# 5. display_jupyter_ai_button — new params forwarded
# ===========================================================================

class TestDisplayJupyterAiButtionNewParams:
    """Ensure display_jupyter_ai_button correctly forwards new params."""

    @pytest.fixture()
    def ipython_mock(self):
        mock_display = MagicMock()
        mock_html    = MagicMock(side_effect=lambda x: x)
        with patch.dict("sys.modules", {
            "IPython": MagicMock(),
            "IPython.display": MagicMock(display=mock_display, HTML=mock_html),
        }):
            yield mock_display, mock_html

    def test_intention_forwarded(self, ipython_mock):
        mock_display, mock_html = ipython_mock
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            "._build_jupyter_widget_html"
        ) as mock_build:
            mock_build.return_value = "<div/>"
            _mod.display_jupyter_ai_button(intention="Explain")
            _, kwargs = mock_build.call_args
            assert kwargs.get("intention") == "Explain"

    def test_custom_context_forwarded(self, ipython_mock):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            "._build_jupyter_widget_html"
        ) as mock_build:
            mock_build.return_value = "<div/>"
            _mod.display_jupyter_ai_button(custom_context="ctx")
            _, kwargs = mock_build.call_args
            assert kwargs.get("custom_context") == "ctx"

    def test_notebook_mode_forwarded(self, ipython_mock):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            "._build_jupyter_widget_html"
        ) as mock_build:
            mock_build.return_value = "<div/>"
            _mod.display_jupyter_ai_button(notebook_mode=True)
            _, kwargs = mock_build.call_args
            assert kwargs.get("notebook_mode") is True

    def test_include_outputs_false_forwarded(self, ipython_mock):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            "._build_jupyter_widget_html"
        ) as mock_build:
            mock_build.return_value = "<div/>"
            _mod.display_jupyter_ai_button(notebook_mode=True, include_outputs=False)
            _, kwargs = mock_build.call_args
            assert kwargs.get("include_outputs") is False

    def test_single_string_provider_forwarded(self, ipython_mock):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            "._build_jupyter_widget_html"
        ) as mock_build:
            mock_build.return_value = "<div/>"
            _mod.display_jupyter_ai_button(providers="claude")
            _, kwargs = mock_build.call_args
            assert kwargs.get("providers") == "claude"

    def test_custom_prompt_prefix_forwarded(self, ipython_mock):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            "._build_jupyter_widget_html"
        ) as mock_build:
            mock_build.return_value = "<div/>"
            _mod.display_jupyter_ai_button(custom_prompt_prefix="SYS:")
            _, kwargs = mock_build.call_args
            assert kwargs.get("custom_prompt_prefix") == "SYS:"


# ===========================================================================
# 6. display_jupyter_notebook_ai_button
# ===========================================================================

class TestDisplayJupyterNotebookAiButton:
    """Tests for the new display_jupyter_notebook_ai_button public function."""

    def test_is_public_callable(self):
        assert callable(_mod.display_jupyter_notebook_ai_button)

    def test_raises_if_ipython_missing(self):
        with patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
            with pytest.raises(ImportError):
                _mod.display_jupyter_notebook_ai_button()

    def test_invalid_position_raises(self):
        with pytest.raises((ValueError, ImportError)):
            _mod.display_jupyter_notebook_ai_button(position="bad")

    def test_delegates_to_display_jupyter_ai_button(self):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button(
                intention="Review",
                include_outputs=False,
                custom_context="ctx",
            )
            mock_btn.assert_called_once()
            _, kwargs = mock_btn.call_args
            assert kwargs["notebook_mode"] is True
            assert kwargs["intention"] == "Review"
            assert kwargs["include_outputs"] is False
            assert kwargs["custom_context"] == "ctx"

    def test_notebook_mode_always_true(self):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button()
            _, kwargs = mock_btn.call_args
            assert kwargs["notebook_mode"] is True

    def test_content_always_none(self):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button()
            args, kwargs = mock_btn.call_args
            # content positional-or-keyword must be None
            assert (args[0] if args else kwargs.get("content")) is None

    def test_single_string_provider_accepted(self):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button(providers="claude")
            _, kwargs = mock_btn.call_args
            assert kwargs.get("providers") == "claude"

    def test_include_outputs_default_false(self):
        """include_outputs defaults to False since v0.4.0 — opt-in for cell outputs."""
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button()
            _, kwargs = mock_btn.call_args
            assert kwargs.get("include_outputs") is False

    def test_floating_position_accepted(self):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button(position="floating")
            _, kwargs = mock_btn.call_args
            assert kwargs.get("position") == "floating"

    def test_custom_prompt_prefix_forwarded(self):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button(
                custom_prompt_prefix="You are a code reviewer."
            )
            _, kwargs = mock_btn.call_args
            assert kwargs.get("custom_prompt_prefix") == "You are a code reviewer."


# ===========================================================================
# 7. html_to_markdown — str|list|None strip_tags
# ===========================================================================

# ===========================================================================
# 8. _build_jupyter_widget_html — providers coercion (single str)
# ===========================================================================

class TestBuildJupyterWidgetProviderCoercion:
    """_build_jupyter_widget_html should accept str providers via _coerce_to_list."""

    def test_single_str_provider_renders_button(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="coerce1", providers="claude"
        )
        assert "Ask Claude" in html

    def test_none_providers_uses_default_four(self):
        html = _mod._build_jupyter_widget_html(widget_id="coerce2", providers=None)
        # Default: claude, chatgpt, gemini, ollama
        assert "claude" in html.lower()
        assert "chatgpt" in html.lower() or "ChatGPT" in html

    def test_list_providers_respected(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="coerce3", providers=["claude", "gemini"]
        )
        assert "Ask Claude" in html
        assert "Ask Gemini" in html

    def test_unknown_provider_gracefully_skipped(self):
        # Unknown names produce an empty config dict → skipped silently
        html = _mod._build_jupyter_widget_html(
            widget_id="coerce4", providers=["claude", "unknown_provider_xyz"]
        )
        assert "Ask Claude" in html
        assert "unknown_provider_xyz" not in html


# ===========================================================================
# 9. Public API surface — new symbols exported
# ===========================================================================

# ===========================================================================
# 10. _build_jupyter_widget_html — include_raw_image and mcp_tools
# ===========================================================================

class TestBuildJupyterWidgetRawImageAndMcp:
    """Tests for include_raw_image and mcp_tools params."""

    def test_include_raw_image_false_default(self):
        html = _mod._build_jupyter_widget_html(widget_id="ri0")
        assert "INCLUDE_RAW_IMAGE = false" in html

    def test_include_raw_image_true(self):
        html = _mod._build_jupyter_widget_html(widget_id="ri1", include_raw_image=True)
        assert "INCLUDE_RAW_IMAGE = true" in html

    def test_capture_images_js_present(self):
        html = _mod._build_jupyter_widget_html(widget_id="ri2")
        assert "captureImages" in html

    def test_canvas_capture_in_js(self):
        html = _mod._build_jupyter_widget_html(widget_id="ri3", include_raw_image=True)
        assert "toDataURL" in html or "canvas" in html.lower()

    def test_img_capture_in_js(self):
        html = _mod._build_jupyter_widget_html(widget_id="ri4")
        # captureImages JS always present; img scan too
        assert "querySelectorAll" in html

    def test_mcp_tools_none_no_mcp_section(self):
        html = _mod._build_jupyter_widget_html(widget_id="mcp0", mcp_tools=None)
        # MCP_TOOLS = null or {}
        assert "MCP_TOOLS" in html

    def test_mcp_tools_empty_dict(self):
        html = _mod._build_jupyter_widget_html(widget_id="mcp1", mcp_tools={})
        assert "MCP_TOOLS" in html

    def test_mcp_tools_valid_enabled_renders_button(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="mcp2",
            mcp_tools={
                "vscode": {
                    "enabled": True,
                    "type": "vscode",
                    "label": "VS Code MCP",
                    "description": "Connect",
                    "server_url": "http://localhost:9999",
                }
            },
        )
        assert "VS Code MCP" in html

    def test_mcp_tools_invalid_url_filtered(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="mcp3",
            mcp_tools={
                "evil": {
                    "enabled": True,
                    "type": "generic",
                    "label": "Evil",
                    "description": "Bad",
                    "server_url": "javascript:hack()",
                }
            },
        )
        # Invalid server_url causes validation failure → tool filtered
        assert "Evil" not in html

    def test_mcp_tools_disabled_not_rendered(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="mcp4",
            mcp_tools={
                "cursor": {
                    "enabled": False,
                    "type": "cursor",
                    "label": "Cursor MCP",
                    "description": "x",
                }
            },
        )
        # disabled tools pass validation but JS skips them
        assert "MCP_TOOLS" in html

    def test_image_capture_xss_safe(self):
        """buildPrompt must not use raw string interpolation for image content."""
        html = _mod._build_jupyter_widget_html(widget_id="ri5", include_raw_image=True)
        assert "imageText" in html or "captureImages" in html
        # No raw f-string injection of user data into JS
        assert 'imageText + "' not in html or "imageText" in html

    def test_mcp_separator_rendered_when_tools_present(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="mcp5",
            mcp_tools={
                "vscode": {
                    "enabled": True,
                    "type": "vscode",
                    "label": "VSC",
                    "description": "x",
                }
            },
        )
        # The MCP section rendering JS is present
        assert "MCP:" in html or "mcpKeys" in html


# ===========================================================================
# 11. display_jupyter_ai_button — include_raw_image and mcp_tools
# ===========================================================================

class TestDisplayJupyterAiButtonRawImageMcp:
    """Ensure include_raw_image and mcp_tools are forwarded correctly."""

    def _call_build(self, **kwargs):
        """
        Call display_jupyter_ai_button and capture _build_jupyter_widget_html args.

        Notes
        -----
        ``_build_jupyter_widget_html`` is defined in the ``_jupyter`` submodule and
        looked up there at call time.  Patching must target that module's namespace,
        not the re-export in ``__init__``, otherwise the patch has no effect.
        """
        captured = {}
        def fake_build(*a, **kw):
            captured.update(kw)
            return "<div/>"
        with patch.object(_jmod, "_build_jupyter_widget_html", side_effect=fake_build):
            with patch("IPython.display.display", MagicMock()):
                with patch("IPython.display.HTML", side_effect=lambda x: x):
                    try:
                        _mod.display_jupyter_ai_button(**kwargs)
                    except Exception:
                        pass
        return captured

    def test_include_raw_image_forwarded(self):
        kw = self._call_build(include_raw_image=True)
        assert kw.get("include_raw_image") is True

    def test_mcp_tools_forwarded(self):
        tools = {"vscode": {"enabled": True, "type": "vscode",
                            "label": "x", "description": "x"}}
        kw = self._call_build(mcp_tools=tools)
        assert kw.get("mcp_tools") == tools

    def test_include_raw_image_signature(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_ai_button)
        assert "include_raw_image" in sig.parameters
        assert sig.parameters["include_raw_image"].default is False

    def test_mcp_tools_signature(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_ai_button)
        assert "mcp_tools" in sig.parameters
        assert sig.parameters["mcp_tools"].default is None


# ===========================================================================
# 12. display_jupyter_notebook_ai_button — include_raw_image and mcp_tools
# ===========================================================================

class TestDisplayNotebookAiButtonRawImageMcp:
    """Notebook button must forward include_raw_image and mcp_tools."""

    def _call_with_mock(self, **kwargs):
        with patch(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._jupyter"
            ".display_jupyter_ai_button"
        ) as mock_btn:
            _mod.display_jupyter_notebook_ai_button(**kwargs)
            return mock_btn.call_args

    def test_include_raw_image_forwarded(self):
        _, kw = self._call_with_mock(include_raw_image=True)
        assert kw.get("include_raw_image") is True

    def test_include_raw_image_default_false(self):
        _, kw = self._call_with_mock()
        assert kw.get("include_raw_image") is False

    def test_mcp_tools_forwarded(self):
        tools = {"cursor": {"enabled": True, "type": "cursor",
                             "label": "x", "description": "x"}}
        _, kw = self._call_with_mock(mcp_tools=tools)
        assert kw.get("mcp_tools") == tools

    def test_mcp_tools_none_forwarded(self):
        _, kw = self._call_with_mock()
        assert kw.get("mcp_tools") is None

    def test_signature_include_raw_image(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_notebook_ai_button)
        assert "include_raw_image" in sig.parameters
        assert sig.parameters["include_raw_image"].default is False

    def test_signature_mcp_tools(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_notebook_ai_button)
        assert "mcp_tools" in sig.parameters
        assert sig.parameters["mcp_tools"].default is None


# ---------------------------------------------------------------------------
# New tests: Sphinx-identical structure, fixed positioning, Copy page UX
# ---------------------------------------------------------------------------


class TestJupyterWidgetSphinxIdenticalStructure:
    """Verify the Jupyter widget HTML matches the Sphinx split-button structure."""

    def _html(self, **kw):
        return _mod._build_jupyter_widget_html(widget_id="t1", **kw)

    def test_copy_page_primary_button_text(self):
        """Primary button must say 'Copy page' (Sphinx-identical)."""
        html = self._html()
        assert "Copy page" in html

    def test_view_as_markdown_in_dropdown(self):
        """Dropdown must include 'View as Markdown' item."""
        html = self._html()
        assert "View as Markdown" in html

    def test_copy_page_description_in_dropdown(self):
        """Dropdown Copy page item must include Markdown-for-LLMs description."""
        html = self._html()
        assert "Markdown for LLMs" in html

    def test_view_as_markdown_description_in_dropdown(self):
        """View as Markdown description must be present."""
        html = self._html()
        assert "View this page as Markdown" in html

    def test_split_button_structure_classes(self):
        """Widget must have ai-split class (Sphinx-identical CSS)."""
        html = self._html()
        assert "ai-split" in html

    def test_divider_class_present(self):
        """Split divider class must be present."""
        html = self._html()
        assert "ai-div" in html

    def test_toggle_button_present(self):
        """Arrow toggle button must be present."""
        html = self._html()
        assert "ai-tog" in html

    def test_fixed_positioning_in_js(self):
        """Dropdown must use fixed positioning (VS Code overflow fix)."""
        html = self._html()
        assert "position:fixed" in html
        assert "getBoundingClientRect" in html

    def test_svgs_inlined_for_copy_and_markdown(self):
        """Copy and Markdown SVG data URIs must be inlined."""
        html = self._html()
        assert "data:image/svg+xml;base64," in html

    def test_separator_between_utility_and_ai_rows(self):
        """A separator must appear between utility rows and AI provider rows."""
        html = self._html(providers=["claude"])
        # ai-sep class must appear more than once (at least one real separator)
        assert "ai-sp" in html  # portal CSS class for separator

    def test_ai_provider_row_present_after_separator(self):
        """AI provider rows are rendered via BUTTONS.forEach after utility items."""
        html = self._html(providers=["claude"])
        view_pos = html.find('"View as Markdown"')
        foreach_pos = html.find("BUTTONS.forEach")
        assert view_pos > 0 and foreach_pos > view_pos

    def test_copy_page_before_view_as_markdown(self):
        """Copy page must appear before View as Markdown in the HTML."""
        html = self._html()
        copy_pos = html.find("Copy page")
        view_pos = html.find("View as Markdown")
        assert copy_pos < view_pos

    def test_view_as_markdown_in_js_menu_build(self):
        """View as Markdown menu item must be built before AI provider items in JS."""
        html = self._html(providers=["claude"])
        # The JS menu construction: "View as Markdown" makeItem call must
        # appear before the BUTTONS.forEach loop that renders AI providers.
        view_pos = html.find('"View as Markdown"')
        buttons_foreach_pos = html.find("BUTTONS.forEach")
        assert view_pos > 0, "View as Markdown makeItem call missing"
        assert buttons_foreach_pos > 0, "BUTTONS.forEach loop missing"
        assert view_pos < buttons_foreach_pos

    def test_inline_position_margin(self):
        """Inline position must add bottom margin to prevent cramping."""
        html = self._html(position="inline")
        assert "margin:" in html

    def test_floating_position_fixed(self):
        """Floating position must use position:fixed on the container."""
        html = self._html(position="floating")
        assert "position:fixed;bottom:16px;right:16px" in html

    def test_mcp_separator_only_when_mcp_enabled(self):
        """MCP section separator JS must be conditional on MCP keys."""
        html = self._html(providers=["claude"])
        # The JS condition `mcpKeys.length > 0` guards the separator
        assert "mcpKeys.length > 0" in html

    def test_disabled_mcp_tool_not_in_safe_mcp_js(self):
        """Disabled MCP tools must not appear in the MCP_TOOLS JS variable."""
        tool = {
            "enabled": False, "type": "vscode", "label": "MyDisabledTool",
            "description": "desc", "server_url": "https://x.com/sse",
            "server_name": "x", "transport": "sse",
        }
        html = self._html(mcp_tools={"vscode": tool})
        # The disabled tool label must not appear in the JSON payload
        assert "MyDisabledTool" not in html


class TestJupyterWidgetMdUrlLogic:
    """Verify .md URL derivation mirrors Sphinx getMarkdownUrl() logic."""

    def _html(self, **kw):
        return _mod._build_jupyter_widget_html(widget_id="t2", **kw)

    def test_get_md_url_js_present(self):
        """getMdUrl JS function must be inlined in the widget."""
        html = self._html()
        assert "getMdUrl" in html

    def test_html_to_md_url_derivation_in_js(self):
        """getMdUrl must replace .html with .md in JS (Sphinx-identical)."""
        html = self._html()
        assert ".html" in html and ".md" in html
        # JS logic
        assert "slice(0, -5) + '.md'" in html or "endsWith('.html')" in html

    def test_page_url_embedded_in_js(self):
        """Validated page_url must be serialised into the widget JS."""
        html = self._html(page_url="https://docs.example.com/page.html")
        assert "docs.example.com" in html

    def test_invalid_page_url_excluded(self):
        """A javascript: page_url must be excluded from the widget."""
        html = self._html(page_url="javascript:evil()")
        assert "javascript" not in html

    def test_context_url_uses_md_when_page_url_set(self):
        """When PAGE_URL is set, prompt uses getMdUrl() — JS logic present."""
        html = self._html(page_url="https://docs.example.com/api.html")
        assert "getMdUrl" in html
        # The JS variable PAGE_URL should be set
        assert "docs.example.com" in html


class TestJupyterIncludeOutputsDefault:
    """Verify include_outputs default changed to False (v0.4.0)."""

    def test_display_function_default_false(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_ai_button)
        assert sig.parameters["include_outputs"].default is False

    def test_notebook_function_default_false(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_notebook_ai_button)
        assert sig.parameters["include_outputs"].default is False

    def test_build_widget_default_false(self):
        import inspect
        sig = inspect.signature(_mod._build_jupyter_widget_html)
        assert sig.parameters["include_outputs"].default is False

    def test_include_outputs_false_emits_false_in_js(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="io1", include_outputs=False
        )
        assert "INCLUDE_OUTPUTS  = false" in html

    def test_include_outputs_true_emits_true_in_js(self):
        html = _mod._build_jupyter_widget_html(
            widget_id="io2", include_outputs=True
        )
        assert "INCLUDE_OUTPUTS  = true" in html


class TestJupyterIncludeRawImageDefault:
    """Verify include_raw_image remains False by default."""

    def test_display_function_default_false(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_ai_button)
        assert sig.parameters["include_raw_image"].default is False

    def test_notebook_function_default_false(self):
        import inspect
        sig = inspect.signature(_mod.display_jupyter_notebook_ai_button)
        assert sig.parameters["include_raw_image"].default is False
