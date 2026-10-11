from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "_static" / "sphinx-feedback.js").read_text(encoding="utf-8")
CSS = (ROOT / "_static" / "sphinx-feedback.css").read_text(encoding="utf-8")
SPHINX = (ROOT / "_sphinx.py").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_runtime_has_no_localstorage_cookie_or_pageview_telemetry():
    assert "localStorage" not in JS
    assert "document.cookie" not in JS
    assert "navigator.userAgent" not in JS
    assert "document.referrer" not in JS
    assert "window.location" not in JS


def test_runtime_uses_secure_192_bit_event_nonce():
    assert "new Uint8Array(24)" in JS
    assert "cryptoApi.getRandomValues" in JS
    assert "feedback-${" in JS


def test_request_transport_omits_credentials_referrer_cache_and_redirects():
    for expected in [
        "credentials: 'omit'",
        "cache: 'no-store'",
        "redirect: 'error'",
        "referrerPolicy: 'no-referrer'",
        "AbortController",
    ]:
        assert expected in JS


def test_no_fetch_is_called_by_boot_or_mount_paths():
    assert JS.count("fetch(config.endpoint") == 1
    assert "async function submitRequest" in JS


def test_pending_and_accepted_state_are_session_only_revision_scoped_and_authority_bound():
    assert "sessionStorage" in JS
    assert "PENDING_PREFIX" in JS
    assert "ACCEPTED_PREFIX" in JS
    assert "page_revision" in JS
    assert "parsed.endpoint !== config.endpoint" in JS
    assert JS.count("endpoint: config.endpoint") >= 2


def test_quick_mode_feature_flag_is_honored():
    assert "if (this.config.quick_enabled)" in JS
    assert "mode === 'quick' && !this.config.quick_enabled" in JS


def test_duplicate_mounts_share_one_controller():
    assert "__SPHINX_FEEDBACK_CONTROLLER__" in JS
    assert "this.views = new Set()" in JS
    assert "current.addView" in JS


def test_ambiguous_retry_reuses_existing_feedback_id():
    assert "Retry will reuse the same feedback id" in JS
    assert "this.pending && this.pending.fingerprint === fingerprint" in JS
    assert "if (this.pending)" in JS


def test_success_receipt_is_verified_before_acceptance():
    assert "page.feedback-receipt.v1" in JS
    assert "receipt.feedback_id !== request.feedback_id" in JS
    assert "HASH_RE.test" in JS


def test_unknown_counter_is_hidden_and_counts_use_compact_kmbt_formatter():
    assert "summary.hidden = true" in JS
    assert "function formatCompactCount" in JS
    for marker in ["1K", "1,000", "1M", "1,000,000", "1B", "1,000,000,000", "1T", "1,000,000,000,000"]:
        assert marker in JS
    assert "rounded >= 1000" in JS
    assert "sphinx-feedback-quick-count" in JS
    assert "positive_count" in JS and "negative_count" in JS


def test_generic_quick_counts_match_ai_learn_dom_contract_when_distribution_is_known():
    assert "'data-feedback-count': hasDistribution ? String(negativeCount) : null" in JS
    assert "'data-feedback-count': hasDistribution ? String(positiveCount) : null" in JS
    assert "'data-sphinx-feedback-quick-count': kind === 'down' ? '-1' : '1'" in JS
    assert 'aggregate_meta.get("complete") is True' in SPHINX
    assert '"positive_count": 0' in SPHINX
    assert '"negative_count": 0' in SPHINX
    assert '`"complete": true`' in README


def test_empty_status_has_zero_geometry():
    assert ".sphinx-feedback-status[hidden]" in CSS
    assert "display:none!important" in CSS


def test_rating_values_are_only_visible_for_selected_rating():
    assert ".sphinx-feedback-rating-value{display:none" in CSS
    assert '[aria-pressed="true"] .sphinx-feedback-rating-value{display:inline}' in CSS


def test_layout_is_container_responsive_for_narrow_sidebars():
    assert "container-type:inline-size" in CSS
    assert "@container" in CSS
    assert "repeat(auto-fit" in CSS


def test_css_is_fully_namespaced_and_has_no_theme_dependency():
    assert ".learn-" not in CSS
    assert ".bd-" not in CSS
    assert ".sphinx-feedback" in CSS


def test_sphinx_assets_are_external_not_inline_executable_script():
    assert 'type="application/json"' in SPHINX
    assert 'app.add_js_file("sphinx-feedback.js"' in SPHINX
    assert "<script>" not in SPHINX


def test_extension_does_not_import_ai_learn_or_ai_assistant():
    # _example_conf.py names the AI assistant to show a combined conf.py; it
    # is documentation and imports nothing (checked below).
    package_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*.py")
        if "tests" not in path.parts and path.name != "_example_conf.py"
    )
    assert "_sphinx_ai_learn" not in package_sources
    assert "_sphinx_ai_assistant" not in package_sources


def test_example_conf_imports_nothing():
    import ast  # noqa: PLC0415

    tree = ast.parse((ROOT / "_example_conf.py").read_text(encoding="utf-8"))
    assert not [
        node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_readme_states_event_not_person_invariant():
    assert "feedback event represents a page reaction, never a person" in README
    assert "No network request occurs on page view" in README


def test_pending_quick_retry_remains_visibly_selected_until_definitive_rejection():
    assert "if (pending.request.mode === 'quick') this.quick = pending.request.rating" in JS
    assert "if (pending.request.mode === 'quick') this.quick = null" in JS


def test_browser_asset_names_are_extension_namespaced():
    assert (ROOT / "_static" / "sphinx-feedback.js").is_file()
    assert (ROOT / "_static" / "sphinx-feedback.css").is_file()
    assert not (ROOT / "_static" / "feedback.js").exists()
    assert not (ROOT / "_static" / "feedback.css").exists()


def test_disabled_extension_does_not_register_browser_assets():
    assert 'if not normalized or not normalized["page_enabled"]' in SPHINX
    assert 'normalized["page_enabled"]\n            and normalized["counter_enabled"]' in SPHINX


def test_session_state_bounds_cover_full_valid_endpoint_and_cleanup_stale_values():
    assert "MAX_ACCEPTED_STATE_CHARS = 4096" in JS
    assert "MAX_PENDING_STATE_CHARS = 12000" in JS
    assert "sessionStorage.removeItem(key);" in JS


def test_pending_retry_is_invalidated_when_interaction_policy_changes():
    assert "request.mode === 'quick' && config.quick_enabled !== true" in JS
    assert "request.mode === 'detailed' && config.detailed_enabled !== true" in JS
    assert "comment && config.comment_enabled !== true" in JS
    assert "contributor && config.contributor_enabled !== true" in JS


def test_controller_rebinds_when_page_config_changes_and_removes_stale_auto_mounts():
    assert "canonicalJson(current.config) === canonicalJson(config)" in JS
    assert "[data-sphinx-feedback-auto]" in JS
    assert "mount.remove()" in JS
    assert "if (view.mount === mount) return" in JS


def test_controller_prunes_detached_theme_mounts():
    assert "view.mount?.isConnected === false" in JS
    assert "this.views.delete(view)" in JS


def test_browser_reserves_anonymous_sentinel_and_bounds_response_fragmentation():
    assert "text.toLowerCase() === 'anonymous'" in JS
    assert "Anonymous” is reserved" in JS
    assert "MAX_RESPONSE_CHUNKS = 512" in JS
    assert "chunkCount > MAX_RESPONSE_CHUNKS" in JS
    assert "response was too fragmented" in JS


def test_http_error_classification_keeps_only_retryable_client_statuses_pending():
    assert "![408, 425, 429].includes(this.status)" in JS
    assert "if (!response.ok)" in JS
    assert "Feedback service returned HTTP ${response.status}." in JS


def test_malformed_session_state_is_removed_instead_of_poisoning_future_boots():
    assert JS.count("try { sessionStorage.removeItem(key); } catch {}") >= 2


def test_browser_response_verification_fails_closed_without_streaming_body_reader():
    assert "Streaming feedback response verification is unavailable" in JS
    assert "await response.text()" not in JS

def test_quick_actions_use_inline_svg_with_unicode_fallback():
    assert "document.createElementNS(SVG_NS, 'svg')" in JS
    assert "sphinx-feedback-icon-fallback" in JS
    for fallback in ["👎", "👍", "⌄"]:
        assert fallback in JS
    assert "M7.653 15.369" in JS
    assert "M8.347.631" in JS
    assert "6 9 12 15 18 9" in JS
    assert "innerHTML" not in JS


def test_quick_action_icons_are_decorative_and_choice_labels_preserve_state_clarity():
    assert "svg.setAttribute('aria-hidden', 'true')" in JS
    assert "svg.setAttribute('focusable', 'false')" in JS
    assert "sphinx-feedback-choice-label" in JS
    assert '.sphinx-feedback-choice-label{display:none' in CSS
    assert '[aria-pressed=\"true\"]>.sphinx-feedback-choice-label' in CSS
    assert 'sphinx-feedback-expand[aria-expanded=\"true\"]>.sphinx-feedback-choice-label' in CSS


def test_quick_action_selected_tones_are_namespaced_and_theme_independent():
    assert "'data-tone': 'negative'" in JS
    assert "'data-tone': 'positive'" in JS
    assert '.sphinx-feedback [data-tone=\"negative\"][aria-pressed=\"true\"]' in CSS
    assert '.sphinx-feedback [data-tone=\"positive\"][aria-pressed=\"true\"]' in CSS



def test_quick_action_selected_state_respects_forced_colors():
    assert '@media(forced-colors:active)' in CSS
    assert 'border-color:Highlight' in CSS
    assert 'color:HighlightText' in CSS


def test_quick_counts_use_logical_divider_and_require_consistent_distribution():
    assert "sphinx-feedback-quick-count" in CSS
    assert "border-inline-start" in CSS
    assert "border-inline-end" in CSS
    assert "counter.neutral_count" in JS
    assert "counter.negative_count + counter.positive_count + counter.neutral_count === counter.count" in JS
    assert "fillQuickButton(down, 'down', negativeCount, quickCountPosition(this.config, 'down'))" in JS
    assert "fillQuickButton(up, 'up', positiveCount, quickCountPosition(this.config, 'up'))" in JS
    assert 'for field in ("positive_count", "negative_count", "neutral_count")' in SPHINX


def test_quick_count_position_is_configurable_per_button_with_balanced_fallback():
    assert "function quickCountPosition(config, kind)" in JS
    assert "kind === 'down' ? 'left' : 'right'" in JS
    assert "config?.buttons_ratings?.[key]" in JS
    assert "data-rating-position" in JS
    assert "quickCountPosition(this.config, 'down')" in JS
    assert "quickCountPosition(this.config, 'up')" in JS
    assert '.sphinx-feedback-quick-count[data-rating-position="left"]' in CSS
    assert '.sphinx-feedback-quick-count[data-rating-position="right"]' in CSS
    assert '"buttons_ratings": dict(normalized["buttons_ratings"])' in SPHINX
    assert '"feedback_buttons_ratings"' in SPHINX


def test_readme_documents_all_supported_quick_count_layouts():
    assert 'feedback_buttons_ratings' in README
    assert 'left_button_rating' in README
    assert 'right_button_rating' in README
    assert '[0 | 👎]' in README
    assert '[👎 | 0]' in README
