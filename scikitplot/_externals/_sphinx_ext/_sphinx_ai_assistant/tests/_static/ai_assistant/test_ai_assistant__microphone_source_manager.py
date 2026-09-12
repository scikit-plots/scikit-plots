from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT
from pathlib import Path

ROOT = RUNTIME_ROOT
JS = (ROOT / "_static" / "ai-assistant.js").read_text(encoding="utf-8")
CSS = (ROOT / "_static" / "ai-assistant.css").read_text(encoding="utf-8")
INIT = (ROOT / "__init__.py").read_text(encoding="utf-8")


def test_picker_is_radiogroup_with_real_radio_buttons():
    # Scope the ARIA-role contract to the microphone picker. Other independent
    # menus (for example Run 171's quick model chooser) may correctly use
    # menuitemradio without weakening the microphone source manager.
    build_start = JS.index("function _buildMicHoverPopup")
    refresh_start = JS.index("function _refreshMicDeviceList")
    refresh_end = JS.index("// ── Speech recognition", refresh_start)
    body = JS[build_start:refresh_start] + JS[refresh_start:refresh_end]
    assert "devList.setAttribute('role', 'radiogroup')" in body
    assert "document.createElement('button')" in body
    assert "item.setAttribute('role', 'radio')" in body
    assert "item.setAttribute('tabindex', dev.deviceId === effectiveId ? '0' : '-1')" in body
    assert "role', 'menuitemradio'" not in body


def test_popup_refresh_does_not_prompt_for_microphone():
    start = JS.index("function _refreshMicDeviceList")
    end = JS.index("// ── Speech recognition", start)
    body = JS[start:end]
    assert "_enumMicDevices" in body
    assert ".getUserMedia(" not in body
    assert "Opening/pinning the popup" in body


def test_permission_prompt_requires_explicit_allow_action():
    assert "ai-assistant-mic-perm-allow" in JS
    assert "Allow microphone and show available devices" in JS
    assert "navigator.mediaDevices.getUserMedia({ audio: true })" in JS


def test_device_selection_commits_immediately_and_capture_still_fails_closed():
    start = JS.index("function _selectMicDevice")
    end = JS.index("function _syncMicDeviceUI", start)
    body = JS[start:end]
    assert "_setMicDevice(requestedId)" in body
    assert ".getUserMedia(" not in body
    assert "verified when recording starts" in body
    shared = JS[JS.index("function _toggleSpeechRecognition"):JS.index("function _stopSpeechRecognition")]
    assert "getUserMedia(_micConstraintsForDevice(_micDeviceId))" in shared
    assert "using browser default" not in shared.lower()
    assert "Selected microphone is unavailable" in shared


def test_selected_track_is_passed_directly_when_supported():
    assert "function _micTrackInputSupported()" in JS
    assert "_speechRecognition.start(_micPinTrack)" in JS
    assert "this browser may still use its system-default input" in JS


def test_live_capture_has_bounded_release_lifecycle():
    assert "var _MIC_STREAM_GRACE_MS = 3000" in JS
    assert "function _scheduleMicStreamRelease()" in JS
    assert JS.count("_scheduleMicStreamRelease();") >= 3


def test_banner_and_footer_share_one_visible_recognition_controller():
    banner_start = JS.index("speakBannerEl.addEventListener('click'")
    banner_body = JS[banner_start:banner_start + 400]
    assert "_toggleSpeechRecognition()" in banner_body
    assert "_bannerToggle()" not in banner_body


def test_space_push_to_talk_is_scoped_and_safe():
    assert "function _bindMicSpacePushToTalk()" in JS
    assert "if (_isTextEntryTarget(e.target)) return" in JS
    assert "blockedSurfaceOpen()" in JS
    assert "e.repeat" in JS
    assert "e.isComposing" in JS
    assert "window.addEventListener('blur'" in JS
    assert "document.addEventListener('visibilitychange'" in JS
    assert "_micSpaceHeld = false" in JS
    assert "aria-keyshortcuts', 'Space'" in JS


def test_space_shortcut_is_configurable_from_sphinx():
    assert '"panelMicSpaceShortcut": _cfg_bool(' in INIT
    assert '"ai_assistant_panel_mic_space_shortcut", True' in INIT
    assert 'app.add_config_value("ai_assistant_panel_mic_space_shortcut", True, "html")' in INIT


def test_keyboard_shortcut_sheet_documents_microphone_controls():
    assert "sectionTitle('Microphone')" in JS
    assert "shortcutRow('Hold to speak', ['Space']" in JS
    assert "shortcutRow('Stop microphone capture', ['Escape'])" in JS


def test_css_resets_radio_buttons_and_styles_capability():
    assert ".ai-assistant-mic-device-item {" in CSS
    assert "width: 100%;" in CSS
    assert "background: transparent;" in CSS
    assert '.ai-assistant-mic-device-item[aria-checked="true"] {' in CSS
    assert "box-shadow: inset 0 0 0 1px" in CSS
    assert ".ai-assistant-mic-device-capability" in CSS
    assert ".ai-assistant-mic-perm-allow" in CSS
