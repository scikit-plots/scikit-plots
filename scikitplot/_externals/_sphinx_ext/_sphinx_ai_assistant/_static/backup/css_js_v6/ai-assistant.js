/**
 * Sphinx AI Assistant — ai-assistant.js
 *
 * Features
 * ────────
 *   • Markdown export (clipboard copy + view as .md)
 *   • AI chat deep-links (Claude, ChatGPT, Gemini, …)
 *   • MCP tool integration (VS Code, Claude Desktop, …)
 *   • PDF export with URL-mode / Print-mode toggle
 *   • Floating AI panel with:
 *       – Minimize (hide to floating "Ask Us" trigger pill)
 *       – Maximize (expand to full viewport height)
 *       – Quick-suggestion chips (customizable, 0–5 items)
 *       – "Speak with assistant" banner / microphone button
 *         (Web Speech API — degrades gracefully when unavailable)
 *       – Icon-button footer: [textarea] [mic] [send]
 *
 * All behaviour driven by window.AI_ASSISTANT_CONFIG injected by the
 * Python extension's add_ai_assistant_context().
 *
 * Security
 * ────────
 *   – All user-facing HTML via textContent / setAttribute, never innerHTML.
 *   – window.open() passes 'noopener,noreferrer' on all external links.
 *   – sessionStorage for PDF-mode persistence (no cross-origin leak).
 *
 * Developer notes
 * ───────────────
 *   – Every public function is at module scope inside the IIFE.
 *   – The global 'click' listener for dropdown-close is registered once only
 *     (guarded by _listenersAttached).
 *   – Turndown is loaded lazily from CDN on first use.
 *   – Speech recognition is lazy-started on first mic click (no permission
 *     prompts until the user explicitly clicks the mic icon).
 */

(function () {
    'use strict';

    // Guard against multiple injections
    if (window.SphinxAIAssistantInitialized) return;
    window.SphinxAIAssistantInitialized = true;

    // ── Module-level singletons ───────────────────────────────────────────────

    /** Captured synchronously before any async boundary — see note in v1. */
    var _selfSrc = (document.currentScript && document.currentScript.src) || null;

    /** True once the global document 'click' listener has been registered. */
    var _listenersAttached = false;

    /**
     * Lazy singleton: the floating AI panel element.
     * Declared at IIFE scope so the falsy check in toggleAIPanel() is safe.
     * @type {HTMLElement|null}
     */
    var _aiPanelEl = null;

    /**
     * Lazy singleton: the floating "Ask Us" trigger pill (shown when minimized).
     * @type {HTMLElement|null}
     */
    var _aiTriggerEl = null;

    /** sessionStorage key for PDF mode persistence. */
    var _PDF_MODE_KEY = 'ai-assistant-pdf-mode';

    /**
     * Stable radio-group name for the model sheet.
     *
     * Using a deterministic constant (not Math.random) so:
     *   • DevTools always show the same name across page loads.
     *   • External code that queries by name works reliably.
     *   • Sheet rebuilds on config hot-reload can be correlated correctly.
     *
     * The value mirrors the sheet's element id so the two can be trivially
     * correlated from outside this IIFE.
     */
    var _MODEL_RADIO_GROUP = 'ai-assistant-model-group';

    /**
     * Whether hold-to-record mode is active.
     * In hold mode: pointerdown on the mic button starts recognition,
     * pointerup / pointerleave stops it — matching the Claude.ai interaction.
     * In toggle mode (default): click toggles recognition on/off.
     *
     * Persisted in localStorage so the preference survives page reloads.
     * Falls back gracefully when localStorage is unavailable (private mode,
     * storage quota exceeded, cross-origin iframe, etc.).
     */
    var _micHoldMode = (function () {
        try {
            return localStorage.getItem('ai-assistant-mic-hold-mode') === 'true';
        } catch (_) {
            return false;
        }
    }());

    /**
     * Selected microphone device ID.
     *
     * The Web Speech API exposes no direct device-selection parameter.
     * Workaround: acquire a getUserMedia stream on the chosen device BEFORE
     * calling SpeechRecognition.start() — the browser reuses the active track.
     *
     * Empty string = browser default (no getUserMedia pre-pin).
     * Persisted to localStorage so the preference survives page reloads.
     * Silently falls back when storage is unavailable (private mode, quota, etc.).
     */
    var _micDeviceId = (function () {
        try {
            return localStorage.getItem('ai-assistant-mic-device-id') || '';
        } catch (_) {
            return '';
        }
    }());

    /**
     * Cached list of available audio input devices.
     * Populated by _enumMicDevices(); empty until first popup open.
     *
     * @type {Array<{deviceId:string, label:string}>}
     */
    var _micDevices = [];

    /**
     * MediaStreamTrack acquired to pin a non-default device for Web Speech API.
     *
     * Kept alive ACROSS hold-to-record presses for the same device so the
     * browser never needs to re-prompt.  Released only when the selected device
     * changes (via _setMicDevice) or when _releaseMicPinTrack is called explicitly.
     *
     * @type {MediaStreamTrack|null}
     */
    var _micPinTrack = null;

    /**
     * Persistent warm MediaStream that keeps the microphone permission live for
     * the entire page session.
     *
     * Acquired once (on first popup open or first recording) via
     * _acquireMicWarmStream().  Holding an active track prevents Chromium and
     * Safari from revoking the origin's mic permission between recognition
     * sessions, which is what causes the browser to re-prompt on every
     * hold-to-record press.
     *
     * Released only via _releaseMicWarmStream() when the selected device changes
     * so the next call to _acquireMicWarmStream() re-acquires on the new device.
     *
     * @type {MediaStream|null}
     */
    var _micWarmStream = null;

    // ── Web Audio API visualisation — module-level singletons ────────────────

    /**
     * Idle sinusoidal heights for the 100 mic popup level bars (px).
     *
     * 100-element look-up table: values rise from 2 px at the edges to 13 px
     * at the centre following a half-sine arch
     *   h[i] = max(2, round(2 + 11 * sin(π × i / 99)))
     * This mirrors the natural envelope of a spoken-word audio waveform and
     * provides a visually balanced resting state.
     *
     * Shared between _buildMicHoverPopup (initial bar heights on DOM creation)
     * and _stopVizLoops (bar reset when recording stops) so there is exactly
     * one source of truth for the idle shape.
     *
     * @type {number[]}
     */
    var _IDLE_LEVEL_HEIGHTS = [
        2,2,3,3,3,4,4,4,5,5,
        5,6,6,6,7,7,7,8,8,8,
        9,9,9,9,10,10,10,10,11,11,
        11,11,11,12,12,12,12,12,12,12,
        13,13,13,13,13,13,13,13,13,13,
        13,13,13,13,13,13,13,13,13,13,
        12,12,12,12,12,12,12,11,11,11,
        11,11,10,10,10,10,9,9,9,9,
        8,8,8,7,7,7,6,6,6,5,
        5,5,4,4,4,3,3,3,2,2
    ];

    /** Number of bars in the footer soundbar ring buffer. @type {number} */
    var _SOUNDBAR_BARS    = 20;

    /** Silence / minimum bar height in px. @type {number} */
    var _SOUNDBAR_MIN_H   = 2;

    /** Full-scale / maximum bar height in px (capped to container). @type {number} */
    var _SOUNDBAR_MAX_H   = 16;

    /** Milliseconds between ring-buffer amplitude snapshots. @type {number} */
    var _SOUNDBAR_TICK_MS = 80;

    /** Silence floor for mic popup level bars in px. @type {number} */
    var _MIC_LEVEL_MIN_H  = 2;

    /** Full-scale peak for mic popup level bars in px. @type {number} */
    var _MIC_LEVEL_MAX_H  = 18;

    /** Web Audio AudioContext — created per session, closed in _disconnectWebAudio. @type {AudioContext|null} */
    var _audioCtx         = null;

    /** AnalyserNode fed from the mic stream — provides frequency / time-domain data. @type {AnalyserNode|null} */
    var _analyserNode     = null;

    /** MediaStreamAudioSourceNode that connects the mic track to _analyserNode. @type {MediaStreamAudioSourceNode|null} */
    var _audioSrcNode     = null;

    /** requestAnimationFrame handle for the mic popup level-bars loop. @type {number|null} */
    var _vizRafId         = null;

    /** setInterval handle for the footer soundbar ring-buffer tick. @type {number|null} */
    var _soundbarTickId   = null;

    /** Ring buffer of pixel heights for the footer soundbar (length = _SOUNDBAR_BARS). @type {number[]} */
    var _soundbarHeights  = [];

    /**
     * Number of bars in the mic popup VU meter level visualiser.
     *
     * 100 bars represent the full standard 0-100 dB level scale.
     * At 2 px per bar + 1 px gap the strip is 299 px — fits inside the
     * 330 px popup min-width with 31 px to spare.
     *
     * Rendering cost: 100 style.height writes per rAF frame (~16 ms budget)
     * is negligible on all modern and legacy mobile devices.
     *
     * @type {number}
     */
    var _MIC_LEVEL_BAR_COUNT = 100;

    // ── VU meter constants (module-level, immutable) ──────────────────────
    //
    // Standard dBFS-to-bar mapping for the 100-bar popup level visualiser.
    //
    // dBFS floor  = -40 dBFS  — practical voice/mic lower bound.
    //               Signals below this threshold are treated as silence.
    // dBFS range  =  40 dB    — floor (-40) to full scale (0 dBFS).
    // Bar index   = round((dBFS - floor) / range × 100), clamped 0-100.
    //
    // Colour zones (standard broadcast/audio convention):
    //   bars  0-59  primary/blue   → -40…-16.4 dBFS  (safe speech level)
    //   bars 60-79  amber #f59e0b  → -16…-8.4  dBFS  (loud / hot)
    //   bars 80-99  red   #ef4444  →  -8…-0.4  dBFS  (peak / clip risk)

    /** dBFS silence floor (signals below this → bar index 0). @type {number} */
    var _VU_DB_FLOOR    = -40;

    /** dBFS working range (floor to full scale). @type {number} */
    var _VU_DB_RANGE    =  40;

    /** First bar index of the amber (loud) zone. @type {number} */
    var _VU_ZONE_AMBER  = 60;

    /** First bar index of the red (peak) zone. @type {number} */
    var _VU_ZONE_RED    = 80;

    /** Amber zone bar colour. @type {string} */
    var _VU_COLOR_AMBER = '#f59e0b';

    /** Red / peak zone bar colour. @type {string} */
    var _VU_COLOR_RED   = '#ef4444';

    /**
     * Frames a newly raised peak bar is held before decay begins (~670 ms at
     * 60 fps; 40 frames × 16.7 ms).
     * @type {number}
     */
    var _VU_PEAK_HOLD   = 40;

    /**
     * Frames between each 1-bar downward decay step once hold expires
     * (~50 ms per bar at 60 fps; full 100-bar sweep ≈ 5 s).
     * @type {number}
     */
    var _VU_PEAK_DECAY  = 3;

    // ── VU meter mutable state (reset on each _startVizLoops call) ────────

    /** Highest bar index seen in the current hold window. @type {number} */
    var _vuPeakBar      = 0;

    /** Frames elapsed since _vuPeakBar was last raised. @type {number} */
    var _vuPeakHold     = 0;

    /** Frame counter driving the decay cadence. @type {number} */
    var _vuPeakDecay    = 0;
    /**
     * requestAnimationFrame polyfill.
     *
     * Standard since Chrome 24 / Firefox 23 / Safari 6.1 / IE 10.
     * Falls back to a 16 ms setTimeout (≈ 60 fps) on legacy Android 4.x
     * WebView and very old desktop browsers.
     *
     * @type {function}
     */
    var _RAF = (window.requestAnimationFrame
             || window.webkitRequestAnimationFrame
             || window.mozRequestAnimationFrame
             || function (cb) { return window.setTimeout(cb, 16); });
    /**
     * cancelAnimationFrame polyfill paired with _RAF above.
     *
     * @type {function}
     */
    var _CAF = (window.cancelAnimationFrame
             || window.webkitCancelAnimationFrame
             || window.mozCancelAnimationFrame
             || window.clearTimeout);


    /**
     * AbortController for the current in-flight panel API fetch.
     * Cancelled (and replaced) whenever a new question is submitted so
     * stale streaming responses never race with the new turn.
     * @type {AbortController|null}
     */
    var _fetchAbortController = null;

    /**
     * Maximum number of turns stored in `_transcript` (and persisted to
     * sessionStorage).  Prevents unbounded memory growth in very long sessions.
     * Each turn is one user message or one assistant/error reply; this cap is
     * applied BEFORE the new turn is appended so the array never exceeds it.
     * Value is intentionally generous (200 turns ≈ 100 Q&A pairs) — most
     * real-world sessions are under 20.  Configurable via
     * ``cfg.panelMaxTranscriptTurns`` in conf.py.
     */
    var _TRANSCRIPT_MAX_TURNS_DEFAULT = 200;

    /**
     * Feature-flag defaults — last line of defence when the injected
     * window.AI_ASSISTANT_CONFIG.features dict is missing/partial.
     *
     * CRITICAL — SINGLE SOURCE OF TRUTH CONTRACT
     * ──────────────────────────────────────────
     * This object MUST stay byte-for-byte equivalent to
     * ``_DEFAULT_FEATURES`` in __init__.py.  They were previously out of
     * sync (JS said ai_panel:false while Python said ai_panel:true), which
     * silently hid the AI panel whenever the inline config script was
     * stripped/reordered by a CDN, CSP, or downstream theme.  That is the
     * exact "BUG-1" the Python merge claims to fix — the merge is necessary
     * but NOT sufficient; the two default tables must also agree.
     *
     * If you change a default here, change __init__.py:_DEFAULT_FEATURES in
     * the same commit, and update the parity test.
     */
    var FEATURE_DEFAULTS = {
        markdown_export: true,
        view_markdown:   true,
        ai_chat:         true,
        mcp_integration: true,   // mirrors Python; set False if no MCP tools
        theme_toggle:    true,
        pdf_export:      true,
        ai_panel:        true,
    };

    /**
     * Regex: matches any icon value that is already an absolute URI — must
     * NOT be prefixed with the _static path.
     */
    var _ABSOLUTE_ICON_RE = /^(?:https?:|data:|blob:|\/)/;

    // ── SVG templates (inline, no external file dependency) ──────────────────

    /**
     * Inline SVG strings used for icon buttons inside the panel.
     * Stored as strings so they can be set via innerHTML on a single wrapper
     * element (not on user-controlled data — safe).
     */
    var ICONS = {
        minimize: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>',
        maximize: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>',
        restore:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="8 3 3 3 3 8"/><polyline points="21 8 21 3 16 3"/><polyline points="3 16 3 21 8 21"/><polyline points="16 21 21 21 21 16"/></svg>',
        close:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
        mic:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="9" y1="22" x2="15" y2="22"/></svg>',
        send:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>',
        chat:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><circle cx="9" cy="11" r="0.8" fill="currentColor" stroke="none"/><circle cx="12" cy="11" r="0.8" fill="currentColor" stroke="none"/><circle cx="15" cy="11" r="0.8" fill="currentColor" stroke="none"/></svg>',
        // ── v0.3 additions — mirror _ICON_META in _static/__init__.py ──────────
        newChat:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>',
        exportTxt:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        copyAns:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
        privacy:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        // ── Listen / Text-to-Speech ───────────────────────────────────────────
        // Speaker-wave icon used for TTS "Listen" button in the action row.
        // Three arc lines indicate audio output (commonly used for "speaker" or
        // "volume" across all major design systems — Feather, Heroicons, Lucide).
        listen:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>',
        // Same icon in "stop" state (filled polygon → playing indicator).
        listenStop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="currentColor" opacity="0.2"/><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="16" y1="8" x2="22" y2="8"/><line x1="16" y1="12" x2="22" y2="12"/><line x1="16" y1="16" x2="22" y2="16"/></svg>',
        // Vertical three-dot "more" icon for the expandable action-row submenu.
        moreVert: '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="5"  r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="19" r="1.5"/></svg>',
        searchAI: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><path d="M8 11h6M11 8v6" stroke-width="1.5"/></svg>',
        keyboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M8 14h8"/></svg>',
        retry:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>',
        // Share-up icon: tray with arrow emerging upward — the universal
        // "share" symbol on iOS / macOS / Android. Used for per-answer sharing.
        shareAns: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>',
        // ── Phase B additions — mirror _ICON_META in _static/__init__.py ──
        model:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="2" x2="9" y2="4"/><line x1="15" y1="2" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="22"/><line x1="15" y1="20" x2="15" y2="22"/><line x1="2" y1="9" x2="4" y2="9"/><line x1="2" y1="15" x2="4" y2="15"/><line x1="20" y1="9" x2="22" y2="9"/><line x1="20" y1="15" x2="22" y2="15"/></svg>',
        terms:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="15" y2="17"/></svg>',
        share:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>',
        menu:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>',
        info:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        chevronDown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>',
        // ── UI-improvement additions ──────────────────────────────────────────
        plus:        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>',
        overflowH:   '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/></svg>',
        // ── Project links additions ───────────────────────────────────────────
        // GitHub mark (official path — monochromatic, works on any background).
        github:      '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>',
        // Globe / world icon — website / documentation home.
        globe:       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        // External-link arrow — shown inside link cards as a launch indicator.
        externalLink:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>',
    };

    // ── Provider accent colours (mirrors _PROVIDER_COLORS in __init__.py) ──────
    //
    // These are merged with any cfg.providerColors injected by the Python side
    // so either side can extend the map without the other breaking.  The JS
    // defaults are the authoritative fallback when the page config is absent
    // (e.g. a CDN-stripped inline script).
    //
    // Sentinel: providers not in the map render with CSS --ai-badge-default.
    var _PROVIDER_COLORS_JS = {
        anthropic:   '#c96442',
        openai:      '#74aa9c',
        google:      '#4285f4',
        mistral:     '#ff7000',
        deepseek:    '#4d6bfe',
        huggingface: '#ff9d00',
        ollama:      '#222222',
        groq:        '#f55036',
        cerebras:    '#8c52ff',
        together:    '#4b5563',
        fireworks:   '#ef4444',
        sambanova:   '#e95b2e',
        cloudflare:  '#f38020',
        perplexity:  '#20b2aa',
        azure_openai:'#0078d4',
    };

    /**
     * Return the hex accent colour for *provider*, merging cfg overrides.
     * @param {string} provider
     * @returns {string}  hex colour, or '' when unknown.
     */
    function _providerColor(provider) {
        if (!provider) return '';
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var merged = Object.assign({}, _PROVIDER_COLORS_JS, cfg.providerColors || {});
        return merged[provider] || '';
    }

    /**
     * Validate a URL for safe use as an ``href`` attribute value.
     *
     * Accepts: ``https://``, ``http://``, and root-relative paths (``/path``).
     * Rejects: ``javascript:``, ``data:``, ``vbscript:``, protocol-relative
     * (``//``), and any other scheme.
     *
     * Defense-in-depth: the Python side filters ``info_url`` before embedding it
     * in the page config.  This guard covers the case where
     * ``window.AI_ASSISTANT_CONFIG`` is injected at CDN/embed level — bypassing
     * the Python filter — or is tampered with post-load.
     *
     * Parameters
     * ----------
     * url : any
     *     Candidate URL value to validate.
     *
     * Returns
     * -------
     * boolean
     *     ``true`` when the URL is safe to assign to ``element.href``.
     */
    function _isSafeHref(url) {
        if (typeof url !== 'string' || !url) return false;
        // Check the scheme prefix (up to first 10 chars covers all dangerous
        // schemes including 'javascript:' at 11 chars — lowercase the slice only).
        var prefix = url.slice(0, 11).toLowerCase();
        if (/^javascript:/i.test(prefix)) return false;
        if (/^data:/i.test(prefix))       return false;
        if (/^vbscript:/i.test(prefix))   return false;
        // Reject protocol-relative URLs (//example.com) — scheme is inherited
        // from the page and may be unexpected.
        if (/^\/\//.test(url))             return false;
        // Accept https, http, and root-relative paths (/path/...).
        return /^https?:\/\//i.test(url) || /^\/[^/]/.test(url) || url === '/';
    }

    /**
     * Parse *val* as a bounded integer, returning *fallback* when the value is
     * absent, non-numeric, non-finite, or outside [min, max].  Never throws.
     *
     * Parameters
     * ----------
     * val : any
     *     Candidate value — typically from ``window.AI_ASSISTANT_CONFIG``.
     * min : number
     *     Inclusive lower bound.
     * max : number
     *     Inclusive upper bound.
     * fallback : number
     *     Returned when *val* fails validation.
     *
     * Returns
     * -------
     * number
     *     A finite integer inside [min, max].
     */
    function _safeInt(val, min, max, fallback) {
        var n = parseInt(val, 10);
        if (!isFinite(n) || n < min || n > max) return fallback;
        return n;
    }

    // ── Touch utilities ───────────────────────────────────────────────────────
    //
    // Three pure helpers that power the haptic feedback system:
    //
    //   _isTouchDevice()               — detects touch-primary input.
    //   _hapticFeedback(pattern)       — fires the Web Vibration API safely.
    //   _attachLongPress(el, …)        — distinguishes short-tap from long-press
    //                                    using Pointer Events, with ghost-click
    //                                    prevention and per-gesture haptics.
    //
    // Integration points (where _hapticFeedback / _attachLongPress are wired):
    //   • All header icon buttons  (close / minimize / maximize / new-chat / export)
    //   • Keyboard-shortcut hint   (subbar left cluster)
    //   • Hamburger kbdRow         (hamburger menu minimize row)
    //   • Hamburger button + right-overflow button
    //   • Speak-with-assistant banner
    //   • Attach button
    //   • Mic button               (pointerdown in hold mode; click in toggle mode)
    //   • Send button
    //   • Floating trigger pill    (tap = restore; long-press = close fully)

    /**
     * Detect whether the primary input device is a touchscreen.
     *
     * Two independent checks are combined:
     *
     *   1. ``navigator.maxTouchPoints > 0`` — hardware capability register
     *      (Chrome, Firefox, Edge, Safari 13+; reliable on physical devices).
     *   2. ``matchMedia('(pointer: coarse)')`` — CSS Level 4 interaction media
     *      feature; true when the primary pointer is imprecise (finger/stylus).
     *
     * Combining both guards against:
     *   • Windows machines with a touch digitizer but mouse attached
     *     (maxTouchPoints > 0, pointer: fine → returns false correctly).
     *   • Browsers that implement one API but not the other.
     *
     * Returns
     * -------
     * boolean
     *     ``true`` when a touch-primary device is detected.
     *
     * Notes
     * -----
     * Developer: Result is evaluated fresh on every call (not cached) to
     *   stay accurate across tests and unusual hybrid-input environments.
     *   Cache the result at the call site when calling in a hot path.
     * User: Detection is best-effort; pointer emulators and hybrid
     *   touch/mouse devices may occasionally return false positives.
     */
    function _isTouchDevice() {
        return (
            navigator.maxTouchPoints > 0 ||
            (window.matchMedia && window.matchMedia('(pointer: coarse)').matches)
        );
    }

    /**
     * Fire a haptic vibration pattern via the Web Vibration API.
     *
     * Silently no-ops when any of the following conditions are true:
     *   – ``navigator.vibrate`` is absent (iOS Safari, all desktop browsers).
     *   – ``prefers-reduced-motion: reduce`` is set — mirrors the CSS contract
     *     used to suppress animations throughout this file.
     *   – The API throws for any reason (iframe sandboxing, document hidden,
     *     OS vibration override, quota, unknown browser quirk).
     *
     * Parameters
     * ----------
     * pattern : number | number[]
     *     Vibration duration in milliseconds, or an alternating
     *     ``[vibrate, pause, vibrate, …]`` array passed directly to
     *     ``navigator.vibrate()``.  Longer values produce stronger haptic
     *     feedback on most hardware.  Recommended constants:
     *
     *         ``[8]``           — light confirmatory tap (button press).
     *         ``[12]``          — medium tap (start / stop recording).
     *         ``[12, 40, 12]``  — double-tap pulse (long-press confirm).
     *
     * Returns
     * -------
     * void
     *
     * Notes
     * -----
     * Developer: The Vibration API requires an active user-gesture in the
     *   call stack.  Always invoke from a synchronous ``click`` or
     *   ``pointerdown`` handler — never from a ``setTimeout``, ``Promise``
     *   resolution, or async callback.
     * User: Vibration respects the OS "do not disturb" / vibration switch;
     *   the browser cannot override a hardware-level silence setting.
     *
     * References
     * ----------
     * https://developer.mozilla.org/en-US/docs/Web/API/Navigator/vibrate
     */
    function _hapticFeedback(pattern) {
        if (!navigator.vibrate) return;
        try {
            if (window.matchMedia &&
                    window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                return;
            }
            navigator.vibrate(pattern);
        } catch (_) {}
    }

    /**
     * Attach long-press and short-tap pointer handlers to an element.
     *
     * Uses the W3C Pointer Events API (not Touch Events) so the same code
     * path handles mouse, stylus, and touch without user-agent sniffing.
     *
     * Interaction model
     * -----------------
     * ``pointerdown`` → start hold timer (``opts.threshold`` ms)
     *
     *   ├─ ``pointerup`` before timer fires
     *   │     → short tap: ``hapticTap`` + ``onShortTap(e)``
     *   │
     *   ├─ ``pointermove`` > ``opts.maxMovePx`` CSS px
     *   │     → cancel timer (scroll-intent detected); no action fires
     *   │
     *   ├─ ``pointercancel``
     *   │     → cancel timer; no action
     *   │
     *   └─ timer fires (hold complete)
     *         → long press: ``hapticLongPress`` + ``onLongPress(e)``
     *         → on touch devices: ghost-click absorber set for 400 ms
     *
     * Ghost-click prevention
     * ----------------------
     * On touch devices the browser synthesises a ``click`` event ≈300 ms
     * after ``pointerup`` even when ``touch-action: manipulation`` is set
     * (manipulation only removes the double-tap-zoom delay, not the
     * synthetic click).  After a long-press fires, a one-shot capture-phase
     * ``click`` listener on ``document`` absorbs that synthetic click so
     * existing ``click`` handlers (e.g. the trigger-pill restore listener)
     * cannot accidentally re-trigger after the long-press action completes.
     *
     * Parameters
     * ----------
     * element : HTMLElement
     *     Target element.
     * onShortTap : function | null
     *     Called on short tap after haptic fires.  Pass ``null`` when an
     *     existing ``click`` listener already handles the primary action so
     *     the two do not double-invoke the same logic.
     * onLongPress : function | null
     *     Called after the hold threshold.  Pass ``null`` when only haptic
     *     feedback is needed for the long-press gesture.
     * opts : object, optional
     *     threshold       : number          — hold duration in ms (default 500).
     *     hapticTap       : number|number[]|null
     *                         — vibration for short tap (default ``[8]``).
     *                           Pass ``null`` to suppress (use when a
     *                           ``pointerdown`` listener already fires haptic).
     *     hapticLongPress : number|number[] — vibration for long press
     *                                          (default ``[12, 40, 12]``).
     *     maxMovePx       : number          — drift-cancel threshold in CSS px
     *                                          (default 8).
     *
     * Returns
     * -------
     * function
     *     Cleanup function.  Removes all attached listeners and clears any
     *     pending timer.  Call when the element leaves the DOM to prevent
     *     ghost listeners accumulating across panel rebuilds.
     *
     * Notes
     * -----
     * Developer: Do NOT attach to the mic button — it owns its own dedicated
     *   ``pointerdown``/``pointerup`` contract for hold-to-record.  Add haptic
     *   calls inline inside those handlers instead (see mic section below).
     * Developer: The cleanup return value mirrors the pattern used by
     *   ``_attachResizer`` so teardown is consistent across utilities.
     * User: Moving the pointer/finger more than ``opts.maxMovePx`` CSS px
     *   during the hold cancels the long-press timer (scroll-intent detection)
     *   without suppressing the normal short-tap action from any click handler.
     *
     * Examples
     * --------
     * >>> // Trigger pill: tap = restore panel (existing click handler),
     * >>> //               long-press = close panel fully.
     * >>> _attachLongPress(triggerEl, null, function () { closeAIPanel(); }, {
     * ...     hapticTap:       null,          // pointerdown listener handles it
     * ...     hapticLongPress: [12, 40, 12],
     * ... });
     */
    function _attachLongPress(element, onShortTap, onLongPress, opts) {
        var threshold = (opts && opts.threshold       != null) ? opts.threshold       : 500;
        var hapticTap = (opts && opts.hapticTap       !== undefined)
                            ? opts.hapticTap : [8];
        var hapticLP  = (opts && opts.hapticLongPress != null) ? opts.hapticLongPress : [12, 40, 12];
        var maxMovePx = (opts && opts.maxMovePx       != null) ? opts.maxMovePx       : 8;

        var _timer         = null;
        var _startX        = 0;
        var _startY        = 0;
        var _longFired     = false;
        var _active        = false;
        var _captureHandle = null;

        function _cancelTimer() {
            if (_timer) { clearTimeout(_timer); _timer = null; }
            _longFired = false;
            _active    = false;
        }

        function _onPointerDown(e) {
            // Left-button / touch only — ignore secondary buttons (right-click).
            if (e.button != null && e.button !== 0) return;
            _active    = true;
            _longFired = false;
            _startX    = e.clientX;
            _startY    = e.clientY;
            // Capture the pointer stream so pointermove / pointerup always reach
            // this element even if the finger drifts off it.
            try { element.setPointerCapture(e.pointerId); } catch (_) {}

            _timer = setTimeout(function () {
                if (!_active) return;
                _longFired = true;
                _active    = false;
                _hapticFeedback(hapticLP);
                if (typeof onLongPress === 'function') onLongPress(e);

                // Absorb the synthetic click that fires ≈300 ms after pointerup
                // on touch devices so no existing click handler re-triggers.
                if (_isTouchDevice()) {
                    _captureHandle = function (ev) {
                        ev.stopPropagation();
                        ev.preventDefault();
                    };
                    document.addEventListener('click', _captureHandle, true);
                    setTimeout(function () {
                        if (_captureHandle) {
                            document.removeEventListener('click', _captureHandle, true);
                            _captureHandle = null;
                        }
                    }, 400);
                }
            }, threshold);
        }

        function _onPointerMove(e) {
            if (!_active) return;
            var dx = e.clientX - _startX;
            var dy = e.clientY - _startY;
            // Cancel the long-press if the pointer drifts (scroll intent).
            if (Math.sqrt(dx * dx + dy * dy) > maxMovePx) {
                _cancelTimer();
            }
        }

        function _onPointerUp() {
            // If the timer already fired (long press) or was cancelled (move),
            // both _active and !_longFired would be false — nothing to do.
            if (!_active && !_longFired) return;
            var wasActive = _active;
            _cancelTimer();   // clears timer; resets _longFired, _active
            if (wasActive) {
                // Short tap: long-press timer never fired.
                if (hapticTap != null) _hapticFeedback(hapticTap);
                if (typeof onShortTap === 'function') onShortTap();
            }
        }

        function _onPointerCancel() {
            _cancelTimer();
        }

        element.addEventListener('pointerdown',   _onPointerDown);
        element.addEventListener('pointermove',   _onPointerMove);
        element.addEventListener('pointerup',     _onPointerUp);
        element.addEventListener('pointercancel', _onPointerCancel);

        // Return a cleanup handle so the caller can remove listeners if the
        // element is ever detached (prevents ghost-listener accumulation).
        return function cleanup() {
            _cancelTimer();
            if (_captureHandle) {
                document.removeEventListener('click', _captureHandle, true);
                _captureHandle = null;
            }
            element.removeEventListener('pointerdown',   _onPointerDown);
            element.removeEventListener('pointermove',   _onPointerMove);
            element.removeEventListener('pointerup',     _onPointerUp);
            element.removeEventListener('pointercancel', _onPointerCancel);
        };
    }

    // ── Lightweight Markdown → safe HTML renderer ─────────────────────────────
    //
    // Renders a strict subset of Markdown that AI assistants commonly emit:
    //   • Fenced code blocks (``` ... ```)
    //   • Inline code (`...`)
    //   • Bold (**text**)
    //   • Italic (*text*)
    //   • Links [label](url) — http/https only, noopener
    //   • Unordered lists (- item / * item)
    //   • Ordered lists (1. item)
    //   • Headers (# / ## / ###)
    //   • Horizontal rules (---)
    //   • Paragraphs (blank-line separation)
    //
    // Security contract:
    //   All text nodes are escaped through _escapeHtml BEFORE pattern matching.
    //   Pattern matches only produce known-safe HTML tags.
    //   Link URLs are validated — only http/https accepted, everything else
    //   is rendered as plain text.
    //
    // @param {string} text  Raw markdown string from the AI model.
    // @returns {string}     Safe HTML string for innerHTML assignment on a
    //                       trusted wrapper element (never on user-controlled
    //                       target attributes).
    function _mdToHtml(text) {
        if (!text) return '';

        // ── 1. Extract fenced code blocks → placeholders ──────────────────
        // Must happen first so inner backtick/asterisk patterns are not
        // processed by the inline rules below.
        var codeBlocks = [];
        var result = text.replace(/```(\w*)\n?([\s\S]*?)```/g, function (_, lang, code) {
            var idx = codeBlocks.length;
            codeBlocks.push({ lang: lang || '', code: code });
            return '\x00CB' + idx + '\x00';   // null-byte placeholder (safe)
        });

        // ── 2. Escape all remaining text (prevents HTML injection) ────────
        result = _escapeHtml(result);

        // ── 3. Inline code (after escaping so & < > inside are safe) ─────
        result = result.replace(/`([^`]+)`/g, '<code class="ai-md-inline-code">$1</code>');

        // ── 4. Headers ────────────────────────────────────────────────────
        result = result.replace(/^### (.+)$/gm, '<h3 class="ai-md-h">$1</h3>');
        result = result.replace(/^## (.+)$/gm,  '<h2 class="ai-md-h">$1</h2>');
        result = result.replace(/^# (.+)$/gm,   '<h1 class="ai-md-h">$1</h1>');

        // ── 5. Horizontal rules ───────────────────────────────────────────
        result = result.replace(/^---+$/gm, '<hr class="ai-md-hr">');

        // ── 6. Bold / italic ─────────────────────────────────────────────
        result = result.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        result = result.replace(/\*([^*]+)\*/g,     '<em>$1</em>');

        // ── 7. Links (http / https only) ──────────────────────────────────
        // Using a function replacement so we can validate the URL scheme
        // before emitting an <a> tag.  Everything else becomes plain text.
        result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, lbl, url) {
            // _escapeHtml ran on url already (step 2); decode &amp; back for
            // scheme check, then re-escape for the attribute value.
            var rawUrl = url.replace(/&amp;/g, '&');
            if (!/^https?:\/\//i.test(rawUrl)) return lbl; // strip unsafe link
            return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' +
                lbl + '</a>';
        });

        // ── 8. Lists ──────────────────────────────────────────────────────
        // Process contiguous bullet-list blocks as a unit so a single <ul>
        // wraps all items.  Same for ordered lists.
        result = result.replace(/((?:^[ \t]*[-*][ \t].+\n?)+)/gm, function (block) {
            var items = block.trim().split('\n').map(function (l) {
                return '<li>' + l.replace(/^[ \t]*[-*][ \t]/, '').trim() + '</li>';
            }).join('');
            return '<ul class="ai-md-ul">' + items + '</ul>';
        });
        result = result.replace(/((?:^[ \t]*\d+\.[ \t].+\n?)+)/gm, function (block) {
            var items = block.trim().split('\n').map(function (l) {
                return '<li>' + l.replace(/^[ \t]*\d+\.[ \t]/, '').trim() + '</li>';
            }).join('');
            return '<ol class="ai-md-ol">' + items + '</ol>';
        });

        // ── 9. Paragraphs ─────────────────────────────────────────────────
        // Blank lines → paragraph breaks.  Single newlines → <br> inside
        // a paragraph.  HTML block elements inserted above are left intact.
        var paras = result.split(/\n{2,}/);
        result = paras.map(function (p) {
            p = p.trim();
            if (!p) return '';
            // Already a block element — don't wrap in <p>
            if (/^<(?:ul|ol|h[1-3]|hr|pre)/i.test(p)) return p;
            // Placeholder line — restore below
            if (/^\x00CB/.test(p)) return p;
            return '<p class="ai-md-p">' + p.replace(/\n/g, '<br>') + '</p>';
        }).join('\n');

        // ── 10. Restore fenced code blocks ────────────────────────────────
        result = result.replace(/\x00CB(\d+)\x00/g, function (_, idx) {
            var cb = codeBlocks[+idx];
            var langAttr = cb.lang ? ' data-lang="' + _escapeHtml(cb.lang) + '"' : '';
            var langBadge = cb.lang
                ? '<span class="ai-md-code-lang">' + _escapeHtml(cb.lang) + '</span>'
                : '';
            // The code content was NOT escaped in step 2 (it was extracted
            // before escaping).  We escape it now.
            return '<pre class="ai-md-pre"' + langAttr + '>' +
                langBadge +
                '<code>' + _escapeHtml(cb.code) + '</code></pre>';
        });

        return result;
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    function loadTurndown(callback) {
        if (typeof TurndownService !== 'undefined') { callback(); return; }
        var script = document.createElement('script');
        // Primary CDN
        script.src = 'https://cdn.jsdelivr.net/npm/turndown@7.1.2/dist/turndown.min.js';
        script.onload  = callback;
        script.onerror = function () {
            console.warn('AI Assistant: Primary CDN failed, attempting fallback...');
            // Fallback to another CDN or a local path
            var fallback = document.createElement('script');
            fallback.src = 'https://unpkg.com/turndown@7.1.2/dist/turndown.js';
            fallback.onload = callback;
            fallback.onerror = function() { console.error('AI Assistant: Critical failure loading Turndown'); };
            document.head.appendChild(fallback);
        };
        document.head.appendChild(script);
    }

    function initAIAssistant() {
        loadTurndown(function () { createAIAssistantUI(); });
    }

    // ── DOM construction ──────────────────────────────────────────────────────

    function createAIAssistantUI() {
        var container = createContainer();
        var button    = createButton();
        var dropdown  = createDropdown();

        container.appendChild(button);
        container.appendChild(dropdown);

        var position = (window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.position) || 'sidebar';
        insertContainer(container, position);
        setupEventListeners(button, dropdown);

        // v0.3: only wire panel-dependent extras when the AI panel feature
        // is actually enabled (respects the FEATURE_DEFAULTS contract).
        var cfg      = window.AI_ASSISTANT_CONFIG || {};
        var features = Object.assign({}, FEATURE_DEFAULTS, cfg.features || {});
        if (features.ai_panel) {
            _bindShortcut();    // R7 — no-op if disabled/invalid in config
            _mountSearchBar();  // R8 — no-op unless explicitly enabled

            // panelStartMinimized (default true): eagerly create the floating
            // trigger pill so users get 1-click access to the panel on every
            // page load — without needing to open the panel first.
            // When false, the pill is created lazily inside createAIPanel()
            // and only becomes visible after the user minimizes the panel.
            if (cfg.panelStartMinimized !== false) {
                if (!_aiTriggerEl) {
                    var title = cfg.panelTitle || 'AI Assistant';
                    _aiTriggerEl = _createTriggerPill(title);
                    // Mirror what minimizeAIPanel() does: mark as minimized
                    // and make it visible so CSS rules apply correctly.
                    _aiTriggerEl.setAttribute('data-minimized', 'true');
                    _aiTriggerEl.style.display = 'flex';
                    document.body.appendChild(_aiTriggerEl);
                }
            }
        }
    }

    function createContainer() {
        var el = document.createElement('div');
        el.className = 'ai-assistant-container';
        el.id = 'ai-assistant-container';
        return el;
    }

    function createButton() {
        var container = document.createElement('div');
        container.className = 'ai-assistant-button';
        container.id = 'ai-assistant-button';

        var staticPath = getStaticPath();

        var mainBtn = document.createElement('button');
        mainBtn.className = 'ai-assistant-button-main';
        mainBtn.id = 'ai-assistant-button-main';
        mainBtn.type = 'button';
        mainBtn.setAttribute('aria-label', 'Copy page as Markdown');
        mainBtn.title = 'Copy this page as Markdown for AI / LLMs';

        var mainIcon = document.createElement('img');
        mainIcon.src = staticPath + '/copy-to-clipboard.svg';
        mainIcon.className = 'ai-assistant-icon';
        mainIcon.setAttribute('aria-hidden', 'true');
        mainIcon.alt = '';

        var mainText = document.createElement('span');
        mainText.className = 'ai-assistant-button-text';
        mainText.textContent = 'Copy page';

        mainBtn.appendChild(mainIcon);
        mainBtn.appendChild(mainText);

        var divider = document.createElement('span');
        divider.className = 'ai-assistant-button-divider';

        var dropBtn = document.createElement('button');
        dropBtn.className = 'ai-assistant-button-dropdown';
        dropBtn.id = 'ai-assistant-button-dropdown';
        dropBtn.type = 'button';
        dropBtn.setAttribute('aria-label', 'More options');
        dropBtn.setAttribute('aria-expanded', 'false');
        dropBtn.setAttribute('aria-haspopup', 'true');

        var dropIcon = document.createElement('img');
        dropIcon.src = staticPath + '/arrow-down.svg';
        dropIcon.className = 'ai-assistant-dropdown-icon';
        dropIcon.setAttribute('aria-hidden', 'true');
        dropIcon.alt = '';

        dropBtn.appendChild(dropIcon);

        container.appendChild(mainBtn);
        container.appendChild(divider);
        container.appendChild(dropBtn);

        return container;
    }

    function createDropdown() {
        var dropdown = document.createElement('div');
        dropdown.className = 'ai-assistant-dropdown';
        dropdown.id = 'ai-assistant-dropdown';
        dropdown.setAttribute('role', 'menu');
        // dropdown.style.display = 'none';

        var cfg        = window.AI_ASSISTANT_CONFIG || {};
        var features   = Object.assign({}, FEATURE_DEFAULTS, cfg.features || {});
        var staticPath = getStaticPath();
        var hasItems   = false;

        // 1. Markdown export
        if (features.markdown_export) {
            dropdown.appendChild(createMenuItem('copy-markdown', 'Copy page', 'Copy this page as Markdown for LLMs.', staticPath + '/copy-to-clipboard.svg'));
            hasItems = true;
        }

        if (features.view_markdown) {
            dropdown.appendChild(createMenuItem('view-markdown', 'View as Markdown', 'View this page as Markdown.', staticPath + '/markdown.svg'));
            hasItems = true;
        }

        // 2. AI chat
        if (features.ai_chat) {
            var providers = cfg.providers || {};
            var enabledProviders = Object.entries(providers).filter(function (kv) { return kv[1].enabled; });
            if (enabledProviders.length > 0) {
                if (hasItems) dropdown.appendChild(createSeparator());
                enabledProviders.forEach(function (kv) {
                    var key  = kv[0], provider = kv[1];
                    var icon = provider.icon || 'comment-discussion.svg';
                    var iconPath = _ABSOLUTE_ICON_RE.test(icon) ? icon : (staticPath + '/' + icon);
                    var item = createMenuItem('ai-chat-' + key, provider.label, provider.description || 'Open AI chat with this page context.', iconPath);
                    item.dataset.provider = key;
                    dropdown.appendChild(item);
                });
                hasItems = true;
            }
        }

        // 3. MCP integration
        if (features.mcp_integration) {
            var mcpTools = cfg.mcp_tools || {};
            var enabledTools = Object.entries(mcpTools).filter(function (kv) { return kv[1].enabled; });
            if (enabledTools.length > 0) {
                if (hasItems) dropdown.appendChild(createSeparator());
                enabledTools.forEach(function (kv) {
                    var key = kv[0], tool = kv[1];
                    var icon = tool.icon || 'ai-tools.svg';
                    var iconPath = _ABSOLUTE_ICON_RE.test(icon) ? icon : (staticPath + '/' + icon);
                    var item = createMenuItem('mcp-' + key, tool.label, tool.description || 'Install MCP server.', iconPath);
                    item.dataset.mcpTool = key;
                    dropdown.appendChild(item);
                });
                hasItems = true;
            }
        }

        // 4. PDF export
        if (features.pdf_export) {
            if (hasItems) dropdown.appendChild(createSeparator());
            dropdown.appendChild(createPdfSection(staticPath, cfg));
            hasItems = true;
        }

        // 5. AI panel
        if (features.ai_panel) {
            var panelTitle = cfg.panelTitle || 'AI Assistant';
            if (hasItems) dropdown.appendChild(createSeparator());
            var panelItem = createMenuItem('ai-panel-open', panelTitle, 'Ask ' + panelTitle + ' about this page', staticPath + '/ai-panel.svg');
            dropdown.appendChild(panelItem);
        }

        return dropdown;
    }

    // ── PDF section ───────────────────────────────────────────────────────────

    function createPdfSection(staticPath, cfg) {
        cfg = cfg || {};
        var pdfUrl     = (cfg.pdfExportUrl || '').trim();
        var showToggle = cfg.pdfUrlModeToggle !== false;

        var savedMode = null;
        try { savedMode = sessionStorage.getItem(_PDF_MODE_KEY); } catch (_e) {}

        var initialMode = (savedMode === 'url' || savedMode === 'print')
            ? savedMode
            : (pdfUrl ? 'url' : 'print');

        var section = document.createElement('div');
        section.className = 'ai-assistant-pdf-section';

        var btn = document.createElement('button');
        btn.className = 'ai-assistant-menu-item ai-assistant-pdf-btn';
        btn.id = 'ai-assistant-pdf-export';
        btn.type = 'button';
        btn.setAttribute('role', 'menuitem');

        var btnContent = document.createElement('div');
        btnContent.className = 'ai-assistant-menu-item-content ai-assistant-pdf-content';

        var btnTitle = document.createElement('div');
        btnTitle.className = 'ai-assistant-menu-item-title';

        var pdfIcon = document.createElement('img');
        pdfIcon.src = staticPath + '/file-pdf.svg';
        pdfIcon.className = 'ai-assistant-menu-icon';
        pdfIcon.setAttribute('aria-hidden', 'true');
        pdfIcon.alt = '';

        var pdfLabel = document.createElement('span');
        pdfLabel.textContent = 'Export as PDF';

        btnTitle.appendChild(pdfIcon);
        btnTitle.appendChild(pdfLabel);

        var btnDesc = document.createElement('div');
        btnDesc.className = 'ai-assistant-menu-item-description ai-assistant-pdf-desc';
        btnDesc.id = 'ai-assistant-pdf-desc';
        btnDesc.textContent = _pdfModeDescription(initialMode, pdfUrl);

        btnContent.appendChild(btnTitle);
        btnContent.appendChild(btnDesc);
        btn.appendChild(btnContent);
        section.appendChild(btn);

        if (showToggle) {
            var toggleRow = document.createElement('div');
            toggleRow.className = 'ai-assistant-pdf-toggle';
            toggleRow.id = 'ai-assistant-pdf-toggle';
            toggleRow.setAttribute('role', 'group');
            toggleRow.setAttribute('aria-label', 'PDF export mode');

            var toggleLabel = document.createElement('span');
            toggleLabel.className = 'ai-assistant-pdf-toggle-label';
            toggleLabel.textContent = 'Mode:';

            var urlBtn = document.createElement('button');
            urlBtn.className = 'ai-assistant-pdf-mode-btn' + (initialMode === 'url' ? ' active' : '');
            urlBtn.id = 'ai-assistant-pdf-mode-url';
            urlBtn.type = 'button';
            urlBtn.textContent = 'URL';
            urlBtn.title = 'Open PDF URL in new tab';
            if (!pdfUrl) urlBtn.disabled = true;

            var printBtn = document.createElement('button');
            printBtn.className = 'ai-assistant-pdf-mode-btn' + (initialMode === 'print' ? ' active' : '');
            printBtn.id = 'ai-assistant-pdf-mode-print';
            printBtn.type = 'button';
            printBtn.textContent = 'Print';
            printBtn.title = 'Use browser print dialog (Save as PDF)';

            urlBtn.addEventListener('click', function (e) { e.stopPropagation(); if (!urlBtn.disabled) _setPdfMode('url'); });
            printBtn.addEventListener('click', function (e) { e.stopPropagation(); _setPdfMode('print'); });

            toggleRow.appendChild(toggleLabel);
            toggleRow.appendChild(urlBtn);
            toggleRow.appendChild(printBtn);
            section.appendChild(toggleRow);
        }

        return section;
    }

    function _pdfModeDescription(mode, pdfUrl) {
        return (mode === 'url' && pdfUrl) ? 'Opens PDF in a new tab.' : 'Save as PDF via browser print dialog.';
    }

    function _setPdfMode(mode) {
        try { sessionStorage.setItem(_PDF_MODE_KEY, mode); } catch (_e) {}
        var urlBtn   = document.getElementById('ai-assistant-pdf-mode-url');
        var printBtn = document.getElementById('ai-assistant-pdf-mode-print');
        var descEl   = document.getElementById('ai-assistant-pdf-desc');
        var pdfUrl   = ((window.AI_ASSISTANT_CONFIG || {}).pdfExportUrl || '').trim();
        if (urlBtn)   urlBtn.classList.toggle('active',   mode === 'url');
        if (printBtn) printBtn.classList.toggle('active', mode === 'print');
        if (descEl)   descEl.textContent = _pdfModeDescription(mode, pdfUrl);
    }

    function _getPdfMode() {
        var pdfUrl = ((window.AI_ASSISTANT_CONFIG || {}).pdfExportUrl || '').trim();
        try {
            var saved = sessionStorage.getItem(_PDF_MODE_KEY);
            if (saved === 'url' || saved === 'print') return saved;
        } catch (_e) {}
        return pdfUrl ? 'url' : 'print';
    }

    // ── Menu helpers ──────────────────────────────────────────────────────────

    function createMenuItem(id, text, desc, iconSrc) {
        var item = document.createElement('button');
        item.className = 'ai-assistant-menu-item';
        item.id = 'ai-assistant-' + id;
        item.type = 'button';
        item.setAttribute('role', 'menuitem');

        var content  = document.createElement('div');
        content.className = 'ai-assistant-menu-item-content';

        var titleRow = document.createElement('div');
        titleRow.className = 'ai-assistant-menu-item-title';

        var icon = document.createElement('img');
        icon.src = iconSrc;
        icon.className = 'ai-assistant-menu-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.alt = '';

        var label = document.createElement('span');
        label.textContent = text;

        titleRow.appendChild(icon);
        titleRow.appendChild(label);

        var descEl = document.createElement('div');
        descEl.className = 'ai-assistant-menu-item-description';
        descEl.textContent = desc;

        content.appendChild(titleRow);
        content.appendChild(descEl);
        item.appendChild(content);

        return item;
    }

    function createSeparator() {
        var sep = document.createElement('div');
        sep.className = 'ai-assistant-menu-separator';
        return sep;
    }

    // ── Static path detection ─────────────────────────────────────────────────

    function getStaticPath() {
        if (_selfSrc && _selfSrc.indexOf('_static') !== -1) {
            return _selfSrc.substring(0, _selfSrc.indexOf('_static') + 7);
        }
        var scripts = document.querySelectorAll('script[src]');
        for (var i = 0; i < scripts.length; i++) {
            var src = scripts[i].getAttribute('src') || '';
            if (src.indexOf('_static') !== -1) return src.substring(0, src.indexOf('_static') + 7);
        }
        var links = document.querySelectorAll('link[href]');
        for (var j = 0; j < links.length; j++) {
            var href = links[j].getAttribute('href') || '';
            if (href.indexOf('_static') !== -1) return href.substring(0, href.indexOf('_static') + 7);
        }
        return '_static';
    }

    // ── Widget placement ──────────────────────────────────────────────────────

    function insertContainer(container, position) {
        if (position === 'none') return;

        if (position === 'sidebar') {
            var sidebarSelectors = [
                '.bd-sidebar-secondary', '.bd-toc',
                'aside.toc-sidebar', '.sidebar-secondary',
                '.toc-drawer', 'aside[role="complementary"]',
            ];
            for (var k = 0; k < sidebarSelectors.length; k++) {
                var sidebar = document.querySelector(sidebarSelectors[k]);
                if (sidebar) {
                    console.debug('AI Assistant: Inserting into sidebar:', sidebarSelectors[k]);
                    sidebar.insertBefore(container, sidebar.firstChild);
                    return;
                }
            }
            console.debug('AI Assistant: No sidebar found, falling back to title position');
            insertInTitlePosition(container);
            return;
        }

        if (position === 'title') {
            insertInTitlePosition(container);
            return;
        }

        var article = document.querySelector('article, [role="main"], .document, .body');
        if (article) article.insertBefore(container, article.firstChild);
    }

    function insertInTitlePosition(container) {
        var article = document.querySelector('article, [role="main"]');
        var heading = article ? article.querySelector('h1') : null;

        if (!heading) {
            if (article) article.insertBefore(container, article.firstChild);
            return;
        }

        // Already wrapped — just append button
        if (heading.parentNode && heading.parentNode.dataset.aiAssistantWrapped) {
            heading.parentNode.appendChild(container);
            return;
        }

        var wrapper = document.createElement('div');
        wrapper.className = 'ai-assistant-title-wrapper';
        wrapper.dataset.aiAssistantWrapped = '1';

        // ── Layout: title grows, button stays fixed beside it ──
        wrapper.style.cssText = [
            'display: flex',
            'flex-direction: row',
            'align-items: center',
            'gap: 0.75rem',
            'flex-wrap: nowrap',
            'width: 100%',
            'min-width: 0',          // allow flex children to shrink below content size
        ].join(';');

        // Title must shrink/truncate on overflow, never push button down
        heading.style.cssText += [
            ';flex: 1 1 auto',
            'min-width: 0',          // critical: lets text shrink inside flex
            'margin: 0',
            'overflow-wrap: break-word',
            'word-break: break-word',
        ].join(';');

        // Button stays fixed width, never shrinks
        container.style.cssText += [
            ';flex: 0 0 auto',
            'align-self: center',
        ].join(';');

        heading.parentNode.insertBefore(wrapper, heading);
        wrapper.appendChild(heading);
        wrapper.appendChild(container);
    }

    // ── Event wiring ──────────────────────────────────────────────────────────

    function setupEventListeners(button, dropdown) {
        var mainButton     = document.getElementById('ai-assistant-button-main');
        var dropdownButton = document.getElementById('ai-assistant-button-dropdown');

        if (mainButton) {
            mainButton.addEventListener('click', function (e) {
                e.stopPropagation();
                handleCopyMarkdown(true);
            });
        }

        if (dropdownButton) {
            dropdownButton.addEventListener('click', function (e) {
                e.stopPropagation();
                // var isOpen = dropdown.style.display !== 'block';
                var isOpen = window.getComputedStyle(dropdown).display !== 'none';
                if (isOpen) {
                    dropdown.style.display = 'none';
                    dropdownButton.setAttribute('aria-expanded', 'false');
                } else {
                    // Make visible first so offsetHeight is measurable, then
                    // decide direction — all in the same frame, no flicker.
                    dropdown.style.display = 'block';
                    _repositionDropdown(dropdown);
                    dropdownButton.setAttribute('aria-expanded', 'true');
                }
            });
        }

        if (!_listenersAttached) {
            _listenersAttached = true;
            document.addEventListener('click', function (e) {
                var btn  = document.getElementById('ai-assistant-button');
                var drop = document.getElementById('ai-assistant-dropdown');
                var dropBtn = document.getElementById('ai-assistant-button-dropdown');
                if (btn && drop && !btn.contains(e.target) && !drop.contains(e.target)) {
                    drop.style.display = 'none';
                    if (dropBtn) dropBtn.setAttribute('aria-expanded', 'false');
                }
            });
        }

        if (button) {
            button.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') closeDropdown();
            });
        }

        var copyMarkdownBtn = document.getElementById('ai-assistant-copy-markdown');
        if (copyMarkdownBtn) copyMarkdownBtn.addEventListener('click', function () { handleCopyMarkdown(false); });

        var viewMarkdownBtn = document.getElementById('ai-assistant-view-markdown');
        if (viewMarkdownBtn) viewMarkdownBtn.addEventListener('click', function () { handleViewMarkdown(); });

        var aiChatBtns = dropdown.querySelectorAll('[id^="ai-assistant-ai-chat-"]');
        aiChatBtns.forEach(function (btn) {
            btn.addEventListener('click', function () { handleAIChat(this.dataset.provider); });
        });

        var mcpBtns = dropdown.querySelectorAll('[id^="ai-assistant-mcp-"]');
        mcpBtns.forEach(function (btn) {
            btn.addEventListener('click', function () { handleMCPInstall(this.dataset.mcpTool); });
        });

        var pdfExportBtn = document.getElementById('ai-assistant-pdf-export');
        if (pdfExportBtn) pdfExportBtn.addEventListener('click', function () { handlePdfExport(); });

        var aiPanelOpenBtn = document.getElementById('ai-assistant-ai-panel-open');
        if (aiPanelOpenBtn) {
            aiPanelOpenBtn.addEventListener('click', function () {
                closeDropdown();
                toggleAIPanel();
            });
        }
    }

    // ── Markdown conversion ───────────────────────────────────────────────────

    function convertToMarkdown() {
        var contentSelector = (window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.content_selector) || 'article';
        var content = document.querySelector(contentSelector);

        if (!content) return Promise.reject(new Error('Could not find page content (selector: ' + contentSelector + ')'));

        var cloned = content.cloneNode(true);
        ['.headerlink', '.ai-assistant-container', 'script', 'style', '.sidebar', 'nav'].forEach(function (sel) {
            cloned.querySelectorAll(sel).forEach(function (el) { el.remove(); });
        });

        var ts = new TurndownService({ headingStyle: 'atx', codeBlockStyle: 'fenced', emDelimiter: '*' });
        ts.addRule('preserveCodeBlocks', {
            filter: ['pre'],
            replacement: function (content, node) {
                var code = node.querySelector('code');
                if (code) {
                    var langMatch = code.className.match(/language-(\w+)/);
                    return '\n\n```' + (langMatch ? langMatch[1] : '') + '\n' + code.textContent + '\n```\n\n';
                }
                return '\n\n```\n' + content + '\n```\n\n';
            },
        });

        return Promise.resolve(ts.turndown(cloned.innerHTML));
    }

    function getMarkdownUrl() {
        // Strip query string AND fragment before rewriting the extension.
        // Rationale: window.location.href.split('#')[0] removes fragments but
        // leaves ?query=params.  A URL like "/page.html?v=2" ends in "?v=2",
        // so /\.html$/ would never match — the .md URL would be wrong.
        // Splitting on both '?' and '#' gives the bare path every time.
        var bare = window.location.href.split('?')[0].split('#')[0];
        if (bare.endsWith('.html')) return bare.replace(/\.html$/, '.md');
        if (bare.endsWith('/'))     return bare + 'index.md';
        return bare + '.md';
    }

    // ── Action handlers ───────────────────────────────────────────────────────

    function handleCopyMarkdown(showInlineConfirmation) {
        convertToMarkdown()
            .then(function (markdown) {
                copyToClipboard(markdown, showInlineConfirmation);
                closeDropdown();
            })
            .catch(function (err) {
                console.error('AI Assistant: Failed to convert to Markdown:', err);
                showNotification('Failed to convert page to Markdown.', true);
            });
    }

    function handleViewMarkdown() {
        window.open(getMarkdownUrl(), '_blank', 'noopener,noreferrer');
        closeDropdown();
    }

    function handleAIChat(providerKey) {
        try {
            var providers = ((window.AI_ASSISTANT_CONFIG || {}).providers) || {};
            var provider  = providers[providerKey];
            if (!provider) { showNotification('AI provider "' + providerKey + '" not configured.', true); return; }

            var promptTpl = typeof provider.prompt_template === 'string' ? provider.prompt_template : 'Read this documentation page: {url}';
            var urlTpl    = typeof provider.url_template    === 'string' ? provider.url_template    : null;
            if (!urlTpl)  { showNotification('AI provider "' + providerKey + '" has no url_template.', true); return; }

            var prompt = promptTpl.replace('{url}', getMarkdownUrl());
            var aiUrl  = urlTpl.replace('{prompt}', encodeURIComponent(prompt));

            // Client-side URL-scheme guard (belt-and-suspenders; server validates
            // url_template at build time, but defence-in-depth is cheap here).
            // Only http:// and https:// are safe to window.open; anything else
            // (javascript:, data:, blob:, vbscript:, …) must be rejected.
            if (!/^https?:\/\//i.test(aiUrl)) {
                console.error('AI Assistant: Blocked unsafe URL scheme in provider "' + providerKey + '":', aiUrl.slice(0, 50));
                showNotification('AI provider URL is not a valid HTTP(S) address.', true);
                return;
            }

            window.open(aiUrl, '_blank', 'noopener,noreferrer');
            closeDropdown();
        } catch (err) {
            console.error('AI Assistant: Failed to open AI chat:', err);
            showNotification('Failed to open AI chat. Please try again.', true);
        }
    }

    function handleMCPInstall(toolKey) {
        try {
            var mcpTools = ((window.AI_ASSISTANT_CONFIG || {}).mcp_tools) || {};
            var tool     = mcpTools[toolKey];
            if (!tool)   { showNotification('MCP tool configuration not found.', true); return; }

            if (tool.type === 'claude_desktop') {
                var mcpbUrl = (typeof tool.mcpb_url === 'string') ? tool.mcpb_url.trim() : '';
                // Scheme validation: only mcpb:// or https:// are legitimate download URLs.
                // Reject javascript:, data:, blob:, ftp:, or any other scheme before
                // triggering a file download — prevents malicious package substitution.
                if (!mcpbUrl) {
                    showNotification('MCP tool "' + toolKey + '" has no mcpb_url.', true);
                    return;
                }
                if (!/^(?:mcpb|https):\/\//i.test(mcpbUrl)) {
                    console.error('AI Assistant: Blocked unsafe mcpb_url scheme for tool "' + toolKey + '":', mcpbUrl.slice(0, 60));
                    showNotification('MCP tool download URL must use mcpb:// or https://.', true);
                    return;
                }
                var urlPath;
                try { urlPath = new URL(mcpbUrl).pathname; }
                catch (_urlErr) { showNotification('MCP tool "' + toolKey + '" has an invalid mcpb_url.', true); return; }
                var a = document.createElement('a');
                a.href = mcpbUrl;
                a.download = urlPath.split('/').pop() || (toolKey + '.zip');
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                showNotification('MCP tool download started.');
                closeDropdown();
                return;
            }

            if (tool.type === 'vscode') {
                var mcpCfg = { name: tool.server_name || toolKey, type: tool.transport || 'sse' };
                if (tool.transport === 'stdio') { mcpCfg.command = tool.command; if (tool.args) mcpCfg.args = tool.args; }
                else mcpCfg.url = tool.server_url;

                // vscode:// is a custom URI scheme — window.open(_self) would try to
                // navigate away from the page, which silently fails in most browsers
                // (the page stays, but the user gets no feedback if VS Code is not
                // installed).  Use a hidden <a> click instead: it triggers the OS
                // protocol handler without changing window.location, and is the
                // documented VS Code MCP installation mechanism.
                var vsUrl = 'vscode:mcp/install?' + encodeURIComponent(JSON.stringify(mcpCfg));
                var vsLink = document.createElement('a');
                vsLink.href = vsUrl;
                // rel="noopener" is a no-op on same-frame navigations but harmless;
                // omit target so the protocol handler fires without page navigation.
                document.body.appendChild(vsLink);
                vsLink.click();
                document.body.removeChild(vsLink);
                showNotification('Opening VS Code MCP install…');
                closeDropdown();
                return;
            }

            console.warn('AI Assistant: Unknown MCP tool type:', tool.type);
            showNotification('Unknown MCP tool type: ' + tool.type, true);
        } catch (err) {
            console.error('AI Assistant: Failed to install MCP tool:', err);
            showNotification('Failed to install MCP tool. Please try again.', true);
        }
    }

    function handlePdfExport() {
        var cfg    = window.AI_ASSISTANT_CONFIG || {};
        var pdfUrl = (cfg.pdfExportUrl || '').trim();
        var mode   = _getPdfMode();
        closeDropdown();
        if (mode === 'url' && pdfUrl) window.open(pdfUrl, '_blank', 'noopener,noreferrer');
        else window.print();
    }

    // ── Clipboard ─────────────────────────────────────────────────────────────

    function copyToClipboard(text, showInlineConfirmation) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text)
                .then(function () {
                    showInlineConfirmation ? showInlineSuccessState() : showNotification('Markdown copied to clipboard!');
                })
                .catch(function (err) {
                    console.error('AI Assistant: Clipboard API failed:', err);
                    fallbackCopy(text, showInlineConfirmation);
                });
        } else {
            fallbackCopy(text, showInlineConfirmation);
        }
    }

    function showInlineSuccessState() {
        var mainButton = document.getElementById('ai-assistant-button-main');
        if (!mainButton) return;
        var iconEl   = mainButton.querySelector('.ai-assistant-icon');
        var textSpan = mainButton.querySelector('.ai-assistant-button-text');
        if (!iconEl || !textSpan) return;
        var origSrc  = iconEl.src;
        var origText = textSpan.textContent;
        iconEl.src           = getStaticPath() + '/checked.svg';
        textSpan.textContent = 'Copied';
        mainButton.classList.add('ai-assistant-button-success');
        setTimeout(function () {
            iconEl.src = origSrc;
            textSpan.textContent = origText;
            mainButton.classList.remove('ai-assistant-button-success');
        }, 2000);
    }

    function fallbackCopy(text, showInlineConfirmation) {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity  = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            showInlineConfirmation ? showInlineSuccessState() : showNotification('Markdown copied to clipboard!');
        } catch (err) {
            console.error('AI Assistant: Fallback copy failed:', err);
            showNotification('Failed to copy to clipboard.', true);
        }
        document.body.removeChild(textarea);
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    function closeDropdown() {
        var dropdown   = document.getElementById('ai-assistant-dropdown');
        var dropButton = document.getElementById('ai-assistant-button-dropdown');
        if (dropdown)   dropdown.style.display = 'none';
        if (dropButton) dropButton.setAttribute('aria-expanded', 'false');
    }

    /**
     * Flip the dropdown above the button when there is not enough viewport
     * space below — prevents menu items from being hidden under the secondary
     * sidebar or clipped by the viewport bottom edge.
     *
     * Called immediately after `display:block` so `offsetHeight` is readable.
     * Both directions set explicitly on every open call so repeated opens
     * (resize, scroll) always pick the correct side.
     *
     * Decision rule:
     *   spaceBelow >= dropHeight  → open downward (default, most readable)
     *   spaceBelow <  dropHeight  → open upward if there is more room above,
     *                               otherwise still open downward (best effort)
     *
     * @param {HTMLElement} dropdown  The dropdown panel element.
     */
    function _repositionDropdown(dropdown) {
        var container = dropdown.parentElement;
        if (!container) return;

        var rect       = container.getBoundingClientRect();
        var dropH      = dropdown.offsetHeight;
        var spaceBelow = window.innerHeight - rect.bottom;
        var spaceAbove = rect.top;
        var GAP        = 'calc(100% + 0.2rem)';

        if (spaceBelow >= dropH || spaceBelow >= spaceAbove) {
            // Enough room below — default downward direction.
            dropdown.style.top    = GAP;
            dropdown.style.bottom = 'auto';
        } else {
            // Not enough room below and more room above — flip upward so all
            // items are visible without scrolling or sliding under a sidebar.
            dropdown.style.top    = 'auto';
            dropdown.style.bottom = GAP;
        }
    }

    function showNotification(message, isError) {
        var el = document.createElement('div');
        el.className = 'ai-assistant-notification' + (isError ? ' error' : '');
        el.textContent = message;
        document.body.appendChild(el);
        setTimeout(function () { el.classList.add('show'); }, 10);
        setTimeout(function () {
            el.classList.remove('show');
            setTimeout(function () { el.remove(); }, 300);
        }, 3000);
    }

    function _escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    // ═══════════════════════════════════════════════════════════════════════
    // v0.3 MODULE — conversation model, persistence, resize, shortcuts,
    //               feedback, privacy sheet, AI search-bar.
    //
    // Design: ONE source of truth (`_transcript`).  Clear / export / copy /
    // restore are all pure views over it — no duplicated state.  Every
    // feature is gated by a config flag and every default reproduces the
    // pre-v0.3 behaviour exactly (no surprise for existing users).
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Single source of truth for the conversation.
     * Each entry: { role: 'user'|'assistant'|'error', text: string, ts: number }
     * @type {Array<{role:string,text:string,ts:number}>}
     */
    var _transcript = [];

    /** sessionStorage key for the persisted transcript. */
    var _TRANSCRIPT_KEY = 'ai-assistant-transcript';

    /**
     * Set of answer indices (0-based) for which feedback has been submitted
     * this session.  Replaces the old single boolean so each answer's
     * feedback block is tracked independently — enabling granular per-answer
     * model-training data collection.
     *
     * @type {Set<number>}
     */
    var _feedbackGivenSet = new Set();

    /**
     * Whether transcript persistence is enabled (config-driven, default on).
     * @returns {boolean}
     */
    function _persistEnabled() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        return cfg.panelPersist !== false;   // default true
    }

    /** Safely read sessionStorage (private-mode / disabled storage safe). */
    function _ssGet(key) {
        try { return window.sessionStorage.getItem(key); } catch (_) { return null; }
    }
    function _ssSet(key, val) {
        try { window.sessionStorage.setItem(key, val); } catch (_) { /* ignore */ }
    }
    function _ssDel(key) {
        try { window.sessionStorage.removeItem(key); } catch (_) { /* ignore */ }
    }

    /** Persist `_transcript` if persistence is enabled. */
    function _saveTranscript() {
        if (!_persistEnabled()) return;
        try { _ssSet(_TRANSCRIPT_KEY, JSON.stringify(_transcript)); } catch (_) {}
    }

    /**
     * Load any persisted transcript into `_transcript`.
     * Defensive: any malformed entry is dropped, never thrown.
     */
    function _loadTranscript() {
        if (!_persistEnabled()) return;
        var raw = _ssGet(_TRANSCRIPT_KEY);
        if (!raw) return;
        try {
            var arr = JSON.parse(raw);
            if (Array.isArray(arr)) {
                _transcript = arr.filter(function (e) {
                    return e && typeof e.text === 'string' &&
                        (e.role === 'user' || e.role === 'assistant' || e.role === 'error');
                });
            }
        } catch (_) { _transcript = []; }
    }

    /**
     * Record a message in the single source of truth and persist.
     *
     * Enforces a configurable maximum turn count to prevent unbounded
     * sessionStorage growth and JSON serialisation slowdown.  Oldest turns
     * are evicted from the head when the cap is exceeded.
     *
     * @param {string} role  'user' | 'assistant' | 'error'
     * @param {string} text
     */
    function _recordMessage(role, text) {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var maxTurns = (typeof cfg.panelMaxTranscriptTurns === 'number' &&
                        cfg.panelMaxTranscriptTurns > 0)
            ? Math.floor(cfg.panelMaxTranscriptTurns)
            : _TRANSCRIPT_MAX_TURNS_DEFAULT;

        _transcript.push({ role: role, text: text, ts: Date.now() });

        // Trim head (oldest entries) when cap is exceeded.
        // Removing pairs (user + assistant) keeps conversations coherent, but
        // a simple slice from the left is safe — the welcome screen is not in
        // the transcript array, only actual message turns.
        if (_transcript.length > maxTurns) {
            _transcript = _transcript.slice(_transcript.length - maxTurns);
        }

        _saveTranscript();
    }

    /**
     * R3 — Clear the conversation WITHOUT a page refresh.
     * Resets the single source of truth and rebuilds the body to its
     * initial welcome/suggestions state.
     *
     * Notes
     * -----
     * Developer: The speak banner lives on the panel element (not the body),
     *   so body.innerHTML = '' does NOT remove it.  However _dismissSpeakBanner()
     *   sets banner.style.display = 'none' when the user sends their first message.
     *   That inline style persists across resets, leaving the banner invisible on
     *   a fresh conversation.  We must explicitly restore it here so the banner
     *   reappears every time the user starts a new chat session.
     */
    function clearConversation() {
        _transcript = [];
        _feedbackGivenSet = new Set();
        _ssDel(_TRANSCRIPT_KEY);
        var body = document.getElementById('ai-assistant-panel-body');
        if (!body) return;
        body.innerHTML = '';
        _renderWelcome(body);
        // Restore speak banner — _dismissSpeakBanner() sets inline display:none
        // when the user sends a message; clear it so the banner is visible again
        // on a fresh conversation exactly as it was on first page load.
        var banner = document.getElementById('ai-assistant-panel-speak-banner');
        if (banner) { banner.style.display = ''; }
        var input = document.getElementById('ai-assistant-panel-input');
        if (input) { input.value = ''; _updateSendBtnState(); input.focus(); }
        showNotification('Conversation cleared', false);
    }

    /**
     * R4 — Export the conversation as a plain-text download.
     * Reads ONLY `_transcript` (the single source of truth).
     */
    function exportConversation() {
        if (_transcript.length === 0) {
            showNotification('Nothing to export yet', true);
            return;
        }
        var cfg   = window.AI_ASSISTANT_CONFIG || {};
        var title = cfg.panelTitle || 'AI Assistant';
        var lines = [
            title + ' — conversation export',
            'Page: ' + (location ? location.href : ''),
            'Exported: ' + new Date().toISOString(),
            '',
            '----------------------------------------',
            '',
        ];
        _transcript.forEach(function (m) {
            var who = m.role === 'user' ? 'You'
                : m.role === 'assistant' ? title
                : 'Error';
            // ISO 8601 tag appended when available — enables unambiguous
            // chronological attribution in feedback reports, diffs, and imports.
            var tsTag = m.ts ? '  [' + new Date(m.ts).toISOString() + ']' : '';
            lines.push('[' + who + ']' + tsTag);
            lines.push(m.text);
            lines.push('');
        });
        var blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href = url;
        a.download = 'ai-conversation-' +
            new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    }

    /**
     * R6 — Copy a single answer's text to the clipboard.
     * Prefers data-raw (the original markdown string) over the rendered HTML
     * text content so the copy is clean and reusable outside the panel.
     * @param {string} text  The exact bubble text (from `_transcript`).
     * @param {HTMLElement} [bubbleEl]  Optional bubble element for data-raw.
     */
    function copyAnswer(text, bubbleEl) {
        var raw = (bubbleEl && bubbleEl.getAttribute('data-raw')) || text;
        copyToClipboard(raw, false);
    }

    /**
     * Share a single answer — with its paired question for recipient context.
     *
     * Delivery tiers (tried in order, first available wins):
     *
     * 1. Web Share API — native share sheet on Android, iOS, and Chromium
     *    desktop (Windows / macOS).  Lets the user pick their own target
     *    (messaging app, email, notes, …).
     * 2. Async Clipboard API — modern browsers, same-origin HTTPS or localhost.
     * 3. ``execCommand('copy')`` — legacy fallback (older browsers, HTTP).
     *
     * Payload format
     * --------------
     * ::
     *
     *     Q: <question>
     *
     *     A: <answer (plain markdown)>
     *
     *     — AI Assistant · https://docs.example.com/page
     *
     * The question prefix and attribution footer are omitted when unavailable.
     * Using raw markdown (``data-raw``) keeps the payload clean and reusable
     * outside the panel — no HTML entities or render artefacts.
     *
     * Parameters
     * ----------
     * answerText : string
     *     The exact bubble text from ``_transcript``.
     * questionText : string | null
     *     The paired user question, or ``null`` when unavailable.
     * bubbleEl : HTMLElement | null
     *     The assistant bubble element; used to read ``data-raw``.
     * btn : HTMLElement | null
     *     The Share button; its label is flashed ("Shared!" / "Copied!") for
     *     1.6 s as a visual confirmation then restored.
     *
     * Notes
     * -----
     * User: On mobile the native share sheet appears immediately.  On desktop
     *   without Web Share API the Q+A block is placed on the clipboard and a
     *   brief toast confirms it.  Paste anywhere to share.
     *
     * Developer: ``writeClipboard`` and ``execCmdCopy`` are nested closures —
     *   they capture ``payload``, ``flash``, and ``showNotification`` from the
     *   enclosing scope.  No module-level state is written.  ``AbortError`` /
     *   ``NotAllowedError`` from ``navigator.share`` are intentionally silent
     *   (user cancelled — no error toast needed).
     */
    function _shareAnswer(answerText, questionText, bubbleEl, btn) {
        var raw     = (bubbleEl && bubbleEl.getAttribute('data-raw')) || answerText;
        var cfg     = window.AI_ASSISTANT_CONFIG || {};
        var aiName  = cfg.panelTitle || 'AI Assistant';
        var pageUrl = (typeof location !== 'undefined') ? location.href : '';

        // Self-contained Q+A payload — recipient reads it without the source page.
        var payload = (questionText ? 'Q: ' + questionText + '\n\nA: ' : '') + raw;
        if (pageUrl) payload += '\n\n\u2014 ' + aiName + ' \u00b7 ' + pageUrl;

        // Flash button label for 1.6 s then restore — primary visual confirmation.
        function flash(label) {
            if (!btn) return;
            var lbl = btn.querySelector('span');
            if (!lbl) return;
            var orig = lbl.textContent;
            lbl.textContent = label;
            btn.disabled = true;
            setTimeout(function () { lbl.textContent = orig; btn.disabled = false; }, 1600);
        }

        // Tier 2b: textarea execCommand fallback (no Clipboard API).
        function execCmdCopy() {
            var ta = document.createElement('textarea');
            ta.value = payload;
            ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none';
            document.body.appendChild(ta);
            ta.select();
            try {
                document.execCommand('copy');
                showNotification('Q & A copied \u2014 ready to share.');
                flash('Copied!');
            } catch (_) {
                showNotification('Could not copy \u2014 please try again.', true);
            }
            document.body.removeChild(ta);
        }

        // Tier 2a: async Clipboard API (modern browsers, HTTPS / localhost).
        function writeClipboard() {
            if (typeof navigator !== 'undefined' &&
                    navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(payload)
                    .then(function () {
                        showNotification('Q & A copied \u2014 ready to share.');
                        flash('Copied!');
                    })
                    .catch(function () { execCmdCopy(); });
            } else {
                execCmdCopy();
            }
        }

        // Tier 1: Web Share API — native share sheet (mobile + modern desktop).
        if (typeof navigator !== 'undefined' && typeof navigator.share === 'function') {
            navigator.share({ title: questionText || aiName, text: payload })
                .then(function () { flash('Shared!'); })
                .catch(function (err) {
                    // AbortError / NotAllowedError = user cancelled — stay silent.
                    if (!err || (err.name !== 'AbortError' && err.name !== 'NotAllowedError')) {
                        writeClipboard();
                    }
                });
            return;
        }

        writeClipboard();
    }

    // ── TTS: Text-to-Speech (Web Speech Synthesis API) ───────────────────────
    //
    // Uses window.speechSynthesis — the correct Web platform TTS API (NOT
    // SpeechRecognition, which handles mic input, not audio output).
    // Degrades gracefully: if the API is absent the Listen button is simply
    // not rendered by _buildBubbleMore().
    //
    // Three states cycle on repeated button clicks:
    //   idle    → playing   (aria-pressed="true"  + pulse animation via CSS)
    //   playing → paused    (aria-pressed="paused" + yellow tint via CSS)
    //   paused  → playing   (resumes from paused position)
    //   playing → idle      (auto-reset when speech ends or on cancel)
    //
    // Only one utterance plays at a time — starting TTS on a new bubble
    // cancels any previously playing one.

    /** Currently active TTS button element (null when idle). */
    var _activeTTSBtn = null;

    /**
     * Reset the TTS button to its idle state.
     * Called when speech ends, errors, or is interrupted by a new utterance.
     * @param {HTMLElement} btn  The Listen button element.
     */
    function _resetTTSBtn(btn) {
        if (!btn) return;
        btn.setAttribute('aria-pressed', 'false');
        btn.innerHTML = ICONS.listen;
        var lbl = document.createElement('span');
        lbl.textContent = 'Listen';
        btn.appendChild(lbl);
        if (_activeTTSBtn === btn) _activeTTSBtn = null;
    }

    /**
     * Toggle TTS playback for an assistant bubble.
     *
     * Parameters
     * ----------
     * btn : HTMLElement
     *     The "Listen" button element.  Its aria-pressed attribute drives
     *     the CSS state machine (idle / playing / paused).
     * text : string
     *     Raw text of the assistant reply (plain-text; markdown stripped).
     */
    function _panelTTSToggle(btn, text) {
        // Guard: API unavailable (old browser, iOS WKWebView, etc.)
        if (!('speechSynthesis' in window)) {
            showNotification('Text-to-speech is not supported in this browser.', true);
            return;
        }

        var state = btn.getAttribute('aria-pressed') || 'false';

        // ── Idle → Playing ────────────────────────────────────────────────
        if (state === 'false') {
            // Cancel any other active utterance first.
            if (_activeTTSBtn && _activeTTSBtn !== btn) {
                window.speechSynthesis.cancel();
                _resetTTSBtn(_activeTTSBtn);
            }

            // Strip markdown syntax for cleaner speech (bold, code fences, etc.)
            var cleanText = text
                .replace(/```[\s\S]*?```/g, 'code block.')
                .replace(/`([^`]+)`/g, '$1')
                .replace(/#{1,6}\s/g, '')
                .replace(/\*\*([^*]+)\*\*/g, '$1')
                .replace(/\*([^*]+)\*/g, '$1')
                .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
                .replace(/---+/g, '.')
                .trim();

            var utterance = new window.SpeechSynthesisUtterance(cleanText);
            utterance.lang  = document.documentElement.lang || 'en-US';
            utterance.rate  = 1.0;
            utterance.pitch = 1.0;

            utterance.onstart = function () {
                btn.setAttribute('aria-pressed', 'true');
                btn.innerHTML = ICONS.listenStop;
                var lbl = document.createElement('span');
                lbl.textContent = 'Stop';
                btn.appendChild(lbl);
                _activeTTSBtn = btn;
            };

            utterance.onend = utterance.onerror = function () {
                _resetTTSBtn(btn);
            };

            utterance.onpause = function () {
                btn.setAttribute('aria-pressed', 'paused');
                // Keep listenStop icon; CSS provides the yellow tint.
            };

            utterance.onresume = function () {
                btn.setAttribute('aria-pressed', 'true');
            };

            window.speechSynthesis.speak(utterance);
            return;
        }

        // ── Playing → Paused ──────────────────────────────────────────────
        if (state === 'true') {
            window.speechSynthesis.pause();
            btn.setAttribute('aria-pressed', 'paused');
            btn.innerHTML = ICONS.listen;
            var lblP = document.createElement('span');
            lblP.textContent = 'Resume';
            btn.appendChild(lblP);
            return;
        }

        // ── Paused → Playing ──────────────────────────────────────────────
        if (state === 'paused') {
            window.speechSynthesis.resume();
            btn.setAttribute('aria-pressed', 'true');
            btn.innerHTML = ICONS.listenStop;
            var lblR = document.createElement('span');
            lblR.textContent = 'Stop';
            btn.appendChild(lblR);
            return;
        }
    }

    /**
     * Build the expandable "⋯ More ▾" action-row menu button and its submenu.
     *
     * Architecture (data-first, extensible):
     *   • `.ai-assistant-panel-bubble-action-more` wraps toggle + submenu.
     *   • `data-open="true|false"` on the submenu drives CSS visibility —
     *     no JS state needed beyond toggling the attribute.
     *   • New actions (Translate, Export, Share) are added as buttons inside
     *     the menu without changing this function's signature.
     *
     * @param {string} answerText  Plain-text answer for TTS playback.
     * @returns {HTMLElement}  The wrapper element (relative-positioned anchor).
     */
    /**
     * Build a semantic ``<time>`` element for an assistant / error bubble's
     * action row.  Placed **before** the Copy button so information precedes
     * interactive controls (WCAG 1.3.2 meaningful sequence).
     *
     * Display
     * -------
     * ``HH:mm``  — 24-hour clock, both fields zero-padded (e.g. ``09:04``).
     *
     * Accessibility
     * -------------
     * ``datetime`` attribute: ISO 8601 extended string from
     * ``Date.prototype.toISOString()`` (e.g. ``2024-01-15T14:30:00.000Z``).
     * Used by assistive technology, parsers, and export consumers.
     *
     * ``aria-label``: locale-aware long-form description for screen readers
     * (e.g. "Sent at Mon, 15 Jan 2024, 14:30").
     *
     * ``title``: initially the same long-form date; updated to a relative
     * string (``"5 minutes ago"``) on ``mouseenter`` for sighted hover users.
     *
     * Parameters
     * ----------
     * ts : number
     *     Unix timestamp in milliseconds (``Date.now()``).  When absent or
     *     non-finite the element is returned empty with ``aria-hidden="true"``
     *     — the ``margin-right: auto`` spacer still holds the layout constant
     *     so Copy / Retry / More stay right-aligned regardless.
     *
     * Returns
     * -------
     * HTMLTimeElement
     *     Fully attributed ``<time>`` ready to prepend to the action row.
     *
     * Notes
     * -----
     * User: The element is purely informational — it has no role, no tabindex,
     *   and no click handler.  Keyboard and pointer users interact only with
     *   the sibling action buttons.
     *
     * Developer: ``tabular-nums`` prevents the ``HH:mm`` width from jittering
     *   when digits change (relevant when replaying transcripts quickly).
     *   The relative-time listener captures ``ts`` and ``fullLabel`` in its
     *   closure — no external state is needed.
     */
    function _buildBubbleTimeEl(ts) {
        var el = document.createElement('time');
        el.className = 'ai-assistant-panel-bubble-time';

        // Guard: missing or invalid timestamp — return invisible spacer only.
        if (!ts || !isFinite(ts)) {
            el.setAttribute('aria-hidden', 'true');
            return el;
        }

        var d  = new Date(ts);
        var hh = ('0' + d.getHours()).slice(-2);
        var mm = ('0' + d.getMinutes()).slice(-2);

        // Short display: HH:mm (24-hour, leading zeros).
        el.textContent = hh + ':' + mm;

        // datetime attribute — ISO 8601 extended (machine-readable, HTML spec
        // compliant, consumed by assistive tech and conversation export).
        el.setAttribute('datetime', d.toISOString());

        // Human-readable label for screen readers; locale-aware, 24-hour clock.
        var fullLabel = d.toLocaleString(
            (typeof navigator !== 'undefined' && navigator.language) || 'en',
            {
                weekday: 'short',
                year:    'numeric',
                month:   'short',
                day:     'numeric',
                hour:    '2-digit',
                minute:  '2-digit',
                hour12:  false,
            }
        );
        el.setAttribute('aria-label', 'Sent at ' + fullLabel);
        el.title = fullLabel;   // static full-date tooltip by default

        // ── Relative-time tooltip on hover ────────────────────────────────
        // Dynamically computes "X min ago" / "X hours ago" / date-only on
        // mouseenter so the tooltip stays accurate for long conversations.
        // mouseleave restores the full absolute date (consistent with the
        // aria-label so screen-reader and pointer users see the same info).
        el.addEventListener('mouseenter', function () {
            var diff = Date.now() - ts;
            var mins = Math.round(diff / 60000);
            var rel;
            if (mins < 1) {
                rel = 'Just now';
            } else if (mins < 60) {
                rel = mins + (mins === 1 ? ' min ago' : ' mins ago');
            } else {
                var hrs = Math.round(mins / 60);
                if (hrs < 24) {
                    rel = hrs + (hrs === 1 ? ' hour ago' : ' hours ago');
                } else {
                    rel = d.toLocaleDateString(
                        (typeof navigator !== 'undefined' && navigator.language) || 'en',
                        { weekday: 'short', month: 'short', day: 'numeric' }
                    );
                }
            }
            el.title = rel;
        });
        el.addEventListener('mouseleave', function () { el.title = fullLabel; });

        return el;
    }

    function _buildBubbleMore(answerText) {
        var wrapper = document.createElement('div');
        wrapper.className = 'ai-assistant-panel-bubble-action-more';

        // ── "⋯ More ▾" toggle button ──────────────────────────────────────
        var toggleBtn = document.createElement('button');
        toggleBtn.className =
            'ai-assistant-panel-bubble-action ' +
            'ai-assistant-panel-bubble-action--more-toggle';
        toggleBtn.type = 'button';
        toggleBtn.setAttribute('aria-label', 'More actions');
        toggleBtn.setAttribute('aria-expanded', 'false');
        toggleBtn.setAttribute('aria-haspopup', 'true');
        toggleBtn.innerHTML = ICONS.moreVert;
        var moreLbl = document.createElement('span');
        moreLbl.textContent = 'More';
        toggleBtn.appendChild(moreLbl);

        // ── Submenu (hidden by default via data-open="false") ─────────────
        var menu = document.createElement('div');
        menu.className = 'ai-assistant-panel-bubble-action-more-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('data-open', 'false');

        // ── Listen (TTS) button inside the menu ───────────────────────────
        // Only rendered when the Web Speech Synthesis API is available.
        if ('speechSynthesis' in window) {
            var listenBtn = document.createElement('button');
            listenBtn.className =
                'ai-assistant-panel-bubble-action ' +
                'ai-assistant-panel-bubble-action--listen';
            listenBtn.type = 'button';
            listenBtn.setAttribute('role', 'menuitem');
            listenBtn.setAttribute('aria-pressed', 'false');
            listenBtn.setAttribute('aria-label', 'Read this answer aloud');
            listenBtn.title = 'Listen — read answer aloud';
            listenBtn.innerHTML = ICONS.listen;
            var listenLbl = document.createElement('span');
            listenLbl.textContent = 'Listen';
            listenBtn.appendChild(listenLbl);
            (function (btn, text) {
                btn.addEventListener('click', function (e) {
                    e.stopPropagation();   // don't bubble to global click-close
                    _panelTTSToggle(btn, text);
                });
            }(listenBtn, answerText));
            menu.appendChild(listenBtn);
        }

        // ── Toggle click handler ──────────────────────────────────────────
        toggleBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            var isOpen = menu.getAttribute('data-open') === 'true';
            menu.setAttribute('data-open', isOpen ? 'false' : 'true');
            toggleBtn.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
        });

        // ── Close menu when focus leaves the wrapper ──────────────────────
        wrapper.addEventListener('focusout', function (e) {
            if (!wrapper.contains(e.relatedTarget)) {
                menu.setAttribute('data-open', 'false');
                toggleBtn.setAttribute('aria-expanded', 'false');
            }
        });

        wrapper.appendChild(toggleBtn);
        wrapper.appendChild(menu);
        return wrapper;
    }

    /**
     * Render the initial welcome + quick-suggestion chips into the body.
     * Extracted so clearConversation() can rebuild without duplicating logic.
     * @param {HTMLElement} body
     */
    function _renderWelcome(body) {
        var cfg     = window.AI_ASSISTANT_CONFIG || {};
        var title   = cfg.panelTitle || 'AI Assistant';
        var quickQs = Array.isArray(cfg.panelQuickQuestions)
            ? cfg.panelQuickQuestions.slice(0, 5) : [];

        var welcome = document.createElement('div');
        welcome.className = 'ai-assistant-panel-welcome';
        var p1 = document.createElement('p');
        var strong = document.createElement('strong');
        strong.textContent = title;
        p1.appendChild(document.createTextNode('Hi! I\u2019m '));
        p1.appendChild(strong);
        p1.appendChild(document.createTextNode('.'));
        var p2 = document.createElement('p');
        p2.textContent = 'Ask me anything about this documentation page.';
        welcome.appendChild(p1);
        welcome.appendChild(p2);
        body.appendChild(welcome);

        if (quickQs.length > 0) {
            var suggestionsEl = document.createElement('div');
            suggestionsEl.className = 'ai-assistant-panel-suggestions';
            suggestionsEl.id = 'ai-assistant-panel-suggestions';
            quickQs.forEach(function (q) {
                var chip = document.createElement('button');
                chip.className = 'ai-assistant-panel-chip';
                chip.type = 'button';
                chip.textContent = q;
                chip.addEventListener('click', function () {
                    var input = document.getElementById('ai-assistant-panel-input');
                    if (input) { input.value = q; _updateSendBtnState(); input.focus(); }
                });
                suggestionsEl.appendChild(chip);
            });
            body.appendChild(suggestionsEl);
        }
    }

    /**
     * Replay a persisted transcript into the panel body (no welcome).
     * @param {HTMLElement} body
     */
    function _replayTranscript(body) {
        _transcript.forEach(function (m) {
            _renderBubble(body, m.text, m.role, undefined, m.ts);
        });
        body.scrollTop = body.scrollHeight;
    }

    // ── R7: keyboard shortcut parsing ─────────────────────────────────────────

    /**
     * Parse a human shortcut string ("Alt+Shift+A", "Ctrl+/") into a
     * predicate over a KeyboardEvent.  Defensive: returns null for empty or
     * unparsable input so the caller can simply skip binding.
     *
     * Rationale: a single un-modified key must NEVER be bound — it would
     * hijack typing site-wide.  At least one modifier is required.
     *
     * @param {string} spec
     * @returns {(e:KeyboardEvent)=>boolean|null}
     */
    function _parseShortcut(spec) {
        if (typeof spec !== 'string' || !spec.trim()) return null;
        var parts = spec.toLowerCase().split('+').map(function (s) { return s.trim(); });
        var want  = { ctrl: false, alt: false, shift: false, meta: false, key: null };
        parts.forEach(function (p) {
            if (p === 'ctrl' || p === 'control') want.ctrl = true;
            else if (p === 'alt' || p === 'option') want.alt = true;
            else if (p === 'shift') want.shift = true;
            else if (p === 'meta' || p === 'cmd' || p === 'command' || p === 'win') want.meta = true;
            else if (p) want.key = p;
        });
        if (!want.key) return null;
        // Require at least one modifier — never bind a bare key.
        if (!(want.ctrl || want.alt || want.meta)) return null;
        return function (e) {
            var k = (e.key || '').toLowerCase();
            return e.ctrlKey === want.ctrl &&
                   e.altKey === want.alt &&
                   e.shiftKey === want.shift &&
                   e.metaKey === want.meta &&
                   k === want.key;
        };
    }

    /** True once the global shortcut handler is installed (idempotency). */
    var _shortcutBound = false;

    /**
     * Install the configurable open/toggle shortcut. Default "Alt+Shift+A".
     * Empty string in config → feature disabled (no global listener at all).
     */
    function _bindShortcut() {
        if (_shortcutBound) return;
        var cfg  = window.AI_ASSISTANT_CONFIG || {};
        var spec = typeof cfg.panelShortcut === 'string'
            ? cfg.panelShortcut : 'Alt+Shift+A';
        var pred = _parseShortcut(spec);
        if (!pred) return;             // disabled or invalid → safe no-op
        _shortcutBound = true;
        document.addEventListener('keydown', function (e) {
            if (pred(e)) { e.preventDefault(); toggleAIPanel(); }
        });
    }

    /** Human-readable shortcut label for the hint chip (or '' if disabled). */
    function _shortcutLabel() {
        var cfg  = window.AI_ASSISTANT_CONFIG || {};
        var spec = typeof cfg.panelShortcut === 'string'
            ? cfg.panelShortcut : 'Alt+Shift+A';
        return _parseShortcut(spec) ? spec : '';
    }

    // ── R1: mouse resize (top-left grip) ──────────────────────────────────────

    /** sessionStorage key for the persisted manual panel size. */
    var _PANEL_SIZE_KEY = 'ai-assistant-panel-size';

    /**
     * Attach a top-left resize grip to the panel.  Width grows leftwards and
     * height upwards (because the panel is anchored bottom-right), clamped to
     * the viewport, and persisted.  Pointer events so mouse + touch + pen all
     * work with one code path.
     *
     * @param {HTMLElement} panel
     */
    function _attachResizer(panel) {
        var grip = document.createElement('div');
        grip.className = 'ai-assistant-panel-resizer';
        grip.setAttribute('aria-hidden', 'true');
        grip.title = 'Drag to resize';
        panel.appendChild(grip);

        // Restore persisted size (only in default, non-maximized state).
        var saved = _ssGet(_PANEL_SIZE_KEY);
        if (saved) {
            try {
                var s = JSON.parse(saved);
                if (s && s.w && s.h) {
                    panel.style.width  = s.w + 'px';
                    panel.style.height = s.h + 'px';   /* height (not maxHeight) so panel fills to saved value */
                }
            } catch (_) {}
        }

        var dragging = false, startX = 0, startY = 0, startW = 0, startH = 0;
        var MIN_W = 300, MIN_H = 320;

        grip.addEventListener('pointerdown', function (e) {
            if (panel.getAttribute('data-maximized') === 'true') return;
            dragging = true;
            startX = e.clientX; startY = e.clientY;
            var rect = panel.getBoundingClientRect();
            startW = rect.width; startH = rect.height;
            document.body.classList.add('ai-assistant-resizing', 'ai-assistant-resizing-xy');
            grip.setPointerCapture(e.pointerId);
            e.preventDefault();
        });

        grip.addEventListener('pointermove', function (e) {
            if (!dragging) return;
            var dw = startX - e.clientX;
            var dh = startY - e.clientY;
            var maxW = window.innerWidth  - 32;
            var maxH = window.innerHeight - 32;
            var newW = Math.max(MIN_W, Math.min(maxW, startW + dw));
            var newH = Math.max(MIN_H, Math.min(maxH, startH + dh));
            panel.style.width  = newW + 'px';
            /* Use height (not maxHeight): maxHeight only sets a ceiling — the
               panel never actually grows past its content height unless an
               explicit height is set on the fixed-position flex container. */
            panel.style.height = newH + 'px';
        });

        function _endDrag(e) {
            if (!dragging) return;
            dragging = false;
            document.body.classList.remove(
                'ai-assistant-resizing', 'ai-assistant-resizing-xy',
                'ai-assistant-resizing-x', 'ai-assistant-resizing-y'
            );
            try { grip.releasePointerCapture(e.pointerId); } catch (_) {}
            var rect = panel.getBoundingClientRect();
            _ssSet(_PANEL_SIZE_KEY, JSON.stringify({
                w: Math.round(rect.width), h: Math.round(rect.height),
            }));
        }
        grip.addEventListener('pointerup', _endDrag);
        grip.addEventListener('pointercancel', _endDrag);

        // ── Left-edge grip (X / width only) ───────────────────────────────────
        var gripLeft = document.createElement('div');
        gripLeft.className = 'ai-assistant-panel-resizer-left';
        gripLeft.setAttribute('aria-hidden', 'true');
        gripLeft.title = 'Drag to resize width';
        panel.appendChild(gripLeft);

        var draggingL = false, startXL = 0, startWL = 0;

        gripLeft.addEventListener('pointerdown', function (e) {
            if (panel.getAttribute('data-maximized') === 'true') return;
            draggingL = true;
            startXL = e.clientX;
            startWL = panel.getBoundingClientRect().width;
            document.body.classList.add('ai-assistant-resizing', 'ai-assistant-resizing-x');
            gripLeft.setPointerCapture(e.pointerId);
            e.preventDefault();
        });

        gripLeft.addEventListener('pointermove', function (e) {
            if (!draggingL) return;
            var dw   = startXL - e.clientX;
            var maxW = window.innerWidth - 32;
            var newW = Math.max(MIN_W, Math.min(maxW, startWL + dw));
            panel.style.width = newW + 'px';
        });

        function _endDragLeft(e) {
            if (!draggingL) return;
            draggingL = false;
            document.body.classList.remove(
                'ai-assistant-resizing', 'ai-assistant-resizing-x'
            );
            try { gripLeft.releasePointerCapture(e.pointerId); } catch (_) {}
            var rect = panel.getBoundingClientRect();
            _ssSet(_PANEL_SIZE_KEY, JSON.stringify({
                w: Math.round(rect.width),
                /* Preserve current height in the saved key so the corner grip's
                   persisted h value is not lost when only width changes. */
                h: Math.round(rect.height),
            }));
        }
        gripLeft.addEventListener('pointerup', _endDragLeft);
        gripLeft.addEventListener('pointercancel', _endDragLeft);

        // ── Top-edge grip (Y / height only) ───────────────────────────────────
        var gripTop = document.createElement('div');
        gripTop.className = 'ai-assistant-panel-resizer-top';
        gripTop.setAttribute('aria-hidden', 'true');
        gripTop.title = 'Drag to resize height';
        panel.appendChild(gripTop);

        var draggingT = false, startYT = 0, startHT = 0;

        gripTop.addEventListener('pointerdown', function (e) {
            if (panel.getAttribute('data-maximized') === 'true') return;
            draggingT = true;
            startYT = e.clientY;
            startHT = panel.getBoundingClientRect().height;
            document.body.classList.add('ai-assistant-resizing', 'ai-assistant-resizing-y');
            gripTop.setPointerCapture(e.pointerId);
            e.preventDefault();
        });

        gripTop.addEventListener('pointermove', function (e) {
            if (!draggingT) return;
            var dh   = startYT - e.clientY;
            var maxH = window.innerHeight - 32;
            var newH = Math.max(MIN_H, Math.min(maxH, startHT + dh));
            panel.style.height = newH + 'px';
        });

        function _endDragTop(e) {
            if (!draggingT) return;
            draggingT = false;
            document.body.classList.remove(
                'ai-assistant-resizing', 'ai-assistant-resizing-y'
            );
            try { gripTop.releasePointerCapture(e.pointerId); } catch (_) {}
            var rect = panel.getBoundingClientRect();
            _ssSet(_PANEL_SIZE_KEY, JSON.stringify({
                w: Math.round(rect.width),
                h: Math.round(rect.height),
            }));
        }
        gripTop.addEventListener('pointerup', _endDragTop);
        gripTop.addEventListener('pointercancel', _endDragTop);
    }

    // ── R5: feedback block ────────────────────────────────────────────────────

    /**
     * Default emoji option set (3 options, signed-integer scale [-1, 0, +1]).
     *
     * Why signed integers and not strings (e.g. "positive"/"neutral"/"negative"):
     *
     *   • Strings cannot be averaged, thresholded, or subtracted to compute
     *     deltas — they are unusable as a model-training signal.
     *   • Signed integers centred on zero are the canonical Likert encoding;
     *     downstream consumers can group, average, or threshold directly.
     *   • The numeric value is computed server-side (see
     *     ``_resolve_feedback_scale`` in __init__.py) and shipped to the
     *     widget as ``cfg.panelFeedbackScale`` — a parallel list aligned with
     *     ``cfg.panelFeedbackOptions`` so the two arrays cannot drift.
     *
     * The defaults below stay as a UI-only fallback for the case where the
     * doc author has not configured any options at all.  When the config
     * supplies options but no parallel scale (older builds) the JS still
     * derives one client-side via _deriveDefaultScale so older injected
     * configs do not silently break.
     */
    /**
     * Default emoji option set — 11 options on a signed-integer scale
     * (-5 … -1, 0, +1 … +5).  Odd-11 includes the neutral midpoint (0) so
     * the user always has a clear "no opinion" option.  The visual gradient
     * goes from most negative (😡) through neutral (😐) to most positive (🤩),
     * giving the user an intuitive left-to-right sweep.
     *
     * Counts 2+ are supported.  Emoji size is auto-scaled by CSS
     * (data-count + data-tier on the options row) so all buttons fit,
     * wrapping to multiple rows for counts above 10.
     */
    var _FEEDBACK_DEFAULTS = [
        { emoji: '\uD83D\uDE21', title: 'Terrible',       value: 'terrible'          },  // 😡 -5
        { emoji: '\uD83D\uDE1E', title: 'Poor',           value: 'poor'              },  // 😞 -4
        { emoji: '\uD83D\uDE1F', title: 'Unsatisfied',    value: 'unsatisfied'       },  // 😟 -3
        { emoji: '\uD83D\uDE41', title: 'No',             value: 'negative'          },  // 🙁 -2
        { emoji: '\uD83D\uDE11', title: 'Not really',     value: 'slightly_negative' },  // 😑 -1
        { emoji: '\uD83D\uDE10', title: 'Neutral',        value: 'neutral'           },  // 😐  0
        { emoji: '\uD83D\uDE42', title: 'Somewhat',       value: 'slightly_positive' },  // 🙂 +1
        { emoji: '\uD83D\uDE0A', title: 'Mostly yes',     value: 'mostly_positive'   },  // 😊 +2
        { emoji: '\uD83D\uDE04', title: 'Good',           value: 'good'              },  // 😄 +3
        { emoji: '\uD83D\uDE01', title: 'Very good',      value: 'very_good'         },  // 😁 +4
        { emoji: '\uD83E\uDD29', title: 'Excellent!',     value: 'excellent'         },  // 🤩 +5
    ];

    /**
     * Client-side fallback scale for the no-server-side-scale case.
     * Matches the server-side ``_generate_symmetric_scale`` exactly:
     *
     *   odd  N -> (-k, ..., -1, 0, +1, ..., +k)  (sum = 0, midpoint 0)
     *   even N -> (-k, ..., -1, +1, ..., +k)     (sum = 0, no midpoint)
     *
     * @param {number} n  Number of options (>=2).
     * @returns {number[]}
     */
    function _deriveDefaultScale(n) {
        if (typeof n !== 'number' || n < 2) return [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]; // 11-emoji fallback
        var k = Math.floor(n / 2);
        var out = [];
        if (n % 2 === 1) {
            for (var i = -k; i <= k; i++) out.push(i);
        } else {
            for (var j = -k; j < 0; j++) out.push(j);
            for (var m = 1; m <= k; m++) out.push(m);
        }
        return out;
    }

    /**
     * Map an emoji count to a CSS layout tier (1–8).
     *
     * Tiers drive ``data-tier`` on the options row, which the CSS uses to
     * pick font-size, padding, and flex-wrap rules independently of
     * ``data-count``.  ``data-count`` is preserved for backward
     * compatibility with any external CSS that targets it directly.
     *
     * Tier map:
     *   1 → 2–3    (single row, largest buttons)
     *   2 → 4–5    (single row)
     *   3 → 6–7    (single row)
     *   4 → 8–9    (single row)
     *   5 → 10     (single row, smallest single-row size)
     *   6 → 11–15  (multi-row wrap, medium buttons)
     *   7 → 16–20  (multi-row wrap, smaller buttons)
     *   8 → 21+    (multi-row wrap, compact buttons)
     *
     * @param {number} n  Number of emoji options (≥ 2).
     * @returns {number}  Tier index 1–8.
     */
    function _getFeedbackTier(n) {
        if (n <= 3)  return 1;
        if (n <= 5)  return 2;
        if (n <= 7)  return 3;
        if (n <= 9)  return 4;
        if (n <= 10) return 5;
        if (n <= 15) return 6;
        if (n <= 20) return 7;
        return 8;
    }

    /**
     * Build a per-answer feedback block.  Options, question, and thanks copy
     * are all config-driven (ai_assistant_panel_feedback_*).  11 emoji options
     * by default, 2+ supported (any count ≥ 2).  Rendered inline under each
     * assistant bubble for granular per-answer model-training data collection.
     *
     * Developer note: on submit, the rating + free text are dispatched as a
     * `ai-assistant-feedback` CustomEvent on `document` AND, if configured,
     * console-logged.  Doc authors hook the event for their own analytics —
     * the extension itself stores nothing and sends nothing.
     *
     * The event payload (``event.detail``) shape — version 1:
     *
     *     {
     *       schemaVersion : 1,                  // for forward compatibility
     *       ratingValue   : -1 | 0 | +1 | ...,  // SIGNED INT (training signal)
     *       ratingLabel   : "negative" | ...,   // string (humans / dashboards)
     *       rating        : "negative" | ...,   // legacy alias = ratingLabel
     *       message       : "free-text...",
     *       query         : "the user's question",   // NEW
     *       answer        : "the model's full reply", // NEW
     *       model         : { id, provider, model } | null,  // NEW (Phase B)
     *       answerIndex   : 0,
     *       page          : "https://docs.example.com/x.html",
     *       ts            : 1716517200000,
     *       sessionId     : "c0c5f8a0-..."    // crypto.randomUUID — idempotency
     *     }
     *
     * Backward compatibility: the legacy ``detail.rating`` string field is
     * preserved as an alias of ``ratingLabel`` so existing listeners keep
     * working without change.
     *
     * User note: each answer carries its own independent feedback block so
     * every exchange can be rated separately.  The numeric value of each
     * emoji is shown on hover (title attribute + data-value), so the final
     * user is always aware of what they are submitting.
     *
     * @param {number} answerIndex     Zero-based index of this assistant
     *                                 answer (tracks which answers have been
     *                                 rated independently).
     * @param {string} [answerText]    The assistant reply text being rated;
     *                                 forwarded in the event payload so the
     *                                 data is usable as a (q, a, rating)
     *                                 training tuple.
     * @param {string} [questionText]  The paired user question for the same
     *                                 reason.  Both fields are optional so
     *                                 the legacy callsite (no args beyond
     *                                 answerIndex) keeps working unchanged.
     * @returns {HTMLElement|null}
     */
    function _buildFeedbackBlock(answerIndex, answerText, questionText) {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        if (cfg.panelFeedback === false) return null;     // opt-out
        if (_feedbackGivenSet.has(answerIndex)) return null;

        var question = (typeof cfg.panelFeedbackQuestion === 'string' &&
            cfg.panelFeedbackQuestion) || 'Was this helpful?';
        var thanks = (typeof cfg.panelFeedbackThanks === 'string' &&
            cfg.panelFeedbackThanks) || 'Thanks for your feedback!';

        var opts = Array.isArray(cfg.panelFeedbackOptions) &&
            cfg.panelFeedbackOptions.length >= 2
            ? cfg.panelFeedbackOptions.slice()   // defensive copy, no upper cap — any count ≥ 2
            : _FEEDBACK_DEFAULTS;

        // Parallel numeric scale.  Prefer the server-resolved
        // cfg.panelFeedbackScale when present (canonical) and only fall back
        // to the client-side derivation for older builds without that field.
        // We never trust cfg.panelFeedbackScale if its length disagrees with
        // opts.length — explicit mismatch ⇒ re-derive (no silent truncation).
        var scale;
        if (Array.isArray(cfg.panelFeedbackScale) &&
            cfg.panelFeedbackScale.length === opts.length &&
            cfg.panelFeedbackScale.every(function (v) { return typeof v === 'number'; })) {
            scale = cfg.panelFeedbackScale.slice();
        } else {
            scale = _deriveDefaultScale(opts.length);
        }

        var wrap = document.createElement('div');
        wrap.className = 'ai-assistant-panel-feedback ai-assistant-panel-feedback--inline';

        var q = document.createElement('p');
        q.className = 'ai-assistant-panel-feedback-q';
        q.textContent = question;
        wrap.appendChild(q);

        var optRow = document.createElement('div');
        optRow.className = 'ai-assistant-panel-feedback-options';

        // Track BOTH the label (legacy) and the numeric value.  The numeric
        // value is the training signal; the label is for humans.
        var chosen = { label: null, value: null };
        opts.forEach(function (o, idx) {
            var b = document.createElement('button');
            b.className = 'ai-assistant-panel-feedback-btn';
            b.type = 'button';

            // Numeric value for this emoji (signed integer).
            var num = scale[idx];

            // ── Sentiment attribute drives CSS colour rules ────────────────
            // positive (>0) → lime-green, negative (<0) → red, neutral → brand
            var sentiment = num > 0 ? 'positive' : (num < 0 ? 'negative' : 'neutral');
            b.setAttribute('data-sentiment', sentiment);

            // ── Inner markup: emoji span + hidden score chip ───────────────
            // The score chip (".ai-fbk-score") is kept hidden via CSS and only
            // revealed (display:inline) when aria-pressed="true", giving instant
            // visual feedback: selected button, sentiment colour, and numeric value.
            var emojiSpan = document.createElement('span');
            emojiSpan.setAttribute('aria-hidden', 'true');   // emoji is decorative
            emojiSpan.textContent = o.emoji || '\u2753';

            var scoreSpan = document.createElement('span');
            scoreSpan.className = 'ai-fbk-score';
            scoreSpan.textContent = num > 0 ? ('+' + num) : String(num);
            scoreSpan.setAttribute('aria-hidden', 'true');   // announced via aria-label

            b.appendChild(emojiSpan);
            b.appendChild(scoreSpan);

            // Hover/aria tooltip: ALWAYS surfaces the signed numeric value so
            // the final user knows what they are sending.  Format:
            //     "<title> (+1)"      or
            //     "<title> (-1)"      or
            //     "<title> ( 0)"      ← single space pad to align in tooltips
            // When the option has no title we still show "(+N)" alone.
            var sign = num > 0 ? '+' + num : (num === 0 ? ' 0' : String(num));
            var tip = o.title ? (o.title + ' (' + sign + ')') : ('(' + sign + ')');
            b.title = tip;
            b.setAttribute('aria-label', tip);
            b.setAttribute('data-value', String(num));
            b.setAttribute('aria-pressed', 'false');

            b.addEventListener('click', function () {
                chosen.label = o.value || o.title || o.emoji;
                chosen.value = num;
                optRow.querySelectorAll('button').forEach(function (x) {
                    x.setAttribute('aria-pressed', 'false');
                });
                b.setAttribute('aria-pressed', 'true');
            });
            optRow.appendChild(b);
        });
        // ── Count + tier attributes drive CSS adaptive sizing ─────────────
        // data-count: exact emoji count (backward compat for any external CSS).
        // data-tier:  coarse layout tier (1–8) used by the built-in CSS rules
        //             to pick font-size, padding, gap, and flex-wrap strategy.
        optRow.setAttribute('data-count', String(opts.length));
        optRow.setAttribute('data-tier',  String(_getFeedbackTier(opts.length)));
        wrap.appendChild(optRow);

        var ta = document.createElement('textarea');
        ta.className = 'ai-assistant-panel-feedback-text';
        ta.placeholder = (typeof cfg.panelFeedbackPlaceholder === 'string' &&
            cfg.panelFeedbackPlaceholder) ||
            'Optional: tell us more (what worked, what didn\u2019t)\u2026';
        ta.setAttribute('aria-label', 'Feedback details');
        wrap.appendChild(ta);

        var submit = document.createElement('button');
        submit.className = 'ai-assistant-panel-feedback-submit';
        submit.type = 'button';
        submit.textContent = (typeof cfg.panelFeedbackSubmit === 'string' &&
            cfg.panelFeedbackSubmit) || 'Send feedback';
        submit.addEventListener('click', function () {
            // Compute a stable session-scoped idempotency key.  We use the
            // built-in crypto.randomUUID when available (modern browsers);
            // a deterministic fallback keyed on page + ts + answerIndex is
            // safe for older browsers because the event consumer can dedupe
            // on (page, answerIndex, ts) just as well.
            var sid;
            try {
                if (window.crypto && typeof window.crypto.randomUUID === 'function') {
                    sid = window.crypto.randomUUID();
                }
            } catch (_) {}
            if (!sid) {
                sid = 'fb-' + (location ? location.pathname : 'p') +
                      '-' + answerIndex + '-' + Date.now();
            }

            // ── Phase B: model attribution ────────────────────────────
            //
            // Resolution order (deterministic; mirrors _panelApiCall):
            //   1. Active panel-model from cfg.panelApiModels (sessionStorage
            //      → default flag → first entry).  This is the canonical
            //      source when the multi-model contract is in use.
            //   2. Legacy single-string cfg.panelApiModel (Phase A path).
            //      Provider tag remains "anthropic" only because that was
            //      the documented assumption of the original API-mode contract
            //      (proxy → /v1/messages); doc authors who use the single-
            //      string path with a non-Anthropic proxy should migrate to
            //      cfg.panelApiModels so this label is accurate.
            //   3. Null when neither is configured (stub-mode reply).
            //
            // The training pipeline reads ``model.id`` and ``model.provider``
            // to group ratings per model; the ``answerIndex`` + ``sessionId``
            // pair below is the idempotency key.
            var modelInfo = null;
            var activeModel = _getActiveModel(cfg);
            if (activeModel) {
                modelInfo = {
                    id:       activeModel.id,
                    provider: activeModel.provider || 'custom',
                    model:    activeModel.model || activeModel.id,
                };
            } else if (typeof cfg.panelApiModel === 'string' && cfg.panelApiModel) {
                modelInfo = {
                    id:       cfg.panelApiModel,
                    provider: 'anthropic',      // legacy single-model assumption
                    model:    cfg.panelApiModel,
                };
            }

            var detail = {
                schemaVersion: 1,
                ratingValue:   chosen.value,        // SIGNED INT
                ratingLabel:   chosen.label,        // string
                rating:        chosen.label,        // legacy alias (back-compat)
                message:       ta.value.trim(),
                query:         (typeof questionText === 'string') ? questionText : '',
                answer:        (typeof answerText === 'string') ? answerText : '',
                model:         modelInfo,
                answerIndex:   answerIndex,
                page:          location ? location.href : '',
                ts:            Date.now(),
                sessionId:     sid,
            };

            // Dev-friendly hook — doc authors attach their own analytics.
            try {
                document.dispatchEvent(new CustomEvent(
                    'ai-assistant-feedback', { detail: detail }));
            } catch (_) {}
            if (cfg.panelFeedbackLog) {
                // eslint-disable-next-line no-console
                console.log('[ai-assistant] feedback', detail);
            }
            _feedbackGivenSet.add(answerIndex);
            wrap.innerHTML = '';
            var done = document.createElement('p');
            done.className = 'ai-assistant-panel-feedback-thanks';
            done.textContent = thanks;
            wrap.appendChild(done);
        });
        wrap.appendChild(submit);

        return wrap;
    }

    // ── R2: privacy / responsibility sheet ────────────────────────────────────

    /**
     * Build the default privacy/responsibility copy.  Every section can be
     * overridden whole via ai_assistant_panel_privacy_html (raw, trusted —
     * authored by the doc owner, NOT user input), or the title via
     * ai_assistant_panel_privacy_title.  The default text is deliberately
     * structured for a beginner→expert reader and explicitly separates the
     * extension's responsibility from the integrated model's.
     *
     * @returns {HTMLElement}
     */
    function _buildPrivacySheet() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var title = (typeof cfg.panelPrivacyTitle === 'string' &&
            cfg.panelPrivacyTitle) || 'Privacy & Responsibility';

        var sheet = document.createElement('div');
        sheet.className = 'ai-assistant-panel-privacy';
        sheet.id = 'ai-assistant-panel-privacy';
        sheet.setAttribute('data-open', 'false');

        var head = document.createElement('div');
        head.className = 'ai-assistant-panel-privacy-head';
        var hStrong = document.createElement('strong');
        hStrong.textContent = title;
        var hClose = _createIconBtn('privacy-close', 'Close ' + title, ICONS.close);
        hClose.addEventListener('click', function () {
            sheet.setAttribute('data-open', 'false');
        });
        head.appendChild(hStrong);
        head.appendChild(hClose);
        sheet.appendChild(head);

        var bodyEl = document.createElement('div');
        bodyEl.className = 'ai-assistant-panel-privacy-body';

        if (typeof cfg.panelPrivacyHtml === 'string' && cfg.panelPrivacyHtml) {
            // Trusted, author-supplied (from conf.py, not end-user input).
            bodyEl.innerHTML = cfg.panelPrivacyHtml;
        } else {
            var apiOn = !!cfg.panelApiEnabled;
            bodyEl.innerHTML =
                '<h4>What this assistant is</h4>' +
                '<p>This is a documentation helper added by a Sphinx ' +
                'extension. Its purpose is to make this page easier to use ' +
                'with AI tools \u2014 nothing more.</p>' +

                '<h4>What the extension does</h4>' +
                '<ul>' +
                '<li>Formats the current page as Markdown in your browser.</li>' +
                '<li>Shows this chat panel and its controls.</li>' +
                '<li>Stores your conversation only in this browser tab ' +
                '(sessionStorage) so it survives navigation; it is cleared ' +
                'when you press \u201cStart a new chat\u201d or close the tab.</li>' +
                '</ul>' +

                '<h4>What the extension does NOT do</h4>' +
                '<ul>' +
                '<li>It runs no server and keeps no database.</li>' +
                '<li>In stub mode it makes <strong>zero network calls</strong>.</li>' +
                '<li>It cannot see, store, or control any AI provider\u2019s ' +
                'internal logs.</li>' +
                '</ul>' +

                '<h4>Where the boundary is</h4>' +
                (apiOn
                    ? ('<p><strong>API mode is enabled.</strong> When you ' +
                       'send a question, your text and an extract of this ' +
                       'page are sent to the configured AI endpoint. From ' +
                       'that point the answer, any data retention, and all ' +
                       'logging are the responsibility of <em>that ' +
                       'provider</em>, governed by <em>their</em> privacy ' +
                       'policy \u2014 not this extension. We cannot inspect ' +
                       'or delete their logs.</p>')
                    : ('<p><strong>Stub mode (default).</strong> No question ' +
                       'leaves your browser. Enabling API mode would send ' +
                       'your text to a third-party model whose logging and ' +
                       'retention are governed by that provider, not by this ' +
                       'extension.</p>')) +

                '<h4>Your control</h4>' +
                '<ul>' +
                '<li>\u201cStart a new chat\u201d erases the stored conversation.</li>' +
                '<li>\u201cExport as txt\u201d gives you a full local copy.</li>' +
                '<li>Closing the tab clears in-browser history.</li>' +
                '</ul>';
        }
        sheet.appendChild(bodyEl);
        return sheet;
    }

    // ── Phase B: Active-model state (sessionStorage-backed) ───────────────────

    /**
     * sessionStorage key for the active panel-model id.  Per tab — never
     * shared across tabs and cleared on tab close.  Mirrors the same
     * persistence philosophy as the chat transcript.
     */
    var _PANEL_MODEL_KEY = 'ai-assistant-active-model-id';

    /**
     * Return the active model id (sessionStorage → cfg default → first valid).
     *
     * Resolution order (deterministic):
     *   1. sessionStorage value, if it still matches a valid id.
     *   2. The entry with ``default: true``, if any.
     *   3. The first entry in the list.
     *   4. ``null`` when the list is empty.
     *
     * @param {Array<object>} models  cfg.panelApiModels (already validated).
     * @returns {string|null}
     */
    function _getActiveModelId(models) {
        if (!Array.isArray(models) || models.length === 0) return null;
        var ids = {};
        models.forEach(function (m) { ids[m.id] = m; });

        var stored = null;
        try { stored = sessionStorage.getItem(_PANEL_MODEL_KEY); } catch (_) {}
        if (stored && ids[stored]) return stored;

        for (var i = 0; i < models.length; i++) {
            if (models[i].default === true) return models[i].id;
        }
        return models[0].id;
    }

    /**
     * Persist *id* as the active model.  Best-effort: sessionStorage may be
     * blocked (Safari private mode, etc.) — failures are silent and the
     * runtime selection still works for the rest of the tab session.
     *
     * @param {string} id
     */
    function _setActiveModelId(id) {
        if (typeof id !== 'string' || !id) return;
        try { sessionStorage.setItem(_PANEL_MODEL_KEY, id); } catch (_) {}
    }

    /**
     * Look up a panel-model entry by id.
     * @param {Array<object>} models
     * @param {string} id
     * @returns {object|null}
     */
    function _findModel(models, id) {
        if (!Array.isArray(models) || !id) return null;
        for (var i = 0; i < models.length; i++) {
            if (models[i].id === id) return models[i];
        }
        return null;
    }

    /**
     * Resolve the model object that should be used for the current turn.
     * Returns an object with the same {id, provider, model, endpoint, ...}
     * shape as a cfg.panelApiModels entry — or null if no panel-models are
     * configured (caller falls back to legacy single-model path).
     *
     * @param {object} cfg  window.AI_ASSISTANT_CONFIG
     * @returns {object|null}
     */
    function _getActiveModel(cfg) {
        if (!cfg || !Array.isArray(cfg.panelApiModels) ||
            cfg.panelApiModels.length === 0) return null;
        var id = _getActiveModelId(cfg.panelApiModels);
        return _findModel(cfg.panelApiModels, id);
    }

    // ── Phase C: Effort level, extended-thinking, and coming-soon features ──────

    /**
     * Ordered effort-level definitions displayed inside the model sheet.
     *
     * Each object carries:
     *   ``id``    – canonical storage key (lowercase, stable across releases).
     *   ``label`` – visible button text.
     *   ``hint``  – one-word sub-label beneath the button.
     *   ``desc``  – one-sentence description shown below the segmented control.
     *
     * Extending: append entries here and the builder loop handles them
     * automatically.  The grid column count is hard-coded to 4 in CSS; adding
     * a fifth entry requires updating ``grid-template-columns`` on
     * ``.ai-assistant-panel-effort-seg``.
     */
    var _EFFORT_LEVELS = [
        { id: 'low',    label: 'Low',    hint: 'Quick',
          desc: 'Fast, concise answers. Best for simple lookups and short questions.' },
        { id: 'medium', label: 'Medium', hint: 'Balanced',
          desc: 'Balanced quality and speed — the sweet spot for most tasks.' },
        { id: 'high',   label: 'High',   hint: 'Deep',
          desc: 'Thorough analysis. Best for research, writing, and code review.' },
        { id: 'max',    label: 'Max',    hint: 'Best',
          desc: 'Maximum reasoning quality. Slowest, but most complete and accurate.' },
    ];

    /**
     * Coming-soon feature placeholders shown in the model sheet footer.
     *
     * Each object carries:
     *   ``label`` – feature name (also used as lookup key in _FUTURE_ICONS).
     *   ``desc``  – one-sentence description.
     *
     * Extend by appending here.  Icons are resolved via ``_FUTURE_ICONS`` below.
     */
    var _FUTURE_FEATURES = [
        { label: 'Temperature',    desc: 'Creativity vs. precision dial' },
        { label: 'Context window', desc: 'Set max tokens for context' },
        { label: 'Tool routing',   desc: 'Enable or disable individual tools' },
        { label: 'System prompt',  desc: 'Per-session custom instructions' },
    ];

    /**
     * SVG icon map for coming-soon feature items.
     *
     * Each value must be safe inner-HTML (no user content, only hardcoded SVG
     * paths).  Unmapped labels fall back to ``ICONS.model`` so every item
     * always shows an icon.
     */
    var _FUTURE_ICONS = {
        'Temperature':
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"' +
            ' stroke-linecap="round" stroke-linejoin="round">' +
            '<circle cx="12" cy="12" r="3"/>' +
            '<path d="M12 1v4m0 14v4M4.22 4.22l2.83 2.83m9.9 9.9 2.83 2.83' +
            'M1 12h4m14 0h4M4.22 19.78l2.83-2.83m9.9-9.9 2.83-2.83"/></svg>',
        'Context window':
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"' +
            ' stroke-linecap="round" stroke-linejoin="round">' +
            '<path d="M8 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3' +
            'm8-18h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3m-4-9H9m6 4H9m6-8H9"/></svg>',
        'Tool routing':
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"' +
            ' stroke-linecap="round" stroke-linejoin="round">' +
            '<line x1="4" y1="6" x2="20" y2="6"/>' +
            '<line x1="4" y1="12" x2="20" y2="12"/>' +
            '<line x1="4" y1="18" x2="12" y2="18"/></svg>',
        'System prompt':
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"' +
            ' stroke-linecap="round" stroke-linejoin="round">' +
            '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>' +
            '<path d="m18.5 2.5 3 3-10 10H8v-3.5l10.5-9.5z"/></svg>',
    };

    // ── Effort: sessionStorage-backed persistence ──────────────────────────────

    /** sessionStorage key for the user's chosen effort level. */
    var _EFFORT_KEY = 'ai-assistant-effort-level';

    /**
     * Return the persisted effort level id, defaulting to ``'medium'``.
     * Falls back silently when sessionStorage is blocked.
     *
     * @returns {string}
     */
    function _getEffortLevel() {
        try { return sessionStorage.getItem(_EFFORT_KEY) || 'medium'; } catch (_) { return 'medium'; }
    }

    /**
     * Persist the chosen effort level.  Best-effort — failures are silent.
     *
     * @param {string} id  One of the ids defined in ``_EFFORT_LEVELS``.
     */
    function _setEffortLevel(id) {
        if (typeof id !== 'string' || !id) return;
        try { sessionStorage.setItem(_EFFORT_KEY, id); } catch (_) {}
    }

    // ── Thinking: sessionStorage-backed persistence ────────────────────────────

    /** sessionStorage key for the extended-reasoning toggle state. */
    var _THINKING_KEY = 'ai-assistant-thinking-on';

    /** sessionStorage key for the thinking token budget. */
    var _THINKING_BUDGET_KEY = 'ai-assistant-thinking-budget';

    /**
     * Return whether extended reasoning is currently enabled.
     *
     * @returns {boolean}
     */
    function _getThinkingOn() {
        try { return sessionStorage.getItem(_THINKING_KEY) === 'true'; } catch (_) { return false; }
    }

    /**
     * Persist the thinking toggle state.
     *
     * @param {boolean} on
     */
    function _setThinkingOn(on) {
        try { sessionStorage.setItem(_THINKING_KEY, on ? 'true' : 'false'); } catch (_) {}
    }

    /**
     * Return the persisted token budget, clamped to [500, 16000].
     * Defaults to 5000 when unset or out-of-range.
     *
     * @returns {number}
     */
    function _getThinkingBudget() {
        try {
            var v = parseInt(sessionStorage.getItem(_THINKING_BUDGET_KEY), 10);
            return (isFinite(v) && v >= 500 && v <= 16000) ? v : 5000;
        } catch (_) { return 5000; }
    }

    /**
     * Persist the thinking token budget.
     *
     * @param {number} v  Tokens; caller ensures value is within [500, 16000].
     */
    function _setThinkingBudget(v) {
        try { sessionStorage.setItem(_THINKING_BUDGET_KEY, String(v)); } catch (_) {}
    }

    // ── Sheet-section DOM helpers ──────────────────────────────────────────────

    /**
     * Build a labeled section divider for the model sheet.
     *
     * Renders as:
     *   LABEL ───────────────────
     *
     * The rule line is ``aria-hidden`` so screen readers skip it.
     *
     * Parameters
     * ----------
     * label : string
     *     Section heading text (uppercase by CSS, not by content).
     *
     * Returns
     * -------
     * HTMLElement
     *     A ``<div class="ai-assistant-panel-sheet-section">`` containing the
     *     label + rule row, ready for content to be appended by the caller.
     */
    function _buildSheetSection(label) {
        var section = document.createElement('div');
        section.className = 'ai-assistant-panel-sheet-section';

        var head = document.createElement('div');
        head.className = 'ai-assistant-panel-sheet-section-head';

        var lbl = document.createElement('span');
        lbl.className = 'ai-assistant-panel-sheet-section-label';
        lbl.textContent = label;

        var rule = document.createElement('span');
        rule.className = 'ai-assistant-panel-sheet-section-rule';
        rule.setAttribute('aria-hidden', 'true');

        head.appendChild(lbl);
        head.appendChild(rule);
        section.appendChild(head);
        return section;
    }

    /**
     * Append the effort, thinking, and coming-soon sections to a model sheet.
     *
     * This helper is called from both the normal and stub-mode (no models
     * configured) code paths inside ``_buildModelSheet()`` to guarantee the
     * sections appear regardless of whether any models are defined.
     *
     * Dispatches custom events on the document:
     *   ``ai-assistant-effort-change``         – when effort level changes.
     *   ``ai-assistant-thinking-change``       – when thinking toggle flips.
     *   ``ai-assistant-thinking-budget-change``– when token budget changes.
     *
     * All event details are plain objects safe for structured-clone.
     * Callers can react via ``document.addEventListener()``.
     *
     * Parameters
     * ----------
     * sheet : HTMLElement
     *     The model-sheet root element to append sections to.
     *
     * Notes
     * -----
     * Developer: CustomEvent is wrapped in try/catch so the function never
     *   throws in environments where CustomEvent is unavailable (old WebViews).
     *
     * Developer: sessionStorage access is always wrapped in try/catch because
     *   it may throw in Safari private mode, cross-origin iframes, and when
     *   storage quota is exceeded.
     */
    function _appendModelSheetSections(sheet) {

        // ── §A  Effort level ───────────────────────────────────────────────────

        var effortSection = _buildSheetSection('Effort');

        var effortSeg = document.createElement('div');
        effortSeg.className = 'ai-assistant-panel-effort-seg';
        effortSeg.setAttribute('role', 'radiogroup');
        effortSeg.setAttribute('aria-label', 'Response effort level');

        var effortDesc = document.createElement('p');
        effortDesc.className = 'ai-assistant-panel-effort-desc';

        var activeEffort = _getEffortLevel();

        _EFFORT_LEVELS.forEach(function (ef) {
            var btn = document.createElement('button');
            btn.className = 'ai-assistant-panel-effort-btn';
            btn.setAttribute('role', 'radio');
            btn.setAttribute('aria-checked', ef.id === activeEffort ? 'true' : 'false');
            btn.dataset.effortId = ef.id;
            btn.type = 'button';

            var efLbl = document.createElement('span');
            efLbl.className = 'ai-assistant-panel-effort-lbl';
            efLbl.textContent = ef.label;

            var efHint = document.createElement('span');
            efHint.className = 'ai-assistant-panel-effort-hint';
            efHint.textContent = ef.hint;
            efHint.setAttribute('aria-hidden', 'true');

            btn.appendChild(efLbl);
            btn.appendChild(efHint);

            // Set initial description for the pre-selected level.
            if (ef.id === activeEffort) { effortDesc.textContent = ef.desc; }

            btn.addEventListener('click', function () {
                activeEffort = ef.id;
                _setEffortLevel(ef.id);
                effortDesc.textContent = ef.desc;
                effortSeg.querySelectorAll('.ai-assistant-panel-effort-btn')
                    .forEach(function (b) {
                        b.setAttribute('aria-checked',
                            b.dataset.effortId === activeEffort ? 'true' : 'false');
                    });
                try {
                    document.dispatchEvent(new CustomEvent(
                        'ai-assistant-effort-change',
                        { detail: { id: ef.id }, bubbles: false }
                    ));
                } catch (_) {}
            });

            effortSeg.appendChild(btn);
        });

        effortSection.appendChild(effortSeg);
        effortSection.appendChild(effortDesc);
        sheet.appendChild(effortSection);

        // ── §B  Extended reasoning (thinking) ─────────────────────────────────

        var thinkingSection = _buildSheetSection('Thinking');

        var thinkingRow = document.createElement('div');
        thinkingRow.className = 'ai-assistant-panel-thinking-row';

        // Left side: title + hint
        var thinkingText = document.createElement('div');
        thinkingText.className = 'ai-assistant-panel-thinking-text';

        var thinkingTitle = document.createElement('div');
        thinkingTitle.className = 'ai-assistant-panel-thinking-title';
        thinkingTitle.textContent = 'Extended reasoning';

        var thinkingHint = document.createElement('div');
        thinkingHint.className = 'ai-assistant-panel-thinking-hint';

        var thinkingOn = _getThinkingOn();
        thinkingHint.textContent = thinkingOn
            ? 'Deeper analysis, slightly slower responses'
            : 'Faster, more concise responses';

        thinkingText.appendChild(thinkingTitle);
        thinkingText.appendChild(thinkingHint);

        // Right side: toggle pill
        var thinkingToggle = document.createElement('button');
        thinkingToggle.className = 'ai-assistant-panel-thinking-toggle';
        thinkingToggle.type = 'button';
        thinkingToggle.setAttribute('role', 'switch');
        thinkingToggle.setAttribute('aria-pressed', thinkingOn ? 'true' : 'false');
        thinkingToggle.setAttribute('aria-label', 'Enable extended reasoning');

        var thinkingThumb = document.createElement('span');
        thinkingThumb.className = 'ai-assistant-panel-thinking-toggle-thumb';
        thinkingThumb.setAttribute('aria-hidden', 'true');
        thinkingToggle.appendChild(thinkingThumb);

        thinkingRow.appendChild(thinkingText);
        thinkingRow.appendChild(thinkingToggle);

        // Token budget area (visible only when thinking is on)
        var budgetArea = document.createElement('div');
        budgetArea.className = 'ai-assistant-panel-budget-area';
        if (thinkingOn) { budgetArea.setAttribute('data-visible', 'true'); }

        var budgetHeader = document.createElement('div');
        budgetHeader.className = 'ai-assistant-panel-budget-header';

        var budgetLabel = document.createElement('span');
        budgetLabel.className = 'ai-assistant-panel-budget-label';
        budgetLabel.textContent = 'Token budget';

        var budgetValue = document.createElement('span');
        budgetValue.className = 'ai-assistant-panel-budget-value';
        var currentBudget = _getThinkingBudget();
        budgetValue.textContent = currentBudget.toLocaleString();

        budgetHeader.appendChild(budgetLabel);
        budgetHeader.appendChild(budgetValue);

        var budgetRange = document.createElement('input');
        budgetRange.type = 'range';
        budgetRange.className = 'ai-assistant-panel-budget-range';
        budgetRange.min = '500';
        budgetRange.max = '16000';
        budgetRange.step = '500';
        budgetRange.value = String(currentBudget);
        budgetRange.setAttribute('aria-label', 'Token budget for extended reasoning');

        budgetRange.addEventListener('input', function () {
            var v = parseInt(budgetRange.value, 10);
            budgetValue.textContent = v.toLocaleString();
            _setThinkingBudget(v);
            try {
                document.dispatchEvent(new CustomEvent(
                    'ai-assistant-thinking-budget-change',
                    { detail: { budget: v }, bubbles: false }
                ));
            } catch (_) {}
        });

        var budgetTicks = document.createElement('div');
        budgetTicks.className = 'ai-assistant-panel-budget-ticks';
        budgetTicks.setAttribute('aria-hidden', 'true');

        var tickMin = document.createElement('span');
        tickMin.className = 'ai-assistant-panel-budget-tick';
        tickMin.textContent = '500';
        var tickMax = document.createElement('span');
        tickMax.className = 'ai-assistant-panel-budget-tick';
        tickMax.textContent = '16 000';

        budgetTicks.appendChild(tickMin);
        budgetTicks.appendChild(tickMax);
        budgetArea.appendChild(budgetHeader);
        budgetArea.appendChild(budgetRange);
        budgetArea.appendChild(budgetTicks);

        // Wire the toggle: flip state, update UI, persist, dispatch event.
        thinkingToggle.addEventListener('click', function () {
            thinkingOn = !thinkingOn;
            _setThinkingOn(thinkingOn);
            thinkingToggle.setAttribute('aria-pressed', thinkingOn ? 'true' : 'false');
            thinkingHint.textContent = thinkingOn
                ? 'Deeper analysis, slightly slower responses'
                : 'Faster, more concise responses';
            budgetArea.setAttribute('data-visible', thinkingOn ? 'true' : 'false');
            try {
                document.dispatchEvent(new CustomEvent(
                    'ai-assistant-thinking-change',
                    { detail: { on: thinkingOn, budget: _getThinkingBudget() },
                      bubbles: false }
                ));
            } catch (_) {}
        });

        thinkingSection.appendChild(thinkingRow);
        thinkingSection.appendChild(budgetArea);
        sheet.appendChild(thinkingSection);

        // ── §C  Coming-soon feature placeholders ───────────────────────────────

        var futureSection = _buildSheetSection('Coming soon');

        var futureList = document.createElement('div');
        futureList.className = 'ai-assistant-panel-future-list';
        // Entire section is decorative — no interactive targets inside.
        futureList.setAttribute('aria-hidden', 'true');

        _FUTURE_FEATURES.forEach(function (f) {
            var item = document.createElement('div');
            item.className = 'ai-assistant-panel-future-item';

            var icon = document.createElement('span');
            icon.className = 'ai-assistant-panel-future-icon';
            // Only hardcoded SVG — no user content reaches innerHTML here.
            icon.innerHTML = _FUTURE_ICONS[f.label] || ICONS.model;

            var text = document.createElement('div');
            text.className = 'ai-assistant-panel-future-text';

            var name = document.createElement('div');
            name.className = 'ai-assistant-panel-future-name';
            name.textContent = f.label;       // textContent — XSS-safe.

            var sub = document.createElement('div');
            sub.className = 'ai-assistant-panel-future-sub';
            sub.textContent = f.desc;         // textContent — XSS-safe.

            text.appendChild(name);
            text.appendChild(sub);

            var badge = document.createElement('span');
            badge.className = 'ai-assistant-panel-future-badge';
            badge.textContent = 'Soon';

            item.appendChild(icon);
            item.appendChild(text);
            item.appendChild(badge);
            futureList.appendChild(item);
        });

        futureSection.appendChild(futureList);
        sheet.appendChild(futureSection);
    }

    // ── Phase B: Model selection sheet (sibling of privacy sheet) ─────────────

    /**
     * Build the model-selection slide-over.  Same pattern as _buildPrivacySheet
     * so the open/close animation, escape handling, and a11y semantics match.
     *
     * The sheet lists every entry of cfg.panelApiModels with:
     *   • a radio button (single-select),
     *   • a label (entry.label or entry.id),
     *   • the wire model name as a subtitle,
     *   • an optional one-line description,
     *   • an external-link icon to entry.info_url (if any).
     *
     * Selecting an entry persists the id and closes the sheet.  An
     * ``ai-assistant-model-change`` CustomEvent is dispatched on document
     * so doc-authors can react (e.g. show a toast).
     *
     * Stub-mode behaviour: when cfg.panelApiModels is empty the sheet still
     * builds and renders a "No models configured" notice — never throws.
     *
     * @returns {HTMLElement}
     */
    function _buildModelSheet() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var sheet = document.createElement('div');
        sheet.className = 'ai-assistant-panel-privacy ai-assistant-panel-model-sheet';
        sheet.id = 'ai-assistant-panel-model-sheet';
        sheet.setAttribute('data-open', 'false');
        // FIX Issue 5: dialog role + aria-modal so screen readers announce the
        // overlay context and restrict virtual cursor to sheet content.
        sheet.setAttribute('role', 'dialog');
        sheet.setAttribute('aria-modal', 'true');
        // aria-labelledby resolved by hStrong.id below.
        var _SHEET_TITLE_ID = 'ai-assistant-model-sheet-title';
        sheet.setAttribute('aria-labelledby', _SHEET_TITLE_ID);

        var head = document.createElement('div');
        head.className = 'ai-assistant-panel-privacy-head';
        var hStrong = document.createElement('strong');
        // FIX Issue 5: stable id so aria-labelledby resolves correctly.
        hStrong.id = _SHEET_TITLE_ID;
        hStrong.textContent = 'Choose a model';
        var hClose = _createIconBtn('model-close', 'Close model picker', ICONS.close);
        hClose.addEventListener('click', function () {
            sheet.setAttribute('data-open', 'false');
        });
        head.appendChild(hStrong);
        head.appendChild(hClose);
        sheet.appendChild(head);

        var bodyEl = document.createElement('div');
        bodyEl.className = 'ai-assistant-panel-privacy-body ai-assistant-panel-model-list';
        // FIX Issue 5: radiogroup role so AT announces single-select semantics.
        bodyEl.setAttribute('role', 'radiogroup');
        bodyEl.setAttribute('aria-label', 'Choose a model');

        var models = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
        if (models.length === 0) {
            var empty = document.createElement('p');
            empty.textContent =
                'No models are configured. Set ' +
                'ai_assistant_panel_api_models in conf.py to enable the picker.';
            bodyEl.appendChild(empty);
            // Wrap in the unified scroll container even for the stub case so
            // that effort / thinking / future sections are always inside the
            // scrollable region and never overflow the sheet frame.
            var scrollElEmpty = document.createElement('div');
            scrollElEmpty.className = 'ai-assistant-panel-sheet-scroll';
            scrollElEmpty.appendChild(bodyEl);
            sheet.appendChild(scrollElEmpty);
            _appendModelSheetSections(scrollElEmpty);
            return sheet;
        }

        var activeId = _getActiveModelId(models);
        // FIX Issue 6: deterministic constant — Math.random() produced a
        // different name on every build, breaking external correlation and
        // making DevTools output unpredictable.
        var groupName = _MODEL_RADIO_GROUP;

        models.forEach(function (m) {
            var row = document.createElement('label');
            row.className = 'ai-assistant-panel-model-row';
            row.setAttribute('data-id', m.id);
            row.setAttribute('data-provider', m.provider || 'custom');

            var radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = groupName;
            radio.value = m.id;
            radio.checked = (m.id === activeId);
            radio.className = 'ai-assistant-panel-model-radio';
            // FIX Issue 15: data-checked mirrors radio.checked at build time so
            // the :has() fallback (Safari < 15.4, Chrome < 105) shows the
            // selected-row highlight without requiring a DOM change event.
            if (m.id === activeId) {
                row.setAttribute('data-checked', 'true');
            }

            // ── Provider badge (coloured circle) ────────────────────────
            var badge = document.createElement('span');
            badge.className = 'ai-assistant-panel-model-badge';
            badge.setAttribute('aria-hidden', 'true');
            badge.title = m.provider || '';
            var bColor = _providerColor(m.provider || '');
            if (bColor) badge.style.background = bColor;

            var textWrap = document.createElement('div');
            textWrap.className = 'ai-assistant-panel-model-text';

            var title = document.createElement('div');
            title.className = 'ai-assistant-panel-model-title';
            title.textContent = (m.label || m.id);

            var sub = document.createElement('div');
            sub.className = 'ai-assistant-panel-model-sub';
            // ``provider · model-wire-name`` — textContent only, no innerHTML.
            sub.textContent = (m.provider || '') +
                (m.model && m.model !== m.id ? ' \u00B7 ' + m.model : '');

            textWrap.appendChild(title);
            textWrap.appendChild(sub);

            if (m.description) {
                var desc = document.createElement('div');
                desc.className = 'ai-assistant-panel-model-desc';
                desc.textContent = m.description;
                textWrap.appendChild(desc);
            }

            row.appendChild(radio);
            row.appendChild(badge);
            row.appendChild(textWrap);

            if (m.info_url && _isSafeHref(m.info_url)) {
                // Public info page link (e.g. anthropic.com/claude).
                // Defense-in-depth: _isSafeHref() validates the scheme client-side
                // even when the Python-side filter was bypassed (CDN injection,
                // multi-tenant embed, tampered config).
                var info = document.createElement('a');
                info.className = 'ai-assistant-panel-model-info';
                info.href = m.info_url;
                info.target = '_blank';
                info.rel = 'noopener noreferrer';
                info.setAttribute('aria-label',
                    'Open model info page for ' + (m.label || m.id));
                info.title = 'Open model info page';
                info.innerHTML = ICONS.info;     // ICONS constant — safe.
                row.appendChild(info);
            }

            row.addEventListener('change', function () {
                if (!radio.checked) return;
                // Capture the per-row id once — stable regardless of config changes.
                var id = m.id;
                _setActiveModelId(id);
                // FIX Issue 3: Read the live config at dispatch time so the
                // CustomEvent payload is never stale.  window.AI_ASSISTANT_CONFIG
                // may be updated post-DOMContentLoaded in hot-reload and SPA
                // environments; the build-time `m` snapshot would then carry wrong
                // provider / model values into analytics handlers.
                try {
                    var liveModels = (window.AI_ASSISTANT_CONFIG || {}).panelApiModels;
                    var liveM = _findModel(
                        Array.isArray(liveModels) ? liveModels : models, id
                    );
                    document.dispatchEvent(new CustomEvent(
                        'ai-assistant-model-change',
                        { detail: liveM
                            ? { id: liveM.id, provider: liveM.provider,
                                model: liveM.model }
                            : { id: id } }
                    ));
                } catch (_) {}
                // FIX Issue 15: keep data-checked in sync so the :has() fallback
                // (Issue 15 CSS) correctly highlights the newly selected row.
                sheet.querySelectorAll('.ai-assistant-panel-model-row[data-checked]')
                    .forEach(function (r) { r.removeAttribute('data-checked'); });
                row.setAttribute('data-checked', 'true');
                // Sync inline picker if present.
                _syncInlinePickers(id);
                // Sheet closure is handled by the row click listener wired in
                // createAIPanel after _closeSheet is in scope.  Using that path
                // ensures aria-expanded is reset and focus returns to the opener
                // for every close path (different model, same model, × button,
                // Escape) — Issue 4 parity for model row selection.
            });

            bodyEl.appendChild(row);
        });

        // ── Unified scroll wrapper ─────────────────────────────────────────
        // All model rows + effort/thinking/future sections are placed inside
        // a single scrollEl so the entire sheet body scrolls together.
        // CSS: .ai-assistant-panel-sheet-scroll  (flex:1; overflow-y:auto)
        // The bodyEl's own overflow is overridden to `visible` by the
        // .ai-assistant-panel-privacy-body.ai-assistant-panel-model-list rule.
        var scrollEl = document.createElement('div');
        scrollEl.className = 'ai-assistant-panel-sheet-scroll';
        scrollEl.appendChild(bodyEl);
        _attachModelFilter(scrollEl, bodyEl, models);
        sheet.appendChild(scrollEl);
        _appendModelSheetSections(scrollEl);
        return sheet;
    }

    /**
     * Attach a filter bar and pagination controls to the model-sheet scroll wrapper.
     *
     * Injected DOM structure inside scrollEl after this call:
     *
     *   scrollEl
     *     filterBar   ← inserted BEFORE bodyEl (search + scope chips + meta row)
     *     bodyEl      ← model-row labels (never removed; only display toggled)
     *     _pagBar     ← inserted AFTER bodyEl (page buttons; hidden when not needed)
     *     … effort / thinking / future sections (added later)
     *
     * Invariants
     * ----------
     * • The currently-checked radio row is NEVER hidden — it stays visible with a
     *   "selected" badge when it falls outside the current filter result set.
     * • ARIA radiogroup semantics are preserved: rows remain in the DOM; only
     *   their CSS display property is toggled (display:flex / display:none).
     * • Filter state (query, scope, sort, page) persists while the sheet is open.
     * • Calling this function more than once on the same scrollEl is a safe no-op
     *   (guarded by dataset.filterAttached).
     *
     * Parameters
     * ----------
     * scrollEl : HTMLElement
     *     The .ai-assistant-panel-sheet-scroll wrapper element.
     * bodyEl : HTMLElement
     *     The .ai-assistant-panel-model-list radiogroup container.
     * models : Array
     *     The panelApiModels config array passed to _buildModelSheet.
     *
     * Notes
     * -----
     * Developer: Filter state resets only via explicit user action (clear button,
     *   Escape key).  It intentionally persists across sheet open/close cycles so
     *   the user returns to their last search context.
     * Developer: _safeInt() is used (already defined in IIFE scope) to bound the
     *   configurable page size.
     */
    function _attachModelFilter(scrollEl, bodyEl, models) {

        // ── Guard: idempotent — safe to call more than once ───────────────────
        if (scrollEl.dataset.filterAttached === 'true') return;

        // ── Threshold: only attach when there are enough models to warrant it ─
        // Configurable: set panelFilterThreshold in conf.py (default 2).
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var THRESHOLD = _safeInt(cfg.panelFilterThreshold, 1, 9999, 2);
        if (!models || models.length < THRESHOLD) return;

        scrollEl.dataset.filterAttached = 'true';

        // Page size: configurable via conf.py panelFilterPageSize; bounded 1–200.
        // Developer note: values > 200 are clamped to 200 to prevent unresponsive
        // rendering on very large model lists.
        var PAGE_SIZE = _safeInt(cfg.panelFilterPageSize, 1, 200, 10);

        // ── Filter state ──────────────────────────────────────────────────────
        var _query = '';        // lowercase trimmed search string; '' = no filter
        var _scope = 'all';     // which field to search: 'all'|'title'|'provider'|'desc'|'id'
        var _sort  = 'default'; // sort order: 'default'|'az'|'za'|'provider'
        var _page  = 0;         // 0-indexed current page

        // ── Snapshot rows from DOM once at init ───────────────────────────────
        // Rows are never added/removed by this module; only display is toggled.
        var _rows = Array.prototype.slice.call(
            bodyEl.querySelectorAll('.ai-assistant-panel-model-row')
        );
        if (_rows.length === 0) return;

        // Pre-extract lowercase text per row (O(n) once; avoids repeated DOM reads).
        var _rowData = _rows.map(function (row, i) {
            var titleEl = row.querySelector('.ai-assistant-panel-model-title');
            var subEl   = row.querySelector('.ai-assistant-panel-model-sub');
            var descEl  = row.querySelector('.ai-assistant-panel-model-desc');
            var id      = (row.getAttribute('data-id') || '').toLowerCase();
            var title   = titleEl ? titleEl.textContent.toLowerCase() : '';
            var sub     = subEl   ? subEl.textContent.toLowerCase()   : '';
            var desc    = descEl  ? descEl.textContent.toLowerCase()  : '';
            return {
                row:     row,
                id:      id,
                title:   title,
                sub:     sub,
                desc:    desc,
                all:     title + ' ' + sub + ' ' + desc + ' ' + id,
                origIdx: i   // stable original order for 'Default' sort
            };
        });

        // ── SVG constants (inline; no external file dependency; not user input) ─
        var _SVG_SEARCH =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
            ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"' +
            ' aria-hidden="true"><circle cx="11" cy="11" r="8"/>' +
            '<line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>';

        var _SVG_CLOSE =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
            ' stroke-width="2" stroke-linecap="round" aria-hidden="true">' +
            '<line x1="18" y1="6" x2="6" y2="18"/>' +
            '<line x1="6" y1="6" x2="18" y2="18"/></svg>';

        var _SVG_PREV =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
            ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"' +
            ' aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>';

        var _SVG_NEXT =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
            ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"' +
            ' aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>';

        // ── Empty state element (appended to bodyEl; shown when 0 results) ────
        // All user-visible text is set via textContent — never innerHTML.
        var _emptyEl = document.createElement('div');
        _emptyEl.className = 'ai-assistant-panel-filter-empty';
        _emptyEl.style.display = 'none';

        var _emptyIcon = document.createElement('span');
        _emptyIcon.className = 'ai-assistant-panel-filter-empty-icon';
        _emptyIcon.setAttribute('aria-hidden', 'true');
        _emptyIcon.innerHTML = _SVG_SEARCH;   // safe: static SVG constant

        var _emptyMsg = document.createElement('p');

        var _emptyClearBtn = document.createElement('button');
        _emptyClearBtn.type = 'button';
        _emptyClearBtn.className = 'ai-assistant-panel-filter-clear-all';
        _emptyClearBtn.textContent = 'Clear filter';
        _emptyClearBtn.addEventListener('click', function () { _clearFilter(); });

        _emptyEl.appendChild(_emptyIcon);
        _emptyEl.appendChild(_emptyMsg);
        _emptyEl.appendChild(_emptyClearBtn);
        bodyEl.appendChild(_emptyEl);

        // ── Outside-filter badge: added to the checked row when it is filtered out ─
        function _ensureOutsideBadge(row) {
            if (row.querySelector('.ai-assistant-panel-filter-outside-badge')) return;
            var badge = document.createElement('span');
            badge.className = 'ai-assistant-panel-filter-outside-badge';
            badge.textContent = 'selected';
            badge.setAttribute('aria-label', 'currently selected — not in current filter');
            row.appendChild(badge);
        }
        function _removeOutsideBadge(row) {
            var b = row.querySelector('.ai-assistant-panel-filter-outside-badge');
            if (b) b.parentNode.removeChild(b);
        }

        // ── Filter helpers ────────────────────────────────────────────────────
        function _getSearchField(d) {
            switch (_scope) {
                case 'title':    return d.title;
                case 'provider': return d.sub;
                case 'desc':     return d.desc;
                case 'id':       return d.id;
                default:         return d.all;
            }
        }

        function _matchesQuery(d) {
            if (!_query) return true;
            return _getSearchField(d).indexOf(_query) !== -1;
        }

        function _applySort(items) {
            var copy = items.slice();
            if (_sort === 'az') {
                copy.sort(function (a, b) {
                    return a.title.localeCompare(b.title);
                });
            } else if (_sort === 'za') {
                copy.sort(function (a, b) {
                    return b.title.localeCompare(a.title);
                });
            } else if (_sort === 'provider') {
                copy.sort(function (a, b) {
                    return a.sub.localeCompare(b.sub);
                });
            } else {
                // Default: restore original config order.
                copy.sort(function (a, b) {
                    return a.origIdx - b.origIdx;
                });
            }
            return copy;
        }

        // ── Smart page-range builder ──────────────────────────────────────────
        //
        // Produces a mixed array of page indices (numbers) and '...' sentinels
        // for display in the pagination row.  Always shows: first page, last page,
        // current page, and one page either side of current.  No Set/Array.from
        // dependency — compatible with the ES5-syntax style of this IIFE.
        //
        // Examples (0-indexed internally, 1-indexed in UI):
        //   _buildPageRange(0, 3)  → [0, 1, 2]
        //   _buildPageRange(5, 10) → [0, '...', 4, 5, 6, '...', 9]
        //   _buildPageRange(1, 10) → [0, 1, 2, '...', 9]
        function _buildPageRange(current, total) {
            if (total <= 7) {
                var all = [];
                for (var i = 0; i < total; i++) all.push(i);
                return all;
            }
            // Collect unique "always-show" page indices into a plain object.
            var shown = {};
            shown[0] = true;
            shown[total - 1] = true;
            shown[current] = true;
            if (current - 1 >= 0)    shown[current - 1] = true;
            if (current + 1 < total) shown[current + 1] = true;

            var keys = [];
            var k;
            for (k in shown) {
                if (Object.prototype.hasOwnProperty.call(shown, k)) {
                    keys.push(parseInt(k, 10));
                }
            }
            keys.sort(function (a, b) { return a - b; });

            var result = [];
            for (var j = 0; j < keys.length; j++) {
                if (j > 0 && keys[j] - keys[j - 1] > 1) result.push('...');
                result.push(keys[j]);
            }
            return result;
        }

        // ── Debounce utility ──────────────────────────────────────────────────
        function _debounce(fn, ms) {
            var timer = null;
            return function () {
                var ctx = this, args = arguments;
                clearTimeout(timer);
                timer = setTimeout(function () { fn.apply(ctx, args); }, ms);
            };
        }

        // ── Clear all filter state ────────────────────────────────────────────
        function _clearFilter() {
            _query = '';
            _scope = 'all';
            _sort  = 'default';
            _page  = 0;
            _input.value = '';
            _updateChips();
            _updateSortLabel();
            _render();
            _input.focus();
        }

        // ── Sync chip active states to _scope ─────────────────────────────────
        function _updateChips() {
            _chips.forEach(function (chip) {
                var active = chip.getAttribute('data-scope') === _scope;
                chip.setAttribute('aria-pressed', active ? 'true' : 'false');
                chip.classList.toggle('ai-assistant-panel-filter-chip--active', active);
            });
        }

        // ── Sync sort button label to _sort ───────────────────────────────────
        var _SORT_LABELS = {
            'default':  'Sort',
            'az':       'A \u2192 Z',
            'za':       'Z \u2192 A',
            'provider': 'Provider'
        };

        function _updateSortLabel() {
            _sortLabelSpan.textContent = _SORT_LABELS[_sort] || 'Sort';
            _sortBtn.classList.toggle(
                'ai-assistant-panel-filter-sort--active',
                _sort !== 'default'
            );
        }

        // ── Main render ───────────────────────────────────────────────────────
        //
        // Pure display pass: derives visibility from state, never mutates state.
        // Called after every state change (query, scope, sort, page).
        function _render() {
            // 1. Find currently checked row — must always remain visible.
            var checkedRow = null;
            var ci;
            for (ci = 0; ci < _rows.length; ci++) {
                var radio = _rows[ci].querySelector('input[type="radio"]');
                if (radio && radio.checked) { checkedRow = _rows[ci]; break; }
            }

            // 2. Filter: apply query against selected scope field.
            var filtered = _rowData.filter(_matchesQuery);

            // 3. Sort filtered results.
            var sorted = _applySort(filtered);

            // 4. Clamp page index within valid range (handles filter narrowing pages).
            var totalPages = sorted.length > 0
                ? Math.ceil(sorted.length / PAGE_SIZE)
                : 1;
            if (_page >= totalPages) _page = totalPages - 1;
            if (_page < 0) _page = 0;

            // 5. Compute current page slice.
            var start = _page * PAGE_SIZE;
            var pageSlice = sorted.slice(start, start + PAGE_SIZE);

            // Build O(1) page-membership lookup keyed on origIdx.
            var inSlice = {};
            pageSlice.forEach(function (d) { inSlice[d.origIdx] = true; });

            // 6. Apply visibility to every row.
            _rowData.forEach(function (d) {
                var onPage    = !!inSlice[d.origIdx];
                var isChecked = (d.row === checkedRow);
                var show      = onPage || isChecked;

                d.row.style.display = show ? 'flex' : 'none';

                if (isChecked && !onPage) {
                    _ensureOutsideBadge(d.row);
                } else {
                    _removeOutsideBadge(d.row);
                }
            });

            // 6b. Reorder visible rows in the DOM to match the sorted page
            // order.  When all rows fit on a single page every row is visible
            // regardless of sort, so only toggling display has no effect —
            // the browser keeps the original insertion order.
            // bodyEl.appendChild on an already-attached node MOVES it (no
            // clone, no removal event), making this O(n) and allocation-free.
            // Hidden rows are left in place; their position is irrelevant.
            pageSlice.forEach(function (d) {
                bodyEl.appendChild(d.row);
            });
            // If the checked row is outside the current page slice it was
            // rendered at the top of the loop above — keep it visually
            // anchored before the page slice by prepending it now.
            if (checkedRow && !inSlice[_rowData.filter(function (d) {
                return d.row === checkedRow;
            })[0].origIdx]) {
                bodyEl.insertBefore(checkedRow, bodyEl.querySelector(
                    '.ai-assistant-panel-model-row[style*="flex"]'
                ) || _emptyEl);
            }

            // 7. Empty state — shown when the query matches 0 rows.
            if (sorted.length === 0) {
                _emptyEl.style.display = 'flex';
                // textContent is safe for user-supplied _query.
                _emptyMsg.textContent = _query
                    ? 'No models match \u201C' + _query + '\u201D'
                    : 'No models in this view';
            } else {
                _emptyEl.style.display = 'none';
            }

            // 8. Update ARIA-live meta line (announced by screen readers).
            _updateMeta(sorted.length, totalPages);

            // 9. Update pagination bar.
            _renderPagination(totalPages, sorted.length);

            // 10. Toggle clear (×) button — only visible when there is a query.
            _clearBtn.style.display = _query ? 'flex' : 'none';
        }

        function _updateMeta(totalFiltered, totalPages) {
            var allCount = _rowData.length;
            var text;
            if (_query || _sort !== 'default') {
                text = totalFiltered + ' of ' + allCount +
                    ' model' + (allCount !== 1 ? 's' : '');
            } else {
                text = allCount + ' model' + (allCount !== 1 ? 's' : '');
            }
            _countEl.textContent = text;

            if (totalFiltered > PAGE_SIZE && totalPages > 1) {
                _pageInfoEl.textContent =
                    '\u00B7 Page ' + (_page + 1) + ' of ' + totalPages;
                _pageInfoEl.style.display = '';
            } else {
                _pageInfoEl.style.display = 'none';
            }
        }

        function _renderPagination(totalPages, totalFiltered) {
            // innerHTML cleared — rebuilt from static constants only (no user data).
            _pagBar.innerHTML = '';
            if (totalPages <= 1 || totalFiltered <= PAGE_SIZE) {
                _pagBar.style.display = 'none';
                return;
            }
            _pagBar.style.display = 'flex';

            // Previous button
            var prevBtn = document.createElement('button');
            prevBtn.type = 'button';
            prevBtn.className = 'ai-assistant-panel-filter-pg-btn';
            prevBtn.setAttribute('aria-label', 'Previous page');
            prevBtn.disabled = (_page === 0);
            prevBtn.innerHTML = _SVG_PREV;   // safe: static constant
            prevBtn.addEventListener('click', function () {
                if (_page > 0) { _page--; _render(); }
            });
            _pagBar.appendChild(prevBtn);

            // Page number buttons with smart ellipsis
            var numsWrap = document.createElement('div');
            numsWrap.className = 'ai-assistant-panel-filter-pg-numbers';

            var pageRange = _buildPageRange(_page, totalPages);
            pageRange.forEach(function (p) {
                if (p === '...') {
                    var ellipsis = document.createElement('span');
                    ellipsis.className = 'ai-assistant-panel-filter-pg-ellipsis';
                    ellipsis.textContent = '\u2026';
                    ellipsis.setAttribute('aria-hidden', 'true');
                    numsWrap.appendChild(ellipsis);
                } else {
                    var pgBtn = document.createElement('button');
                    pgBtn.type = 'button';
                    var isActive = (p === _page);
                    pgBtn.className = 'ai-assistant-panel-filter-pg-num' +
                        (isActive ? ' ai-assistant-panel-filter-pg-num--active' : '');
                    // textContent: p is a number from _buildPageRange (not user input).
                    pgBtn.textContent = String(p + 1);  // 1-indexed for display
                    pgBtn.setAttribute('aria-label', 'Page ' + (p + 1));
                    pgBtn.setAttribute('aria-current', isActive ? 'page' : 'false');
                    (function (pageIdx) {
                        pgBtn.addEventListener('click', function () {
                            if (_page !== pageIdx) { _page = pageIdx; _render(); }
                        });
                    }(p));
                    numsWrap.appendChild(pgBtn);
                }
            });
            _pagBar.appendChild(numsWrap);

            // Next button
            var nextBtn = document.createElement('button');
            nextBtn.type = 'button';
            nextBtn.className = 'ai-assistant-panel-filter-pg-btn';
            nextBtn.setAttribute('aria-label', 'Next page');
            nextBtn.disabled = (_page === totalPages - 1);
            nextBtn.innerHTML = _SVG_NEXT;   // safe: static constant
            nextBtn.addEventListener('click', function () {
                if (_page < totalPages - 1) { _page++; _render(); }
            });
            _pagBar.appendChild(nextBtn);
        }

        // ── Build the filter bar DOM ──────────────────────────────────────────

        var filterBar = document.createElement('div');
        filterBar.className = 'ai-assistant-panel-filter-bar';
        filterBar.setAttribute('role', 'search');
        filterBar.setAttribute('aria-label', 'Filter models');

        // Row 1 ── search input + sort button ─────────────────────────────────
        var row1 = document.createElement('div');
        row1.className = 'ai-assistant-panel-filter-row';

        // Search wrap: icon + input + clear button
        var searchWrap = document.createElement('div');
        searchWrap.className = 'ai-assistant-panel-filter-search-wrap';

        var searchIconEl = document.createElement('span');
        searchIconEl.className = 'ai-assistant-panel-filter-search-icon';
        searchIconEl.setAttribute('aria-hidden', 'true');
        searchIconEl.innerHTML = _SVG_SEARCH;   // safe: static constant

        var _input = document.createElement('input');
        _input.type = 'search';
        _input.className = 'ai-assistant-panel-filter-input';
        _input.placeholder = 'Search models\u2026';
        _input.setAttribute('aria-label', 'Search models');
        _input.setAttribute('autocomplete', 'off');
        _input.setAttribute('spellcheck', 'false');

        var _clearBtn = document.createElement('button');
        _clearBtn.type = 'button';
        _clearBtn.className = 'ai-assistant-panel-filter-clear';
        _clearBtn.setAttribute('aria-label', 'Clear search');
        _clearBtn.title = 'Clear search';
        _clearBtn.style.display = 'none';
        _clearBtn.innerHTML = _SVG_CLOSE;   // safe: static constant
        _clearBtn.addEventListener('click', function () { _clearFilter(); });

        searchWrap.appendChild(searchIconEl);
        searchWrap.appendChild(_input);
        searchWrap.appendChild(_clearBtn);

        // Sort wrap: button + dropdown menu
        var sortWrap = document.createElement('div');
        sortWrap.className = 'ai-assistant-panel-filter-sort-wrap';

        var _sortBtn = document.createElement('button');
        _sortBtn.type = 'button';
        _sortBtn.className = 'ai-assistant-panel-filter-sort';
        _sortBtn.setAttribute('aria-label', 'Sort models');
        _sortBtn.setAttribute('aria-expanded', 'false');
        _sortBtn.setAttribute('aria-haspopup', 'listbox');

        var _sortLabelSpan = document.createElement('span');
        _sortLabelSpan.textContent = 'Sort';

        var sortChevron = document.createElement('span');
        sortChevron.className = 'ai-assistant-panel-filter-sort-chevron';
        sortChevron.setAttribute('aria-hidden', 'true');
        sortChevron.innerHTML =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
            ' stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">' +
            '<polyline points="6 9 12 15 18 9"/></svg>';

        _sortBtn.appendChild(_sortLabelSpan);
        _sortBtn.appendChild(sortChevron);

        var _sortMenu = document.createElement('div');
        _sortMenu.className = 'ai-assistant-panel-filter-sort-menu';
        _sortMenu.setAttribute('role', 'listbox');
        _sortMenu.setAttribute('aria-label', 'Sort order');
        _sortMenu.style.display = 'none';

        var _SORT_OPTIONS = [
            { id: 'default',  label: 'Default order'  },
            { id: 'az',       label: 'Name A \u2192 Z' },
            { id: 'za',       label: 'Name Z \u2192 A' },
            { id: 'provider', label: 'By provider'     }
        ];

        _SORT_OPTIONS.forEach(function (opt) {
            var item = document.createElement('button');
            item.type = 'button';
            item.className = 'ai-assistant-panel-filter-sort-item';
            item.setAttribute('role', 'option');
            item.setAttribute('data-sort-id', opt.id);
            item.textContent = opt.label;   // safe: static constant
            (function (sortId) {
                item.addEventListener('click', function () {
                    _sort = sortId;
                    _page = 0;
                    _updateSortLabel();
                    _closeSortMenu();
                    _render();
                });
            }(opt.id));
            _sortMenu.appendChild(item);
        });

        // Sort menu open / close helpers
        var _sortMenuOpen = false;

        function _openSortMenu() {
            _sortMenuOpen = true;
            _sortMenu.style.display = 'block';
            _sortBtn.setAttribute('aria-expanded', 'true');
        }
        function _closeSortMenu() {
            _sortMenuOpen = false;
            _sortMenu.style.display = 'none';
            _sortBtn.setAttribute('aria-expanded', 'false');
        }

        _sortBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            _sortMenuOpen ? _closeSortMenu() : _openSortMenu();
        });
        _sortBtn.addEventListener('keydown', function (e) {
            if ((e.key === 'Escape' || e.keyCode === 27) && _sortMenuOpen) {
                e.stopPropagation();
                _closeSortMenu();
            }
        });
        // Prevent menu-item clicks from bubbling to document (which closes the menu).
        _sortMenu.addEventListener('click', function (e) { e.stopPropagation(); });

        sortWrap.appendChild(_sortBtn);
        sortWrap.appendChild(_sortMenu);

        row1.appendChild(searchWrap);
        row1.appendChild(sortWrap);

        // Row 2 ── scope chips ─────────────────────────────────────────────────
        var row2 = document.createElement('div');
        row2.className = 'ai-assistant-panel-filter-chips-row';
        row2.setAttribute('role', 'group');
        row2.setAttribute('aria-label', 'Search scope');

        var _CHIP_DEFS = [
            { scope: 'all',      label: 'All'         },
            { scope: 'title',    label: 'Name'        },
            { scope: 'provider', label: 'Provider'    },
            { scope: 'desc',     label: 'Description' },
            { scope: 'id',       label: 'ID'          }
        ];

        var _chips = [];
        _CHIP_DEFS.forEach(function (def) {
            var chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'ai-assistant-panel-filter-chip';
            chip.setAttribute('data-scope', def.scope);
            chip.setAttribute('aria-pressed', def.scope === 'all' ? 'true' : 'false');
            if (def.scope === 'all') {
                chip.classList.add('ai-assistant-panel-filter-chip--active');
            }
            chip.textContent = def.label;   // safe: static constant
            (function (scopeId) {
                chip.addEventListener('click', function () {
                    _scope = scopeId;
                    _page  = 0;
                    _updateChips();
                    _render();
                });
            }(def.scope));
            row2.appendChild(chip);
            _chips.push(chip);
        });

        // Row 3 ── meta: count + page info (ARIA live region) ─────────────────
        var row3 = document.createElement('div');
        row3.className = 'ai-assistant-panel-filter-meta';
        row3.setAttribute('aria-live', 'polite');
        row3.setAttribute('aria-atomic', 'true');

        var _countEl    = document.createElement('span');
        _countEl.className = 'ai-assistant-panel-filter-count';

        var _pageInfoEl = document.createElement('span');
        _pageInfoEl.className = 'ai-assistant-panel-filter-page-info';
        _pageInfoEl.style.display = 'none';

        row3.appendChild(_countEl);
        row3.appendChild(_pageInfoEl);

        filterBar.appendChild(row1);
        filterBar.appendChild(row2);
        filterBar.appendChild(row3);

        // ── Pagination bar (inserted after bodyEl, before effort sections) ─────
        var _pagBar = document.createElement('div');
        _pagBar.className = 'ai-assistant-panel-filter-pagination';
        _pagBar.style.display = 'none';

        // ── Wire input events ─────────────────────────────────────────────────
        var _debouncedRender = _debounce(function () {
            _page = 0;
            _render();
        }, 300);

        _input.addEventListener('input', function () {
            _query = _input.value.trim().toLowerCase();
            _debouncedRender();
        });

        // Escape key: clear search when a query is active.
        _input.addEventListener('keydown', function (e) {
            if ((e.key === 'Escape' || e.keyCode === 27) && _query) {
                e.stopPropagation();
                _clearFilter();
            }
        });

        // Close sort menu on any outside click (global listener).
        document.addEventListener('click', function () { _closeSortMenu(); });

        // ── Badge refresh on model selection ─────────────────────────────────
        // The row `change` listener in _buildModelSheet cannot reach _render()
        // (different closure scope).  A delegated listener on bodyEl handles
        // every radio `change` that bubbles up and re-renders the badge state
        // so the outside-badge moves to (or off) the newly selected row
        // immediately — without waiting for the next filter / sort / page event.
        bodyEl.addEventListener('change', function (e) {
            if (e.target && e.target.type === 'radio') {
                _render();
            }
        });

        // ── Inject into DOM ───────────────────────────────────────────────────
        // filterBar goes BEFORE bodyEl inside scrollEl.
        scrollEl.insertBefore(filterBar, bodyEl);
        // _pagBar goes AFTER bodyEl (bodyEl.nextSibling is null at this call site
        // because _appendModelSheetSections has not run yet).
        if (bodyEl.nextSibling) {
            scrollEl.insertBefore(_pagBar, bodyEl.nextSibling);
        } else {
            scrollEl.appendChild(_pagBar);
        }

        // ── Initial render ────────────────────────────────────────────────────
        _render();
    }

    /**
     * Update all inline model pickers in the DOM to reflect a new active id.
     * Called whenever the model changes via the sheet so the inline picker
     * stays in sync (and vice-versa via _buildInlineModelPicker).
     *
     * Supports both the button variant (btn._syncState) and the legacy
     * <select> variant (p.value) for backwards compatibility.
     *
     * @param {string} id
     */
    function _syncInlinePickers(id) {
        var pickers = document.querySelectorAll('.ai-assistant-panel-inline-model-picker');
        pickers.forEach(function (p) {
            // Button variant: update label, dot, and aria-label via stored sync fn.
            if (typeof p._syncState === 'function') {
                p._syncState(id);
            }
            // Legacy <select> variant — keep for backward compat if any remain.
            else if (p.tagName === 'SELECT' && p.value !== id) {
                p.value = id;
            }
        });
    }

    /**
     * Update the model-sheet radio buttons to reflect a new active id.
     * Called whenever the model changes via the inline picker so the
     * sheet stays in sync — the symmetric counterpart to _syncInlinePickers.
     *
     * Also maintains the ``data-checked`` attribute used as the :has()
     * fallback for legacy browsers (Safari < 15.4, Chrome < 105).
     *
     * The sheet may not exist yet when this is called (lazy-built on first
     * open), so the querySelector is intentionally deferred to call time and
     * silently no-ops when the sheet is absent.
     *
     * @param {string} id  Model id that should be checked.
     */
    function _syncModelSheet(id) {
        var sheet = document.getElementById('ai-assistant-panel-model-sheet');
        if (!sheet) return;
        /* Query only within the sheet so unrelated radios elsewhere on the
           page are never accidentally touched. */
        var radios = sheet.querySelectorAll('input[type="radio"]');
        radios.forEach(function (r) {
            if (r.value === id && !r.checked) {
                r.checked = true;
            }
        });
        // Keep data-checked attribute in sync for :has() fallback.
        sheet.querySelectorAll('.ai-assistant-panel-model-row[data-checked]').forEach(
            function (r) { r.removeAttribute('data-checked'); }
        );
        var activeRow = sheet.querySelector(
            '.ai-assistant-panel-model-row[data-id="' + id + '"]'
        );
        if (activeRow) activeRow.setAttribute('data-checked', 'true');
    }

    // ── Phase B: Terms of Service sheet (sibling of privacy sheet) ────────────

    /**
     * Build the Terms-of-Service slide-over.  Pattern-equivalent to
     * _buildPrivacySheet — they share CSS classes and behaviour, so the
     * theme styling of one automatically applies to the other.  Author may
     * override the entire body via cfg.panelTermsHtml (trusted, from conf.py).
     *
     * @returns {HTMLElement}
     */
    function _buildTermsSheet() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var title = (typeof cfg.panelTermsTitle === 'string' &&
            cfg.panelTermsTitle) || 'Terms of Service';

        var sheet = document.createElement('div');
        sheet.className = 'ai-assistant-panel-privacy ai-assistant-panel-terms';
        sheet.id = 'ai-assistant-panel-terms';
        sheet.setAttribute('data-open', 'false');

        var head = document.createElement('div');
        head.className = 'ai-assistant-panel-privacy-head';
        var hStrong = document.createElement('strong');
        hStrong.textContent = title;
        var hClose = _createIconBtn('terms-close', 'Close ' + title, ICONS.close);
        hClose.addEventListener('click', function () {
            sheet.setAttribute('data-open', 'false');
        });
        head.appendChild(hStrong);
        head.appendChild(hClose);
        sheet.appendChild(head);

        var bodyEl = document.createElement('div');
        bodyEl.className = 'ai-assistant-panel-privacy-body';

        if (typeof cfg.panelTermsHtml === 'string' && cfg.panelTermsHtml) {
            // Trusted, author-supplied (from conf.py, not end-user input).
            bodyEl.innerHTML = cfg.panelTermsHtml;
        } else {
            bodyEl.innerHTML =
                '<h4>Documentation context</h4>' +
                '<p>This assistant ships as part of the documentation. ' +
                'It is offered "as is", without warranty of any kind. ' +
                'Use of any answer it produces is at your own risk.</p>' +

                '<h4>Acceptable use</h4>' +
                '<ul>' +
                '<li>Do not submit confidential, regulated, or personally ' +
                'identifiable information through the chat input.</li>' +
                '<li>Do not use the assistant to attempt to bypass access ' +
                'controls or extract content you are not entitled to.</li>' +
                '<li>Generated answers may be inaccurate. Verify against ' +
                'the actual documentation before relying on them.</li>' +
                '</ul>' +

                '<h4>Model providers</h4>' +
                '<p>When API mode is enabled, your question and an extract ' +
                'of this page are forwarded to the configured AI provider ' +
                'via the documentation owner\u2019s proxy. Each provider has ' +
                'its own terms; consult the model\u2019s information page ' +
                '(\u2139 icon in the model picker) for the canonical link.</p>' +

                '<h4>Feedback</h4>' +
                '<p>If you submit feedback through the \u201cWas this ' +
                'helpful?\u201d block, your rating, optional message, the ' +
                'question, and the model\u2019s answer may be collected by ' +
                'the documentation owner for the purpose of improving the ' +
                'documentation or the model. The documentation owner\u2019s ' +
                'privacy policy governs that collection.</p>';
        }
        sheet.appendChild(bodyEl);
        return sheet;
    }

    // ── Phase B: Share sheet (small modal) ────────────────────────────────────

    /**
     * Build a small share modal listing copy-link + intent-share targets.
     * Targets are sanitised server-side by ``_filter_share_targets`` so
     * every url_template here is guaranteed http/https/mailto:.  The
     * special ``copy_link`` id writes ``location.href`` to the clipboard
     * instead of opening a URL.
     *
     * @returns {HTMLElement}
     */
    function _buildShareSheet() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var label = (typeof cfg.panelShareLabel === 'string' &&
            cfg.panelShareLabel) || 'Share';

        var sheet = document.createElement('div');
        sheet.className = 'ai-assistant-panel-privacy ai-assistant-panel-share';
        sheet.id = 'ai-assistant-panel-share-sheet';
        sheet.setAttribute('data-open', 'false');

        var head = document.createElement('div');
        head.className = 'ai-assistant-panel-privacy-head';
        var hStrong = document.createElement('strong');
        hStrong.textContent = label;
        var hClose = _createIconBtn('share-close', 'Close ' + label, ICONS.close);
        hClose.addEventListener('click', function () {
            sheet.setAttribute('data-open', 'false');
        });
        head.appendChild(hStrong);
        head.appendChild(hClose);
        sheet.appendChild(head);

        var bodyEl = document.createElement('div');
        bodyEl.className = 'ai-assistant-panel-privacy-body ai-assistant-panel-share-list';

        // Always include the current URL prominently — copyable on click.
        var urlRow = document.createElement('div');
        urlRow.className = 'ai-assistant-panel-share-url';
        var urlInput = document.createElement('input');
        urlInput.type = 'text';
        urlInput.readOnly = true;
        urlInput.value = (typeof location !== 'undefined') ? location.href : '';
        urlInput.setAttribute('aria-label', 'Page URL');
        urlInput.addEventListener('focus', function () { urlInput.select(); });
        urlRow.appendChild(urlInput);
        bodyEl.appendChild(urlRow);

        var targets = Array.isArray(cfg.panelShareTargets) ? cfg.panelShareTargets : [];
        targets.forEach(function (t) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'ai-assistant-panel-share-target';
            btn.setAttribute('aria-label', t.label || t.id);

            var icon = document.createElement('span');
            icon.className = 'ai-assistant-panel-share-target-icon';
            icon.setAttribute('aria-hidden', 'true');
            icon.innerHTML = ICONS.share;     // ICONS constant — safe.
            btn.appendChild(icon);

            var lbl = document.createElement('span');
            lbl.textContent = t.label || t.id;
            btn.appendChild(lbl);

            btn.addEventListener('click', function () {
                if (t.id === 'copy_link' || !t.url_template) {
                    _copyShareLink();
                    sheet.setAttribute('data-open', 'false');
                    return;
                }
                // Build target URL with {url} and {title} placeholders.
                var pageUrl   = (typeof location !== 'undefined') ? location.href : '';
                var pageTitle = (typeof document !== 'undefined' && document.title) || '';
                var u = String(t.url_template)
                    .replace(/\{url\}/g,   encodeURIComponent(pageUrl))
                    .replace(/\{title\}/g, encodeURIComponent(pageTitle));
                try {
                    var w = window.open(u, '_blank', 'noopener,noreferrer');
                    // Newer browsers honour rel via the third arg above; for
                    // older ones, fall back to clearing opener defensively.
                    if (w) { try { w.opener = null; } catch (_) {} }
                } catch (_) {}
                sheet.setAttribute('data-open', 'false');
            });

            bodyEl.appendChild(btn);
        });

        sheet.appendChild(bodyEl);
        return sheet;
    }


    // ── Project Links sheet ───────────────────────────────────────────────────

    /**
     * Build the "Project Links" slide-over sheet.
     *
     * Shows two hero cards — Source Repository (GitHub) and Project Website —
     * each rendered as a tappable card that opens the configured URL in a new
     * tab.  Both cards and the sheet itself are fully customisable via
     * ``window.AI_ASSISTANT_CONFIG``.
     *
     * Configuration keys read from cfg
     * ---------------------------------
     * panelLinks          bool    Master switch. false → sheet not built.
     * panelLinksTitle     string  Sheet header text.  Default: "Project Links".
     * panelLinksHtml      string  When non-empty, replaces the built-in cards
     *                             with trusted author HTML (same pattern as
     *                             panelPrivacyHtml / panelTermsHtml).
     * panelSourceUrl      string  GitHub / source repository URL.
     * panelSourceLabel    string  Card heading.  Default: "Source Repository".
     * panelSourceDesc     string  Card sub-text shown beneath the heading.
     * panelSiteUrl        string  Project website URL.
     * panelSiteLabel      string  Card heading.  Default: "Project Website".
     * panelSiteDesc       string  Card sub-text shown beneath the heading.
     *
     * Returns
     * -------
     * HTMLElement
     *     The assembled sheet element (data-open="false" initially).
     */
    function _buildLinksSheet() {
        var cfg    = window.AI_ASSISTANT_CONFIG || {};
        var title  = (typeof cfg.panelLinksTitle === 'string' && cfg.panelLinksTitle)
                     || 'Project Links';

        var sheet = document.createElement('div');
        sheet.className = 'ai-assistant-panel-privacy ai-assistant-panel-links-sheet';
        sheet.id        = 'ai-assistant-panel-links-sheet';
        sheet.setAttribute('data-open', 'false');

        var head   = document.createElement('div');
        head.className = 'ai-assistant-panel-privacy-head';
        var hStrong = document.createElement('strong');
        hStrong.textContent = title;
        var hClose  = _createIconBtn('links-close', 'Close ' + title, ICONS.close);
        hClose.addEventListener('click', function () {
            sheet.setAttribute('data-open', 'false');
        });
        head.appendChild(hStrong);
        head.appendChild(hClose);
        sheet.appendChild(head);

        var bodyEl = document.createElement('div');
        bodyEl.className = 'ai-assistant-panel-privacy-body ai-assistant-panel-links-body';

        if (typeof cfg.panelLinksHtml === 'string' && cfg.panelLinksHtml) {
            bodyEl.innerHTML = cfg.panelLinksHtml;
        } else {
            function _buildLinkCard(iconHtml, heading, desc, url, accent) {
                if (!url || !_isSafeHref(url)) return null;
                var card = document.createElement('a');
                card.className  = 'ai-assistant-panel-link-card';
                card.href       = url;
                card.target     = '_blank';
                card.rel        = 'noopener noreferrer';
                card.setAttribute('aria-label', heading + ' \u2014 opens in a new tab');
                if (accent) card.style.setProperty('--ai-link-card-accent', accent);

                var ic = document.createElement('span');
                ic.className = 'ai-assistant-panel-link-card-icon';
                ic.setAttribute('aria-hidden', 'true');
                ic.innerHTML = iconHtml;
                card.appendChild(ic);

                var txt = document.createElement('span');
                txt.className = 'ai-assistant-panel-link-card-text';

                var h = document.createElement('strong');
                h.textContent = heading;
                txt.appendChild(h);

                if (desc) {
                    var d = document.createElement('span');
                    d.className = 'ai-assistant-panel-link-card-desc';
                    d.textContent = desc;
                    txt.appendChild(d);
                }

                var urlBadge = document.createElement('span');
                urlBadge.className = 'ai-assistant-panel-link-card-url';
                try {
                    var parsed = new URL(url);
                    urlBadge.textContent = parsed.hostname + parsed.pathname.replace(/\/$/, '');
                } catch (_) { urlBadge.textContent = url; }
                txt.appendChild(urlBadge);
                card.appendChild(txt);

                var arrow = document.createElement('span');
                arrow.className = 'ai-assistant-panel-link-card-arrow';
                arrow.setAttribute('aria-hidden', 'true');
                arrow.innerHTML = ICONS.externalLink;
                card.appendChild(arrow);

                return card;
            }

            var sourceUrl   = (typeof cfg.panelSourceUrl   === 'string') ? cfg.panelSourceUrl   : '';
            var sourceLabel = (typeof cfg.panelSourceLabel  === 'string' && cfg.panelSourceLabel)
                              ? cfg.panelSourceLabel : 'Source Repository';
            var sourceDesc  = (typeof cfg.panelSourceDesc  === 'string') ? cfg.panelSourceDesc  : '';
            var siteUrl     = (typeof cfg.panelSiteUrl     === 'string') ? cfg.panelSiteUrl     : '';
            var siteLabel   = (typeof cfg.panelSiteLabel   === 'string' && cfg.panelSiteLabel)
                              ? cfg.panelSiteLabel   : 'Project Website';
            var siteDesc    = (typeof cfg.panelSiteDesc    === 'string') ? cfg.panelSiteDesc    : '';

            var sourceCard = _buildLinkCard(
                ICONS.github, sourceLabel, sourceDesc, sourceUrl,
                'var(--ai-github-accent, #24292f)'
            );
            // Extra class lets CSS dark-mode override target this card only.
            if (sourceCard) sourceCard.classList.add('ai-assistant-panel-source-card');

            var siteCard   = _buildLinkCard(
                ICONS.globe,  siteLabel,   siteDesc,   siteUrl,
                'var(--ai-site-accent, var(--ai-accent, #2563eb))'
            );
            if (siteCard) siteCard.classList.add('ai-assistant-panel-site-card');

            if (sourceCard) bodyEl.appendChild(sourceCard);
            if (siteCard)   bodyEl.appendChild(siteCard);

            if (!sourceCard && !siteCard) {
                var empty = document.createElement('p');
                empty.className = 'ai-assistant-panel-links-empty';
                empty.textContent =
                    'No project links configured. Set ai_assistant_panel_source_url ' +
                    'and ai_assistant_panel_site_url in conf.py.';
                bodyEl.appendChild(empty);
            }
        }

        sheet.appendChild(bodyEl);
        return sheet;
    }


    /**
     * Copy the current page URL to the clipboard with a small toast.
     * Best-effort; silently no-ops if the browser blocks clipboard access.
     */
    function _copyShareLink() {
        var url = (typeof location !== 'undefined') ? location.href : '';
        if (!url) return;
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(url);
            }
        } catch (_) {}
    }

    // ── Phase B: Hamburger overflow popover ───────────────────────────────────

    /**
     * Build a small overflow popover that duplicates the most-used controls
     * in a single menu.  Purpose: keep the sub-bar uncluttered on narrow
     * viewports / mobile, and give a single entry-point to all sheets.
     *
     * The popover is opened by a hamburger button in the sub-bar.  Each
     * item is a real button — clicking it triggers the same handler as the
     * underlying control.  Closing happens on outside-click or Escape.
     *
     * @param {object} hooks  { onPrivacy, onTerms, onShare, onModel }
     *                        Click handlers, each optional.
     * @returns {HTMLElement}
     */
    function _buildHamburgerMenu(hooks) {
        var pop = document.createElement('div');
        pop.className = 'ai-assistant-panel-hamburger';
        pop.id = 'ai-assistant-panel-hamburger';
        pop.setAttribute('data-open', 'false');
        pop.setAttribute('role', 'menu');

        function addItem(iconHtml, label, handler) {
            if (typeof handler !== 'function') return;
            var item = document.createElement('button');
            item.type = 'button';
            item.className = 'ai-assistant-panel-hamburger-item';
            item.setAttribute('role', 'menuitem');
            var ic = document.createElement('span');
            ic.setAttribute('aria-hidden', 'true');
            ic.innerHTML = iconHtml;
            var sp = document.createElement('span');
            sp.textContent = label;
            item.appendChild(ic);
            item.appendChild(sp);
            item.addEventListener('click', function () {
                pop.setAttribute('data-open', 'false');
                handler();
            });
            pop.appendChild(item);
        }

        addItem(ICONS.model,    'Choose a model',            hooks && hooks.onModel);
        addItem(ICONS.privacy,  'Privacy & Responsibility',  hooks && hooks.onPrivacy);
        addItem(ICONS.terms,    'Terms of Service',          hooks && hooks.onTerms);
        addItem(ICONS.share,    'Share',                     hooks && hooks.onShare);
        addItem(ICONS.github,   'Project Links',             hooks && hooks.onLinks);

        // Keyboard shortcut hint row — shown at the bottom of the menu when a
        // shortcut is configured.  Now interactive: left-click = minimize,
        // right-click = close (same contract as the header minimize icon-btn).
        var kbdHintLabel = _shortcutLabel();
        if (kbdHintLabel) {
            var sep = document.createElement('hr');
            sep.className = 'ai-assistant-panel-hamburger-sep';
            sep.setAttribute('aria-hidden', 'true');
            pop.appendChild(sep);

            var kbdRow = document.createElement('div');
            kbdRow.className = 'ai-assistant-panel-hamburger-kbd-row';
            kbdRow.setAttribute('role', 'menuitem');
            kbdRow.setAttribute('tabindex', '0');
            kbdRow.setAttribute('aria-label', 'Minimize panel (right-click to close)');
            kbdRow.title = 'Left-click: minimize  \u00b7  Right-click: close';

            var kbdIcon = document.createElement('span');
            kbdIcon.setAttribute('aria-hidden', 'true');
            kbdIcon.innerHTML = ICONS.keyboard;  // ICONS constant — safe.
            kbdRow.appendChild(kbdIcon);

            kbdHintLabel.split('+').forEach(function (tok, i, arr) {
                var k = document.createElement('kbd');
                k.textContent = tok.trim();
                kbdRow.appendChild(k);
                if (i < arr.length - 1) {
                    kbdRow.appendChild(document.createTextNode('+'));
                }
            });

            // Left-click: close hamburger menu then minimize panel.
            kbdRow.addEventListener('click', function () {
                _hapticFeedback([8]);
                pop.setAttribute('data-open', 'false');
                minimizeAIPanel();
            });
            // Right-click: close hamburger menu then fully close panel.
            kbdRow.addEventListener('contextmenu', function (e) {
                e.preventDefault();
                pop.setAttribute('data-open', 'false');
                closeAIPanel();
            });
            // Keyboard: Enter / Space mirrors the left-click action.
            kbdRow.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    pop.setAttribute('data-open', 'false');
                    minimizeAIPanel();
                }
            });

            pop.appendChild(kbdRow);
        }

        return pop;
    }

    // ── Phase B: Inline footer model picker (Claude-bar style) ────────────────

    /**
     * Build a compact <select> inline beside the mic + send buttons.  When
     * the user changes it, the active model id is persisted (sessionStorage)
     * and the model-change event is dispatched — same contract as the sheet.
     *
     * Returns null when no models are configured OR when the doc author
     * has disabled the inline picker (cfg.panelInlineModelPicker === false).
     * The dedicated sheet button in the sub-bar remains available regardless.
     *
     * @returns {HTMLElement|null}
     */
    /**
     * Build the inline model picker — a compact pill button that opens the
     * model sheet on click.  Replaces the legacy native ``<select>`` so
     * provider badges, truncation tooltips, and full ARIA semantics work
     * correctly across all platforms (OS-native selects ignore CSS once open
     * on Windows/macOS, cannot carry ``aria-expanded``/``aria-controls``, and
     * cannot display rich per-model metadata).
     *
     * Returns null when ``panelInlineModelPicker === false`` or no models are
     * configured.  The caller is responsible for wiring the returned button to
     * ``_openSheet(modelSheet)`` after both are in scope.
     *
     * Parameters
     * ----------
     * None — reads ``window.AI_ASSISTANT_CONFIG`` at call time.
     *
     * Returns
     * -------
     * HTMLButtonElement|null
     *     Pill button with a ``._syncState(id)`` method for live updates, or
     *     null when the picker should be suppressed.
     *
     * Notes
     * -----
     * Developer: ``btn._syncState(id)`` always reads fresh config at update
     *   time (not the build-time closure) to avoid the stale-models bug
     *   (Issue 3).  The method is stored on the element so ``_syncInlinePickers``
     *   can call it without holding a closure reference.
     *
     * User: The picker shows [badge dot · label · chevron].  On click it
     *   opens the model sheet where the user selects a model via radio button.
     */
    function _buildInlineModelPicker() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        if (cfg.panelInlineModelPicker === false) return null;
        var models = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
        if (models.length === 0) return null;

        var activeId = _getActiveModelId(models);
        var active   = _findModel(models, activeId);

        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className =
            'ai-assistant-panel-footer-btn ai-assistant-panel-inline-model-picker';
        // ARIA: announce as a dialog trigger (Issue 11 / Issue 1).
        btn.setAttribute('aria-haspopup', 'dialog');
        btn.setAttribute('aria-expanded', 'false');
        btn.setAttribute('aria-controls', 'ai-assistant-panel-model-sheet');
        var initLabel = active ? (active.label || active.id) : 'Model';
        btn.setAttribute('aria-label', 'Choose a model \u2014 current: ' + initLabel);
        btn.title = initLabel;

        // ── Small-screen icon-only fallback ────────────────────────────────
        // On narrow viewports (≤ 575 px) the pill collapses to this single
        // model icon, hiding the dot, label, and chevron.  Mirrors the exact
        // same logic used by .ai-assistant-panel-model-link in the sub-bar:
        // the icon is always in the DOM; CSS controls visibility at the
        // breakpoint defined in the @media (max-width: 575px) block.
        var iconOnly = document.createElement('span');
        iconOnly.className = 'ai-assistant-panel-inline-picker-icon';
        iconOnly.setAttribute('aria-hidden', 'true');
        iconOnly.innerHTML = ICONS.model;   // ICONS constant — safe.
        btn.appendChild(iconOnly);

        // ── Provider badge dot ──────────────────────────────────────────────
        var dot = document.createElement('span');
        dot.className = 'ai-assistant-panel-inline-picker-dot';
        dot.setAttribute('aria-hidden', 'true');
        var dotColor = _providerColor((active && active.provider) || '');
        if (dotColor) {
            dot.style.background = dotColor;
        } else {
            dot.style.opacity = '0';
        }
        btn.appendChild(dot);

        // ── Label (truncated via CSS; full name on title) ───────────────────
        var lbl = document.createElement('span');
        lbl.className = 'ai-assistant-panel-inline-picker-label';
        lbl.textContent = initLabel;
        btn.appendChild(lbl);

        // ── Chevron ─────────────────────────────────────────────────────────
        // Class added so the narrow-screen rule can hide it by class name
        // (safer than :last-child which depends on DOM order).
        var chev = document.createElement('span');
        chev.className = 'ai-assistant-panel-inline-picker-chev';
        chev.setAttribute('aria-hidden', 'true');
        chev.innerHTML = ICONS.chevronDown;   // ICONS constant — safe.
        btn.appendChild(chev);

        // ── Live sync method — reads fresh config at update time (Issue 3) ──
        // Stored on the element so _syncInlinePickers can call it without
        // re-querying the closure, and so the button always reflects the live
        // panelApiModels list even when hot-reloaded after DOMContentLoaded.
        btn._syncState = function (id) {
            var freshModels = (window.AI_ASSISTANT_CONFIG || {}).panelApiModels;
            var m = _findModel(
                Array.isArray(freshModels) ? freshModels : models,
                id
            );
            var text = m ? (m.label || m.id) : id;
            lbl.textContent = text;
            btn.title = text;
            btn.setAttribute('aria-label', 'Choose a model \u2014 current: ' + text);
            var c = _providerColor((m && m.provider) || '');
            if (c) {
                dot.style.background = c;
                dot.style.opacity = '1';
            } else {
                dot.style.opacity = '0';
            }
        };

        // NOTE: The click handler (_openSheet) is wired by the caller in
        // createAIPanel after modelSheet is in scope — the button is returned
        // first, then the sheet is built, then the listener is attached.

        return btn;
    }

    // ── R8: standalone AI search-bar (opt-in, additive) ───────────────────────

    /**
     * Build a self-contained mini search input that forwards its text into
     * the AI panel as the first question.  The extension renders its OWN
     * element and never touches the theme\u2019s search DOM, so PyData / Furo /
     * RTD search keep working untouched.  Off by default.
     *
     * @param {boolean} mini  Compact inline variant when true.
     * @returns {HTMLElement}
     */
    function _buildSearchBar(mini) {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        var ph  = (typeof cfg.panelSearchPlaceholder === 'string' &&
            cfg.panelSearchPlaceholder) || 'Ask AI about these docs\u2026';

        var bar = document.createElement('div');
        bar.className = 'ai-assistant-searchbar' +
            (mini ? ' ai-assistant-searchbar--mini' : '');

        var icon = document.createElement('span');
        icon.setAttribute('aria-hidden', 'true');
        icon.innerHTML = ICONS.searchAI;     // ICONS constant — safe.
        bar.appendChild(icon);

        var inp = document.createElement('input');
        inp.type = 'text';
        inp.setAttribute('aria-label', ph);
        inp.placeholder = ph;
        bar.appendChild(inp);

        var kbdLabel = _shortcutLabel();
        if (kbdLabel) {
            var hint = document.createElement('span');
            hint.className = 'ai-assistant-searchbar-kbd-hint';
            var hIcon = document.createElement('span');
            hIcon.setAttribute('aria-hidden', 'true');
            hIcon.innerHTML = ICONS.keyboard;        // ICONS constant — safe.
            hint.appendChild(hIcon);
            // Render each chord token as its own <kbd>.
            kbdLabel.split('+').forEach(function (tok, i, arr) {
                var k = document.createElement('kbd');
                k.textContent = tok.trim();
                hint.appendChild(k);
                if (i < arr.length - 1) {
                    hint.appendChild(document.createTextNode('+'));
                }
            });
            bar.appendChild(hint);
        } else {
            bar.appendChild(document.createElement('span')); // spacer
        }

        function _go() {
            var q = inp.value.trim();
            if (!q) return;
            if (!_aiPanelEl) _aiPanelEl = createAIPanel();
            _openAIPanel();
            var panelInput = document.getElementById('ai-assistant-panel-input');
            if (panelInput) {
                panelInput.value = q;
                _updateSendBtnState();
                handleAIPanelSubmit();
            }
            inp.value = '';
        }
        inp.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') { e.preventDefault(); _go(); }
        });
        return bar;
    }

    /**
     * Mount the standalone search-bar into a host selector (config-driven).
     * Default OFF.  If the configured selector is not found nothing happens
     * (safe no-op) so a missing element can never break the page.
     *
     * Position is controlled by cfg.searchBarPosition:
     *   "top"    → insertBefore(bar, host.firstChild)  — prepend at sidebar top.
     *   "bottom" → appendChild(bar)                    — append (default).
     * Any value other than "top" falls back to "bottom" (pre-existing behaviour).
     */
    function _mountSearchBar() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        if (!cfg.searchBar) return;                       // default off
        var sel = (typeof cfg.searchBarSelector === 'string' &&
            cfg.searchBarSelector) || '';
        if (!sel) return;
        var host = document.querySelector(sel);
        if (!host) return;
        if (host.querySelector('.ai-assistant-searchbar')) return;  // idempotent
        var bar = _buildSearchBar(cfg.searchBarMini === true);
        if (cfg.searchBarPosition === 'top') {
            // Prepend: place before the first existing child so the search bar
            // appears at the very top of the sidebar — above navigation links.
            // Default "top"
            host.insertBefore(bar, host.firstChild);
        } else {
            // "bottom": append after all existing children.
            host.appendChild(bar);
        }
    }

    // ── AI Panel ──────────────────────────────────────────────────────────────

    /**
     * Create a small icon button for the panel header.
     *
     * @param {string} id           Element id (without prefix).
     * @param {string} ariaLabel    Accessible label.
     * @param {string} iconHtml     Raw SVG string (from ICONS constant — not user data).
     * @returns {HTMLButtonElement}
     */
    function _createIconBtn(id, ariaLabel, iconHtml) {
        var btn = document.createElement('button');
        btn.className = 'ai-assistant-panel-icon-btn';
        btn.id = 'ai-assistant-panel-' + id;
        btn.type = 'button';
        btn.setAttribute('aria-label', ariaLabel);
        btn.innerHTML = iconHtml;   // ICONS constant — no user data, safe.
        return btn;
    }

    /**
     * Create the floating AI assistant panel DOM element (lazy singleton).
     *
     * New in this revision
     * ────────────────────
     *   • Minimize button  — collapses panel to floating trigger pill.
     *   • Maximize button  — toggles full viewport-height expansion.
     *   • Quick suggestion chips  — customizable list from config.
     *   • "Speak with assistant" banner  — dismissable pill above input.
     *   • Redesigned footer  — [textarea] [mic icon] [send icon].
     *   • Floating trigger pill  — "Ask Us" button shown when minimized.
     *
     * @returns {HTMLElement}
     */
    function createAIPanel() {
        var cfg         = window.AI_ASSISTANT_CONFIG || {};
        var title       = cfg.panelTitle       || 'AI Assistant';
        var placeholder = cfg.panelPlaceholder || 'Ask a question about this page\u2026';
        // Quick-suggestion chips are now built by _renderWelcome (shared with
        // clearConversation) so they are no longer constructed here.
        var speakBanner = cfg.panelSpeakBanner !== false;   // default true
        var hasSpeech   = _speechSupported();

        // ── Outer panel ──────────────────────────────────────────────────────
        var panel = document.createElement('div');
        panel.id = 'ai-assistant-panel';
        panel.className = 'ai-assistant-panel';
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-label', title);
        panel.style.display = 'none';

        // ── Header ───────────────────────────────────────────────────────────
        var header = document.createElement('div');
        header.className = 'ai-assistant-panel-header';

        var headerTitle = document.createElement('div');
        headerTitle.className = 'ai-assistant-panel-header-title';

        // Logo — try image file first; inline SVG as fallback attribute
        var logo = document.createElement('img');
        logo.src = getStaticPath() + '/ai-panel.svg';
        logo.className = 'ai-assistant-panel-logo';
        logo.setAttribute('aria-hidden', 'true');
        logo.alt = '';

        var titleSpan = document.createElement('span');
        titleSpan.textContent = title;

        headerTitle.appendChild(logo);
        headerTitle.appendChild(titleSpan);

        // Header action buttons
        var headerActions = document.createElement('div');
        headerActions.className = 'ai-assistant-panel-header-actions';

        var minimizeBtn = _createIconBtn('minimize', 'Minimize panel', ICONS.minimize);
        var maximizeBtn = _createIconBtn('maximize', 'Maximize panel', ICONS.maximize);
        var closeBtn    = _createIconBtn('close',    'Close ' + _escapeHtml(title), ICONS.close);

        // R3: clear conversation without page refresh ("Start a new chat").
        var newChatBtn = _createIconBtn('new-chat', 'Start a new chat', ICONS.newChat);
        newChatBtn.title = 'Start a new chat';
        newChatBtn.addEventListener('pointerdown', function () { _hapticFeedback([8]); });
        newChatBtn.addEventListener('click', clearConversation);

        // R4: export the conversation as a plain-text download.
        var exportBtn = _createIconBtn(
            'export', 'Export AI conversation as txt', ICONS.exportTxt);
        exportBtn.title = 'Export AI conversation as txt';
        exportBtn.addEventListener('pointerdown', function () { _hapticFeedback([8]); });
        exportBtn.addEventListener('click', exportConversation);

        headerActions.appendChild(newChatBtn);
        headerActions.appendChild(exportBtn);
        headerActions.appendChild(minimizeBtn);
        headerActions.appendChild(maximizeBtn);
        headerActions.appendChild(closeBtn);

        header.appendChild(headerTitle);
        header.appendChild(headerActions);

        // ── Sub-bar: hamburger (Phase B) + keyboard hint (R7) + privacy link (R2) ──
        //
        // Layout (left → right):
        //
        //    [☰ hamburger]  [⌨ kbd-hint]   . . .   [Privacy] [Terms] [model ▾] [↗ Share]
        //
        // The hamburger button is ADDITIVE: it duplicates each sheet entry-
        // point in a single popover so narrow viewports can collapse the
        // right-hand cluster gracefully via CSS without losing access to
        // any control.  The pre-existing sub-bar layout is preserved
        // exactly when ``cfg.panelHamburger === false``.
        var cfgRef = window.AI_ASSISTANT_CONFIG || {};
        var subbar = document.createElement('div');
        subbar.className = 'ai-assistant-panel-subbar';

        // ── Left cluster: hamburger (optional) + keyboard hint ──
        var leftCluster = document.createElement('div');
        leftCluster.className = 'ai-assistant-panel-subbar-left';

        var hamburgerBtn = null;
        var hamburgerMenu = null;
        if (cfgRef.panelHamburger !== false) {
            hamburgerBtn = _createIconBtn(
                'hamburger', 'Open menu', ICONS.menu);
            hamburgerBtn.title = 'Open menu';
            leftCluster.appendChild(hamburgerBtn);
        }

        var kbdLabel = _shortcutLabel();
        if (kbdLabel) {
            var hint = document.createElement('span');
            hint.className = 'ai-assistant-panel-kbd-hint';
            hint.setAttribute('role', 'button');
            hint.setAttribute('tabindex', '0');
            hint.setAttribute('aria-label', 'Minimize panel (right-click to close)');
            hint.title = 'Left-click: minimize  \u00b7  Right-click: close';
            var hIcon = document.createElement('span');
            hIcon.setAttribute('aria-hidden', 'true');
            hIcon.innerHTML = ICONS.keyboard;        // ICONS constant — safe.
            hint.appendChild(hIcon);
            // Render each chord token as its own <kbd>.
            kbdLabel.split('+').forEach(function (tok, i, arr) {
                var k = document.createElement('kbd');
                k.textContent = tok.trim();
                hint.appendChild(k);
                if (i < arr.length - 1) {
                    hint.appendChild(document.createTextNode('+'));
                }
            });
            // Left-click: minimize panel.
            hint.addEventListener('click', function () { _hapticFeedback([8]); minimizeAIPanel(); });
            // Right-click: fully close panel.
            hint.addEventListener('contextmenu', function (e) {
                e.preventDefault();
                closeAIPanel();
            });
            // Keyboard: Enter / Space mirrors the left-click action.
            hint.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    minimizeAIPanel();
                }
            });
            leftCluster.appendChild(hint);
        }

        // ── Left cluster: Source (GitHub) button ──────────────────────────────
        // Shown when panelSource !== false AND panelSourceUrl is a valid URL.
        // Clicking opens the Links sheet (same _openSheet contract as all other
        // sheets).  Built here so the element is available to wire() below.
        var sourceBtn = null;
        if (cfgRef.panelSource !== false) {
            sourceBtn = document.createElement('button');
            sourceBtn.type = 'button';
            sourceBtn.className =
                'ai-assistant-panel-subbar-link-btn ai-assistant-panel-source-btn';
            var sourceIc = document.createElement('span');
            sourceIc.setAttribute('aria-hidden', 'true');
            sourceIc.innerHTML = ICONS.github;   // ICONS constant — safe.
            sourceBtn.appendChild(sourceIc);
            var sourceLbl = document.createElement('span');
            sourceLbl.textContent =
                (cfgRef.panelSourceBtnLabel) || 'Source';
            sourceBtn.appendChild(sourceLbl);
            sourceBtn.setAttribute(
                'aria-label',
                (cfgRef.panelSourceBtnLabel || 'Source') + ' \u2014 open project links'
            );
            sourceBtn.title = 'View project source & links';
            leftCluster.appendChild(sourceBtn);
        }

        subbar.appendChild(leftCluster);

        // ── Right cluster: model · privacy · terms · share ──
        // Order matches the user-requested layout: model button sits BEFORE
        // the Privacy link; Terms sits after Privacy; Share is the rightmost
        // entry-point.  Each control is independently toggleable via cfg.
        var rightCluster = document.createElement('div');
        rightCluster.className = 'ai-assistant-panel-subbar-right';

        var privacyLink = document.createElement('button');
        privacyLink.className = 'ai-assistant-panel-privacy-link';
        privacyLink.type = 'button';
        privacyLink.textContent =
            (window.AI_ASSISTANT_CONFIG &&
             window.AI_ASSISTANT_CONFIG.panelPrivacyLinkText) ||
            'Privacy & Responsibility';

        // Terms-of-Service link — sibling of Privacy, same CSS class so the
        // theme styling cascades automatically.
        var termsLink = null;
        if (cfgRef.panelTerms !== false) {
            termsLink = document.createElement('button');
            termsLink.className = 'ai-assistant-panel-privacy-link ai-assistant-panel-terms-link';
            termsLink.type = 'button';
            termsLink.textContent =
                (cfgRef.panelTermsLinkText) || 'Terms of Service';
            termsLink.setAttribute('aria-label', 'Open Terms of Service');
        }

        // Model picker button (sheet entry-point).  Shown when any panel
        // models are configured — otherwise the button would have no effect.
        var modelLink = null;
        if (Array.isArray(cfgRef.panelApiModels) && cfgRef.panelApiModels.length > 0) {
            modelLink = document.createElement('button');
            modelLink.className =
                'ai-assistant-panel-privacy-link ai-assistant-panel-model-link';
            modelLink.type = 'button';
            // Label format: "<icon> <Active model label> ▾"
            var modelIc = document.createElement('span');
            modelIc.setAttribute('aria-hidden', 'true');
            modelIc.innerHTML = ICONS.model;
            modelLink.appendChild(modelIc);

            var modelLbl = document.createElement('span');
            modelLbl.className = 'ai-assistant-panel-model-link-label';
            var activeNow = _getActiveModel(cfgRef);
            modelLbl.textContent = activeNow
                ? (activeNow.label || activeNow.id)
                : 'Model';
            modelLink.appendChild(modelLbl);

            var modelChev = document.createElement('span');
            modelChev.setAttribute('aria-hidden', 'true');
            modelChev.innerHTML = ICONS.chevronDown;
            modelLink.appendChild(modelChev);

            // Dynamic aria-label and title: mirrors the inline-picker format
            // "Choose a model — current: <label>" so screen readers and the
            // browser tooltip both surface the currently-selected model name.
            var _initModelText = activeNow ? (activeNow.label || activeNow.id) : 'Model';
            modelLink.setAttribute('aria-label', 'Choose a model \u2014 current: ' + _initModelText);
            modelLink.title = _initModelText;
        }

        // Share button — opens the Share sheet.
        var shareLink = null;
        if (cfgRef.panelShare !== false) {
            shareLink = document.createElement('button');
            shareLink.className = 'ai-assistant-panel-privacy-link ai-assistant-panel-share-link';
            shareLink.type = 'button';
            var shareIc = document.createElement('span');
            shareIc.setAttribute('aria-hidden', 'true');
            shareIc.innerHTML = ICONS.share;
            shareLink.appendChild(shareIc);
            var shareLbl = document.createElement('span');
            shareLbl.textContent = (cfgRef.panelShareLabel) || 'Share';
            shareLink.appendChild(shareLbl);
            shareLink.setAttribute('aria-label', 'Share this page');
            shareLink.title = 'Share this page';
        }

        // Append right-cluster items in visual left→right order:
        //   Model | Privacy | Terms | Share | Site | ⋯
        // Items that are null (feature-flagged off) are silently skipped.
        if (modelLink)  rightCluster.appendChild(modelLink);
        rightCluster.appendChild(privacyLink);
        if (termsLink)  rightCluster.appendChild(termsLink);
        if (shareLink)  rightCluster.appendChild(shareLink);

        // ── Right cluster: Site (website) button — after Share ────────────────
        // Counterpart to sourceBtn.  Opens the same Links sheet from the right
        // side so the user can reach project links from either subbar edge.
        var siteBtn = null;
        if (cfgRef.panelSite !== false) {
            siteBtn = document.createElement('button');
            siteBtn.type = 'button';
            siteBtn.className =
                'ai-assistant-panel-subbar-link-btn ai-assistant-panel-site-btn';
            var siteIc = document.createElement('span');
            siteIc.setAttribute('aria-hidden', 'true');
            siteIc.innerHTML = ICONS.globe;   // ICONS constant — safe.
            siteBtn.appendChild(siteIc);
            var siteLbl = document.createElement('span');
            siteLbl.textContent = (cfgRef.panelSiteBtnLabel) || 'Website';
            siteBtn.appendChild(siteLbl);
            siteBtn.setAttribute(
                'aria-label',
                (cfgRef.panelSiteBtnLabel || 'Website') + ' \u2014 open project links'
            );
            siteBtn.title = 'Visit project website & links';
            rightCluster.appendChild(siteBtn);
        }

        // Right overflow toggle button — collapsed representation of the
        // entire right cluster when the panel is too narrow to show individual
        // items.  Visibility is CSS-driven via data-narrow on the panel root.
        var rightOverflowBtn = document.createElement('button');
        rightOverflowBtn.className = 'ai-assistant-panel-subbar-overflow-btn';
        rightOverflowBtn.type = 'button';
        rightOverflowBtn.setAttribute('aria-label', 'More options');
        rightOverflowBtn.setAttribute('aria-haspopup', 'menu');
        rightOverflowBtn.title = 'More options';
        rightOverflowBtn.innerHTML = ICONS.overflowH;  // ICONS constant — safe.
        rightCluster.appendChild(rightOverflowBtn);

        subbar.appendChild(rightCluster);

        // ── Body ─────────────────────────────────────────────────────────────
        var body = document.createElement('div');
        body.className = 'ai-assistant-panel-body';
        body.id = 'ai-assistant-panel-body';
        // aria-live="polite" — screen readers announce new content (AI replies,
        // typing indicators) without interrupting the user's current speech.
        // "polite" is correct here; "assertive" would be intrusive.
        body.setAttribute('aria-live', 'polite');
        body.setAttribute('aria-relevant', 'additions');

        // Load any persisted conversation (single source of truth).  If it
        // is non-empty, replay it; otherwise show the welcome + suggestions.
        // _renderWelcome is the SAME path used by clearConversation() so the
        // welcome markup is defined exactly once (no duplication).
        _loadTranscript();
        if (_transcript.length > 0) {
            _replayTranscript(body);
        } else {
            _renderWelcome(body);
        }

        // ── Speak banner (optional) ───────────────────────────────────────────
        //
        // Shown above the footer when: panelSpeakBanner=true AND browser
        // supports Web Speech API.  Clicking it starts mic recognition.
        // The banner is dismissed after the first user message or mic use.

        var speakBannerEl = null;
        if (speakBanner && hasSpeech) {
            speakBannerEl = document.createElement('button');
            speakBannerEl.className = 'ai-assistant-panel-speak-banner';
            speakBannerEl.id = 'ai-assistant-panel-speak-banner';
            speakBannerEl.type = 'button';
            speakBannerEl.setAttribute('aria-label', 'Speak with your assistant');
            speakBannerEl.innerHTML = ICONS.mic;   // ICONS constant — safe.

            var speakText = document.createElement('span');
            speakText.textContent = 'Speak with your assistant';
            speakBannerEl.appendChild(speakText);

            speakBannerEl.addEventListener('click', function () {
                _hapticFeedback([8]);
                _bannerToggle();
            });
        }

        // ── Footer ────────────────────────────────────────────────────────────
        var footer = document.createElement('div');
        footer.className = 'ai-assistant-panel-footer';

        // ── Input group: Claude-style column layout ───────────────────────────
        //
        //   ┌────────────────────────────────────────────────────────────────┐
        //   │  [textarea — grows vertically, full width]                     │
        //   ├────────────────────────────────────────────────────────────────┤
        //   │  [+ attach]          ·····         [model ▾?] [🎤 mic?] [➤ send] │
        //   └────────────────────────────────────────────────────────────────┘
        //
        // The action bar mirrors Claude's input UI: a single + on the left
        // for attachments/context, and the send controls on the right.
        var inputGroup = document.createElement('div');
        inputGroup.className = 'ai-assistant-panel-input-group';

        var input = document.createElement('textarea');
        input.id = 'ai-assistant-panel-input';
        input.className = 'ai-assistant-panel-input';
        input.rows = 2;
        input.placeholder = placeholder;
        input.setAttribute('aria-label', 'Your question');

        inputGroup.appendChild(input);

        // ── Action bar ────────────────────────────────────────────────────────
        var footerActions = document.createElement('div');
        footerActions.className = 'ai-assistant-panel-footer-actions';

        // + (attach / add context) button — left anchor, mirrors Claude.ai.
        // Dispatches a custom event so doc authors can hook file-upload flows.
        var attachBtn = document.createElement('button');
        attachBtn.className = 'ai-assistant-panel-footer-btn ai-assistant-panel-footer-btn--attach';
        attachBtn.type = 'button';
        attachBtn.setAttribute('aria-label', 'Add attachment or context');
        attachBtn.setAttribute('title', 'Add attachment or context');
        attachBtn.innerHTML = ICONS.plus;   // ICONS constant — safe.
        attachBtn.addEventListener('click', function () {
            _hapticFeedback([8]);
            panel.dispatchEvent(new CustomEvent('ai-assistant-attach', {
                bubbles: true, cancelable: true,
            }));
        });
        footerActions.appendChild(attachBtn);

        // Right-side action cluster: [soundbar?] | model ▾ | mic | send
        var footerActionsRight = document.createElement('div');
        footerActionsRight.className = 'ai-assistant-panel-footer-actions-right';

        // ── Footer soundbar: container only — bars built dynamically in _startVizLoops
        //
        // Bar count is determined at recording-start from the viewport width
        // (_computeSoundbarBarCount) so it is always right for the device.
        // CSS controls visibility (max-width: 0 → active max-width);
        // JS manages bar elements, heights, and opacity per tick.
        (function () {
            var soundbar = document.createElement('div');
            soundbar.className = 'ai-assistant-footer-soundbar';
            soundbar.id        = 'ai-assistant-footer-soundbar';
            soundbar.setAttribute('aria-hidden', 'true');
            // No bars here — _rebuildSoundbarBars() creates them in _startVizLoops.
            footerActionsRight.appendChild(soundbar);
        }());

        // Inline model picker (Claude-bar style): [model ▾?]
        // Returns null when no models are configured or panelInlineModelPicker=false.
        var inlinePicker = _buildInlineModelPicker();
        if (inlinePicker) footerActionsRight.appendChild(inlinePicker);

        // Microphone button (shown only when speech is supported): [🎤 mic?]
        //
        // Structure:
        //   .ai-assistant-mic-wrapper                 — position:relative anchor
        //     .ai-assistant-mic-popup                 — hover popup (right-side)
        //       .ai-assistant-mic-popup-row--level     — voice level bars (top)
        //       .ai-assistant-mic-popup-sep            — separator
        //       .ai-assistant-mic-popup-row--hold      — hold-to-record toggle (bottom)
        //     button.ai-assistant-panel-footer-btn--mic — the mic button itself
        //
        // Hover logic: CSS shows the popup when the wrapper is hovered.
        // Hold-to-record: when _micHoldMode is true, pointerdown/up drives recognition
        //   instead of click-toggle — matching the Claude.ai "press and hold" pattern.
        var micBtnEl = null;
        if (hasSpeech) {
            // ── Wrapper ───────────────────────────────────────────────────────
            var micWrapper = document.createElement('div');
            micWrapper.className = 'ai-assistant-mic-wrapper';

            // ── Hover popup (right-anchored, floats above action bar) ─────────
            var micPopup = _buildMicHoverPopup();
            micWrapper.appendChild(micPopup);

            // ── Mic button ────────────────────────────────────────────────────
            micBtnEl = document.createElement('button');
            micBtnEl.className = 'ai-assistant-panel-footer-btn ai-assistant-panel-footer-btn--mic';
            micBtnEl.id = 'ai-assistant-panel-mic';
            micBtnEl.type = 'button';

            // Initial label depends on persisted hold mode
            var _micInitLabel = _micHoldMode ? 'Press and hold to record' : 'Speak your question';
            micBtnEl.setAttribute('aria-label', _micInitLabel);
            micBtnEl.setAttribute('title', _micInitLabel);
            micBtnEl.setAttribute('data-hold', _micHoldMode ? 'true' : 'false');
            micBtnEl.innerHTML = ICONS.mic;   // ICONS constant — safe.

            // Hold-to-record: pointerdown → start, pointerup/pointerleave → stop
            micBtnEl.addEventListener('pointerdown', function (e) {
                if (!_micHoldMode) return;
                _hapticFeedback([12]);   // medium pulse — confirms recording has started
                e.preventDefault();
                _micPointerHeld = true;
                try {
                    micBtnEl.setPointerCapture(e.pointerId);
                } catch (_) {}
                if (!_isListening) {
                    _toggleSpeechRecognition();
                }
            });
            var _STOP_FLUSH_MS = 600;
            var _stopTimer = null;
            micBtnEl.addEventListener('pointerup', function () {
                // if (_micHoldMode && _isListening) { _stopSpeechRecognition(); }
                _micPointerHeld = false;
                if (_micHoldMode) {
                    _stopSpeechRecognition();
                }
            });
            micBtnEl.addEventListener('pointercancel', function () {
                _micPointerHeld = false;
                if (_micHoldMode) {
                    _stopSpeechRecognition();
                }
            });

            // Click-toggle mode (default / hold=false).
            //
            // Correct interaction chain:
            //   1. Hover mic wrapper     → CSS reveals the right-side expand-chevron button.
            //   2. Click expand-chevron  → JS toggles data-pinned on #ai-assistant-mic-popup.
            //   3. Popup visible only    → when data-pinned="true" (never from hover alone).
            //
            // The mic button controls speech recognition ONLY.
            // Popup visibility is the exclusive responsibility of the expand-chevron button
            // (.ai-assistant-mic-expand-btn) so the two concerns are fully decoupled.
            //
            // NOTE: _dismissSpeakBanner() must NOT be called here.
            //   The speak banner serves dual purpose: it is the "Speak with your
            //   assistant" discovery prompt AND the recording-state indicator
            //   (.recording class + pulse animation + text change via _bannerSetRecording).
            //   Hiding it on mic click destroys the recording feedback the user expects
            //   to see on the banner while speaking.  The correct and only dismissal
            //   point is _submitQuestion() — after the user has actually sent a message,
            //   which signals they have engaged with the feature.
            micBtnEl.addEventListener('click', function () {
                if (!_micHoldMode) {
                    _hapticFeedback([8]);
                    _toggleSpeechRecognition();
                }
            });

            micWrapper.appendChild(micBtnEl);

            // ── Right-side expand button (Claude-style hover reveal) ──────────
            //
            // Adds a small chevron button to the RIGHT of the mic button that:
            //   • stays hidden (width 0 / overflow hidden) when not hovering
            //   • slides in to 2rem on wrapper hover (CSS drives the animation)
            //   • when clicked, pins / unpins the mic popup via data-pinned
            //
            // This mirrors Claude.ai's left-side settings-reveal — placed on
            // the right so the layout reads: [🎤 mic] [▲ expand]
            //
            // DOM structure added:
            //   .ai-assistant-mic-expand-wrapper   (overflow:hidden, width:0→2rem)
            //     .ai-assistant-mic-expand-btn     (chevron, aria-expanded)
            //
            // Popup interactivity contract:
            //   data-pinned="true"  → popup visible + pointer-events:auto (CSS)
            //   data-pinned="false" → back to hover-only visibility
            //
            // Keyboard: focusin inside the popup pins it; focusout unpins when
            //   focus truly leaves (relatedTarget outside popup + wrapper).
            //   Outside-click handler closes pinned popup.  The outside-click
            //   listener removes itself when the wrapper leaves the DOM so
            //   no ghost listeners accumulate across panel rebuilds.

            var micExpandWrapper = document.createElement('div');
            micExpandWrapper.className = 'ai-assistant-mic-expand-wrapper';
            micExpandWrapper.setAttribute('aria-hidden', 'true');   // wrapper is decorative

            var micExpandBtn = document.createElement('button');
            micExpandBtn.className = 'ai-assistant-mic-expand-btn';
            micExpandBtn.type = 'button';
            micExpandBtn.setAttribute('aria-label', 'Microphone options');
            micExpandBtn.setAttribute('title', 'Microphone options');
            micExpandBtn.setAttribute('aria-haspopup', 'true');
            micExpandBtn.setAttribute('aria-expanded', 'false');
            micExpandBtn.setAttribute('aria-controls', 'ai-assistant-mic-popup');
            micExpandBtn.removeAttribute('aria-hidden');   // focusable — override wrapper

            // Chevron-up SVG: points up (popup appears above); rotates on open
            micExpandBtn.innerHTML =
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
                + ' stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"'
                + ' aria-hidden="true">'
                + '<polyline points="18 15 12 9 6 15"/>'
                + '</svg>';

            // ── Click: pin / unpin popup ──────────────────────────────────────
            micExpandBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                var nowPinned = micPopup.getAttribute('data-pinned') === 'true';
                var next = nowPinned ? 'false' : 'true';
                micPopup.setAttribute('data-pinned', next);
                micExpandBtn.setAttribute('aria-expanded', next);
                // When the popup is closed (next="false"), clear data-dragged so
                // the next open uses the slide-in animation rather than the
                // drag-mode opacity-only transition.  If not cleared, a popup
                // that was dragged in a previous session would permanently suppress
                // its entry animation on every subsequent open.
                if (next === 'false') {
                    micPopup.removeAttribute('data-dragged');
                }
            });

            // ── Keyboard: pin while focus is inside popup ─────────────────────
            // focusin fires when any descendant receives focus (bubbles).
            micPopup.addEventListener('focusin', function () {
                micPopup.setAttribute('data-pinned', 'true');
                micExpandBtn.setAttribute('aria-expanded', 'true');
            });
            // focusout fires when focus leaves any descendant.
            // relatedTarget is the element that WILL receive focus next.
            // Only unpin when focus truly leaves both the popup and the wrapper.
            //
            // Guard: skip while the popup is being dragged.  mousedown on the
            // level row calls e.preventDefault() which usually prevents focus
            // movement, but an external event (OS permission dialog, window blur
            // on some browsers) can still emit focusout with relatedTarget=null
            // — which matches the !focusTarget branch and closes the popup
            // mid-drag.  data-dragged="true" is set by the drag IIFE (inside
            // _buildMicHoverPopup) the moment a real drag begins (≥3px move),
            // so reading it here is a zero-extra-variable guard that fully
            // decouples the two code sections.
            micPopup.addEventListener('focusout', function (e) {
                if (micPopup.getAttribute('data-dragged') === 'true') { return; }
                var focusTarget = e.relatedTarget;
                if (!focusTarget
                        || (!micPopup.contains(focusTarget)
                            && !micWrapper.contains(focusTarget))) {
                    micPopup.setAttribute('data-pinned', 'false');
                    micExpandBtn.setAttribute('aria-expanded', 'false');
                    micPopup.removeAttribute('data-dragged');   // Bug 4: reset drag state on close
                }
            });

            // ── Outside-click: close pinned popup ─────────────────────────────
            // Self-cleaning: removes itself once the wrapper leaves the DOM so
            // no ghost handlers accumulate if createAIPanel() is called again.
            (function () {
                function _closePinnedMicPopup(e) {
                    // Self-clean when the wrapper has been removed from the DOM
                    if (!micWrapper.isConnected) {
                        document.removeEventListener('click', _closePinnedMicPopup, true);
                        return;
                    }
                    if (micWrapper.contains(e.target)) return;
                    micPopup.setAttribute('data-pinned', 'false');
                    micExpandBtn.setAttribute('aria-expanded', 'false');
                    micPopup.removeAttribute('data-dragged');   // Bug 4: reset drag state on close
                }
                // Use capture so the handler fires before any inner stopPropagation
                document.addEventListener('click', _closePinnedMicPopup, true);
            }());

            // ── Escape: close from keyboard ───────────────────────────────────
            micExpandBtn.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') {
                    micPopup.setAttribute('data-pinned', 'false');
                    micExpandBtn.setAttribute('aria-expanded', 'false');
                    micPopup.removeAttribute('data-dragged');   // Bug 4: reset drag state on close
                    micExpandBtn.focus();
                }
            });

            micExpandWrapper.appendChild(micExpandBtn);
            micWrapper.appendChild(micExpandWrapper);
            footerActionsRight.appendChild(micWrapper);
        }

        // Send icon button: [➤ send]
        var sendBtn = document.createElement('button');
        sendBtn.className = 'ai-assistant-panel-footer-btn ai-assistant-panel-footer-btn--send';
        sendBtn.id = 'ai-assistant-panel-send';
        sendBtn.type = 'button';
        sendBtn.setAttribute('aria-label', 'Send question');
        sendBtn.setAttribute('title', 'Send (Enter)');
        sendBtn.innerHTML = ICONS.send;   // ICONS constant — safe.
        footerActionsRight.appendChild(sendBtn);

        footerActions.appendChild(footerActionsRight);
        inputGroup.appendChild(footerActions);
        footer.appendChild(inputGroup);

        // ── Assemble panel ────────────────────────────────────────────────────
        panel.appendChild(header);
        panel.appendChild(subbar);            // R7 kbd hint + R2 privacy link
        panel.appendChild(body);

        // Speak banner goes between body and footer, attached to the panel
        // so it is always visible above the input regardless of scroll.
        if (speakBannerEl) panel.appendChild(speakBannerEl);

        panel.appendChild(footer);
        // NOTE: The routed listener below (via _openSheet) is the single
        // authoritative opener.  A direct listener here was removed — it
        // fired before _openSheet closed any already-open sheet, creating a
        // one-microtask window where two sheets shared data-open="true", and
        // it would bypass the aria-expanded + focus management wired into
        // _openSheet by Phase 2 fixes.
        //
        // ── Phase B: additional slide-over sheets + hamburger popover ──────
        // Each sheet uses the exact same data-open contract as the privacy
        // sheet so the existing CSS animation and a11y semantics cover them
        // for free.  Only one sheet may be open at a time — opening one
        // closes the others, which keeps the panel readable on small screens.
        var modelSheet = _buildModelSheet();
        panel.appendChild(modelSheet);
        // R2: privacy/responsibility slide-over (absolute, covers panel).
        var privacySheet = _buildPrivacySheet();
        panel.appendChild(privacySheet);

        var termsSheet = (cfgRef.panelTerms !== false) ? _buildTermsSheet() : null;
        if (termsSheet) panel.appendChild(termsSheet);

        var shareSheet = (cfgRef.panelShare !== false) ? _buildShareSheet() : null;
        if (shareSheet) panel.appendChild(shareSheet);

        // Links sheet — source repository + project website cards.
        // Built when panelLinks !== false (default true).  Both sourceBtn and
        // siteBtn in the sub-bar open this same sheet via _openSheet.
        var linksSheet = (cfgRef.panelLinks !== false) ? _buildLinksSheet() : null;
        if (linksSheet) panel.appendChild(linksSheet);

        /**
         * Open exactly one sheet at a time.  Pass null to close all.
         * @param {HTMLElement|null} target
         */
        /**
         * Element that held focus immediately before a sheet opened.
         * Restored by _closeSheet so keyboard users return to their trigger.
         * @type {Element|null}
         */
        var _sheetOpenerEl = null;

        /**
         * Open exactly one sheet at a time.  Pass null to close all.
         *
         * Moves focus into the newly-opened sheet (WCAG 2.1 SC 2.4.3) and
         * updates ``aria-expanded`` on the modelLink trigger when the model
         * sheet is the target.
         *
         * Parameters
         * ----------
         * target : HTMLElement|null
         *     Sheet element to open, or null to close every sheet.
         *
         * Notes
         * -----
         * Developer: Focus is moved inside a rAF so the browser has painted
         *   the sheet as visible before the focus call — calling focus() on a
         *   ``visibility: hidden`` element is a no-op in most engines.
         *
         * User: Escape key and the close (×) button both route through
         *   _closeSheet which restores focus to the originating button.
         */
        function _openSheet(target) {
            [modelSheet, privacySheet, termsSheet, shareSheet, linksSheet].forEach(function (s) {
                if (!s) return;
                s.setAttribute('data-open', (s === target) ? 'true' : 'false');
            });

            // Update aria-expanded on the sub-bar model trigger.
            if (modelLink) {
                modelLink.setAttribute('aria-expanded',
                    (target === modelSheet) ? 'true' : 'false');
            }
            // Update aria-expanded on the inline footer picker.
            if (inlinePicker && typeof inlinePicker.setAttribute === 'function') {
                inlinePicker.setAttribute('aria-expanded',
                    (target === modelSheet) ? 'true' : 'false');
            }

            if (target) {
                _sheetOpenerEl = document.activeElement;
                // Defer focus until the visibility transition has rendered.
                requestAnimationFrame(function () {
                    // Prefer the checked radio in the model sheet; fall back to
                    // the close button, then any focusable element.
                    var firstFocus =
                        target.querySelector('input[type="radio"]:checked') ||
                        target.querySelector('button') ||
                        target.querySelector(
                            '[href], input, [tabindex]:not([tabindex="-1"])'
                        );
                    if (firstFocus) firstFocus.focus();
                });
            }
        }

        /**
         * Close a specific sheet and return focus to the element that opened it.
         *
         * Parameters
         * ----------
         * sheet : HTMLElement
         *     The sheet element to close.
         *
         * Notes
         * -----
         * Developer: Always use _closeSheet instead of calling
         *   ``sheet.setAttribute('data-open', 'false')`` directly so that
         *   aria-expanded state and focus restoration are always kept in sync.
         */
        function _closeSheet(sheet) {
            if (!sheet) return;
            sheet.setAttribute('data-open', 'false');
            // Reset aria-expanded on both model triggers.
            if (modelLink) modelLink.setAttribute('aria-expanded', 'false');
            if (inlinePicker && typeof inlinePicker.setAttribute === 'function') {
                inlinePicker.setAttribute('aria-expanded', 'false');
            }
            // Return focus to the element that triggered the sheet.
            if (_sheetOpenerEl && typeof _sheetOpenerEl.focus === 'function') {
                _sheetOpenerEl.focus();
                _sheetOpenerEl = null;
            }
        }

        // Wire the sub-bar buttons.  Each handler routes through _openSheet
        // so the "only one open at a time" invariant is honoured centrally.
        if (modelLink) {
            // Issue 11: Declare popup type and initial collapsed state.
            modelLink.setAttribute('aria-haspopup', 'dialog');
            modelLink.setAttribute('aria-expanded', 'false');
            modelLink.setAttribute('aria-controls', 'ai-assistant-panel-model-sheet');
            modelLink.addEventListener('click', function () { _openSheet(modelSheet); });
        }
        // Issue 1: Wire the inline footer pill button to open the model sheet.
        // This must be done here (after modelSheet is in scope) rather than
        // inside _buildInlineModelPicker which runs before modelSheet exists.
        if (inlinePicker && modelSheet) {
            inlinePicker.addEventListener('click', function () {
                _openSheet(modelSheet);
            });
        }
        privacyLink.addEventListener('click', function () { _openSheet(privacySheet); });
        if (termsLink && termsSheet) {
            termsLink.addEventListener('click', function () { _openSheet(termsSheet); });
        }
        if (shareLink && shareSheet) {
            shareLink.addEventListener('click', function () { _openSheet(shareSheet); });
        }

        // Wire Source and Site buttons → linksSheet (or direct URL fallback).
        // When the links sheet is disabled (panelLinks: false), each button
        // falls back to opening its configured URL directly in a new tab so the
        // user is never stranded with a non-functional button.
        function _openLinksOrUrl(url) {
            if (linksSheet) {
                _openSheet(linksSheet);
            } else if (url && _isSafeHref(url)) {
                try {
                    var w = window.open(url, '_blank', 'noopener,noreferrer');
                    if (w) { try { w.opener = null; } catch (_) {} }
                } catch (_) {}
            }
        }
        if (sourceBtn) {
            sourceBtn.addEventListener('click', function () {
                _openLinksOrUrl(cfgRef.panelSourceUrl || '');
            });
        }
        if (siteBtn) {
            siteBtn.addEventListener('click', function () {
                _openLinksOrUrl(cfgRef.panelSiteUrl || '');
            });
        }

        // Sync the sub-bar model-link label with the active model whenever
        // the user changes it (via sheet or inline picker).  Reusing the
        // same DOM event the helpers already dispatch means there is exactly
        // one source-of-truth for the active model id (sessionStorage), and
        // every UI surface listens to the same change signal.
        if (modelLink) {
            document.addEventListener('ai-assistant-model-change', function (ev) {
                var d = ev && ev.detail;
                if (!d || typeof d.id !== 'string') return;
                var m = _findModel(cfgRef.panelApiModels || [], d.id);
                var text = m ? (m.label || m.id) : d.id;
                var lbl = modelLink.querySelector('.ai-assistant-panel-model-link-label');
                if (lbl) lbl.textContent = text;
                // Keep aria-label and title in sync with the selected model —
                // same format used by the inline-picker btn._syncState().
                modelLink.setAttribute('aria-label', 'Choose a model \u2014 current: ' + text);
                modelLink.title = text;
            });
        }

        // Hamburger overflow popover.  Built ONCE per panel; the same DOM
        // element is reused on every open/close so any internal state is
        // preserved across toggles.  The same popover is shared by both
        // the left hamburger button and the right overflow button — the
        // data-anchor attribute controls which side it appears on.
        var hamburgerMenuEl = null;
        if (hamburgerBtn) {
            hamburgerMenuEl = _buildHamburgerMenu({
                onModel:   modelLink   ? function () { _openSheet(modelSheet); }   : null,
                onPrivacy: function () { _openSheet(privacySheet); },
                onTerms:   termsSheet  ? function () { _openSheet(termsSheet); }   : null,
                onShare:   shareSheet  ? function () { _openSheet(shareSheet); }   : null,
                onLinks:   linksSheet  ? function () { _openSheet(linksSheet); }   : null,
            });
            panel.appendChild(hamburgerMenuEl);

            // Left hamburger: anchor popover to the left edge.
            hamburgerBtn.addEventListener('click', function (e) {
                _hapticFeedback([8]);
                e.stopPropagation();
                hamburgerMenuEl.setAttribute('data-anchor', 'left');
                var open = hamburgerMenuEl.getAttribute('data-open') === 'true';
                hamburgerMenuEl.setAttribute('data-open', open ? 'false' : 'true');
            });

            // Outside-click closes the popover (the panel listener below
            // handles Escape).  Using mousedown rather than click so a
            // click on a menu-item's own handler fires before this closer.
            document.addEventListener('mousedown', function (e) {
                if (!hamburgerMenuEl) return;
                if (hamburgerMenuEl.getAttribute('data-open') !== 'true') return;
                if (hamburgerBtn && hamburgerBtn.contains(e.target)) return;
                if (rightOverflowBtn && rightOverflowBtn.contains(e.target)) return;
                if (hamburgerMenuEl.contains(e.target)) return;
                hamburgerMenuEl.setAttribute('data-open', 'false');
            });
        }

        // Right overflow button: anchor the shared hamburger popover to the
        // right edge of the subbar — visible on narrow panels only (CSS).
        rightOverflowBtn.addEventListener('click', function (e) {
            _hapticFeedback([8]);
            e.stopPropagation();
            if (!hamburgerMenuEl) return;
            hamburgerMenuEl.setAttribute('data-anchor', 'right');
            var open = hamburgerMenuEl.getAttribute('data-open') === 'true';
            hamburgerMenuEl.setAttribute('data-open', open ? 'false' : 'true');
        });

        // R1: top-left resize grip (also restores any persisted size).
        _attachResizer(panel);

        // ResizeObserver: toggle data-narrow on the panel root when the panel
        // is narrower than the breakpoint so CSS collapses the subbar clusters
        // gracefully.  Falls back to the existing viewport-based @media rule on
        // browsers that don't support ResizeObserver (IE11, very old Safari).
        if (typeof ResizeObserver !== 'undefined') {
            /**
             * Progressive right-cluster overflow.
             *
             * Hides subbar-right items one-by-one as the panel narrows instead
             * of collapsing the entire cluster at once.  Priority order (first
             * to last to hide):
             *
             *   modelLink    575 px
             *   privacyLink  475 px
             *   termsLink    350 px
             *   shareLink    300 px
             *
             * Sets [data-overflow-hidden] on each item so CSS max-width + opacity
             * transitions can animate the collapse.  Sets [data-overflow-visible]
             * on the overflow button once any item is hidden so it fades in with a
             * 60 ms delay (items start squeezing before the ⋯ button appears).
             *
             * Items configured as null (feature-flagged off) are silently skipped.
             *
             * @param {number} w - Panel content rect width in pixels.
             */
            function _updateSubbarOverflow(w) {
                var slots = [
                    // Hide order: rightmost carriage departs first.
                    // Visual order left→right: Model | Privacy | Terms | Share
                    { el: modelLink,   px: 575 },   /* leftmost  — exits last   */
                    { el: privacyLink, px: 475 },
                    { el: termsLink,   px: 350 },
                    { el: shareLink,   px: 300 },   /* rightmost — exits first  */
                ];
                var anyHidden = false;
                for (var si = 0; si < slots.length; si++) {
                    var slot = slots[si];
                    if (!slot.el) continue;
                    if (w < slot.px) {
                        slot.el.setAttribute('data-overflow-hidden', '');
                        anyHidden = true;
                    } else {
                        slot.el.removeAttribute('data-overflow-hidden');
                    }
                }
                // Overflow button: fade in as soon as any item is hidden.
                if (anyHidden) {
                    rightOverflowBtn.setAttribute('data-overflow-visible', '');
                } else {
                    rightOverflowBtn.removeAttribute('data-overflow-visible');
                }
                // Left cluster: kbd-hint collapses only at very narrow widths.
                panel.setAttribute('data-narrow', w < 300 ? 'true' : 'false');
            }
            var _subbarRO = new ResizeObserver(function (entries) {
                var w = entries[0] && entries[0].contentRect && entries[0].contentRect.width;
                if (typeof w !== 'number') return;
                _updateSubbarOverflow(w);
            });
            _subbarRO.observe(panel);
        }

        // ── Events ────────────────────────────────────────────────────────────

        closeBtn.addEventListener('pointerdown', function () { _hapticFeedback([8]); });
        closeBtn.addEventListener('click', closeAIPanel);

        minimizeBtn.addEventListener('click', function () { _hapticFeedback([8]); minimizeAIPanel(); });
        // Right-click on the minimize button: fully close (mirrors kbd-hint / kbd-row contract).
        minimizeBtn.addEventListener('contextmenu', function (e) {
            e.preventDefault();
            closeAIPanel();
        });

        maximizeBtn.addEventListener('click', function () {
            _hapticFeedback([8]);
            var isMax = panel.getAttribute('data-maximized') === 'true';
            if (isMax) {
                // ── Restore ────────────────────────────────────────────────────
                panel.removeAttribute('data-maximized');
                maximizeBtn.setAttribute('aria-label', 'Maximize panel');
                maximizeBtn.innerHTML = ICONS.maximize;
                // Re-apply any manually-saved size so the panel returns to
                // exactly where the user left it before maximizing.
                var saved = _ssGet(_PANEL_SIZE_KEY);
                if (saved) {
                    try {
                        var s = JSON.parse(saved);
                        if (s && s.w && s.h) {
                            panel.style.width  = s.w + 'px';
                            panel.style.height = s.h + 'px';
                        }
                    } catch (_) {}
                } else {
                    // No saved size: clear inline styles so CSS defaults apply.
                    panel.style.width  = '';
                    panel.style.height = '';
                }
            } else {
                // ── Maximize ───────────────────────────────────────────────────
                // Clear any inline width/height set by the resize grips so the
                // CSS [data-maximized="true"] rules can take full control of
                // both dimensions — otherwise the inline values win in cascade.
                panel.style.width  = '';
                panel.style.height = '';
                panel.setAttribute('data-maximized', 'true');
                maximizeBtn.setAttribute('aria-label', 'Restore panel size');
                maximizeBtn.innerHTML = ICONS.restore;
            }
        });

        // Issue 4: Re-wire all sheet close (×) buttons to go through _closeSheet
        // so focus is always returned to the opener.  The hClose listeners
        // registered inside each sheet builder call sheet.setAttribute directly
        // (they run before _closeSheet exists); we add a second listener here
        // which performs the focus restoration after the flag is already 'false'.
        [modelSheet, privacySheet, termsSheet, shareSheet, linksSheet].forEach(function (s) {
            if (!s) return;
            var closeBtn = s.querySelector('button[id$="-close"]');
            if (!closeBtn) return;
            closeBtn.addEventListener('click', function () { _closeSheet(s); });
        });

        // Close the model sheet on any model-row click — whether the user
        // selects a different model OR re-clicks the already-active one.
        //
        // Why click instead of change:
        //   A radio 'change' event does NOT fire when the already-checked radio
        //   is clicked, so the change handler alone cannot close the sheet when
        //   the user confirms their current model selection.
        //
        // Why _closeSheet (not setAttribute):
        //   Routes through the same path as the × button and Escape key so
        //   aria-expanded is reset and keyboard focus always returns to the
        //   element that opened the sheet (Issue 4 parity for row selection).
        //
        // Info-link guard:
        //   Clicking the external info <a> inside a row must NOT close the
        //   sheet — the user is reading model information, not confirming a
        //   selection.  e.target.closest() is supported by all browsers that
        //   support the rest of this codebase; no polyfill required.
        if (modelSheet) {
            modelSheet.querySelectorAll('.ai-assistant-panel-model-row')
                .forEach(function (row) {
                    row.addEventListener('click', function (e) {
                        // Guard: ignore clicks that land on or inside the info link.
                        if (e.target &&
                            typeof e.target.closest === 'function' &&
                            e.target.closest('.ai-assistant-panel-model-info')) {
                            return;
                        }
                        _closeSheet(modelSheet);
                    });
                });
        }

        sendBtn.addEventListener('pointerdown', function () { _hapticFeedback([8]); });
        sendBtn.addEventListener('click', handleAIPanelSubmit);

        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAIPanelSubmit(); }
        });

        input.addEventListener('input', _updateSendBtnState);

        panel.addEventListener('keydown', function (e) {
            // Phase B: Escape closes the topmost overlay first, then the panel.
            // Priority (highest first):
            //   1. Hamburger popover (lightest overlay)
            //   2. Any open sheet (privacy / model / terms / share)
            //   3. The panel itself
            // The "any sheet" branch checks each in turn; only one is open
            // at a time per the _openSheet invariant, so the check is O(4).
            if (e.key !== 'Escape') return;
            if (hamburgerMenuEl &&
                hamburgerMenuEl.getAttribute('data-open') === 'true') {
                hamburgerMenuEl.setAttribute('data-open', 'false');
                return;
            }
            var openSheets = [privacySheet, modelSheet, termsSheet, shareSheet]
                .filter(function (s) {
                    return s && s.getAttribute('data-open') === 'true';
                });
            if (openSheets.length > 0) {
                // Issue 4: Use _closeSheet so focus is returned to the opener.
                openSheets.forEach(function (s) { _closeSheet(s); });
                return;
            }
            closeAIPanel();
        });

        document.body.appendChild(panel);

        // ── Floating trigger pill (shown when minimized) ──────────────────────
        // Idempotency guard (C-4): never create a second trigger if one
        // already exists from a prior createAIPanel() call.
        if (!_aiTriggerEl) {
            _aiTriggerEl = _createTriggerPill(title);
            document.body.appendChild(_aiTriggerEl);
        }

        return panel;
    }

    /**
     * Create the floating "Ask Us" trigger pill shown when the panel is minimized.
     *
     * @param {string} title  Panel title for aria-label.
     * @returns {HTMLButtonElement}
     */
    function _createTriggerPill(title) {
        var trigger = document.createElement('button');
        trigger.id = 'ai-assistant-trigger';
        trigger.className = 'ai-assistant-panel-trigger';
        trigger.type = 'button';
        trigger.setAttribute('aria-label', 'Open ' + _escapeHtml(title));
        trigger.setAttribute('title', 'Open ' + _escapeHtml(title));

        // Icon
        var iconWrap = document.createElement('span');
        iconWrap.setAttribute('aria-hidden', 'true');
        iconWrap.innerHTML = ICONS.chat;   // ICONS constant — safe.

        var label = document.createElement('span');
        // BUG-FIX: was hardcoded 'Ask Us' — now reads cfg.panelTriggerLabel
        // so ai_assistant_panel_trigger_label in conf.py is actually applied.
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        label.textContent = cfg.panelTriggerLabel || 'Ask Us';

        trigger.appendChild(iconWrap);
        trigger.appendChild(label);

        // Tap haptic fires on physical press (pointerdown) for immediate feedback.
        trigger.addEventListener('pointerdown', function () { _hapticFeedback([8]); });
        trigger.addEventListener('click', function () { restoreAIPanel(); });

        // Long-press (500 ms hold): close the panel completely rather than restore.
        // onShortTap is null — the click handler above manages the normal tap action.
        // hapticTap is null  — the pointerdown listener above already fired on press.
        // Ghost-click absorber inside _attachLongPress prevents the subsequent
        // synthetic click from re-triggering restoreAIPanel() after closeAIPanel().
        _attachLongPress(trigger, null, function () { closeAIPanel(); }, {
            hapticTap:       null,
            hapticLongPress: [12, 40, 12],
        });

        return trigger;
    }

    // ── Panel state management ────────────────────────────────────────────────

    /** Toggle the AI panel open/closed. Creates it on first call. */
    function toggleAIPanel() {
        if (!_aiPanelEl) _aiPanelEl = createAIPanel();

        var isVisible = _aiPanelEl.style.display !== 'none';
        if (isVisible) {
            closeAIPanel();
        } else {
            _openAIPanel();
        }
    }

    /** Slide the panel in (shared by toggleAIPanel and restoreAIPanel). */
    function _openAIPanel() {
        if (!_aiPanelEl) _aiPanelEl = createAIPanel();

        // If was minimized, hide trigger first
        if (_aiTriggerEl) {
            _aiTriggerEl.removeAttribute('data-minimized');
            _aiTriggerEl.style.display = 'none';
        }
        _aiPanelEl.removeAttribute('data-minimized');
        _aiPanelEl.style.display = 'flex';
        requestAnimationFrame(function () {
            _aiPanelEl.classList.add('ai-assistant-panel--open');
        });
        var inp = document.getElementById('ai-assistant-panel-input');
        if (inp) setTimeout(function () {
            inp.focus();
            /* Resize on open: handles the case where the panel is reopened with
               a pre-filled textarea (retry, chip, or restored draft). */
            _autoResizeInput(inp);
        }, 100);
    }

    /**
     * Minimize: hide the panel but keep conversation state.
     * Show the floating trigger pill so the user can reopen.
     */
    function minimizeAIPanel() {
        if (!_aiPanelEl) return;
        _aiPanelEl.classList.remove('ai-assistant-panel--open');
        setTimeout(function () {
            if (_aiPanelEl) {
                _aiPanelEl.style.display = 'none';
                _aiPanelEl.setAttribute('data-minimized', 'true');
            }
            if (_aiTriggerEl) {
                _aiTriggerEl.setAttribute('data-minimized', 'true');
                _aiTriggerEl.style.display = 'flex';
            }
        }, 280);
    }

    /** Restore from minimized state. */
    function restoreAIPanel() {
        if (!_aiPanelEl) { _openAIPanel(); return; }
        _openAIPanel();
    }

    /** Fully close the panel (slide out animation). */
    function closeAIPanel() {
        if (!_aiPanelEl) return;
        _aiPanelEl.classList.remove('ai-assistant-panel--open');
        _aiPanelEl.removeAttribute('data-minimized');
        if (_aiTriggerEl) {
            _aiTriggerEl.removeAttribute('data-minimized');
            _aiTriggerEl.style.display = 'none';
        }
        // Stop speech recognition if active
        _stopSpeechRecognition();
        setTimeout(function () {
            if (_aiPanelEl) _aiPanelEl.style.display = 'none';
        }, 300);
        var dropBtn = document.getElementById('ai-assistant-button-dropdown');
        if (dropBtn) dropBtn.focus();
    }

    // ── Mic hover popup ───────────────────────────────────────────────────────

    /**
     * Build the mic hover popup element (right-anchored, above the action bar).
     *
     * Contains two rows separated by a thin rule:
     *   1. Voice-level visualization — seven bars driven by CSS animation when
     *      `data-active="true"` is set on `.ai-assistant-mic-level-bars` via
     *      `_setMicActiveState`. Bars are purely decorative (aria-hidden).
     *   2. Hold-to-record row — finger icon + label + on/off toggle switch.
     *      Clicking the row or the toggle calls `_setMicHoldMode`.
     *
     * Visibility contract (CSS-driven):
     *   • `.ai-assistant-mic-wrapper:hover .ai-assistant-mic-popup` → visible.
     *   • The popup stays visible while interacting with the toggle (hover does
     *     not leave the wrapper).
     *
     * @returns {HTMLElement}
     */
    function _buildMicHoverPopup() {
        var popup = document.createElement('div');
        popup.className = 'ai-assistant-mic-popup';
        popup.id = 'ai-assistant-mic-popup';
        popup.setAttribute('role', 'group');
        popup.setAttribute('aria-label', 'Microphone options');

        // ── Row 1: voice level bars ───────────────────────────────────────────
        var levelRow = document.createElement('div');
        levelRow.className = 'ai-assistant-mic-popup-row ai-assistant-mic-popup-row--level';
        levelRow.setAttribute('aria-hidden', 'true');   // decorative

        var levelBars = document.createElement('div');
        levelBars.className = 'ai-assistant-mic-level-bars';
        levelBars.id = 'ai-assistant-mic-level-bars';

        // 100 bars; sinusoidal idle heights (2px edges → 13px centre) create
        // a natural waveform silhouette that mirrors spoken-word audio profiles.
        var BAR_COUNT = _MIC_LEVEL_BAR_COUNT;  // 100 bars — 0-100 dBFS VU meter scale
        var _idleHeights = _IDLE_LEVEL_HEIGHTS;  // module-level constant
        for (var _b = 0; _b < BAR_COUNT; _b++) {
            var bar = document.createElement('span');
            bar.className = 'ai-mic-bar';
            bar.style.height = _idleHeights[_b] + 'px';
            levelBars.appendChild(bar);
        }

        levelRow.appendChild(levelBars);
        popup.appendChild(levelRow);

        // ── Separator ─────────────────────────────────────────────────────────
        var sep = document.createElement('div');
        sep.className = 'ai-assistant-mic-popup-sep';
        sep.setAttribute('aria-hidden', 'true');
        popup.appendChild(sep);

        // ── Devices section ───────────────────────────────────────────────────
        //
        // DOM layout:
        //   .ai-assistant-mic-devices-section
        //     .ai-assistant-mic-devices-header   ← "Microphone" group label
        //     .ai-assistant-mic-device-list       ← async-populated device items

        var devSection = document.createElement('div');
        devSection.className = 'ai-assistant-mic-devices-section';

        // Group header: mic icon + "Microphone" label (decorative only)
        var devHeader = document.createElement('div');
        devHeader.className = 'ai-assistant-mic-devices-header';
        devHeader.setAttribute('aria-hidden', 'true');

        var devHeaderIcon = document.createElement('span');
        devHeaderIcon.className = 'ai-assistant-mic-devices-header-icon';
        devHeaderIcon.setAttribute('aria-hidden', 'true');
        devHeaderIcon.innerHTML =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
            + ' stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
            + '<rect x="9" y="2" width="6" height="11" rx="3"/>'
            + '<path d="M5 10a7 7 0 0 0 14 0"/>'
            + '<line x1="12" y1="19" x2="12" y2="22"/>'
            + '<line x1="9" y1="22" x2="15" y2="22"/>'
            + '</svg>';

        var devHeaderLabel = document.createElement('span');
        devHeaderLabel.className = 'ai-assistant-mic-devices-header-label';
        devHeaderLabel.textContent = 'Microphone';

        devHeader.appendChild(devHeaderIcon);
        devHeader.appendChild(devHeaderLabel);
        devSection.appendChild(devHeader);

        // Permission bar — shows mic permission state + browser URL-bar guidance
        // Built once here; updated every popup open by _refreshMicDeviceList.
        var permBar = _buildMicPermissionBar();
        devSection.appendChild(permBar);

        // Device list — populated asynchronously when popup is pinned open
        var devList = document.createElement('div');
        devList.className = 'ai-assistant-mic-device-list';
        devList.setAttribute('role', 'group');
        devList.setAttribute('aria-label', 'Select microphone');
        devSection.appendChild(devList);

        popup.appendChild(devSection);

        // Separator between devices section and hold-to-record row
        var sep2 = document.createElement('div');
        sep2.className = 'ai-assistant-mic-popup-sep';
        sep2.setAttribute('aria-hidden', 'true');
        popup.appendChild(sep2);

        // ── Row 2: hold-to-record toggle ──────────────────────────────────────
        var holdRow = document.createElement('div');
        holdRow.className = 'ai-assistant-mic-popup-row ai-assistant-mic-popup-row--hold';
        holdRow.setAttribute('role', 'button');
        holdRow.setAttribute('tabindex', '-1');
        holdRow.setAttribute('aria-label', 'Toggle hold-to-record mode');

        var iconSpan = document.createElement('span');
        iconSpan.className = 'ai-assistant-mic-popup-icon';
        iconSpan.setAttribute('aria-hidden', 'true');
        iconSpan.innerHTML =
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
            + ' stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
            + '<path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/>'
            + '<path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/>'
            + '<path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/>'
            + '<path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34'
            + 'l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>'
            + '</svg>';

        var label = document.createElement('span');
        label.className = 'ai-assistant-mic-popup-label';
        label.textContent = 'Hold to record';

        var toggle = document.createElement('button');
        toggle.className = 'ai-assistant-mic-popup-toggle';
        toggle.id = 'ai-assistant-mic-hold-toggle';
        toggle.type = 'button';
        toggle.setAttribute('aria-pressed', _micHoldMode ? 'true' : 'false');
        toggle.setAttribute('aria-label', 'Hold-to-record mode');
        toggle.setAttribute('title',
            _micHoldMode ? 'Hold-to-record: ON' : 'Hold-to-record: OFF');

        var track = document.createElement('span');
        track.className = 'ai-assistant-mic-toggle-track';
        var thumb = document.createElement('span');
        thumb.className = 'ai-assistant-mic-toggle-thumb';
        track.appendChild(thumb);
        toggle.appendChild(track);

        toggle.addEventListener('click', function (e) {
            e.stopPropagation();
            _setMicHoldMode(!_micHoldMode);
        });

        // Clicking the row label/icon (but NOT the toggle button itself) should
        // also toggle hold mode.  The toggle's own click handler calls
        // stopPropagation so normally only one handler fires per click, but we
        // add an explicit guard here so the behaviour is correct even if that
        // assumption ever breaks (e.g. keyboard synthetic click on holdRow,
        // assistive-technology double-dispatch on role="button" + nested button).
        holdRow.addEventListener('click', function (e) {
            if (toggle.contains(e.target)) { return; }   // already handled above
            _setMicHoldMode(!_micHoldMode);
        });

        holdRow.appendChild(iconSpan);
        holdRow.appendChild(label);
        holdRow.appendChild(toggle);
        popup.appendChild(holdRow);

        // ── Device list: populate on popup open via MutationObserver ──────────
        //
        // Re-enumerates on every open so newly plugged-in devices appear without
        // a page reload.  The first open after permission grant will also return
        // real labels (not placeholder "Microphone N" strings).
        (function () {
            var obs = new MutationObserver(function () {
                if (popup.getAttribute('data-pinned') === 'true') {
                    _refreshMicDeviceList(devList);
                }
            });
            obs.observe(popup, { attributes: true, attributeFilter: ['data-pinned'] });
        }());

        // ── Drag-to-move popup ────────────────────────────────────────────────
        //
        // The voice-level row (.ai-assistant-mic-popup-row--level) is the drag
        // handle (grab-cursor affordance set in CSS).  On the FIRST real drag
        // (mousemove ≥ 3px from mousedown origin) the popup's CSS bottom/right
        // anchors are replaced with explicit top/left offsets relative to its
        // offsetParent (.ai-assistant-mic-wrapper, position:relative).
        // Subsequent drags update those top/left values directly.
        //
        // WHY NOT position:fixed?
        //   .ai-assistant-panel carries transform:translateY(0) scale(1) when
        //   open.  Any non-"none" transform — even an identity one — creates a
        //   new CSS containing block for position:fixed descendants (CSS spec
        //   §9.3).  This means a fixed popup's top/left are interpreted as
        //   offsets from the panel's border edge, not the viewport, so setting
        //   them to getBoundingClientRect() viewport values places the popup
        //   far off-screen and it cannot be recovered without a page refresh.
        //   Keeping position:absolute and working in offsetParent coordinates
        //   avoids the issue entirely.  All ancestors have overflow:visible so
        //   the popup travels freely outside its containing box.
        //
        // Behaviour contract:
        //   • Left-button (button === 0) only.
        //   • preventDefault() on mousedown prevents unintended text selection.
        //   • top/left override is deferred until first mousemove (≥ 3px) so
        //     that a plain click on the level row NEVER sets data-dragged — this
        //     is the root-cause fix for "bars disappear when clicking": the old
        //     code set data-dragged on mousedown, which triggered the CSS opacity
        //     transition and briefly hid/flash-showed the popup before
        //     data-pinned reasserted.
        //   • document-level mousemove / mouseup give reliable tracking even
        //     when the cursor momentarily leaves the popup during fast moves.
        //   • The `data-dragged="true"` attribute lets CSS suppress the wrapper-
        //     hover reveal so visibility is controlled exclusively by data-pinned.
        //   • No persistent listeners are added — the two document handlers
        //     are no-ops whenever `_micDragging` is false.
        (function () {
            var _micDragging = false;
            var _originX  = 0, _originY  = 0;
            var _startLeft = 0, _startTop = 0;

            popup.addEventListener('mousedown', function (e) {
                if (e.button !== 0) return;                    // left-button only
                if (!e.target.closest('.ai-assistant-mic-popup-row--level')) return;
                e.preventDefault();                            // no text selection

                _micDragging = true;
                _originX     = e.clientX;
                _originY     = e.clientY;

                // Capture start position now for already-promoted popups;
                // for first-drag cases it is captured in the mousemove handler
                // after promotion (see below), avoiding a stale-rect read.
                if (popup.getAttribute('data-dragged') === 'true') {
                    _startLeft = parseFloat(popup.style.left) || 0;
                    _startTop  = parseFloat(popup.style.top)  || 0;
                }

                popup.style.cursor = 'grabbing';
            });

            document.addEventListener('mousemove', function (e) {
                if (!_micDragging) return;

                // ── First drag: anchor top/left in offsetParent space ─────────
                // Only activate after a meaningful movement (≥ 3px Manhattan
                // distance) so an accidental mousedown+mouseup (click) on the
                // level row never sets data-dragged and never triggers the CSS
                // opacity transition that momentarily hid bars on click.
                if (popup.getAttribute('data-dragged') !== 'true') {
                    var moved = Math.abs(e.clientX - _originX)
                              + Math.abs(e.clientY - _originY);
                    if (moved < 3) { return; }                 // below threshold

                    // Convert the popup's current viewport position into
                    // offsetParent-relative (micWrapper) coordinates.
                    //
                    // getBoundingClientRect() accounts for ancestor transforms
                    // visually.  Since the panel's transform is scale(1) (no
                    // distortion), 1 viewport-px == 1 wrapper-px, so viewport
                    // deltas translate directly to offsetParent deltas.
                    //
                    // clientTop/clientLeft correct for any border on offsetParent
                    // (currently 0 for micWrapper, but included for robustness).
                    var popupRect  = popup.getBoundingClientRect();
                    var opEl       = popup.offsetParent || document.documentElement;
                    var opRect     = opEl.getBoundingClientRect();
                    popup.style.top    = (popupRect.top  - opRect.top  - (opEl.clientTop  || 0)) + 'px';
                    popup.style.left   = (popupRect.left - opRect.left - (opEl.clientLeft || 0)) + 'px';
                    popup.style.bottom = 'auto';
                    popup.style.right  = 'auto';
                    // position stays 'absolute' — do NOT set position:fixed.
                    // See IIFE comment above for the full technical rationale.
                    popup.setAttribute('data-dragged', 'true');

                    // Re-read after layout update (inline styles now authoritative)
                    _startLeft = parseFloat(popup.style.left) || 0;
                    _startTop  = parseFloat(popup.style.top)  || 0;
                    // Reset origin so the first move delta is computed correctly
                    // from the new base position.
                    _originX   = e.clientX;
                    _originY   = e.clientY;
                    // Fall through — delta is 0 this frame so no visible jump,
                    // but removing the early return means drag starts tracking
                    // immediately (no 1-frame stutter / lag on drag-start).
                }

                popup.style.left = (_startLeft + (e.clientX - _originX)) + 'px';
                popup.style.top  = (_startTop  + (e.clientY - _originY)) + 'px';
            });

            document.addEventListener('mouseup', function () {
                if (!_micDragging) return;
                _micDragging       = false;
                popup.style.cursor = '';

                // After a real drag the browser synthesizes a click event on
                // the element under the pointer.  If that element is outside
                // micWrapper the capture-phase _closePinnedMicPopup listener
                // would fire and immediately unpin the popup right after the
                // user releases the drag.
                //
                // Guard: mark the popup as having just finished a drag.  The
                // one-shot capture click listener below reads and clears the
                // flag before _closePinnedMicPopup (also capture) can act on
                // it, because listeners registered later in the same phase fire
                // in registration order.  The flag is also set only when a real
                // drag occurred (data-dragged="true"), so plain clicks are never
                // affected.
                if (popup.getAttribute('data-dragged') === 'true') {
                    popup.setAttribute('data-just-dragged', 'true');
                    document.addEventListener('click', function _absorbPostDragClick(ev) {
                        document.removeEventListener('click', _absorbPostDragClick, true);
                        popup.removeAttribute('data-just-dragged');
                        ev.stopPropagation();   // prevent _closePinnedMicPopup
                    }, true);
                }
            });
        }());

        return popup;
    }

    /**
     * Set hold-to-record mode and sync all dependent UI elements.
     *
     * Hold mode ON  → pointerdown starts / pointerup stops recognition.
     * Hold mode OFF → click toggles recognition (default behaviour).
     *
     * Persists the preference to localStorage so it survives page reloads.
     * Silently ignores storage errors (private mode, quota exceeded, etc.).
     *
     * @param {boolean} enabled  True = enable hold mode; false = toggle mode.
     */
    function _setMicHoldMode(enabled) {
        _micHoldMode = !!enabled;

        // Persist preference
        try {
            localStorage.setItem('ai-assistant-mic-hold-mode', _micHoldMode ? 'true' : 'false');
        } catch (_) {}

        // Sync toggle button
        var toggle = document.getElementById('ai-assistant-mic-hold-toggle');
        if (toggle) {
            toggle.setAttribute('aria-pressed', _micHoldMode ? 'true' : 'false');
            toggle.setAttribute('title',
                _micHoldMode ? 'Hold-to-record: ON' : 'Hold-to-record: OFF');
        }

        // Sync mic button label and data attribute
        var micBtn = document.getElementById('ai-assistant-panel-mic');
        if (micBtn) {
            micBtn.setAttribute('data-hold', _micHoldMode ? 'true' : 'false');
            var newLabel = _isListening
                ? 'Stop recording'
                : (_micHoldMode ? 'Press and hold to record' : 'Speak your question');
            micBtn.setAttribute('aria-label', newLabel);
            micBtn.setAttribute('title', newLabel);
        }
    }

    // ── Mic device management ─────────────────────────────────────────────────

    /**
     * Stop and release the device-pin MediaStreamTrack.
     *
     * Idempotent — safe to call when _micPinTrack is already null.
     * Called from _setMicDevice (on device change) and from the cold-path inside
     * _toggleSpeechRecognition before re-acquiring a fresh pin track.
     *
     * Notes
     * -----
     * Do NOT call this on recognition end/stop — the track must stay alive across
     * hold-to-record presses to prevent browser permission re-prompts.
     */
    function _releaseMicPinTrack() {
        if (_micPinTrack) {
            try { _micPinTrack.stop(); } catch (_) {}
            _micPinTrack = null;
        }
    }

    /**
     * Stop and release the warm permission MediaStream.
     *
     * Idempotent — safe to call when _micWarmStream is already null.
     * Called only from _setMicDevice when the selected device changes so the
     * next _acquireMicWarmStream() re-acquires on the newly selected device.
     *
     * Notes
     * -----
     * Do NOT call this on recognition end/stop.  The stream must stay alive
     * across hold-to-record presses to prevent permission re-prompts.
     */
    function _releaseMicWarmStream() {
        if (_micWarmStream) {
            try {
                _micWarmStream.getTracks().forEach(function (t) { t.stop(); });
            } catch (_) {}
            _micWarmStream = null;
        }
    }

    /**
     * Acquire (or reuse) the persistent warm MediaStream.
     *
     * If _micWarmStream already contains at least one live track the callback
     * is invoked synchronously and no new getUserMedia call is issued — this is
     * the hot path that executes on every subsequent hold press without any
     * latency or permission dialog.
     *
     * On the cold path (first press or after device change), getUserMedia is
     * called once with the constraints for the currently selected device.  The
     * browser shows its permission dialog at most once per origin per device.
     * After the stream is established the callback is invoked; on failure the
     * callback receives null and _micWarmStream remains null (recording falls
     * back to the SpeechRecognition API's own internal stream).
     *
     * Parameters
     * ----------
     * callback : function(MediaStream|null)
     *     Invoked when the stream is ready (or on failure).  May be omitted.
     *
     * Notes
     * -----
     * The constraints honour _micDeviceId so the warm stream is pinned to the
     * user's chosen device from the first press onward.
     */
    function _acquireMicWarmStream(callback) {
        // Hot path: reuse the existing live stream — zero latency, no dialog.
        if (_micWarmStream) {
            var liveTracks = _micWarmStream.getTracks().filter(function (t) {
                return t.readyState === 'live';
            });
            if (liveTracks.length > 0) {
                if (callback) { callback(_micWarmStream); }
                return;
            }
            // All tracks ended (e.g. device unplugged) — discard stale ref.
            _micWarmStream = null;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            if (callback) { callback(null); }
            return;
        }

        // Use exact device constraint when one is selected, otherwise use the
        // browser default.  Matching the constraint to _micDeviceId ensures the
        // warm stream actually pins the right hardware from the first use.
        var constraints = _micDeviceId
            ? { audio: { deviceId: { exact: _micDeviceId } } }
            : { audio: true };

        navigator.mediaDevices.getUserMedia(constraints)
            .then(function (stream) {
                _micWarmStream = stream;
                if (callback) { callback(stream); }
            })
            .catch(function (err) {
                console.warn('AI Assistant: Warm stream acquisition failed:', err);
                _micWarmStream = null;
                if (callback) { callback(null); }
            });
    }

    /**
     * Guard: prevents duplicate devicechange registrations across popup rebuilds.
     * Set to true on first call to _enumMicDevices; survives the page lifetime.
     * @type {boolean}
     */
    var _micDeviceChangeListenerAdded = false;

    /**
     * Enumerate available audio input devices and cache the result.
     *
     * Parameters
     * ----------
     * callback : function(Array<{deviceId:string, label:string}>, string)
     *     Invoked with (deviceList, permissionState).
     *     permissionState is one of: 'granted' | 'denied' | 'prompt' | 'unsupported'.
     *
     * Notes
     * -----
     * Permission API (Chromium/Edge): navigator.permissions.query({ name:'microphone' })
     *   resolves before enumerateDevices so the callback always receives an accurate
     *   state string on first open.
     *
     * devicechange (all modern browsers): registered once for the page lifetime so
     *   newly plugged-in or unplugged devices refresh the list automatically.
     *
     * permissionchange: registered on the PermissionStatus object so permission
     *   revocations in browser Settings update the UI without a page reload.
     *
     * Firefox quirk: the Permissions API does not expose 'microphone' on Firefox;
     *   we catch the rejection and fall back to 'prompt' (safe default).
     *
     * Non-secure context (http://): mediaDevices is undefined in modern browsers;
     *   the function exits immediately with ('unsupported').
     */
    function _enumMicDevices(callback) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            callback([], 'unsupported');
            return;
        }

        // ── devicechange: refresh device list on hot-plug (once per page) ───
        if (!_micDeviceChangeListenerAdded) {
            _micDeviceChangeListenerAdded = true;
            try {
                navigator.mediaDevices.addEventListener('devicechange', function () {
                    var listEl = document.querySelector('.ai-assistant-mic-device-list');
                    if (listEl) { _refreshMicDeviceList(listEl); }
                });
            } catch (_) {}
        }

        // ── Core enumeration logic ────────────────────────────────────────────
        function doEnumerate(permState) {
            navigator.mediaDevices.enumerateDevices().then(function (devices) {
                var counter = 0;
                _micDevices = devices
                    .filter(function (d) { return d.kind === 'audioinput'; })
                    .map(function (d) {
                        counter++;
                        return {
                            deviceId: d.deviceId,
                            label:    d.label || ('Microphone ' + counter)
                        };
                    });
                callback(_micDevices, permState);
            }).catch(function () {
                callback([], 'unsupported');
            });
        }

        // ── Permissions API: query before enumerate so callback gets real state ─
        if (navigator.permissions && navigator.permissions.query) {
            navigator.permissions.query({ name: 'microphone' }).then(function (status) {
                doEnumerate(status.state); // 'granted' | 'denied' | 'prompt'

                // permissionchange: auto-refresh when user revokes in browser Settings
                status.onchange = function () {
                    var listEl = document.querySelector('.ai-assistant-mic-device-list');
                    if (listEl) { _refreshMicDeviceList(listEl); }
                };
            }).catch(function () {
                // Firefox: Permissions API rejects for 'microphone' — fall back gracefully
                doEnumerate('prompt');
            });
        } else {
            doEnumerate('prompt');
        }
    }

    /**
     * Select a microphone device and persist the choice.
     *
     * Updates all .ai-assistant-mic-device-item aria-checked states in the DOM.
     * Safe to call whether or not the popup is currently visible.
     *
     * Parameters
     * ----------
     * deviceId : string
     *     MediaDeviceInfo.deviceId, or '' / 'default' for the browser default.
     */
    function _setMicDevice(deviceId) {
        var newId = (deviceId === 'default') ? '' : (deviceId || '');

        // Only release existing tracks when the device actually changes.
        // Re-selecting the same device must not interrupt a live warm stream.
        if (newId !== _micDeviceId) {
            _releaseMicPinTrack();
            _releaseMicWarmStream();
        }

        _micDeviceId = newId;
        try {
            if (_micDeviceId) {
                localStorage.setItem('ai-assistant-mic-device-id', _micDeviceId);
            } else {
                localStorage.removeItem('ai-assistant-mic-device-id');
            }
        } catch (_) {}
        _syncMicDeviceUI();
    }

    /**
     * Sync all .ai-assistant-mic-device-item aria-checked attributes to the
     * current _micDeviceId.
     *
     * Decoupled from _setMicDevice so _refreshMicDeviceList can call it after
     * re-rendering without triggering a redundant localStorage write cycle.
     */
    function _syncMicDeviceUI() {
        var items = document.querySelectorAll('.ai-assistant-mic-device-item');
        var effectiveId = _micDeviceId || 'default';
        for (var i = 0; i < items.length; i++) {
            var active = items[i].getAttribute('data-device-id') === effectiveId;
            items[i].setAttribute('aria-checked', active ? 'true' : 'false');
        }
    }

    /**
     * Update the permission-bar element to reflect the current permission state.
     *
     * Parameters
     * ----------
     * permState : string
     *     One of 'granted' | 'denied' | 'prompt' | 'unsupported'.
     * hasRealLabels : boolean
     *     True when at least one device has a real OS-assigned label (i.e. the
     *     user has already granted permission in a prior session).
     *
     * Notes
     * -----
     * When permState is 'granted' the bar is hidden via CSS (data-permission="granted").
     * For 'denied' and 'prompt' the URL-bar mockup becomes visible via CSS.
     * The aria-live="polite" on the bar ensures screen readers announce changes.
     */
    function _updateMicPermissionBar(permState, hasRealLabels) {
        var bar = document.getElementById('ai-assistant-mic-permission-bar');
        if (!bar) { return; }

        bar.setAttribute('data-permission', permState);

        var iconEl = bar.querySelector('.ai-assistant-mic-perm-icon');
        var textEl = bar.querySelector('.ai-assistant-mic-perm-text');
        if (!iconEl || !textEl) { return; }

        var SVG_CHECK =
            '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
            + ' stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"'
            + ' aria-hidden="true">'
            + '<polyline points="3 8.5 6.5 12 13 4.5"/>'
            + '</svg>';
        var SVG_INFO =
            '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
            + ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
            + ' aria-hidden="true">'
            + '<circle cx="8" cy="8" r="6.5"/>'
            + '<line x1="8" y1="7" x2="8" y2="11"/>'
            + '<circle cx="8" cy="5" r="0.6" fill="currentColor" stroke="none"/>'
            + '</svg>';
        var SVG_BLOCK =
            '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
            + ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
            + ' aria-hidden="true">'
            + '<circle cx="8" cy="8" r="6.5"/>'
            + '<line x1="10.5" y1="5.5" x2="5.5" y2="10.5"/>'
            + '<line x1="5.5" y1="5.5" x2="10.5" y2="10.5"/>'
            + '</svg>';

        if (permState === 'granted') {
            iconEl.innerHTML = SVG_CHECK;
            textEl.textContent = 'Microphone permitted';
        } else if (permState === 'denied') {
            iconEl.innerHTML = SVG_BLOCK;
            textEl.textContent = 'Microphone blocked \u2014 see instructions below';
        } else if (permState === 'unsupported') {
            iconEl.innerHTML = SVG_INFO;
            textEl.textContent = 'Permission API not available';
        } else {
            // 'prompt': permission not yet decided
            if (hasRealLabels) {
                // enumerateDevices already returns real labels — treat as granted
                iconEl.innerHTML = SVG_CHECK;
                textEl.textContent = 'Microphone permitted';
                bar.setAttribute('data-permission', 'granted');
            } else {
                iconEl.innerHTML = SVG_INFO;
                textEl.textContent = 'Click the mic icon in your address bar to allow';
            }
        }

        // ── Browser-specific settings guide ───────────────────────────────────
        //
        // Populate once per update; the section is hidden for 'granted' and
        // 'unsupported' by the bar's CSS collapse.
        var settSec = bar.querySelector('.ai-assistant-mic-perm-settings');
        if (!settSec) { return; }

        var showSettings = (permState !== 'granted' && permState !== 'unsupported');
        settSec.style.display = showSettings ? '' : 'none';
        if (!showSettings) { return; }

        var info = _getBrowserSettingsInfo();
        console.log('info=', info);

        // Browser badge
        var badge = settSec.querySelector('.ai-assistant-mic-perm-settings-browser');
        if (badge) { badge.textContent = info.displayName; }

        // URL row
        var urlRowEl = settSec.querySelector('.ai-assistant-mic-perm-settings-url-row');
        var urlCode  = settSec.querySelector('.ai-assistant-mic-perm-settings-url');
        var copyBtn  = settSec.querySelector('.ai-assistant-mic-perm-settings-copy');

        if (urlRowEl && urlCode && copyBtn) {
            if (info.settingsUrl) {
                urlCode.textContent = info.settingsUrl;
                // Replace onclick each update so the closure captures the current URL
                copyBtn.onclick = (function (url, btn) {
                    return function () {
                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(url).then(function () {
                                var prev = btn.textContent;
                                btn.textContent = '\u2713 Copied';
                                setTimeout(function () { btn.textContent = prev; }, 1800);
                            }).catch(function () { _selectSettingsUrl(urlCode); });
                        } else {
                            _selectSettingsUrl(urlCode);
                        }
                    };
                }(info.settingsUrl, copyBtn));
                urlRowEl.style.display = '';
            } else {
                urlRowEl.style.display = 'none';
            }
        }

        // Steps list
        var stepsList = settSec.querySelector('.ai-assistant-mic-perm-settings-steps');
        console.log(
        'steps count=',
        stepsList ? stepsList.children.length : 'missing'
        );
        if (stepsList) {
            stepsList.innerHTML = '';
            info.steps.forEach(function (step) {
                var li = document.createElement('li');
                li.textContent = step;
                stepsList.appendChild(li);
            });
        }

        // Legacy steps list
        var legacyBody = settSec.querySelector('.ai-assistant-mic-perm-settings-legacy-body');
        if (legacyBody) {
            legacyBody.innerHTML = '';
            info.legacySteps.forEach(function (step) {
                var li = document.createElement('li');
                li.textContent = step;
                legacyBody.appendChild(li);
            });
        }
    }

    /**
     * Detect the running browser from the user-agent and vendor strings.
     *
     * Returns
     * -------
     * string
     *     One of: 'chrome' | 'edge' | 'firefox' | 'opera' | 'brave' | 'safari' | 'other'.
     *
     * Notes
     * -----
     * Detection order matters: Brave, Edge, and Opera all include "Chrome" in their
     * UA strings.  Brave is identified via navigator.brave (synchronous flag it
     * injects); Edge via /edg\//; Opera via /opr\//; then Chrome via vendor.
     */
    function _detectBrowser() {
        var ua     = navigator.userAgent;
        var vendor = (navigator.vendor || '').toLowerCase();
        if (navigator.brave)             { return 'brave'; }
        if (/edg\//i.test(ua))           { return 'edge'; }
        if (/opr\/|opera/i.test(ua))     { return 'opera'; }
        if (/firefox/i.test(ua))         { return 'firefox'; }
        if (/chrome/i.test(ua) && /google/i.test(vendor)) { return 'chrome'; }
        if (/safari/i.test(ua) && !/chrome/i.test(ua))    { return 'safari'; }
        return 'other';
    }

    /**
     * Return browser-specific microphone permission settings information.
     *
     * Returns
     * -------
     * {displayName, settingsUrl, steps, legacySteps}
     *     displayName  : string        browser name for the UI badge
     *     settingsUrl  : string|null   paste-into-address-bar URL (null = none available)
     *     steps        : string[]      ordered steps for current / new-UI method
     *     legacySteps  : string[]      alternative steps for older browser versions
     *                                  OR address-bar icon method
     *
     * Notes
     * -----
     * Developer: chrome://, edge://, opera://, brave:// URLs cannot be navigated
     *   to from a web page via window.open() or an anchor element — all modern
     *   browsers block cross-scheme navigation from web content.  They must be
     *   COPIED by the user and pasted into the address bar manually.
     *   Firefox's about:preferences#privacy can technically be opened via
     *   window.open() from within Firefox, but we surface it as copy-paste for
     *   consistency across all browsers.
     *   Safari has no internal deep-link URL at all; step-by-step instructions
     *   are the only option.
     *
     * User — "Steps" describes the new/current UI method (Settings URL or menu).
     *   "Alternative method" describes the address-bar icon approach which works
     *   across all recent versions of each browser.
     *
     * Chrome 117+ note: Google replaced the padlock icon with a "site information"
     *   icon that looks like a tune/sliders symbol (⊙) or a circle-i (ⓘ).
     *   Both the new-UI Settings URL and the legacy address-bar icon methods are
     *   provided so users on any Chrome version can follow along.
     *
     * Firefox note: Firefox shows a red crossed microphone icon at the RIGHT end
     *   of the address bar when a site's mic access was blocked. Clicking it
     *   opens an inline permission panel — the fastest per-site fix method.
     *   The about:preferences#privacy route works for all Firefox versions.
     *
     * Safari note: macOS Ventura (13)+ renamed "Preferences" to "Settings".
     *   Both wordings are noted.  iOS permissions live in the system Settings app.
     */
    function _getBrowserSettingsInfo() {
        var browser = _detectBrowser();
        var encoded = encodeURIComponent(window.location.origin);
        var host    = window.location.hostname || 'this site';

        switch (browser) {

            // ── Chrome ────────────────────────────────────────────────────────
            // Supports the direct siteDetails deep-link (all desktop versions).
            // Chrome 117+ replaced the lock icon with a site-info/tune icon.
            case 'chrome':
                return {
                    displayName: 'Chrome',
                    settingsUrl: 'chrome://settings/content/siteDetails?site=' + encoded,
                    steps: [
                        'Copy the URL below and paste it into Chrome\u2019s address bar, then press Enter',
                        'The Site Settings page for \u201c' + host + '\u201d opens directly',
                        'Scroll to \u201cMicrophone\u201d under the Permissions section',
                        'Change the setting from \u201cBlock\u201d to \u201cAllow\u201d',
                        'Close the Settings tab, then reload this page'
                    ],
                    legacySteps: [
                        'Look at the LEFT end of the address bar while on this page:',
                        '\u2022 Chrome 117 and later: click the \u22d9 tune or \u24d8 site-info icon',
                        '\u2022 Older Chrome: click the \uD83D\uDD12 padlock icon',
                        'Select \u201cSite settings\u201d from the dropdown that appears',
                        'Find \u201cMicrophone\u201d and change it to \u201cAllow\u201d',
                        'Close Settings and reload this page',
                        '\u2014 Global alternative: paste chrome://settings/content/microphone into a new tab to view and manage all sites at once'
                    ]
                };

            // ── Microsoft Edge ────────────────────────────────────────────────
            // Edge 87+ supports the same siteDetails deep-link as Chromium.
            // Edge 118+ also replaced the padlock with an info/tune icon.
            case 'edge':
                return {
                    displayName: 'Edge',
                    settingsUrl: 'edge://settings/content/siteDetails?site=' + encoded,
                    steps: [
                        'Copy the URL below and paste it into Edge\u2019s address bar, then press Enter',
                        'The Site Permissions page for \u201c' + host + '\u201d opens directly',
                        'Scroll to \u201cMicrophone\u201d under Permissions',
                        'Change the setting from \u201cBlock\u201d to \u201cAllow\u201d',
                        'Close the Settings tab, then reload this page'
                    ],
                    legacySteps: [
                        'Look at the LEFT end of the address bar while on this page:',
                        '\u2022 Edge 118 and later: click the \u24d8 info or \u22d9 tune icon',
                        '\u2022 Older Edge: click the \uD83D\uDD12 padlock icon',
                        'Click \u201cPermissions for this site\u201d from the panel that opens',
                        'Find \u201cMicrophone\u201d and set it to \u201cAllow\u201d',
                        'Close the panel and reload this page',
                        '\u2014 Global alternative: paste edge://settings/content/microphone into a new tab to view and manage all sites at once'
                    ]
                };

            // ── Opera ─────────────────────────────────────────────────────────
            // Opera Chromium (15+) supports the same siteDetails deep-link.
            case 'opera':
                return {
                    displayName: 'Opera',
                    settingsUrl: 'opera://settings/content/siteDetails?site=' + encoded,
                    steps: [
                        'Copy the URL below and paste it into Opera\u2019s address bar, then press Enter',
                        'The Site Settings page for \u201c' + host + '\u201d opens directly',
                        'Find \u201cMicrophone\u201d under Permissions',
                        'Change the setting to \u201cAllow\u201d',
                        'Close the Settings tab, then reload this page'
                    ],
                    legacySteps: [
                        'Click the \uD83D\uDD12 lock, shield, or \u24d8 info icon at the LEFT of the address bar',
                        'Click \u201cSite settings\u201d or \u201cManage permissions\u201d from the dropdown',
                        'Find \u201cMicrophone\u201d and set it to \u201cAllow\u201d',
                        'Reload this page',
                        '\u2014 Global alternative: paste opera://settings/content/microphone into a new tab to view and manage all sites at once'
                    ]
                };

            // ── Brave ─────────────────────────────────────────────────────────
            // Brave (Chromium-based) supports the same siteDetails deep-link.
            case 'brave':
                return {
                    displayName: 'Brave',
                    settingsUrl: 'brave://settings/content/siteDetails?site=' + encoded,
                    steps: [
                        'Copy the URL below and paste it into Brave\u2019s address bar, then press Enter',
                        'The Site Settings page for \u201c' + host + '\u201d opens directly',
                        'Find \u201cMicrophone\u201d under Permissions',
                        'Change the setting to \u201cAllow\u201d',
                        'Close the Settings tab, then reload this page'
                    ],
                    legacySteps: [
                        'Click the \uD83E\uDD81 lion icon at the RIGHT of the address bar (Brave Shields)',
                        '\u2014 OR \u2014 click the \uD83D\uDD12 padlock or \u24d8 info icon at the LEFT of the address bar',
                        'Click \u201cSite permissions\u201d or \u201cSite settings\u201d',
                        'Find \u201cMicrophone\u201d and set it to \u201cAllow\u201d',
                        'Reload this page',
                        '\u2014 Global alternative: paste brave://settings/content/microphone into a new tab to view and manage all sites at once'
                    ]
                };

            // ── Firefox ───────────────────────────────────────────────────────
            // Firefox does NOT expose a site-specific deep-link URL; the closest
            // is about:preferences#privacy which shows the global Permissions list.
            // Firefox also shows a red mic icon at the RIGHT of the address bar
            // when a site has been blocked — clicking it is the fastest method.
            // about:permissions (per-site) was removed in Firefox 47; use
            // about:preferences#privacy → Microphone → Settings instead.
            case 'firefox':
                return {
                    displayName: 'Firefox',
                    settingsUrl: 'about:preferences#privacy',
                    steps: [
                        'Copy the URL below, open a new Firefox tab, paste it in, then press Enter',
                        'Scroll down to the \u201cPermissions\u201d section',
                        'Click \u201cSettings\u2026\u201d next to \u201cUse the Microphone\u201d',
                        'Find \u201c' + host + '\u201d in the list and change Status to \u201cAllow\u201d',
                        'Click \u201cSave Changes\u201d, then reload this page'
                    ],
                    legacySteps: [
                        'Fastest fix \u2014 look at the RIGHT end of the address bar:',
                        'If you see a red crossed-out microphone icon (\uD83C\uDFA4 with a line), click it',
                        'Select \u201cAllow Microphone\u201d or \u201cTemporarily Blocked\u201d to re-enable',
                        '\u2014 Firefox 128 and later: click the \uD83D\uDD12 padlock icon \u2192 open the Permissions section in the Site Privacy Panel',
                        '\u2014 Older Firefox: click the \uD83D\uDD12 padlock or \uD83D\uDEE1\uFE0F shield icon at the LEFT of the address bar',
                        'Click \u201cConnection secure\u201d \u2192 then the arrow (\u203a) \u2192 \u201cMore Information\u2026\u201d',
                        'In the Page Info dialog, click the \u201cPermissions\u201d tab',
                        'Find \u201cUse the Microphone\u201d \u2192 uncheck \u201cUse Default\u201d \u2192 select \u201cAllow\u201d',
                        'Close the dialog and reload this page'
                    ]
                };

            // ── Safari ────────────────────────────────────────────────────────
            // Safari has no internal URL scheme for site permissions; all paths
            // go through system Settings (iOS) or the Safari Settings menu (macOS).
            // macOS Ventura (13+) renamed \u201cPreferences\u201d to \u201cSettings\u201d.
            case 'safari': {
                var isMobile = /iphone|ipad|ipod/i.test(navigator.userAgent);
                if (isMobile) {
                    // iOS / iPadOS: permissions live in the system Settings app
                    return {
                        displayName: 'Safari (iOS)',
                        settingsUrl: null,
                        steps: [
                            'Open the iOS \u2699\uFE0F Settings app (grey gear icon on the home screen)',
                            'Scroll down and tap \u201cSafari\u201d',
                            'Tap \u201cMicrophone\u201d and set the permission to \u201cAllow\u201d or \u201cAsk\u201d',
                            'Return to this page and try the microphone again'
                        ],
                        legacySteps: [
                            'Alternative per-app path \u2014 iOS Settings app:',
                            'Go to Settings \u2192 Privacy & Security \u2192 Microphone',
                            'Find \u201cSafari\u201d in the list and enable the toggle',
                            'Return to this page',
                            '\u2014 OR for per-site control \u2014',
                            'Open Settings \u2192 Safari \u2192 Settings for Websites \u2192 Microphone',
                            'Find \u201c' + host + '\u201d and set to \u201cAllow\u201d'
                        ]
                    };
                }
                // macOS Safari
                return {
                    displayName: 'Safari',
                    settingsUrl: null,
                    steps: [
                        'While on this page, click \u201cSafari\u201d in the menu bar',
                        'Click \u201cSettings for This Website\u2026\u201d (or press \u2303\u2318S)',
                        'In the permissions sheet that appears, set \u201cMicrophone\u201d to \u201cAllow\u201d',
                        'Close the sheet \u2014 permission takes effect immediately'
                    ],
                    legacySteps: [
                        'Global path (works in all macOS Safari versions):',
                        'Click \u201cSafari\u201d in the menu bar \u2192 \u201cSettings\u2026\u201d or \u201cPreferences\u2026\u201d (\u2318,)',
                        'Click the \u201cWebsites\u201d tab',
                        'Select \u201cMicrophone\u201d in the left sidebar',
                        'Find \u201c' + host + '\u201d in the right panel and set it to \u201cAllow\u201d',
                        '\u2014 OR address-bar method \u2014',
                        'Click the \u201cAA\u201d or page icon in the Smart Search field (address bar)',
                        'Click \u201cWebsite Settings\u2026\u201d',
                        'Set \u201cMicrophone\u201d to \u201cAllow\u201d and close the sheet'
                    ]
                };
            }

            // ── Unknown / other browser ───────────────────────────────────────
            default:
                return {
                    displayName: 'your browser',
                    settingsUrl: null,
                    steps: [
                        'Look at the LEFT end of the address bar for a lock, info, or tune icon',
                        'Click it and look for \u201cSite settings\u201d, \u201cPermissions\u201d, or \u201cSite permissions\u201d',
                        'Find \u201cMicrophone\u201d and set it to \u201cAllow\u201d',
                        'Reload this page'
                    ],
                    legacySteps: [
                        'Open your browser\u2019s Settings (usually \u2039\u22ee\u203a or gear icon in the toolbar)',
                        'Search for \u201cSite permissions\u201d, \u201cContent settings\u201d, or \u201cPrivacy\u201d',
                        'Find \u201cMicrophone\u201d and allow access for \u201c' + host + '\u201d',
                        'Reload this page'
                    ]
                };
        }
    }

    /**
     * Classify an audio-input device by its OS-assigned label.
     *
     * Parameters
     * ----------
     * label : string
     *     MediaDeviceInfo.label as returned by enumerateDevices().
     *
     * Returns
     * -------
     * string
     *     Short human-readable category, or '' if no pattern matches.
     *
     * Notes
     * -----
     * Patterns cover the most common Windows (Stereo Mix, Wave Out), macOS
     * (BlackHole, Soundflower), and Linux (ALSA monitor) loopback/virtual
     * device names.  The empty-string fallback is intentional: unlabelled
     * physical microphones need no category badge.
     */
    function _categorizeMicDevice(label) {
        var l = (label || '').toLowerCase();
        if (/stereo mix|what u hear|wave out|loopback|monitor/i.test(l)) {
            return 'Loopback / monitor';
        }
        if (/virtual|vb-?cable|blackhole|soundflower|voicemeeter|ndi/i.test(l)) {
            return 'Virtual device';
        }
        if (/bluetooth|wireless|airpods|headset.*bt|bt.*headset/i.test(l)) {
            return 'Bluetooth / wireless';
        }
        if (/usb/i.test(l)) { return 'USB microphone'; }
        if (/built.?in|internal/i.test(l)) { return 'Built-in microphone'; }
        if (/hdmi|displayport|display audio/i.test(l)) { return 'Display / HDMI'; }
        // Physical microphones that do not match any specific pattern above
        // (e.g. unlabelled hardware mics, proprietary interface mics, studio
        // interfaces) are still real host-level audio inputs.  Label them so
        // users can distinguish them from virtual/loopback entries in the list.
        return 'Host microphone';
    }

    /**
     * Select all text content of an element so the user can copy it manually.
     *
     * Used as a fallback when the Clipboard API is unavailable (non-secure
     * context or denied by the browser).
     *
     * Parameters
     * ----------
     * el : HTMLElement
     *     Element whose text content should be selected.
     */
    function _selectSettingsUrl(el) {
        try {
            var range = document.createRange();
            range.selectNodeContents(el);
            var sel = window.getSelection();
            if (sel) { sel.removeAllRanges(); sel.addRange(range); }
        } catch (_) {}
    }

    /**
     * Build the permission-state URL-bar indicator element.
     *
     * Returns an .ai-assistant-mic-permission-bar div that is inserted once
     * into the devices section.  Its data-permission attribute is updated by
     * _updateMicPermissionBar() each time the popup is opened.
     *
     * DOM structure
     * -------------
     * .ai-assistant-mic-permission-bar [data-permission]
     *   .ai-assistant-mic-perm-status
     *     .ai-assistant-mic-perm-icon          ← SVG status icon
     *     .ai-assistant-mic-perm-text          ← status message (aria-live)
     *   .ai-assistant-mic-perm-urlbar          ← browser URL-bar mockup
     *     .ai-assistant-mic-perm-urlbar-chrome ← address bar chrome
     *       .ai-assistant-mic-perm-urlbar-lock ← 🔒 lock SVG
     *       .ai-assistant-mic-perm-urlbar-url  ← hostname text
     *       .ai-assistant-mic-perm-urlbar-mic  ← 🎤 permission icon (pulsing/blocked)
     *     .ai-assistant-mic-perm-urlbar-hint   ← "↑ click here" label
     *
     * Notes
     * -----
     * User: The URL-bar mockup mirrors the permission-icon location in Chrome,
     *   Edge, Firefox, and Safari — the camera/mic icon appears at the RIGHT
     *   end of the address bar.  The mockup is aria-hidden; the status text
     *   above it carries the accessible description.
     *
     * Developer: The mockup updates its data-blocked attribute so CSS can swap
     *   between the pulsing-blue (prompt) and crossed-red (denied) mic icon
     *   without JavaScript re-rendering.
     *
     * @returns {HTMLElement}
     */
    function _buildMicPermissionBar() {
        var bar = document.createElement('div');
        bar.className = 'ai-assistant-mic-permission-bar';
        bar.id = 'ai-assistant-mic-permission-bar';
        bar.setAttribute('data-permission', 'prompt');  // updated by _updateMicPermissionBar
        bar.setAttribute('aria-live', 'polite');

        // ── Status row ────────────────────────────────────────────────────────
        var statusRow = document.createElement('div');
        statusRow.className = 'ai-assistant-mic-perm-status';

        var iconEl = document.createElement('span');
        iconEl.className = 'ai-assistant-mic-perm-icon';
        iconEl.setAttribute('aria-hidden', 'true');

        var textEl = document.createElement('span');
        textEl.className = 'ai-assistant-mic-perm-text';
        textEl.textContent = 'Checking\u2026';

        statusRow.appendChild(iconEl);
        statusRow.appendChild(textEl);
        bar.appendChild(statusRow);

        // ── Localhost / file:// warning ──────────────────────────────────────

        var protocol = window.location.protocol;
        var hostname = window.location.hostname;
        var showLocalhostWarning =
            window.location.protocol === 'file:';

        var insecureOrigin =
            protocol === 'file:' ||
            (
                protocol !== 'https:' &&
                hostname !== 'localhost' &&
                hostname !== '127.0.0.1'
            );

        if (insecureOrigin) {

            var warn = document.createElement('div');
            warn.className = 'ai-assistant-mic-origin-warning';

            warn.innerHTML =
                '<strong>Microphone setup warning</strong>'
                + '<br>'
                + 'Speech recognition and microphone permissions are most reliable when this page is served from '
                + '<code>localhost</code> or <code>https://</code>.'
                + '<br><br>'
                + 'Avoid opening the application directly using:'
                + '<br>'
                + '<code>file:///...</code>'
                + '<br><br>'
                + 'Recommended options:'
                + '<ul>'
                + '<li><code>python -m http.server 8000</code></li>'
                + '<li>VS Code Live Server (<code>ritwickdey.LiveServer</code>)</li>'
                + '<li>VS Code Simple Browser + local web server</li>'
                + '<li>Any HTTPS-hosted deployment</li>'
                + '</ul>';

            bar.appendChild(warn);
        }

        // ── URL-bar mockup (visible for prompt + denied states via CSS) ───────
        //
        // Mirrors the browser address-bar permission icon so users immediately
        // understand where to click.  aria-hidden on the entire mockup because
        // the status text above is the accessible description.
        var urlBar = document.createElement('div');
        urlBar.className = 'ai-assistant-mic-perm-urlbar';
        urlBar.setAttribute('aria-hidden', 'true');

        var urlChrome = document.createElement('div');
        urlChrome.className = 'ai-assistant-mic-perm-urlbar-chrome';

        // Lock / security icon (left side of address bar)
        var lockEl = document.createElement('span');
        lockEl.className = 'ai-assistant-mic-perm-urlbar-lock';
        lockEl.innerHTML =
            '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
            + ' stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
            + ' aria-hidden="true">'
            + '<rect x="3.5" y="7" width="9" height="7" rx="1.5"/>'
            + '<path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2"/>'
            + '</svg>';

        // Hostname text (centre of address bar)
        var urlText = document.createElement('span');
        urlText.className = 'ai-assistant-mic-perm-urlbar-url';
        urlText.textContent = (window.location.hostname || 'localhost');

        // Mic permission icon (right end of address bar — the control users click)
        var micIcon = document.createElement('span');
        micIcon.className = 'ai-assistant-mic-perm-urlbar-mic';
        // Mic SVG: body of microphone
        micIcon.innerHTML =
            '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
            + ' stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
            + ' aria-hidden="true">'
            + '<rect x="5" y="1" width="6" height="8" rx="3"/>'
            + '<path d="M2.5 7.5a5.5 5.5 0 0 0 11 0"/>'
            + '<line x1="8" y1="13" x2="8" y2="15"/>'
            + '<line x1="6" y1="15" x2="10" y2="15"/>'
            + '</svg>';

        urlChrome.appendChild(lockEl);
        urlChrome.appendChild(urlText);
        urlChrome.appendChild(micIcon);
        urlBar.appendChild(urlChrome);

        // Arrow hint label below the chrome bar
        var hint = document.createElement('div');
        hint.className = 'ai-assistant-mic-perm-urlbar-hint';
        hint.textContent = '\u2191 click the mic icon to allow';
        urlBar.appendChild(hint);

        bar.appendChild(urlBar);

        // ── Browser-specific settings guide ───────────────────────────────────
        //
        // Populated lazily by _updateMicPermissionBar() so the content always
        // reflects the actual browser at render time.
        //
        // DOM structure:
        //   .ai-assistant-mic-perm-settings
        //     .ai-assistant-mic-perm-settings-header
        //       .ai-assistant-mic-perm-settings-browser  ← badge: "Chrome" etc.
        //       .ai-assistant-mic-perm-settings-title
        //     .ai-assistant-mic-perm-settings-url-row
        //       code.ai-assistant-mic-perm-settings-url  ← paste-in URL
        //       button.ai-assistant-mic-perm-settings-copy
        //     ol.ai-assistant-mic-perm-settings-steps    ← current method
        //     button.ai-assistant-mic-perm-settings-legacy-toggle
        //     ol.ai-assistant-mic-perm-settings-legacy-body ← older method
        var settSec = document.createElement('div');
        settSec.className = 'ai-assistant-mic-perm-settings';

        // Header: browser badge + title
        var settHeader = document.createElement('div');
        settHeader.className = 'ai-assistant-mic-perm-settings-header';

        var browserBadge = document.createElement('span');
        browserBadge.className = 'ai-assistant-mic-perm-settings-browser';

        var settTitle = document.createElement('span');
        settTitle.className = 'ai-assistant-mic-perm-settings-title';
        settTitle.textContent = 'How to allow microphone:';

        settHeader.appendChild(browserBadge);
        settHeader.appendChild(settTitle);
        settSec.appendChild(settHeader);

        // URL row — hidden when no settings URL exists for this browser
        var urlRowEl = document.createElement('div');
        urlRowEl.className = 'ai-assistant-mic-perm-settings-url-row';

        var urlCode = document.createElement('code');
        urlCode.className = 'ai-assistant-mic-perm-settings-url';
        urlCode.setAttribute('aria-label', 'Settings URL — click Copy or select all text to copy');

        var copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.className = 'ai-assistant-mic-perm-settings-copy';
        copyBtn.setAttribute('aria-label', 'Copy settings URL to clipboard');
        copyBtn.textContent = 'Copy';

        urlRowEl.appendChild(urlCode);
        urlRowEl.appendChild(copyBtn);
        settSec.appendChild(urlRowEl);

        // Steps list — populated by _updateMicPermissionBar
        var stepsList = document.createElement('ol');
        stepsList.className = 'ai-assistant-mic-perm-settings-steps';
        settSec.appendChild(stepsList);

        // Legacy / alternative method — hidden by default, toggle to reveal
        var legacyToggle = document.createElement('button');
        legacyToggle.type = 'button';
        legacyToggle.className = 'ai-assistant-mic-perm-settings-legacy-toggle';
        legacyToggle.setAttribute('aria-expanded', 'false');
        legacyToggle.textContent = '\u25B6 Alternative / older-browser method';

        var legacyBody = document.createElement('ol');
        legacyBody.className = 'ai-assistant-mic-perm-settings-legacy-body';
        legacyBody.setAttribute('aria-hidden', 'true');

        legacyToggle.addEventListener('click', function () {
            var expanded = legacyToggle.getAttribute('aria-expanded') === 'true';
            legacyToggle.setAttribute('aria-expanded', String(!expanded));
            legacyBody.setAttribute('aria-hidden', String(expanded));
            legacyToggle.textContent = (!expanded ? '\u25BC ' : '\u25B6 ')
                + 'Alternative / older-browser method';
        });
        settSec.appendChild(legacyToggle);
        settSec.appendChild(legacyBody);

        bar.appendChild(settSec);
        return bar;
    }

    /**
     * Populate the device list element with available microphone options.
     *
     * Renders a loading placeholder, then replaces it with one radio-style item
     * per device once enumeration resolves.  A synthetic "Default microphone"
     * entry (deviceId: 'default') is always prepended so the user can revert to
     * the browser default.  Updates the permission bar after each enumeration.
     *
     * Parameters
     * ----------
     * listEl : HTMLElement
     *     Container element (.ai-assistant-mic-device-list) to populate.
     *
     * Notes
     * -----
     * User: Device labels are empty strings until microphone permission is
     *   granted.  The popup re-enumerates on every open so real labels appear
     *   automatically after the first successful recording session.
     *
     * Developer: Re-entrant calls (e.g. fast open/close) are naturally serialised
     *   because each call clears listEl.innerHTML first — only the last async
     *   result is visible.  If strict serialisation is ever needed, replace the
     *   innerHTML clear with a generation counter guard.
     */
    function _refreshMicDeviceList(listEl) {
        // Show loading state immediately (synchronous)
        listEl.innerHTML = '';
        var loader = document.createElement('div');
        loader.className = 'ai-assistant-mic-devices-loading';
        loader.textContent = 'Loading\u2026';
        listEl.appendChild(loader);

        // Proactively acquire the warm stream before enumerating.
        //
        // Browsers only populate real device labels and non-empty deviceIds in
        // enumerateDevices() AFTER the user has granted microphone permission.
        // Without a prior getUserMedia call the list shows only placeholder
        // "Microphone N" labels with deviceId === '' — which this function
        // filters out, leaving only "Default microphone".
        //
        // By calling _acquireMicWarmStream first we:
        //   (a) trigger the one-time permission dialog (if not yet granted),
        //   (b) ensure enumerateDevices returns all real devices with labels, and
        //   (c) establish the warm stream that prevents re-prompting on recording.
        //
        // On failure (permission denied, no mediaDevices) we fall through to
        // _enumMicDevices which handles denied/unsupported states correctly.
        function doRefresh() {
            _enumMicDevices(function (devices, permState) {
                listEl.innerHTML = '';

                // Determine whether labels are real OS names (permission was granted)
                var hasRealLabels = devices.some(function (d) {
                    return d.label && !(/^Microphone \d+$/.test(d.label));
                });

                // Update permission bar — always, regardless of device count
                _updateMicPermissionBar(permState, hasRealLabels || permState === 'granted');

                // ── Device list assembly ────────────────────────────────────
                //
                // Strategy:
                //   1. "default" entry (deviceId='default') — the OS/browser system
                //      default.  Chrome/Edge return this with a label like
                //      "Default – Microphone (Device Name)".  We use the real label
                //      when available; fall back to "System Default".
                //   2. "communications" entry (deviceId='communications') — Windows
                //      Communications Audio Device (separate from the default).
                //      Included when present; labelled "Communications Device" if
                //      the browser withholds its name before permission is granted.
                //   3. All other real audioinput devices — physical mics, loopback
                //      monitors, virtual cables, Bluetooth headsets, etc.
                //   4. Phantom entries (deviceId='', label='') — browser is hiding
                //      device identity before permission.  Excluded; only the
                //      synthetic "System Default" entry is shown in this state.
                //
                // This guarantees at least one selectable entry (System Default)
                // even before permission is granted, and surfaces every available
                // input once permission is given.

                // Separate the browser-provided special entries from real devices
                var browserDefault = null;
                var browserComms   = null;
                var real           = [];

                devices.forEach(function (d) {
                    if (d.deviceId === 'default') {
                        browserDefault = d;
                    } else if (d.deviceId === 'communications') {
                        browserComms = d;
                    } else if (d.deviceId !== '') {
                        // Real device with a unique ID — include regardless of label
                        real.push(d);
                    }
                    // deviceId === '' with label === '' → pre-permission phantom, skip
                });

                // Build ordered list
                var all = [];

                // 1. System Default — always first
                all.push({
                    deviceId: 'default',
                    label: (browserDefault && browserDefault.label)
                        ? browserDefault.label
                        : 'System Default',
                    subtitle: 'System default'
                });

                // 2. Communications device (Windows)
                if (browserComms) {
                    all.push({
                        deviceId: 'communications',
                        label: browserComms.label || 'Communications Device',
                        subtitle: 'Communications'
                    });
                }

                // 3. Physical, loopback, and virtual devices — with category subtitle
                real.forEach(function (d) {
                    all.push({
                        deviceId: d.deviceId,
                        label:    d.label,
                        subtitle: _categorizeMicDevice(d.label)
                    });
                });

                // Show contextual empty message for denied or no-hardware cases
                if (permState === 'denied') {
                    var deniedEl = document.createElement('div');
                    deniedEl.className = 'ai-assistant-mic-devices-empty';
                    deniedEl.textContent = 'Microphone blocked \u2014 see instructions above.';
                    listEl.appendChild(deniedEl);
                    return;   // do not render device rows when access is blocked
                }

                if (real.length === 0 && !browserComms) {
                    var emptyEl = document.createElement('div');
                    emptyEl.className = 'ai-assistant-mic-devices-empty';
                    emptyEl.textContent = (permState === 'prompt' && !hasRealLabels)
                        ? 'Allow microphone to see all available devices'
                        : 'No microphones found';
                    listEl.appendChild(emptyEl);
                    // Fall through — still render the System Default entry
                }

                all.forEach(function (dev) {
                    var item = document.createElement('div');
                    item.className = 'ai-assistant-mic-device-item';
                    item.setAttribute('role', 'menuitemradio');
                    item.setAttribute('tabindex', '0');
                    item.setAttribute('data-device-id', dev.deviceId);

                    var effectiveId = _micDeviceId || 'default';
                    item.setAttribute('aria-checked', (dev.deviceId === effectiveId) ? 'true' : 'false');

                    // Label + subtitle wrapper
                    var labelWrap = document.createElement('span');
                    labelWrap.className = 'ai-assistant-mic-device-label-wrap';

                    var nameSpan = document.createElement('span');
                    nameSpan.className = 'ai-assistant-mic-device-name';
                    nameSpan.textContent = dev.label;
                    nameSpan.title = dev.label;
                    labelWrap.appendChild(nameSpan);

                    // Subtitle (device category) — omitted when empty
                    if (dev.subtitle) {
                        var subSpan = document.createElement('span');
                        subSpan.className = 'ai-assistant-mic-device-subtitle';
                        subSpan.textContent = dev.subtitle;
                        subSpan.setAttribute('aria-hidden', 'true');
                        labelWrap.appendChild(subSpan);
                    }

                    // Checkmark (CSS opacity: 0 → 1 on aria-checked="true")
                    var checkSpan = document.createElement('span');
                    checkSpan.className = 'ai-assistant-mic-device-check';
                    checkSpan.setAttribute('aria-hidden', 'true');
                    checkSpan.innerHTML =
                        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
                        + ' stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
                        + '<polyline points="3 8 7 12 13 5"/>'
                        + '</svg>';

                    item.appendChild(labelWrap);
                    item.appendChild(checkSpan);

                    // Click + keyboard activation (IIFE captures stable devId)
                    (function (devId) {
                        item.addEventListener('click', function () {
                            _setMicDevice(devId);
                        });
                        item.addEventListener('keydown', function (e) {
                            if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault();
                                _setMicDevice(devId);
                            } else if (e.key === 'ArrowDown') {
                                e.preventDefault();
                                var next = item.nextElementSibling;
                                if (next) { next.focus(); }
                            } else if (e.key === 'ArrowUp') {
                                e.preventDefault();
                                var prev = item.previousElementSibling;
                                if (prev) { prev.focus(); }
                            }
                        });
                    }(dev.deviceId));

                    listEl.appendChild(item);
                });
            });
        }

        // Acquire the warm stream first to ensure real labels are available.
        // If acquisition fails (denied / no API) doRefresh still runs — the
        // _enumMicDevices callback will surface the appropriate denied/empty state.
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            _acquireMicWarmStream(function () { doRefresh(); });
        } else {
            doRefresh();
        }
    }

    // ── Speech recognition──────────────────────────────────────────────────── ────────────────────────────────────────────────────


    /**
     * Web Speech API recognition instance (lazy, created on first mic click).
     * @type {SpeechRecognition|null}
     */
    var _speechRecognition = null;

    /**
     * True when the SpeechRecognition instance has finished its last session
     * and is safe to call .start() again.  Starts true (no instance yet →
     * trivially safe).  Set to false when .start() is called; restored to
     * true inside the onend handler.
     *
     * This bridges the async gap between calling stop() and onend firing:
     * if _doStart() is invoked while the engine is still winding down,
     * it queues the start in _pendingSpeechStart instead of throwing
     * InvalidStateError.
     */
    var _speechRecognitionEnded = true;

    /**
     * True when _doStart() was called while _speechRecognitionEnded was
     * still false (i.e. the engine hadn't fired onend yet).  The onend
     * handler checks this flag and calls _doStart() immediately after the
     * engine becomes idle, so rapid hold-release-hold sequences always start.
     */
    var _pendingSpeechStart = false;

    /** True when speech recognition is actively listening. */
    var _isListening = false;
    var _micPointerHeld = false;
    var _speechStartPending = false;
    var _recognitionFlushing = false;

    /**
     * Textarea value captured at the moment recording begins.
     *
     * Purpose
     * ───────
     * In continuous mode the engine delivers multiple onresult events.
     * Each event supplies only the transcripts produced SINCE the last
     * final result (via e.resultIndex), so we must keep a record of
     * what was already in the textarea when the session started.  Every
     * onresult call rebuilds the displayed value as:
     *   _speechBaseText + _speechFinalText + current-interim
     * which guarantees the user's pre-existing text is never lost and
     * no phrase is ever appended more than once.
     *
     * Reset in onstart — NOT in onend — so that rapid stop-start cycles
     * (user stops, immediately restarts) correctly treat any text
     * committed in the previous session as the new baseline.
     *
     * @type {string}
     */
    var _speechBaseText = '';

    /**
     * Running concatenation of all isFinal transcripts committed during
     * the current recording session.
     *
     * Each time onresult delivers a final result the trimmed transcript
     * is space-joined onto this string.  Together with _speechBaseText
     * it forms the permanent portion of the textarea content; the
     * interim portion is shown live but never persisted here.
     *
     * Reset to '' in onstart so every new session starts clean.
     *
     * @type {string}
     */
    var _speechFinalText = '';

    /**
     * Handle for the 30-second recording auto-stop setTimeout.
     *
     * Lifecycle
     * ─────────
     * • Set   → inside onstart, immediately after _setMicActiveState(true).
     * • Clear → in _stopSpeechRecognition() (manual stop) and in onend
     *           (natural or auto-stop end).  Both sites null the handle so
     *           no stale callback fires after the session has already ended.
     *
     * Why not rely on the browser's built-in silence timeout?
     *   With continuous=true the engine never auto-stops on silence; without
     *   this timer the user would record indefinitely.  30 s is the target
     *   maximum session length — matches the prior 8-10 s browser default
     *   extended to a user-visible "generous" ceiling while still preventing
     *   runaway sessions that silently drain battery and capture audio.
     *
     * @type {number|null}
     */
    var _recordingAutoStopTimer = null;

    /**
     * True when the browser supports the Web Speech API recognition interface.
     * @returns {boolean}
     */
    function _speechSupported() {
        return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    }

    /**
     * Toggle speech recognition on/off.
     * On first call, lazily creates the SpeechRecognition instance.
     * Appends recognised text to the panel textarea.
     */
    function _toggleSpeechRecognition() {
        if (!_speechSupported()) {
            showNotification('Speech recognition is not supported in this browser.', true);
            return;
        }

        if (_isListening) { _stopSpeechRecognition(); return; }

        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!_speechRecognition) {
            _speechRecognition = new SpeechRecognition();
            _speechRecognition.continuous     = true;   // keep engine alive for up to 30 s
            _speechRecognition.interimResults = true;   // required for continuous; final-only commits handled in onresult
            _speechRecognition.lang           = navigator.language || 'en-US';

            _speechRecognition.onstart = function () {
                _speechStartPending = false;
                if (_micHoldMode && !_micPointerHeld) {
                    _speechStartPending = false;
                    _pendingSpeechStart = false;
                    _stopSpeechRecognition();
                    return;
                }
                _isListening = true;
                _setMicActiveState(true);

                // ── Snapshot textarea for repetition-free accumulation ────────
                // Capture the textarea's current value as the immutable base for
                // this session.  onresult rebuilds the displayed value each time
                // as: _speechBaseText + _speechFinalText + current-interim, so
                // no phrase is ever appended twice regardless of how many
                // continuous-mode events fire.
                // Reset _speechFinalText so finals from any prior session do not
                // bleed into this one.  Snapshotting here (onstart) rather than
                // onend means a rapid stop→start cycle sees the post-committed
                // textarea as its new baseline automatically.
                var _inputSnap = document.getElementById('ai-assistant-panel-input');
                _speechBaseText  = _inputSnap ? _inputSnap.value : '';
                _speechFinalText = '';

                // ── 30-second auto-stop ───────────────────────────────────────
                // Clear any stale timer first (guards against rapid start/stop).
                // The callback calls _stopSpeechRecognition() so onend fires
                // normally, delivering any buffered final transcript before the
                // session closes — identical behaviour to the user pressing stop.
                clearTimeout(_recordingAutoStopTimer);
                _recordingAutoStopTimer = setTimeout(function () {
                    _recordingAutoStopTimer = null;
                    if (_isListening) { _stopSpeechRecognition(); }
                }, 30000);
            };

            _speechRecognition.onresult = function (e) {
                // ── Continuous-mode safe accumulation ─────────────────────────
                //
                // Problem with Array.from(e.results):
                //   In continuous mode the engine keeps the results array alive
                //   across events.  Iterating from index 0 every time means each
                //   earlier phrase is re-appended on every subsequent event,
                //   producing rapidly growing duplicate text in the textarea.
                //
                // Fix — iterate from e.resultIndex:
                //   The Web Speech API contract guarantees that results[0 …
                //   resultIndex-1] were delivered in previous events and must not
                //   be re-processed.  Only results[resultIndex … length-1] are new.
                //
                // isFinal gating:
                //   isFinal=true  → committed phrase; append to _speechFinalText.
                //   isFinal=false → interim hypothesis; show as live preview but
                //                   never persist so it never duplicates.
                //
                // Textarea composition on every event:
                //   _speechBaseText  (captured in onstart, never changes mid-session)
                //   + _speechFinalText (all committed phrases this session)
                //   + interimText      (current hypothesis, replaced each event)
                //
                var hasFinal    = false;
                var interimText = '';

                for (var i = e.resultIndex; i < e.results.length; i++) {
                    var r    = e.results[i];
                    var text = r[0].transcript.trim();
                    if (!text) { continue; }

                    if (r.isFinal) {
                        // Space-join committed phrases; guard leading space when
                        // _speechFinalText is empty (first phrase of the session).
                        _speechFinalText = _speechFinalText
                            ? _speechFinalText + ' ' + text
                            : text;
                        hasFinal = true;
                    } else {
                        // Multiple interim segments within one event are joined
                        // with a space to produce a single coherent preview.
                        interimText = interimText
                            ? interimText + ' ' + text
                            : text;
                    }
                }

                // _recognitionFlushing gates the onend restart-guard.
                // Set true ONLY on a final result: interim-only events must not
                // suppress the automatic restart that hold-to-record depends on.
                _recognitionFlushing = hasFinal;

                try {
                    var input = document.getElementById('ai-assistant-panel-input');
                    if (!input) { return; }

                    // Rebuild textarea value from the three layers.
                    // The separation into committed + preview avoids any string
                    // mutation of _speechFinalText for the interim portion.
                    var committed = _speechFinalText;
                    var preview   = interimText
                        ? (committed ? committed + ' ' + interimText : interimText)
                        : committed;
                    var full = _speechBaseText
                        ? (preview ? _speechBaseText + ' ' + preview : _speechBaseText)
                        : preview;

                    input.value = full;
                    _autoResizeInput(input);
                    _updateSendBtnState();
                    // Only steal focus after a committed result — do not interrupt
                    // the user if they are typing alongside an interim preview.
                    if (hasFinal) { input.focus(); }
                } finally {
                    _recognitionFlushing = false;
                }
            };

            // NOTE: Do NOT release _micPinTrack or _micWarmStream here.
            // Both streams must survive across recognition sessions so the browser
            // never re-prompts on the next hold-to-record press.
            //
            // IMPORTANT: Do NOT null _speechRecognition here.  Keeping the same
            // instance alive is the mechanism that prevents Chrome from opening a
            // new audio capture session — and therefore from re-showing the
            // permission indicator — on every hold-to-record press.
            _speechRecognition.onend = function () {

                // ── Clear auto-stop timer (natural or auto end) ───────────────
                // The engine may reach onend via: (a) the user stopping manually
                // through _stopSpeechRecognition(), (b) the 30-second timer
                // callback itself, or (c) the browser's own silence detection.
                // _stopSpeechRecognition() already clears the timer, but cases
                // (b) and (c) arrive here without going through that function, so
                // guard here too.  clearTimeout(null) is a safe no-op.
                clearTimeout(_recordingAutoStopTimer);
                _recordingAutoStopTimer = null;

                _speechStartPending = false;
                _speechRecognitionEnded = true;
                _isListening = false;

                _setMicActiveState(false);

                // Only restart if user is STILL holding
                if (
                    _pendingSpeechStart &&
                    !_recognitionFlushing &&
                    (
                        !_micHoldMode ||
                        _micPointerHeld
                    )
                ) {

                    _pendingSpeechStart = false;

                    setTimeout(function () {
                        _doStart();
                    }, 150);

                } else {
                    _pendingSpeechStart = false;
                }
            };

            _speechRecognition.onerror = function (e) {

                _speechStartPending = false;
                _speechRecognitionEnded = true;
                _isListening = false;

                _setMicActiveState(false);

                if (
                    _pendingSpeechStart &&
                    e.error !== 'aborted' &&
                    (
                        !_micHoldMode ||
                        _micPointerHeld
                    )
                ) {

                    _pendingSpeechStart = false;
                    _doStart();

                } else {
                    _pendingSpeechStart = false;
                }
                if (e.error === 'no-speech') {
                    _speechStartPending = false;
                    _pendingSpeechStart = false;
                    _speechRecognitionEnded = true;
                    return;
                }
                if (
                    e.error !== 'aborted' &&
                    e.error !== 'no-speech'
                ) {
                    showNotification(
                        'Speech recognition error: ' + e.error,
                        true
                    );
                }
            };
        }

        // ── Device pin / warm-stream: guarantee mic permission is held ─────────
        //
        // The Web Speech API has no direct device-selection parameter.
        //
        // Strategy A (specific device selected):
        //   Acquire a getUserMedia pin track for the chosen device BEFORE calling
        //   .start().  The browser reuses the active track for the recognition
        //   session.  The track is kept alive across hold presses (readyState
        //   check) so getUserMedia is called at most once per device per page load.
        //
        // Strategy B (browser default):
        //   Acquire (or reuse) the persistent warm stream via _acquireMicWarmStream.
        //   This holds the permission open so .start() never re-prompts, regardless
        //   of how many times the user presses and releases the hold button.
        //
        // In both cases the streams survive recognition end/error — they are only
        // released when the user changes the selected device in _setMicDevice.

        function _doStart() {
            if (_speechStartPending) {
                return;
            }
            // User already released button while startup was pending
            if (_micHoldMode && !_micPointerHeld) {
                _pendingSpeechStart = false;
                return;
            }
            if (_recognitionFlushing) {
                _pendingSpeechStart = true;
                return;
            }
            if (!_speechRecognitionEnded) {
                _pendingSpeechStart = true;
                return;
            }
            try {
                _speechRecognitionEnded = false;
                _speechStartPending = true;
                _speechRecognition.start();

            } catch (err) {

                _speechStartPending = false;
                _speechRecognitionEnded = true;
                _pendingSpeechStart = false;

                console.error(
                    'AI Assistant: Speech recognition start error:',
                    err
                );

                showNotification(
                    'Could not start microphone. Check browser permissions.',
                    true
                );
            }
        }

        if (_micDeviceId && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Strategy A: specific device.
            // Reuse the existing live pin track — hot path, zero latency.
            if (_micPinTrack && _micPinTrack.readyState === 'live') {
                _doStart();
                return;
            }
            // Cold path: acquire a fresh pin track (first press or after device change).
            _releaseMicPinTrack();
            navigator.mediaDevices.getUserMedia({
                audio: { deviceId: { exact: _micDeviceId } }
            }).then(function (stream) {
                var tracks = stream.getAudioTracks();
                _micPinTrack = tracks.length ? tracks[0] : null;
                // Keep the full stream as the warm stream so _acquireMicWarmStream
                // can reuse it and avoid a redundant getUserMedia call.
                _micWarmStream = stream;
                _doStart();
            }).catch(function (err) {
                // Device unavailable (disconnected, permission denied) — fall back
                // to the browser default silently so recording still works.
                console.warn('AI Assistant: Device pin failed, using browser default:', err);
                _micPinTrack = null;
                _acquireMicWarmStream(function () { _doStart(); });
            });
        } else {
            // Strategy B: browser default — ensure warm stream is live then start.
            _acquireMicWarmStream(function () { _doStart(); });
        }
    }

    function _stopSpeechRecognition() {
        // ── Cancel the 30-second auto-stop timer immediately ─────────────────
        // Must run before _speechRecognition.stop() so the timer callback
        // never fires after the session has already been torn down manually.
        // Nulling the handle prevents a second clearTimeout on a stale id
        // (harmless but needlessly confusing in profiler traces).
        clearTimeout(_recordingAutoStopTimer);
        _recordingAutoStopTimer = null;

        // Do NOT detach handlers and do NOT null the instance.
        //
        // Root cause of the repeated-permission-prompt bug:
        //   Discarding (nulling) the SpeechRecognition instance forces Chrome to
        //   create a fresh instance on the next hold press.  Every new instance
        //   opens a NEW internal audio capture session, which is what triggers
        //   the browser permission indicator to reappear each time.
        //
        // Fix — keep the same instance alive across hold-release cycles:
        //   Chrome reuses the existing audio capture session for .start() calls
        //   on the same instance, so the permission indicator appears only once
        //   (on the very first press) and never again.
        //
        // Use stop() instead of abort():
        //   stop() signals the engine to finish the current utterance, deliver
        //   the final onresult transcript, and then fire onend.
        //   abort() discards the transcript and fires onerror('aborted') before
        //   onend — the user loses any partial speech that was being recognised.
        //
        // Handlers (onresult / onend / onerror) are left attached so they
        // continue to manage _speechRecognitionEnded and _pendingSpeechStart
        // correctly for the next hold press.
        if (_speechRecognition && _isListening) {
            try { _speechRecognition.stop(); } catch (_) {}
        }
        // NOTE: _micPinTrack and _micWarmStream are intentionally NOT released here.
        // Keeping them alive means the browser retains the permission grant between
        // hold-to-record presses so it never re-prompts.  Tracks are released only
        // when the selected device changes (see _setMicDevice).
        _isListening = false;
        _setMicActiveState(false);
    }


    // ── Banner-only independent speech recognition engine ─────────────────────
    //
    // Purpose
    // ───────
    // The speak-banner button (class="ai-assistant-panel-speak-banner") needs
    // a "click to start / click to stop" flow with:
    //   • 30-second hard-cap auto-stop.
    //   • 1.5-second silence auto-stop when Web Audio is available.
    //   • 2-second result-gap silence fallback when Web Audio is unavailable.
    //
    // Why a separate engine and not _toggleSpeechRecognition?
    // ────────────────────────────────────────────────────────
    // The shared engine's _doStart() and onstart both contain this guard:
    //
    //   if (_micHoldMode && !_micPointerHeld) { return; }
    //
    // _micPointerHeld is only set true by pointerdown events on the footer
    // mic button, so banner clicks (which never fire pointerdown) are always
    // blocked when hold-mode is on.  Coupling the banner to hold-mode state
    // is semantically wrong — the banner is always click-to-toggle regardless
    // of what the footer mic is doing.
    //
    // This engine is therefore FULLY INDEPENDENT:
    //   • Its own SpeechRecognition instance (_bannerRec, kept alive to avoid
    //     repeated permission prompts — same rationale as the shared engine).
    //   • Its own state variables (no aliasing of _isListening, _micHoldMode, …).
    //   • Its own Web Audio graph (acquired fresh on each start; released on stop).
    //   • Writes transcript to the same textarea as the footer mic does.
    //   • Toggles .recording on the banner button for visual / ARIA feedback.
    //
    // Cross-browser compatibility
    // ───────────────────────────
    // Chrome / Edge   : full Web Audio + SpeechRecognition — silence detection active.
    // Firefox         : SpeechRecognition absent → _speechSupported() returns false →
    //                   _bannerToggle() shows notification and returns; no crash.
    // iOS Safari      : SpeechRecognition works; getUserMedia may be restricted →
    //                   _bannerBegin() catches the rejection and falls back to
    //                   recognition-only mode (30s timer + result-gap timer).
    // Legacy / unknown: fail _speechSupported() early; no crash.
    // ─────────────────────────────────────────────────────────────────────────

    /** SpeechRecognition instance for the banner (kept alive across sessions). */
    var _bannerRec          = null;

    /** True while the banner engine is actively recognising speech. */
    var _bannerActive       = false;

    /**
     * True when SpeechRecognition.onend (or onerror) has fired and the engine
     * is in the idle state.  Mirrors _speechRecognitionEnded for the shared engine.
     */
    var _bannerEnded        = true;

    /**
     * True between .start() and onstart — the engine is winding up.
     * Guards against double .start() calls.
     */
    var _bannerStarting     = false;

    /**
     * True when _bannerDoStart() was called while _bannerEnded was still false.
     * onend/onerror checks this and calls _bannerDoStart() after the engine idles,
     * matching the _pendingSpeechStart pattern in the shared engine.
     */
    var _bannerPendingStart = false;

    /** Handle for the 30-second hard-cap setTimeout. */
    var _bannerAutoTimer    = null;

    /**
     * Handle for the result-gap silence fallback setTimeout.
     * Reset on every onresult; fires after _BANNER_GAP_MS of no new results.
     * Only used when _bannerAnalyser is unavailable (no Web Audio).
     */
    var _bannerResultTimer  = null;

    /**
     * requestAnimationFrame handle for the RMS silence detection loop.
     * Only used when _bannerAnalyser is available.
     */
    var _bannerSilenceRaf   = null;

    /** AudioContext for banner silence detection. */
    var _bannerAudioCtx     = null;

    /** AnalyserNode for reading RMS amplitude. */
    var _bannerAnalyser     = null;

    /** MediaStreamAudioSourceNode feeding the analyser. */
    var _bannerAudioSrc     = null;

    /** MediaStream from getUserMedia — released on _bannerDisconnectAudio(). */
    var _bannerStream       = null;

    /** Textarea value snapshot taken at session start (avoids duplicate text). */
    var _bannerBaseText     = '';

    /** All isFinal transcripts committed during the current session. */
    var _bannerFinalText    = '';

    /**
     * True once the first onresult event fires in the current session.
     * Gates RMS silence detection so the user can click the banner and take a
     * moment to start speaking without the engine auto-stopping prematurely.
     * Reset to false on every onstart.
     */
    var _bannerHasSpoken    = false;

    /**
     * True while a getUserMedia() call is in-flight (between _bannerBegin()
     * and the .then()/.catch() resolution).  Lets _bannerToggle() detect the
     * "awaiting permission" state and route a second click to _bannerStop().
     * Cleared by the .then()/.catch() handlers and by _bannerStop().
     */
    var _bannerAwaiting     = false;

    /** setInterval handle for the banner mini soundbar animation. */
    var _bannerMiniSbTimer  = null;

    // ── Tuning constants ──────────────────────────────────────────────────────

    /** Session hard-cap in ms. */
    var _BANNER_MAX_MS      = 30000;

    /**
     * Duration of sustained silence (in ms) before auto-stop when Web Audio
     * RMS detection is active.  1500 ms ≈ 1.5 seconds — noticeably shorter
     * than a comfortable sentence pause but long enough to avoid cutting off
     * mid-phrase breath pauses.
     */
    var _BANNER_SILENCE_MS  = 1500;

    /**
     * RMS amplitude threshold.  Samples below this value are treated as silent.
     * 0.015 ≈ −36 dBFS; sits comfortably above thermal noise and well below
     * normal conversational speech so it is robust across microphone hardware.
     */
    var _BANNER_SILENCE_RMS = 0.015;

    /**
     * Duration (ms) of no new recognition results before auto-stop in the
     * Web-Audio-unavailable fallback path.  Slightly longer than _BANNER_SILENCE_MS
     * to compensate for result-delivery latency in the speech engine.
     */
    var _BANNER_GAP_MS      = 2000;

    /**
     * Grace period (ms) after onstart before RMS silence detection can trigger.
     * Silence detection is armed once this period elapses OR the first onresult
     * fires — whichever comes first.  Prevents premature auto-stop when the user
     * clicks the banner and then pauses before beginning to speak.
     */
    var _BANNER_GRACE_MS     = 3000;

    /** Number of vertical bars in the banner mini soundbar. */
    var _BANNER_MINI_SB_BARS = 5;

    /** Tick interval (ms) for the banner mini soundbar animation. */
    var _BANNER_MINI_SB_MS   = 80;

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Toggle banner recognition on/off.
     *
     * Entry point called by the banner button click handler.
     * Click while idle  → calls _bannerBegin() to start.
     * Click while active/starting → calls _bannerStop(false) to stop cleanly.
     */
    function _bannerToggle() {
        if (!_speechSupported()) {
            showNotification(
                'Speech recognition is not supported in this browser.',
                true
            );
            return;
        }
        if (_bannerActive || _bannerStarting || _bannerAwaiting) {
            _bannerStop(false);
        } else {
            _bannerBegin();
        }
    }

    /**
     * Acquire a getUserMedia stream for Web Audio silence detection, then
     * start the recognition engine.
     *
     * Graceful degradation path
     * ─────────────────────────
     * If getUserMedia is unavailable (legacy browser, iOS restriction, or
     * permission denied), _bannerDoStart() is called directly without an
     * audio graph.  Silence detection then falls back to the result-gap
     * timer strategy (reset on every onresult; fires after _BANNER_GAP_MS).
     *
     * iOS Safari compatibility
     * ────────────────────────
     * iOS Safari suspends AudioContext objects created outside a synchronous
     * user-gesture handler, and in some versions silently refuses to resume
     * them.  The AudioContext is therefore created HERE — synchronously within
     * the click → _bannerToggle() → _bannerBegin() call stack — before any
     * async boundary.  _bannerConnectAudio() reuses this pre-created context
     * to attach the MediaStream once getUserMedia resolves.
     *
     * Cancellation race guard
     * ───────────────────────
     * getUserMedia is asynchronous; the user may click Stop before it resolves.
     * _bannerAwaiting is set true here and cleared by _bannerStop() on cancel.
     * The .then() handler checks this flag to detect cancellation and releases
     * the acquired stream immediately without starting recognition.
     */
    function _bannerBegin() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Pre-create AudioContext synchronously while still inside the user
            // gesture.  This is the ONLY reliable path on iOS Safari (≥ 14.5).
            if (!_bannerAudioCtx) {
                var AC = window.AudioContext || window.webkitAudioContext;
                if (AC) {
                    try {
                        _bannerAudioCtx = new AC();
                    } catch (_acErr) {
                        _bannerAudioCtx = null;   // proceed without Web Audio
                    }
                }
            }

            _bannerAwaiting = true;
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function (stream) {
                    // Detect cancellation: _bannerStop() sets _bannerAwaiting=false.
                    // If it is still true, the user has NOT cancelled — proceed.
                    var cancelled = !_bannerAwaiting;
                    _bannerAwaiting = false;

                    if (cancelled) {
                        // Release the stream immediately; don't start recognition.
                        try {
                            stream.getTracks().forEach(function (t) { t.stop(); });
                        } catch (_e) {}
                        _bannerDisconnectAudio();
                        return;
                    }

                    _bannerStream = stream;
                    _bannerConnectAudio(stream);
                    _bannerDoStart();
                })
                .catch(function (err) {
                    _bannerAwaiting = false;
                    // Close the pre-created AudioContext — no stream means no graph.
                    if (_bannerAudioCtx) {
                        try { _bannerAudioCtx.close(); } catch (_e) {}
                        _bannerAudioCtx = null;
                    }
                    // Permission denied or hardware unavailable — proceed without
                    // Web Audio.  The user will see normal recognition behaviour;
                    // silence detection degrades to result-gap timer only.
                    console.warn(
                        'AI Assistant banner: getUserMedia failed, '
                        + 'using recognition-only fallback:',
                        err
                    );
                    _bannerDoStart();
                });
        } else {
            _bannerDoStart();
        }
    }

    /**
     * Internal: call SpeechRecognition.start() once the engine is truly idle.
     *
     * If the engine is still winding down (_bannerEnded === false), sets
     * _bannerPendingStart so that onend/onerror restarts automatically —
     * exactly the same deferred-start pattern used by the shared engine.
     */
    function _bannerDoStart() {
        if (_bannerStarting) { return; }

        if (!_bannerEnded) {
            _bannerPendingStart = true;
            return;
        }

        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;

        // Create the instance once and keep it alive across sessions to avoid
        // Chrome opening a new audio-capture session (which re-shows the
        // permission indicator) on every start.
        if (!_bannerRec) {
            _bannerRec = new SR();

            // continuous=true: engine keeps running until explicitly stopped.
            // interimResults=true: required for real-time textarea preview.
            _bannerRec.continuous     = true;
            _bannerRec.interimResults = true;
            _bannerRec.lang           = navigator.language || 'en-US';

            // ── onstart ──────────────────────────────────────────────────────
            _bannerRec.onstart = function () {
                _bannerStarting     = false;
                _bannerActive       = true;
                _bannerEnded        = false;
                _bannerPendingStart = false;
                _bannerHasSpoken    = false;   // arm grace period; silence detection gates on this

                // Snapshot the textarea so the same continuous-mode accumulation
                // strategy used by the shared engine applies here too:
                //   displayed value = _bannerBaseText + _bannerFinalText + interim
                var inp = document.getElementById('ai-assistant-panel-input');
                _bannerBaseText  = inp ? inp.value : '';
                _bannerFinalText = '';

                // Visual / ARIA feedback
                _bannerSetRecording(true);

                // 30-second hard cap — cleared in _bannerClearTimers()
                clearTimeout(_bannerAutoTimer);
                _bannerAutoTimer = setTimeout(function () {
                    _bannerAutoTimer = null;
                    if (_bannerActive) { _bannerStop(false); }
                }, _BANNER_MAX_MS);

                // Start RMS silence detection when Web Audio graph is live.
                // Falls back to result-gap timer path (set in onresult) otherwise.
                if (_bannerAnalyser) {
                    _bannerStartRmsSilence();
                }
            };

            // ── onresult ─────────────────────────────────────────────────────
            _bannerRec.onresult = function (e) {
                // Arm silence detection — user has begun speaking this session.
                _bannerHasSpoken = true;

                var inp = document.getElementById('ai-assistant-panel-input');
                if (!inp) { return; }

                var hasFinal    = false;
                var interimText = '';

                // Iterate from e.resultIndex (not 0) to avoid re-appending earlier
                // phrases — identical contract to the shared engine's onresult.
                for (var i = e.resultIndex; i < e.results.length; i++) {
                    var r    = e.results[i];
                    var text = r[0].transcript.trim();
                    if (!text) { continue; }

                    if (r.isFinal) {
                        _bannerFinalText = _bannerFinalText
                            ? _bannerFinalText + ' ' + text
                            : text;
                        hasFinal = true;
                    } else {
                        interimText = interimText
                            ? interimText + ' ' + text
                            : text;
                    }
                }

                // Rebuild textarea: base + finals + interim
                var committed = _bannerFinalText;
                var preview   = interimText
                    ? (committed ? committed + ' ' + interimText : interimText)
                    : committed;
                var full = _bannerBaseText
                    ? (preview ? _bannerBaseText + ' ' + preview : _bannerBaseText)
                    : preview;

                inp.value = full;
                _autoResizeInput(inp);
                _updateSendBtnState();
                if (hasFinal) { inp.focus(); }

                // Fallback silence detection: reset the result-gap countdown on
                // every recognition event.  Only active when _bannerAnalyser is
                // unavailable (RMS loop not running).
                if (!_bannerAnalyser) {
                    clearTimeout(_bannerResultTimer);
                    _bannerResultTimer = setTimeout(function () {
                        if (_bannerActive) { _bannerStop(false); }
                    }, _BANNER_GAP_MS);
                }
            };

            // ── onend ─────────────────────────────────────────────────────────
            // Fired after both .stop() and natural engine termination.
            // Do NOT re-enable the session automatically (unlike the shared
            // engine's hold-to-record restart) — the banner is click-to-toggle.
            _bannerRec.onend = function () {
                _bannerClearTimers();
                _bannerStarting     = false;
                _bannerActive       = false;
                _bannerEnded        = true;
                _bannerSetRecording(false);
                _bannerDisconnectAudio();

                // Restart only if _bannerDoStart() was called while we were
                // winding down (rapid double-click, programmatic re-trigger).
                if (_bannerPendingStart) {
                    _bannerPendingStart = false;
                    setTimeout(function () { _bannerDoStart(); }, 150);
                }
            };

            // ── onerror ───────────────────────────────────────────────────────
            _bannerRec.onerror = function (e) {
                _bannerClearTimers();
                _bannerStarting     = false;
                _bannerActive       = false;
                _bannerEnded        = true;
                _bannerSetRecording(false);
                _bannerDisconnectAudio();

                // Surface actionable errors to the user; swallow no-speech /
                // aborted which are informational / programmatic respectively.
                if (e.error === 'not-allowed' ||
                    e.error === 'service-not-allowed') {
                    showNotification(
                        'Microphone access denied. '
                        + 'Allow microphone access in your browser settings.',
                        true
                    );
                } else if (
                    e.error !== 'aborted' &&
                    e.error !== 'no-speech'
                ) {
                    showNotification(
                        'Speech recognition error: ' + e.error,
                        true
                    );
                }

                // Honour a pending restart unless the session was deliberately
                // aborted (e.g. by _bannerStop(true)).
                if (_bannerPendingStart && e.error !== 'aborted') {
                    _bannerPendingStart = false;
                    _bannerDoStart();
                } else {
                    _bannerPendingStart = false;
                }
            };
        }

        try {
            _bannerEnded    = false;
            _bannerStarting = true;
            _bannerRec.start();
        } catch (err) {
            _bannerStarting     = false;
            _bannerEnded        = true;
            _bannerPendingStart = false;
            console.error('AI Assistant banner: recognition start error:', err);
            showNotification(
                'Could not start microphone. Check browser permissions.',
                true
            );
        }
    }

    /**
     * Stop the banner recognition engine.
     *
     * Parameters
     * ----------
     * abort : boolean
     *     True  → SpeechRecognition.abort() — discards any partial transcript
     *             and fires onerror('aborted') then onend.  Use for hard resets.
     *     False → SpeechRecognition.stop()  — flushes the final onresult event
     *             so the user's last phrase is committed before onend.  Use for
     *             normal user-initiated or auto-triggered stops.
     */
    function _bannerStop(abort) {
        _bannerAwaiting = false;   // cancel any in-flight getUserMedia result
        _bannerClearTimers();
        _bannerPendingStart = false;

        if (_bannerRec && (_bannerActive || _bannerStarting)) {
            try {
                if (abort) {
                    _bannerRec.abort();
                } else {
                    _bannerRec.stop();
                }
            } catch (_e) {}
        }

        // Update state and UI immediately — onend will confirm but UI must be
        // responsive (especially on mobile where onend can be slow).
        _bannerActive   = false;
        _bannerStarting = false;
        _bannerSetRecording(false);
        _bannerDisconnectAudio();
    }

    /**
     * Clear all banner timers and the RAF silence loop.
     * Called from _bannerStop(), onend, and onerror to guarantee cleanup
     * regardless of which exit path is taken.
     */
    function _bannerClearTimers() {
        clearTimeout(_bannerAutoTimer);
        _bannerAutoTimer = null;
        clearTimeout(_bannerResultTimer);
        _bannerResultTimer = null;
        cancelAnimationFrame(_bannerSilenceRaf);
        _bannerSilenceRaf = null;
        clearInterval(_bannerMiniSbTimer);
        _bannerMiniSbTimer = null;
    }

    /**
     * Set / clear the .recording class, aria-label, title, visible text span,
     * and mini soundbar on the banner button.
     *
     * Parameters
     * ----------
     * active : boolean
     *     True  → button styled as "recording" (pulsing red); text → "Listening…".
     *     False → button reverts to idle state; text → "Speak with your assistant".
     *
     * Notes
     * -----
     * The span textContent is updated so sighted users see the live state without
     * relying solely on the aria-label (which is announced by screen readers only).
     * The mini soundbar is started when recording with Web Audio available, and its
     * DOM element is removed on idle; the interval handle is cleared by
     * _bannerClearTimers() which always runs before _bannerSetRecording(false).
     */
    function _bannerSetRecording(active) {
        var btn = document.getElementById('ai-assistant-panel-speak-banner');
        if (!btn) { return; }

        btn.classList.toggle('recording', active);

        var label = active ? 'Stop recording'          : 'Speak with your assistant';
        var text  = active ? 'Listening… tap to stop' : 'Speak with your assistant';
        btn.setAttribute('aria-label', label);
        btn.setAttribute('title',      label);

        var span = btn.querySelector('span');
        if (span) { span.textContent = text; }

        // Mini soundbar: create+start when recording + analyser live; remove on idle.
        // The interval is already cleared by _bannerClearTimers() before this call.
        var existingSb = btn.querySelector('.ai-assistant-banner-soundbar');
        if (active) {
            if (!existingSb && _bannerAnalyser) {
                _bannerStartMiniSoundbar(btn);
            }
        } else {
            if (existingSb) { btn.removeChild(existingSb); }
        }
    }

    // ── Banner Web Audio — RMS silence detection ──────────────────────────────

    /**
     * Connect a live MediaStream to the pre-created AnalyserNode for RMS measurement.
     *
     * The graph is: MediaStreamAudioSourceNode → AnalyserNode (no destination).
     * This is a pure analysis graph — no sound is played back and there is zero
     * risk of echo or feedback regardless of speaker / headphone configuration.
     *
     * Parameters
     * ----------
     * stream : MediaStream
     *     Live microphone stream from getUserMedia.
     *
     * Notes
     * -----
     * The AudioContext is pre-created synchronously in _bannerBegin() to satisfy
     * the iOS Safari requirement that AudioContext objects be constructed within
     * a synchronous user-gesture handler.  This function only attaches the
     * MediaStream to the already-live context; it does NOT create a new context.
     *
     * If _bannerAudioCtx is null (pre-creation failed or unavailable), this
     * function is a safe no-op — the caller falls back to result-gap timer
     * silence detection.
     *
     * fftSize 256 → 128 frequency bins; smoothingTimeConstant 0.80 matches the
     * main shared engine for consistent RMS characteristics.
     */
    function _bannerConnectAudio(stream) {
        if (!stream || !_bannerAudioCtx) { return; }
        // Idempotent: skip if already connected (shouldn't occur, but guard it).
        if (_bannerAnalyser) { return; }
        try {
            _bannerAnalyser = _bannerAudioCtx.createAnalyser();
            _bannerAnalyser.fftSize               = 256;
            _bannerAnalyser.smoothingTimeConstant = 0.80;
            _bannerAudioSrc = _bannerAudioCtx.createMediaStreamSource(stream);
            _bannerAudioSrc.connect(_bannerAnalyser);
            // NOT connected to destination — analysis only; zero echo/feedback.
            if (_bannerAudioCtx.state === 'suspended') {
                _bannerAudioCtx.resume().catch(function () {});
            }
        } catch (err) {
            console.warn('AI Assistant banner: Web Audio stream connection failed:', err);
            _bannerAnalyser = null;
            _bannerAudioSrc = null;
        }
    }

    /**
     * Disconnect and close the Web Audio graph.  Idempotent and safe to call
     * when no graph is present.  Also stops all tracks on _bannerStream so the
     * browser removes the microphone indicator in the tab bar.
     */
    function _bannerDisconnectAudio() {
        cancelAnimationFrame(_bannerSilenceRaf);
        _bannerSilenceRaf = null;

        if (_bannerAudioSrc) {
            try { _bannerAudioSrc.disconnect(); } catch (_e) {}
            _bannerAudioSrc = null;
        }
        if (_bannerAnalyser) {
            try { _bannerAnalyser.disconnect(); } catch (_e) {}
            _bannerAnalyser = null;
        }
        if (_bannerAudioCtx) {
            try { _bannerAudioCtx.close(); } catch (_e) {}
            _bannerAudioCtx = null;
        }
        if (_bannerStream) {
            try {
                _bannerStream.getTracks().forEach(function (t) { t.stop(); });
            } catch (_e) {}
            _bannerStream = null;
        }
    }

    /**
     * Read the current RMS amplitude from _bannerAnalyser.
     *
     * Returns
     * -------
     * number
     *     RMS in [0, 1].  Returns 1 (treated as "sound present") when no
     *     analyser is connected — guarantees the silence loop never fires
     *     without a live audio graph.
     *
     * Notes
     * -----
     * Uses time-domain data (getByteTimeDomainData) rather than frequency data.
     * Values are unsigned 8-bit [0..255] centred at 128 (silence = flat line).
     * Subtracting 128 and dividing by 128 normalises to [-1..1].
     * RMS = sqrt( mean( v² ) ).
     */
    function _bannerReadRms() {
        if (!_bannerAnalyser) { return 1; }
        var buf = new Uint8Array(_bannerAnalyser.fftSize);
        _bannerAnalyser.getByteTimeDomainData(buf);
        var sum = 0;
        for (var i = 0; i < buf.length; i++) {
            var v = (buf[i] - 128) / 128;
            sum += v * v;
        }
        return Math.sqrt(sum / buf.length);
    }

    /**
     * requestAnimationFrame loop that auto-stops recognition after
     * _BANNER_SILENCE_MS of sustained RMS silence.
     *
     * Lifecycle
     * ─────────
     * • Started by onstart when _bannerAnalyser is available.
     * • Cancelled by _bannerClearTimers() (called from _bannerStop, onend,
     *   onerror) so it never fires after the session has ended.
     * • Self-terminates by returning without rescheduling when !_bannerActive.
     *
     * Grace period + _bannerHasSpoken guard
     * ──────────────────────────────────────
     * Silence detection is deliberately disabled until one of the following
     * conditions is met (whichever comes first):
     *
     *   1. _bannerHasSpoken becomes true — the first onresult has fired,
     *      confirming the user has started speaking.
     *   2. _BANNER_GRACE_MS elapses — safety valve that arms detection even
     *      when the speech engine never fires onresult (e.g. very quiet
     *      environment or network delay); prevents infinite silent sessions.
     *
     * Without this guard, clicking the banner and pausing > 1.5 s before
     * speaking would cause premature auto-stop before the user had begun.
     *
     * Algorithm (after grace period arms)
     * ─────────
     * Maintains lastSoundAt in closure.  Every frame:
     *   RMS > threshold → reset lastSoundAt (user is speaking).
     *   RMS ≤ threshold → check elapsed since lastSoundAt.
     *     ≥ _BANNER_SILENCE_MS → call _bannerStop(false) and exit loop.
     *     <  _BANNER_SILENCE_MS → reschedule (silence not yet sustained).
     */
    function _bannerStartRmsSilence() {
        cancelAnimationFrame(_bannerSilenceRaf);

        var startedAt   = Date.now();
        var lastSoundAt = Date.now();

        function tick() {
            if (!_bannerActive) { return; }

            var now    = Date.now();
            var graced = (now - startedAt) >= _BANNER_GRACE_MS;

            // Hold off silence counting until the user has spoken OR the grace
            // period has elapsed — prevents premature stop on slow starters.
            if (!_bannerHasSpoken && !graced) {
                lastSoundAt = now;   // keep the silence clock reset
                _bannerSilenceRaf = requestAnimationFrame(tick);
                return;
            }

            var rms = _bannerReadRms();
            if (rms > _BANNER_SILENCE_RMS) {
                lastSoundAt = now;
            } else if (now - lastSoundAt >= _BANNER_SILENCE_MS) {
                // Sustained silence detected — stop cleanly (flush final transcript)
                _bannerStop(false);
                return;
            }

            _bannerSilenceRaf = requestAnimationFrame(tick);
        }

        _bannerSilenceRaf = requestAnimationFrame(tick);
    }

    /**
     * Create and animate the banner mini soundbar inside the banner button.
     *
     * Renders _BANNER_MINI_SB_BARS vertical bars driven by the live RMS
     * amplitude from _bannerAnalyser.  Each bar receives a phase-offset sine
     * factor so bars animate independently and organically even at low RMS.
     *
     * The soundbar container (`<div class="ai-assistant-banner-soundbar">`) is
     * appended to btn and removed by _bannerSetRecording(false).
     * The interval handle is stored in _bannerMiniSbTimer and cleared by
     * _bannerClearTimers() which always runs before _bannerSetRecording(false).
     *
     * Parameters
     * ----------
     * btn : HTMLElement
     *     The banner button element to append the soundbar into.
     *
     * Notes
     * -----
     * prefers-reduced-motion: the tick callback returns immediately when the
     * media query matches, leaving bars at their 2 px silence height.
     */
    function _bannerStartMiniSoundbar(btn) {
        clearInterval(_bannerMiniSbTimer);

        var reducedMotion = window.matchMedia &&
            window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        var sb = document.createElement('div');
        sb.className = 'ai-assistant-banner-soundbar';
        sb.setAttribute('aria-hidden', 'true');

        var bars = [];
        for (var i = 0; i < _BANNER_MINI_SB_BARS; i++) {
            var b = document.createElement('span');
            b.className    = 'ai-assistant-banner-soundbar-bar';
            b.style.height = '2px';
            sb.appendChild(b);
            bars.push(b);
        }
        btn.appendChild(sb);

        if (reducedMotion) { return; }   // bars remain at silence height

        _bannerMiniSbTimer = setInterval(function () {
            if (!_bannerActive) { return; }
            var rms = _bannerReadRms();
            var now = Date.now();
            for (var j = 0; j < bars.length; j++) {
                // Independent phase shift per bar → organic animation
                var phase  = (now / 180) + j * 1.2;
                var factor = 0.45 + 0.55 * Math.abs(Math.sin(phase));
                var h      = Math.round(2 + rms * 10 * factor);
                h = h < 2 ? 2 : (h > 12 ? 12 : h);
                bars[j].style.height = h + 'px';
            }
        }, _BANNER_MINI_SB_MS);
    }

    // ── End banner engine ─────────────────────────────────────────────────────

    /**
     * Create an AudioContext and connect the mic MediaStream to an AnalyserNode.
     *
     * Called by _setMicActiveState(true) after _micWarmStream is live.
     * Idempotent — if _audioCtx already exists the call is a no-op.
     * The audio signal is connected analyser → nowhere (no destination):
     * pure analysis with zero echo or feedback risk.
     *
     * Parameters
     * ----------
     * stream : MediaStream | null
     *     Live microphone stream from _micWarmStream.  When null, the function
     *     returns silently — viz loops will read zeros and bars stay at minimum
     *     height (graceful degradation on unsupported browsers).
     *
     * Notes
     * -----
     * fftSize 256 → 128 frequency bins.  smoothingTimeConstant 0.80 gives
     * visually responsive but non-jittery bar heights.  One AudioContext per
     * recording session keeps total context count within the browser limit (6).
     */
    function _connectWebAudio(stream) {
        if (!stream || _audioCtx) { return; }
        try {
            var AudioCtxCtor = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtxCtor) { return; }
            _audioCtx             = new AudioCtxCtor();
            _analyserNode         = _audioCtx.createAnalyser();
            _analyserNode.fftSize                = 256;   // 128 bins
            _analyserNode.smoothingTimeConstant  = 0.80;
            _audioSrcNode = _audioCtx.createMediaStreamSource(stream);
            _audioSrcNode.connect(_analyserNode);
            // NOT connected to destination — analysis only, zero echo/feedback.
            // iOS Safari suspends AudioContext between gestures; resume immediately.
            // Non-fatal: if the promise rejects, viz loops silently fall back to
            // the "no analyser" path (idle heights only).
            if (_audioCtx.state === 'suspended') {
                _audioCtx.resume().catch(function () {});
            }
        } catch (err) {
            console.warn('AI Assistant: Web Audio connect failed:', err);
            _audioCtx = null; _analyserNode = null; _audioSrcNode = null;
        }
    }

    /**
     * Disconnect and close the Web Audio graph built by _connectWebAudio.
     *
     * Idempotent — safe to call when no graph exists (all null → early-return).
     * Called internally by _stopVizLoops().
     */
    function _disconnectWebAudio() {
        if (_audioSrcNode) {
            try { _audioSrcNode.disconnect(); } catch (_e) {}
            _audioSrcNode = null;
        }
        if (_analyserNode) {
            try { _analyserNode.disconnect(); } catch (_e) {}
            _analyserNode = null;
        }
        if (_audioCtx) {
            try { _audioCtx.close(); } catch (_e) {}
            _audioCtx = null;
        }
    }

    /**
     * Read current RMS amplitude from _analyserNode as a value in [0, 1].
     *
     * Uses the time-domain buffer (getByteTimeDomainData) rather than the
     * frequency buffer because RMS of the raw waveform is a direct proxy
     * for perceived loudness and maps naturally to bar height.
     *
     * Returns
     * -------
     * number
     *     RMS amplitude in [0, 1].  Returns 0 when no analyser is connected.
     *
     * Notes
     * -----
     * Time-domain values are unsigned 8-bit [0..255], centred at 128 (silence).
     * Subtracting 128 normalises to [-128..127]; dividing by 128 gives [-1..1].
     * Squaring, averaging, and square-rooting gives RMS in [0, 1].
     */
    function _readRmsAmplitude() {
        if (!_analyserNode) { return 0; }
        var buf = new Uint8Array(_analyserNode.fftSize);
        _analyserNode.getByteTimeDomainData(buf);
        var sum = 0;
        for (var i = 0; i < buf.length; i++) {
            var v = (buf[i] - 128) / 128;
            sum += v * v;
        }
        return Math.sqrt(sum / buf.length);
    }

    /**
     * Start both real-time visualisation loops.
     *
     * Loop 1 — rAF (level bars):
     *   Maps 128 AnalyserNode frequency bins → 100 mic popup bars at ~60 fps.
     *   Each bar height is proportional to the energy in its frequency bin.
     *
     * Loop 2 — setInterval (footer soundbar):
     *   Appends one RMS amplitude sample to the ring buffer every
     *   _SOUNDBAR_TICK_MS (80 ms).  The buffer scrolls left so new samples
     *   appear on the right, creating a scrolling oscilloscope timeline.
     *
     * Idempotent — already-running loops are not duplicated.
     *
     * Notes
     * -----
     * prefers-reduced-motion: both loops are skipped; bars remain at their
     * current height.  CSS also applies static heights as a fallback.
     */
    /**
     * Return the number of footer soundbar bars that fit for the current
     * viewport width.  Matches the CSS responsive max-width breakpoints so
     * the bar count and CSS-revealed width are always in sync.
     *
     * Returns
     * -------
     * number
     *     One of: 8 (< 360 px) | 12 (360–479 px) | 16 (480–767 px) | 24 (≥ 768 px).
     */
    function _computeSoundbarBarCount() {
        var vw = (window.innerWidth || document.documentElement.clientWidth || 320);
        if (vw < 360) return 8;
        if (vw < 480) return 12;
        if (vw < 768) return 16;
        return 24;
    }

    /**
     * (Re)build soundbar bar elements to match the target count `n`.
     *
     * Idempotent: leaves the DOM untouched when the count is already correct.
     * Each new bar starts at _SOUNDBAR_MIN_H (2 px) and 0.30 opacity.
     *
     * Parameters
     * ----------
     * soundbarEl : HTMLElement
     *     The soundbar container element.
     * n : number
     *     Target bar count (8 | 12 | 16 | 24).
     *
     * Returns
     * -------
     * NodeList
     *     Live NodeList of `.ai-assistant-footer-soundbar-bar` elements.
     */
    function _rebuildSoundbarBars(soundbarEl, n) {
        var existing = soundbarEl.querySelectorAll('.ai-assistant-footer-soundbar-bar');
        if (existing.length === n) { return existing; }
        while (soundbarEl.firstChild) { soundbarEl.removeChild(soundbarEl.firstChild); }
        for (var i = 0; i < n; i++) {
            var b = document.createElement('div');
            b.className       = 'ai-assistant-footer-soundbar-bar';
            b.style.height    = _SOUNDBAR_MIN_H + 'px';
            b.style.opacity   = '0.30';
            soundbarEl.appendChild(b);
        }
        return soundbarEl.querySelectorAll('.ai-assistant-footer-soundbar-bar');
    }

    function _startVizLoops() {
        // ── prefers-reduced-motion: skip all JS animation ─────────────────
        if (window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            return;
        }

        var raf = (_RAF || requestAnimationFrame);

        // ── iOS / legacy AudioContext resume ──────────────────────────────
        // iOS Safari suspends AudioContext between user gestures; resume
        // proactively.  The .catch() guard makes this non-fatal.
        if (_audioCtx && _audioCtx.state === 'suspended') {
            _audioCtx.resume().catch(function () {});
        }

        // ── Loop 1: rAF — mic popup VU meter level bars (~60 fps) ─────────
        //
        // Standard 0-100 dBFS VU meter architecture:
        //
        //   Source   → _readRmsAmplitude() → RMS in [0, 1]
        //   Mapping  → dBFS = 20×log10(rms)
        //   Scale    → level = (dBFS - floor) / range × barCount, clamped 0-barCount
        //   Display  → bars 0…level-1 = active (zone colour, max height)
        //              bars level…N-1 = silent (idle sinusoidal height, muted)
        //   Peak     → _vuPeakBar held for _VU_PEAK_HOLD frames,
        //              then decays 1 bar every _VU_PEAK_DECAY frames.
        //
        // Colour zones (broadcast convention):
        //   0-59   primary (CSS --pst-color-primary)  safe speech  -40…-16.4 dBFS
        //   60-79  amber  _VU_COLOR_AMBER              loud/hot     -16…-8.4  dBFS
        //   80-99  red    _VU_COLOR_RED                peak/clip     -8…-0.4  dBFS
        if (!_vizRafId) {
            var levelBarsEl = document.getElementById('ai-assistant-mic-level-bars');
            if (levelBarsEl) {
                var _lbEls   = levelBarsEl.querySelectorAll('.ai-mic-bar');
                var _lbCount = _lbEls.length;  // _MIC_LEVEL_BAR_COUNT = 100

                // Reset VU peak state at loop start so a new session begins clean.
                _vuPeakBar   = 0;
                _vuPeakHold  = 0;
                _vuPeakDecay = 0;

                (function _rafTick() {
                    _vizRafId = raf(_rafTick);

                    if (!_analyserNode) {
                        // No analyser yet — hold idle sinusoidal heights
                        return;
                    }

                    // ── 1. Compute VU level (0-_lbCount) from RMS ─────────
                    var rms   = _readRmsAmplitude();
                    //   Clamp rms away from zero before log to avoid -Infinity.
                    var dBFS  = 20 * Math.log10(rms < 1e-9 ? 1e-9 : rms);
                    //   Map dBFS → bar index: floor=-40, range=40 dB, scale=_lbCount.
                    var level = Math.round((dBFS - _VU_DB_FLOOR) / _VU_DB_RANGE * _lbCount);
                    level     = level < 0 ? 0 : (level > _lbCount ? _lbCount : level);

                    // ── 2. Peak hold + decay ───────────────────────────────
                    if (level >= _vuPeakBar) {
                        // New or equal peak: reset hold timer.
                        _vuPeakBar   = level;
                        _vuPeakHold  = 0;
                        _vuPeakDecay = 0;
                    } else {
                        _vuPeakHold++;
                        if (_vuPeakHold > _VU_PEAK_HOLD) {
                            // Hold expired: decay one bar every _VU_PEAK_DECAY frames.
                            _vuPeakDecay++;
                            if (_vuPeakDecay >= _VU_PEAK_DECAY) {
                                _vuPeakDecay = 0;
                                if (_vuPeakBar > 0) { _vuPeakBar--; }
                            }
                        }
                    }

                    // ── 3. Render bars ────────────────────────────────────
                    //   Active bar (i < level): max height, zone colour.
                    //   Peak bar  (i === _vuPeakBar, _vuPeakBar > 0, i >= level):
                    //             max height, zone colour (held above active stack).
                    //   Idle bar  (everything else): idle sinusoidal height,
                    //             background cleared so CSS inactive colour applies.
                    for (var i = 0; i < _lbCount; i++) {
                        var isActive = (i < level);
                        var isPeak   = (!isActive && i === _vuPeakBar && _vuPeakBar > 0);

                        if (isActive || isPeak) {
                            _lbEls[i].style.height = _MIC_LEVEL_MAX_H + 'px';
                            // Zone colour via inline style (overrides CSS active default).
                            // Empty string lets the CSS rule supply the primary colour so
                            // it respects --pst-color-primary and dark-mode overrides.
                            if (i >= _VU_ZONE_RED) {
                                _lbEls[i].style.background = _VU_COLOR_RED;
                            } else if (i >= _VU_ZONE_AMBER) {
                                _lbEls[i].style.background = _VU_COLOR_AMBER;
                            } else {
                                _lbEls[i].style.background = '';
                            }
                        } else {
                            // Return to idle sinusoidal height; clear inline colour so
                            // the CSS inactive rule (muted/dark) applies correctly.
                            _lbEls[i].style.height = (
                                _IDLE_LEVEL_HEIGHTS[i] !== undefined
                                    ? _IDLE_LEVEL_HEIGHTS[i]
                                    : _MIC_LEVEL_MIN_H
                            ) + 'px';
                            _lbEls[i].style.background = '';
                        }
                    }
                }());
            }
        }

        // ── Loop 2: interval — footer soundbar ring buffer (80 ms) ────────
        if (!_soundbarTickId) {
            var soundbarEl = document.getElementById('ai-assistant-footer-soundbar');
            if (soundbarEl) {
                // Build bar count optimal for this viewport
                var _sbCount  = _computeSoundbarBarCount();
                var _sbBarEls = _rebuildSoundbarBars(soundbarEl, _sbCount);

                // Pre-warm: seed ring buffer with a soft 2-cycle sine wave
                // (2–4 px) so bars look organic from the very first tick.
                // Real audio amplitude immediately overwrites these values.
                _soundbarHeights = [];
                for (var _si = 0; _si < _sbCount; _si++) {
                    var _t  = _si / Math.max(1, _sbCount - 1); // 0 → 1
                    var _sw = Math.abs(Math.sin(_t * Math.PI * 2));
                    _soundbarHeights.push(_SOUNDBAR_MIN_H + _sw * 2); // 2 – 4 px
                }

                _soundbarTickId = setInterval(function () {
                    var rms = _readRmsAmplitude();
                    var h   = _SOUNDBAR_MIN_H
                        + rms * (_SOUNDBAR_MAX_H - _SOUNDBAR_MIN_H);
                    h = Math.max(_SOUNDBAR_MIN_H, Math.min(_SOUNDBAR_MAX_H, h));

                    // Shift ring buffer left; push newest sample on right
                    _soundbarHeights.shift();
                    _soundbarHeights.push(h);

                    // Apply heights + proportional opacity to all bars
                    for (var i = 0; i < _sbCount; i++) {
                        var bh  = _soundbarHeights[i];
                        // opacity: 0.30 at silence → 1.00 at full scale
                        var opc = (0.30 + 0.70 * (bh / _SOUNDBAR_MAX_H)).toFixed(2);
                        _sbBarEls[i].style.height  = bh + 'px';
                        _sbBarEls[i].style.opacity = opc;
                    }
                }, _SOUNDBAR_TICK_MS);
            }
        }
    }

    /**
     * Stop both visualisation loops and reset bars to idle/silence heights.
     *
     * Also tears down the Web Audio graph (_disconnectWebAudio) so the
     * AudioContext is closed before the next recording session starts.
     * The CSS height transition (0.04–0.06 s) smooths the return to silence
     * height so bars don't snap abruptly.
     *
     * Idempotent — safe to call multiple times or when loops are not running.
     */
    function _stopVizLoops() {
        // ── Cancel rAF loop (mic popup level bars) ────────────────────────
        if (_vizRafId !== null) {
            (_CAF || cancelAnimationFrame)(_vizRafId);
            _vizRafId = null;
        }
        // ── Cancel soundbar tick loop ─────────────────────────────────────
        if (_soundbarTickId !== null) {
            clearInterval(_soundbarTickId);
            _soundbarTickId = null;
        }
        // ── Reset VU meter peak-hold state ────────────────────────────────
        _vuPeakBar   = 0;
        _vuPeakHold  = 0;
        _vuPeakDecay = 0;
        // ── Reset mic popup level bars to idle sinusoidal heights ─────────
        // Clear inline background so the CSS inactive colour (muted) applies.
        // Clear inline height so the idle arch from _IDLE_LEVEL_HEIGHTS shows.
        var levelBarsEl = document.getElementById('ai-assistant-mic-level-bars');
        if (levelBarsEl) {
            var bars = levelBarsEl.querySelectorAll('.ai-mic-bar');
            for (var i = 0; i < bars.length; i++) {
                bars[i].style.height =
                    (_IDLE_LEVEL_HEIGHTS[i] !== undefined
                        ? _IDLE_LEVEL_HEIGHTS[i]
                        : _MIC_LEVEL_MIN_H) + 'px';
                bars[i].style.background = '';  // restore CSS inactive colour
            }
        }
        // ── Reset soundbar bars: silence height + dimmed opacity ──────────
        var soundbarEl = document.getElementById('ai-assistant-footer-soundbar');
        if (soundbarEl) {
            var sbBars = soundbarEl.querySelectorAll('.ai-assistant-footer-soundbar-bar');
            for (var j = 0; j < sbBars.length; j++) {
                sbBars[j].style.height  = _SOUNDBAR_MIN_H + 'px';
                sbBars[j].style.opacity = '0.30';
            }
        }
        // ── Tear down Web Audio graph ─────────────────────────────────────
        _disconnectWebAudio();
    }

    /**
     * Update the mic button, speak-banner, voice-level bars, and footer
     * soundbar visual state.
     *
     * @param {boolean} active  True → recording animation; false → idle.
     */
    function _setMicActiveState(active) {
        var micBtn    = document.getElementById('ai-assistant-panel-mic');
        var bannerBtn = document.getElementById('ai-assistant-panel-speak-banner');
        if (micBtn) {
            micBtn.classList.toggle('recording', active);
            var activeLabel = active
                ? 'Stop recording'
                : (_micHoldMode ? 'Press and hold to record' : 'Speak your question');
            micBtn.setAttribute('aria-label', activeLabel);
            micBtn.setAttribute('title',      activeLabel);
        }
        if (bannerBtn) {
            bannerBtn.classList.toggle('recording', active);
        }

        // Drive voice-level bars colour via data attribute.
        // JS handles heights; CSS only switches the active brand colour.
        var levelBars = document.getElementById('ai-assistant-mic-level-bars');
        if (levelBars) {
            levelBars.dataset.active = active ? 'true' : 'false';
        }

        // Drive footer soundbar visibility.
        // setAttribute / removeAttribute so [data-active] (presence) and
        // [data-active="true"] (value) selectors both match correctly.
        var footerSoundbar = document.getElementById('ai-assistant-footer-soundbar');
        if (footerSoundbar) {
            if (active) {
                footerSoundbar.setAttribute('data-active', 'true');
            } else {
                footerSoundbar.removeAttribute('data-active');
            }
        }

        // Start / stop Web Audio analysis and both visualisation loops.
        // _connectWebAudio uses the already-acquired _micWarmStream; it is a
        // no-op when no stream is available (graceful degradation).
        if (active) {
            _connectWebAudio(_micWarmStream);
            _startVizLoops();
        } else {
            _stopVizLoops();   // also calls _disconnectWebAudio()
        }
    }

    /** Dismiss the speak-with-assistant banner (one-time, on first interaction). */
    function _dismissSpeakBanner() {
        var banner = document.getElementById('ai-assistant-panel-speak-banner');
        if (banner) banner.style.display = 'none';
    }

    /**
     * Auto-resize the panel textarea to fit its content (Claude-style).
     *
     * Algorithm
     * ─────────
     * 1. Reset height to 'auto' so the browser recalculates scrollHeight
     *    against the natural content height (not the previous clamped value).
     * 2. Set height to scrollHeight — the textarea grows to show ALL lines.
     * 3. CSS max-height (200 px) caps the visual box; overflow-y:auto on the
     *    element shows a scrollbar only when content exceeds that cap.
     * 4. On clear (value=''), scrollHeight collapses back to the single-row
     *    natural height, which is ≥ min-height (2.4 rem) via CSS, so the
     *    input snaps back to its starting size automatically.
     *
     * Why not `resize:both` or native auto?
     * ───────────────────────────────────────
     * Native `resize:none` is already set (panel grip handles panel resize).
     * CSS `field-sizing:content` is not yet cross-browser (Chrome 123+ only).
     * The scrollHeight technique works in every modern browser and gives us
     * exact control over min/max bounds without JS math.
     *
     * @param {HTMLTextAreaElement} el  The textarea to resize.
     */
    function _autoResizeInput(el) {
        if (!el) return;
        /* Step 1 — collapse so scrollHeight is unconstrained by prior height */
        el.style.height = 'auto';
        /* Step 2 — expand to full content height; CSS max-height caps the rest */
        el.style.height = el.scrollHeight + 'px';
    }

    /**
     * Update the send icon button's accent state based on textarea content.
     * The send icon gets a brand-colour accent when there is text to submit.
     *
     * Also resizes the textarea to fit current content (auto-grow / auto-shrink)
     * so this single function covers every path that changes the textarea value:
     * typing, chip insert, retry fill, send/clear, new-chat reset.
     */
    function _updateSendBtnState() {
        var input   = document.getElementById('ai-assistant-panel-input');
        var sendBtn = document.getElementById('ai-assistant-panel-send');
        if (!sendBtn || !input) return;
        var hasText = input.value.trim().length > 0;
        sendBtn.classList.toggle('has-text', hasText);
        /* Resize after toggling send state so the layout is already settled */
        _autoResizeInput(input);
    }

    // ── Message bubbles ───────────────────────────────────────────────────────

    /**
     * Render one bubble into `body`.  Pure view helper — does NOT touch the
     * `_transcript` source of truth (callers do).  Reused by live messages
     * and by transcript replay so there is exactly one bubble-building path.
     *
     * For assistant/error bubbles adds:
     *   • R6  action row: [Copy] [Retry] buttons
     *   • R5  per-answer inline feedback block (emoji + optional message)
     *
     * @param {HTMLElement} body
     * @param {string}      text
     * @param {string}      role        'user' | 'assistant' | 'error'
     * @param {string}      [question]  Paired user question for Retry.
     *                                  If omitted, Retry walks _transcript
     *                                  to find the preceding user turn.
     */
    function _renderBubble(body, text, role, question, ts) {
        var bubble = document.createElement('div');
        bubble.className = 'ai-assistant-panel-bubble ai-assistant-panel-bubble--' + role;

        if (role === 'assistant') {
            // Render markdown for assistant replies — safe because _mdToHtml
            // escapes all text before applying pattern replacements and only
            // emits known-safe tags.  bubble is NOT user-controlled.
            bubble.innerHTML = _mdToHtml(text);
            bubble.setAttribute('data-raw', text);  // preserve for copy/export
        } else {
            // User / error bubbles: plain text only (XSS-safe by design).
            bubble.textContent = text;
        }

        body.appendChild(bubble);

        // User bubble: minimal action row — timestamp only.
        //
        // Mirrors the assistant action row layout but contains only the <time>
        // element; no Copy / Share / Retry affordances are needed for outgoing
        // messages.  The `--user` modifier pins the row to the trailing (right)
        // edge via `align-self: flex-end` so it stays visually attached to the
        // user bubble above it, matching the messaging-app convention (iMessage,
        // WhatsApp, Telegram) where send-time sits under the bubble on its side.
        //
        // Guard: skip the row entirely when `ts` is absent or non-finite (old
        // persisted transcript entries that pre-date timestamp recording) so no
        // empty DOM node is introduced.
        if (role === 'user' && ts && isFinite(ts)) {
            var userActs = document.createElement('div');
            userActs.className = 'ai-assistant-panel-bubble-actions ai-assistant-panel-bubble-actions--user';
            userActs.appendChild(_buildBubbleTimeEl(ts));
            body.appendChild(userActs);
        }

        if (role === 'assistant' || role === 'error') {
            // ── R6: action row — Copy + Share + Retry ─────────────────────────
            var actions = document.createElement('div');
            actions.className = 'ai-assistant-panel-bubble-actions';

            // Timestamp — prepended before Copy so information precedes action
            // (WCAG 1.3.2).  margin-right: auto in CSS pushes buttons right.
            actions.appendChild(_buildBubbleTimeEl(ts));

            // Hoist question resolution before Copy so Share (inserted between
            // Copy and Retry) can use it without a second _transcript walk —
            // single source of truth, resolved once.
            // Prefers the explicit `question` param, then walks _transcript back.
            var retryQ = question || (function () {
                for (var i = _transcript.length - 1; i >= 0; i--) {
                    if (_transcript[i].role === 'user') return _transcript[i].text;
                }
                return null;
            }());

            // Copy button
            var copyBtn = document.createElement('button');
            copyBtn.className = 'ai-assistant-panel-bubble-action';
            copyBtn.type = 'button';
            copyBtn.setAttribute('aria-label', 'Copy this answer');
            copyBtn.title = 'Copy this answer';
            copyBtn.innerHTML = ICONS.copyAns;   // ICONS constant — safe.
            var copyLbl = document.createElement('span');
            copyLbl.textContent = 'Copy';
            copyBtn.appendChild(copyLbl);
            copyBtn.addEventListener('click', function () { copyAnswer(text, bubble); });
            actions.appendChild(copyBtn);

            // Share button — between Copy and Retry (OpenAI-inspired).
            // Payload = "Q: <question>\n\nA: <answer>\n\n— AI · <url>"
            // so the recipient receives full context without visiting the source.
            // retryQ already resolved above; direct closure is safe (no loop).
            var shareBtn = document.createElement('button');
            shareBtn.className = 'ai-assistant-panel-bubble-action';
            shareBtn.type = 'button';
            shareBtn.setAttribute('aria-label', 'Share this answer');
            shareBtn.title = 'Share Q \u0026 A \u2014 send question + answer to another app or clipboard';
            shareBtn.innerHTML = ICONS.shareAns;   // ICONS constant — safe.
            var shareLbl = document.createElement('span');
            shareLbl.textContent = 'Share';
            shareBtn.appendChild(shareLbl);
            shareBtn.addEventListener('click', function () { _shareAnswer(text, retryQ, bubble, shareBtn); });
            actions.appendChild(shareBtn);

            // Retry button — re-submits the paired user question.
            // retryQ resolved above (hoisted so Share can use it too).
            if (retryQ) {
                var retryBtn = document.createElement('button');
                retryBtn.className = 'ai-assistant-panel-bubble-action';
                retryBtn.type = 'button';
                retryBtn.setAttribute('aria-label', 'Retry this answer');
                retryBtn.title = 'Retry — re-send the same question';
                retryBtn.innerHTML = ICONS.retry;  // ICONS constant — safe.
                var retryLbl = document.createElement('span');
                retryLbl.textContent = 'Retry';
                retryBtn.appendChild(retryLbl);
                retryBtn.addEventListener('click', function () {
                    var panelInput = document.getElementById('ai-assistant-panel-input');
                    if (!panelInput) return;
                    panelInput.value = retryQ;
                    _updateSendBtnState();
                    handleAIPanelSubmit();
                });
                actions.appendChild(retryBtn);
            }

            // ── "⋯ More ▾" expandable submenu (contains Listen and future actions)
            var moreWrapper = _buildBubbleMore(text);
            actions.appendChild(moreWrapper);

            body.appendChild(actions);

            // ── R5: per-answer inline feedback block ──────────────────────────
            // Count how many assistant answers precede this one so each gets a
            // unique stable index for independent feedback tracking.
            //
            // Pass the answer text (this bubble) and the paired user question
            // (retryQ, resolved a few lines above) so the dispatched event
            // payload is a complete (q, a, rating, message) training tuple.
            var answerIndex = body.querySelectorAll(
                '.ai-assistant-panel-feedback').length;
            var fb = _buildFeedbackBlock(answerIndex, text, retryQ);
            if (fb) body.appendChild(fb);
        }
    }

    // ── Typing indicator ──────────────────────────────────────────────────────

    /**
     * Show a pulsing typing indicator bubble in the panel body.
     * Creates the element (never re-creates if one already exists — idempotent).
     *
     * @param {HTMLElement} body  The panel body element.
     * @returns {HTMLElement}  The typing indicator element (for later removal).
     */
    function _showTypingIndicator(body) {
        var existing = body.querySelector('.ai-assistant-typing');
        if (existing) return existing;

        var el = document.createElement('div');
        el.className = 'ai-assistant-typing';
        el.setAttribute('aria-label', 'AI is thinking');
        el.setAttribute('role', 'status');
        el.setAttribute('aria-live', 'polite');

        for (var i = 0; i < 3; i++) {
            var dot = document.createElement('span');
            dot.className = 'ai-assistant-typing-dot';
            el.appendChild(dot);
        }
        body.appendChild(el);
        body.scrollTop = body.scrollHeight;
        return el;
    }

    /**
     * Remove the typing indicator from the panel body (safe no-op if absent).
     * @param {HTMLElement} body
     */
    function _hideTypingIndicator(body) {
        var el = body && body.querySelector('.ai-assistant-typing');
        if (el) el.remove();
    }

    /**
     * Append a message: records it in the single source of truth and renders
     * it via `_renderBubble`.
     *
     * Ordering guarantee (per turn):
     *   user bubble  →  [assistant bubble → action row (Copy, Retry) → feedback block]
     *
     * The action row and per-answer feedback block are built and inserted
     * inside `_renderBubble` itself — directly after the assistant bubble —
     * so they are always in DOM order with that bubble regardless of how many
     * turns precede or follow.  There is NO second feedback pass here.
     *
     * @param {string} text
     * @param {string} role  'user' | 'assistant' | 'error'
     */
    function _appendPanelMessage(text, role) {
        var body = document.getElementById('ai-assistant-panel-body');
        if (!body) return;

        // Remove welcome + suggestions on first real message.
        var welcome = body.querySelector('.ai-assistant-panel-welcome');
        if (welcome) welcome.remove();
        var suggestions = body.querySelector('.ai-assistant-panel-suggestions');
        if (suggestions) suggestions.remove();

        _recordMessage(role, text);   // single source of truth
        // Read the timestamp just stored — _recordMessage always pushes before
        // returning and JS is single-threaded, so the last entry is ours.
        _renderBubble(body, text, role, undefined, _transcript[_transcript.length - 1].ts);
        // _renderBubble already appended: bubble → action row → feedback block.
        // No further DOM manipulation needed here.

        body.scrollTop = body.scrollHeight;
    }

    // ── Panel submit ──────────────────────────────────────────────────────────

    async function handleAIPanelSubmit() {
        var input   = document.getElementById('ai-assistant-panel-input');
        var sendBtn = document.getElementById('ai-assistant-panel-send');
        if (!input) return;

        var rawText = input.value.trim();
        if (!rawText) return;

        // ── Cancel any in-flight request before starting a new one ───────
        // Without this, rapid submits fire multiple concurrent fetches; the
        // older response can arrive AFTER the newer one, producing
        // out-of-order replies in the panel.  AbortController.abort() causes
        // the pending fetch() to reject with AbortError — caught in the
        // catch block below and silently ignored (no error bubble shown for
        // intentional cancellations).
        if (_fetchAbortController) {
            _fetchAbortController.abort();
        }
        _fetchAbortController = new AbortController();

        // Stop speech if active
        _stopSpeechRecognition();
        _bannerStop(false);
        _dismissSpeakBanner();

        var MAX_CHARS    = 4000;
        var questionText = rawText.length > MAX_CHARS
            ? rawText.slice(0, MAX_CHARS) + '\u2026 [truncated]'
            : rawText;

        _appendPanelMessage(questionText, 'user');
        input.value = '';
        _updateSendBtnState();
        input.disabled = true;
        if (sendBtn) sendBtn.disabled = true;

        // ── Typing indicator ──────────────────────────────────────────────
        var body = document.getElementById('ai-assistant-panel-body');
        var typingEl = body ? _showTypingIndicator(body) : null;

        var cfg = window.AI_ASSISTANT_CONFIG || {};
        try {
            if (cfg.panelApiEnabled) {
                await _panelApiCall(questionText, cfg);
            } else {
                await _panelStubReply(questionText);
            }
        } catch (err) {
            // AbortError is thrown when _fetchAbortController.abort() is
            // called (i.e. the user submitted a new question before this
            // one completed).  Do NOT show an error bubble for intentional
            // cancellations — the new question's handler will show its own reply.
            if (err && err.name === 'AbortError') {
                // Intentional cancellation — swallow silently.
            } else {
                console.error('AI Assistant panel error:', err);
                _appendPanelMessage('Sorry, something went wrong: ' + err.message, 'error');
            }
        } finally {
            if (body) _hideTypingIndicator(body);
            input.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            _updateSendBtnState();
            input.focus();
        }
    }

    // ── API + stub ────────────────────────────────────────────────────────────

    /**
     * Live API call — routed through a USER-SUPPLIED PROXY endpoint.
     *
     * Why a proxy is mandatory (C-2)
     * ──────────────────────────────
     * A browser cannot call any AI provider API directly:
     *   • Providers send no CORS headers for arbitrary web origins, so the
     *     preflight request is blocked before it leaves the browser.
     *   • Every provider requires a secret API token; embedding it in static
     *     JS would expose it to every reader of the page source.
     * Therefore "API mode" MUST point at the doc owner's own thin proxy that
     * injects the token server-side.  Free options (zero ongoing cost):
     *
     *   Option A — HuggingFace Space (CPU tier, always on, free):
     *       endpoint = "https://<org>-ai-proxy.hf.space/v1/chat/completions"
     *       provider = "huggingface"
     *
     *   Option B — Cloudflare Worker (100 000 req/day free tier):
     *       endpoint = "https://hf-proxy.<subdomain>.workers.dev"
     *       provider = "huggingface" | "cloudflare"
     *
     *   Option C — local dev_proxy.py (development only, never deploy):
     *       endpoint = "http://localhost:8787/v1/chat/completions"
     *       provider = "huggingface"
     *
     *   Option D — HuggingFace ZeroGPU Space (free shared GPU, self-host model):
     *       endpoint = "https://<org>-<space>.hf.space/v1/chat/completions"
     *       provider = "huggingface"
     *
     * Provider routing
     * ────────────────
     * provider === "anthropic"
     *   → Anthropic /v1/messages body shape (system at top level, not in messages).
     *     Never streams (Anthropic SSE requires a different event format that
     *     needs a separate implementation; non-streaming is cleaner here).
     *
     * all other providers (OpenAI-compat)
     *   → OpenAI /v1/chat/completions body shape (system as messages[0]).
     *     When cfg.panelApiStreaming !== false AND provider is in
     *     _STREAMING_PROVIDERS: sends stream:true and renders via SSE loop.
     *     Otherwise sends stream:false and waits for the complete JSON.
     *
     * Response parsing (non-streaming path)
     * ──────────────────────────────────────
     * 1. Anthropic shape:  data.content[].text
     * 2. OpenAI shape:     data.choices[0].message.content
     * 3. Generic fallback: data.reply | data.answer | data.text
     * Simpler proxy responses (wrapping models behind a thin shim) work via (3).
     *
     * @param {string} question  User question text (already length-truncated).
     * @param {object} cfg       window.AI_ASSISTANT_CONFIG
     */
    async function _panelApiCall(question, cfg) {
        // ── 1. Resolve active model and endpoint ──────────────────────────
        var activeModel = _getActiveModel(cfg);
        var endpoint = '';
        var modelName = '';
        var provider = '';

        if (activeModel) {
            // Per-model endpoint wins; falls back to shared panelApiUrl so
            // the convenient list[str] config shape still works.
            endpoint  = (activeModel.endpoint || '').trim() ||
                        (typeof cfg.panelApiUrl === 'string'
                            ? cfg.panelApiUrl.trim() : '');
            modelName = activeModel.model || activeModel.id;
            provider  = (activeModel.provider || 'custom').toLowerCase();
        } else {
            // Legacy single-model path (ai_assistant_panel_api_url +
            // ai_assistant_panel_api_model).  Defaults to Anthropic so
            // existing single-model deployments are unaffected.
            endpoint  = (typeof cfg.panelApiUrl === 'string'
                            ? cfg.panelApiUrl.trim() : '');
            modelName = cfg.panelApiModel || 'claude-sonnet-4-20250514';
            provider  = 'anthropic';
        }

        // ── 2. Guard: endpoint is required ────────────────────────────────
        if (!endpoint) {
            throw new Error(
                'API mode is enabled but no proxy endpoint is configured.\n' +
                'The browser cannot call any AI provider API directly — a thin\n' +
                'server-side proxy is required to inject the token.\n\n' +
                'Free options (zero ongoing cost):\n' +
                '  A) HuggingFace Space (CPU, always on):\n' +
                '       endpoint: "https://<org>-ai-proxy.hf.space/v1/chat/completions"\n' +
                '  B) Cloudflare Worker (100k req/day free):\n' +
                '       endpoint: "https://hf-proxy.<subdomain>.workers.dev"\n' +
                '  C) Local dev only — run dev_proxy.py on port 8787:\n' +
                '       endpoint: "http://localhost:8787/v1/chat/completions"\n\n' +
                'Set ai_assistant_panel_api_url (single-model) or add an\n' +
                '"endpoint" key to each ai_assistant_panel_api_models entry.'
            );
        }

        // ── 3. Build page context (best-effort; never throws) ─────────────
        var pageMarkdown = '';
        try { pageMarkdown = await convertToMarkdown(); } catch (_) {}

        // FIX Issue 7: configurable token and context limits.
        // Global defaults come from cfg; per-model overrides take precedence.
        // _safeInt(val, min, max, fallback) — defined at module level.
        var maxTokens    = _safeInt(cfg.panelMaxTokens,    1, 32000,  1000);
        var contextLimit = _safeInt(cfg.panelContextLimit, 100, 200000, 8000);
        if (activeModel) {
            // Per-model entry can override the global limits — e.g. a smaller
            // model on a free-tier endpoint may need a tighter context window.
            if (activeModel.max_tokens)
                maxTokens    = _safeInt(activeModel.max_tokens,    1, 32000,  maxTokens);
            if (activeModel.context_limit)
                contextLimit = _safeInt(activeModel.context_limit, 100, 200000, contextLimit);
        }

        // FIX Issue 7: configurable system prompt via cfg.panelSystemPrompt.
        // Use {context} as the template variable for the page markdown block.
        var defaultSystemPrompt = pageMarkdown
            ? 'You are a helpful documentation assistant. Answer questions ' +
              'about the following documentation page.\n\n---\n' +
              pageMarkdown.slice(0, contextLimit) + '\n---'
            : 'You are a helpful documentation assistant.';
        var systemPrompt = (typeof cfg.panelSystemPrompt === 'string' &&
                            cfg.panelSystemPrompt)
            ? cfg.panelSystemPrompt.replace('{context}',
                pageMarkdown.slice(0, contextLimit))
            : defaultSystemPrompt;

        // ── 4. Build request body ─────────────────────────────────────────
        // Anthropic uses a distinct body shape (system at top level).
        // Every other provider (HuggingFace, Groq, Cloudflare Workers AI,
        // Cerebras, Together, Fireworks, SambaNova, Ollama, custom) uses the
        // OpenAI /v1/chat/completions shape.
        var isAnthropic = (provider === 'anthropic');
        var body;
        if (isAnthropic) {
            body = JSON.stringify({
                model:      modelName,
                max_tokens: maxTokens,
                system:     systemPrompt,
                messages:   [{ role: 'user', content: question }],
            });
        } else {
            body = JSON.stringify({
                model:      modelName,
                max_tokens: maxTokens,
                stream:     false,   // overwritten below when streaming is on
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user',   content: question },
                ],
            });
        }

        // ── 5. SSE streaming path (OpenAI-compat providers only) ──────────
        //
        // Which providers stream:
        //   All OpenAI-compat providers support SSE when stream:true is sent.
        //   Anthropic uses a different SSE format (anthropic-stream) — handled
        //   separately and not yet implemented; stays on the non-streaming path.
        //
        // Master switch: cfg.panelApiStreaming (from ai_assistant_panel_api_streaming
        // in conf.py).  Set False on hosting platforms that buffer SSE frames
        // (some PaaS providers coalesce the stream into a single response).
        //
        // _STREAMING_PROVIDERS is the exhaustive list of OpenAI-compat
        // providers whose proxies are known to forward SSE correctly.
        // "custom" is included so user-defined endpoints stream by default;
        // set ai_assistant_panel_api_streaming = False to opt out.
        var _STREAMING_PROVIDERS = [
            'huggingface',   // HF Inference API, HF Space proxies, ZeroGPU Spaces
            'groq',          // Groq fast-inference cloud (free tier available)
            'cerebras',      // Cerebras Inference (free tier available)
            'together',      // Together AI (free tier available)
            'fireworks',     // Fireworks AI (free tier available)
            'sambanova',     // SambaNova Cloud (free tier available)
            'cloudflare',    // Cloudflare Workers AI (free 10k tokens/day)
            'ollama',        // Local Ollama server (always free)
            'openai',        // OpenAI API (proxied)
            'google',        // Google Gemini OpenAI-compat endpoint (proxied)
            'mistral',       // Mistral AI (proxied)
            'deepseek',      // DeepSeek API (proxied)
            'azure_openai',  // Azure OpenAI (proxied)
            'perplexity',    // Perplexity API (proxied)
            'custom',        // Any user-defined OpenAI-compat endpoint
        ];

        var streamingEnabled = (cfg.panelApiStreaming !== false);

        if (streamingEnabled && !isAnthropic &&
                _STREAMING_PROVIDERS.indexOf(provider) !== -1) {
            var sb = JSON.parse(body);
            sb.stream = true;
            await _panelApiCallStreaming(endpoint, JSON.stringify(sb), provider);
            return;
        }

        // ── 6. Non-streaming path ─────────────────────────────────────────
        var response = await fetch(endpoint, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    body,
            signal:  _fetchAbortController ? _fetchAbortController.signal : undefined,
        });

        if (!response.ok) {
            var errBody = await response.text().catch(function () { return ''; });
            throw new Error('API ' + response.status + ': ' + errBody.slice(0, 120));
        }

        var data = await response.json();
        var reply = '';

        if (isAnthropic) {
            // Anthropic /v1/messages response shape: {content: [{type,text}]}
            if (Array.isArray(data.content)) {
                reply = data.content
                    .filter(function (b) { return b && b.type === 'text'; })
                    .map(function (b) { return b.text; })
                    .join('\n').trim();
            }
        } else {
            // OpenAI /v1/chat/completions shape: {choices:[{message:{content}}]}
            if (Array.isArray(data.choices) && data.choices.length > 0) {
                var msg = data.choices[0].message;
                reply = (msg && typeof msg.content === 'string')
                    ? msg.content.trim() : '';
            }
            // Anthropic-shape fallback: proxy wraps an Anthropic response
            // verbatim even when provider is not 'anthropic'.
            if (!reply && Array.isArray(data.content)) {
                reply = data.content
                    .filter(function (b) { return b && b.type === 'text'; })
                    .map(function (b) { return b.text; })
                    .join('\n').trim();
            }
        }

        // Generic fallback: simpler shim proxies return a flat {reply|answer|text}.
        if (!reply) {
            reply = (typeof data.reply  === 'string' && data.reply)  ||
                    (typeof data.answer === 'string' && data.answer) ||
                    (typeof data.text   === 'string' && data.text)   || '';
        }

        _appendPanelMessage(reply || '(no response)', 'assistant');
    }

    async function _panelApiCallStreaming(endpoint, bodyStr, provider) {
        var response = await fetch(endpoint, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    bodyStr,
            signal:  _fetchAbortController ? _fetchAbortController.signal : undefined,
        });

        if (!response.ok) {
            var errBody = await response.text().catch(function () { return ''; });
            throw new Error('API ' + response.status + ': ' + errBody.slice(0, 120));
        }

        // ── Graceful fallback: proxy returned JSON instead of SSE ─────────
        // Some hosting platforms (certain PaaS providers, ZeroGPU cold-start
        // responses) buffer the stream and return a single JSON blob even when
        // stream:true was requested.  Parse it with the same triple-shape
        // parser used by the non-streaming path so the panel always shows a
        // reply rather than "(no response)".
        var contentType = response.headers.get('content-type') || '';
        if (contentType.indexOf('text/event-stream') === -1 || !response.body) {
            var data2 = await response.json();
            var reply2 = '';
            // OpenAI shape
            if (Array.isArray(data2.choices) && data2.choices.length > 0) {
                var m2 = data2.choices[0].message;
                reply2 = (m2 && typeof m2.content === 'string')
                    ? m2.content.trim() : '';
            }
            // Anthropic shape fallback (proxy forwarded Anthropic verbatim)
            if (!reply2 && Array.isArray(data2.content)) {
                reply2 = data2.content
                    .filter(function (b) { return b && b.type === 'text'; })
                    .map(function (b) { return b.text; })
                    .join('\n').trim();
            }
            // Generic shim fallback
            if (!reply2) {
                reply2 = (typeof data2.reply  === 'string' && data2.reply)  ||
                         (typeof data2.answer === 'string' && data2.answer) ||
                         (typeof data2.text   === 'string' && data2.text)   || '';
            }
            _appendPanelMessage(reply2 || '(no response)', 'assistant');
            return;
        }

        var panelBody = document.getElementById('ai-assistant-panel-body');
        _hideTypingIndicator(panelBody);

        var streamBubble = document.createElement('div');
        streamBubble.className = 'ai-assistant-panel-bubble ai-assistant-panel-bubble--assistant ai-assistant-panel-bubble--streaming';
        if (panelBody) panelBody.appendChild(streamBubble);

        var accumulated = '';
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var sseBuf = '';
        // Track the current SSE event type (RFC 6455 §10.1):
        // Lines beginning with "event:" set the event type for the NEXT
        // "data:" line.  Reset to "message" after each dispatch.
        var sseEventType = 'message';

        try {
            while (true) {
                var chk = await reader.read();
                if (chk.done) break;
                sseBuf += decoder.decode(chk.value, { stream: true });
                var lines = sseBuf.split('\n');
                sseBuf = lines.pop();
                for (var li = 0; li < lines.length; li++) {
                    var ln = lines[li].trim();
                    if (!ln) {
                        // Empty line = SSE event boundary: reset event type
                        sseEventType = 'message';
                        continue;
                    }
                    // Track event: field (sets type for subsequent data:)
                    if (ln.startsWith('event: ')) {
                        sseEventType = ln.slice(7).trim();
                        continue;
                    }
                    if (ln === 'data: [DONE]') {
                        sseEventType = 'message';
                        continue;
                    }
                    if (ln.startsWith('data: ')) {
                        // Server-sent event: error — surface message to user.
                        // Some SSE servers emit "event: error\ndata: {...}" on
                        // rate-limit, auth failure, or upstream API errors.
                        // Without this branch they are silently dropped.
                        if (sseEventType === 'error') {
                            var errPayload = ln.slice(6);
                            var errMsg = '';
                            try {
                                var ep = JSON.parse(errPayload);
                                errMsg = (ep && (ep.error || ep.message || ep.detail)) || errPayload;
                            } catch (_) { errMsg = errPayload; }
                            console.error('AI Assistant: SSE server error event:', errMsg);
                            // Replace streaming bubble with error bubble so the
                            // user sees the failure, not an empty reply.
                            if (streamBubble && streamBubble.parentNode) {
                                streamBubble.parentNode.removeChild(streamBubble);
                            }
                            _appendPanelMessage(
                                'The AI server reported an error: ' +
                                String(errMsg).slice(0, 200),
                                'error'
                            );
                            sseEventType = 'message';
                            return;  // abort further SSE processing
                        }
                        try {
                            var parsed = JSON.parse(ln.slice(6));
                            var delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
                            if (delta && typeof delta.content === 'string') {
                                accumulated += delta.content;
                                streamBubble.innerHTML = _mdToHtml(accumulated);
                                streamBubble.setAttribute('data-raw', accumulated);
                                if (panelBody) panelBody.scrollTop = panelBody.scrollHeight;
                            }
                        } catch (_pe) {}
                        sseEventType = 'message';
                    }
                }
            }
        } finally {
            try { reader.releaseLock(); } catch (_) {}
        }

        streamBubble.classList.remove('ai-assistant-panel-bubble--streaming');
        _recordMessage('assistant', accumulated || '(no response)');
        // Read the timestamp just stored — same single-threaded guarantee as
        // _appendPanelMessage: the last _transcript entry is this streamed reply.
        var streamTs = _transcript[_transcript.length - 1].ts;

        if (panelBody && accumulated) {
            var acts = document.createElement('div');
            acts.className = 'ai-assistant-panel-bubble-actions';

            // Timestamp — prepended before Copy so information precedes action.
            acts.appendChild(_buildBubbleTimeEl(streamTs));

            // Hoist question resolution before Copy so Share can reuse it without
            // a second _transcript walk — single source of truth, resolved once.
            // Array.findLast (ES2023): declarative reverse scan — no mutable
            // sentinel, no manual break — semantically identical to the IIFE
            // used in the non-streaming _renderBubble path above.
            var _lastUser = _transcript.findLast(function (m) { return m.role === 'user'; });
            var retryQ2 = _lastUser ? _lastUser.text : null;

            // Copy button
            var cb2 = document.createElement('button');
            cb2.className = 'ai-assistant-panel-bubble-action';
            cb2.type = 'button';
            cb2.setAttribute('aria-label', 'Copy this answer');
            cb2.title = 'Copy this answer';
            cb2.innerHTML = ICONS.copyAns;
            var cl2 = document.createElement('span'); cl2.textContent = 'Copy';
            cb2.appendChild(cl2);
            (function (ft, bEl) {
                cb2.addEventListener('click', function () { copyAnswer(ft, bEl); });
            }(accumulated, streamBubble));
            acts.appendChild(cb2);

            // Share button — between Copy and Retry (OpenAI-inspired).
            // Payload = "Q: <question>\n\nA: <answer>\n\n— AI · <url>"
            // so the recipient receives full context without visiting the source.
            (function (and, q, bbl) {
                var sb2 = document.createElement('button');
                sb2.className = 'ai-assistant-panel-bubble-action';
                sb2.type = 'button';
                sb2.setAttribute('aria-label', 'Share this answer');
                sb2.title = 'Share Q \u0026 A \u2014 send question + answer to another app or clipboard';
                sb2.innerHTML = ICONS.shareAns;
                var sl2 = document.createElement('span'); sl2.textContent = 'Share';
                sb2.appendChild(sl2);
                sb2.addEventListener('click', function () { _shareAnswer(and, q, bbl, sb2); });
                acts.appendChild(sb2);
            }(accumulated, retryQ2, streamBubble));

            // Retry button — retryQ2 hoisted above so Share can use it too.
            if (retryQ2) {
                var rb2 = document.createElement('button');
                rb2.className = 'ai-assistant-panel-bubble-action';
                rb2.type = 'button';
                rb2.setAttribute('aria-label', 'Retry this answer');
                rb2.title = 'Retry \u2014 re-send the same question';
                rb2.innerHTML = ICONS.retry;
                var rl2 = document.createElement('span'); rl2.textContent = 'Retry';
                rb2.appendChild(rl2);
                (function (q) {
                    rb2.addEventListener('click', function () {
                        var pi = document.getElementById('ai-assistant-panel-input');
                        if (!pi) return;
                        pi.value = q;
                        _updateSendBtnState();
                        handleAIPanelSubmit();
                    });
                }(retryQ2));
                acts.appendChild(rb2);
            }

            // "⋯ More ▾" — extensible submenu (contains Listen + future actions)
            var moreW2 = _buildBubbleMore(accumulated);
            acts.appendChild(moreW2);

            panelBody.appendChild(acts);

            var fbIdx2 = panelBody.querySelectorAll('.ai-assistant-panel-feedback').length;
            var fb2 = _buildFeedbackBlock(fbIdx2, accumulated, retryQ2);
            if (fb2) panelBody.appendChild(fb2);
        }
        if (panelBody) panelBody.scrollTop = panelBody.scrollHeight;
    }

    async function _panelStubReply(_question) {
        await new Promise(function (resolve) { setTimeout(resolve, 400); });
        _appendPanelMessage(
            'This AI assistant panel is running in stub mode (no live API calls).\n\n' +
            'To enable live responses, set in conf.py:\n' +
            '    ai_assistant_panel_api_enabled = True\n' +
            '    ai_assistant_panel_api_models  = [{ "id": "...", "provider": "huggingface",\n' +
            '        "model": "openai/gpt-oss-20b",\n' +
            '        "endpoint": "<your-free-proxy-url>" }]\n\n' +
            'Free proxy options (zero ongoing cost):\n' +
            '  A) HuggingFace Space (CPU, always on) — deploy app.py + Dockerfile\n' +
            '  B) Cloudflare Worker (100 000 req/day) — deploy worker.js\n' +
            '  C) Local dev_proxy.py (development only) — run on port 8787\n' +
            '  D) HuggingFace ZeroGPU Space (free shared GPU, self-host the model)',
            'assistant'
        );
    }

    // ── Bootstrap ─────────────────────────────────────────────────────────────

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAIAssistant);
    } else {
        initAIAssistant();
    }

})();
