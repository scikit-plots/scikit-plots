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
     * Web Speech API recognition instance (lazy, created on first mic click).
     * @type {SpeechRecognition|null}
     */
    var _speechRecognition = null;

    /** True when speech recognition is actively listening. */
    var _isListening = false;

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
                var isOpen = dropdown.style.display !== 'none';
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
     */
    function clearConversation() {
        _transcript = [];
        _feedbackGivenSet = new Set();
        _ssDel(_TRANSCRIPT_KEY);
        var body = document.getElementById('ai-assistant-panel-body');
        if (!body) return;
        body.innerHTML = '';
        _renderWelcome(body);
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
            lines.push('[' + who + ']');
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
            _renderBubble(body, m.text, m.role);
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

        var head = document.createElement('div');
        head.className = 'ai-assistant-panel-privacy-head';
        var hStrong = document.createElement('strong');
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

        var models = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
        if (models.length === 0) {
            var empty = document.createElement('p');
            empty.textContent =
                'No models are configured. Set ' +
                'ai_assistant_panel_api_models in conf.py to enable the picker.';
            bodyEl.appendChild(empty);
            sheet.appendChild(bodyEl);
            return sheet;
        }

        var activeId = _getActiveModelId(models);
        var groupName = 'ai-assistant-model-' + Math.random().toString(36).slice(2, 8);

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

            if (m.info_url && typeof m.info_url === 'string') {
                // Public info page link (e.g. anthropic.com/claude).
                // Validated by ai_assistant_panel_api_models filter so the
                // scheme is guaranteed safe (http/https or site-relative).
                var info = document.createElement('a');
                info.className = 'ai-assistant-panel-model-info';
                info.href = m.info_url;
                info.target = '_blank';
                info.rel = 'noopener noreferrer';
                info.setAttribute('aria-label', 'Open model info page');
                info.title = 'Open model info page';
                info.innerHTML = ICONS.info;     // ICONS constant — safe.
                row.appendChild(info);
            }

            row.addEventListener('change', function () {
                if (!radio.checked) return;
                _setActiveModelId(m.id);
                // Notify doc authors so they can react (e.g. analytics).
                try {
                    document.dispatchEvent(new CustomEvent(
                        'ai-assistant-model-change',
                        { detail: { id: m.id, provider: m.provider,
                                    model: m.model } }));
                } catch (_) {}
                // Sync inline picker if present.
                _syncInlinePickers(m.id);
                // Close the sheet on selection.
                sheet.setAttribute('data-open', 'false');
            });

            bodyEl.appendChild(row);
        });

        sheet.appendChild(bodyEl);
        return sheet;
    }

    /**
     * Update all inline model pickers in the DOM to reflect a new active id.
     * Called whenever the model changes via the sheet so the inline picker
     * stays in sync (and vice-versa via _buildInlineModelPicker).
     *
     * @param {string} id
     */
    function _syncInlinePickers(id) {
        var pickers = document.querySelectorAll('.ai-assistant-panel-inline-model-picker');
        pickers.forEach(function (p) {
            if (p.value !== id) p.value = id;
        });
    }

    /**
     * Update the model-sheet radio buttons to reflect a new active id.
     * Called whenever the model changes via the inline <select> picker so the
     * sheet stays in sync — the symmetric counterpart to _syncInlinePickers.
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

        addItem(ICONS.privacy,  'Privacy & Responsibility',  hooks && hooks.onPrivacy);
        addItem(ICONS.terms,    'Terms of Service',          hooks && hooks.onTerms);
        addItem(ICONS.model,    'Choose a model',            hooks && hooks.onModel);
        addItem(ICONS.share,    'Share',                     hooks && hooks.onShare);

        // Keyboard shortcut hint row — shown at the bottom of the menu when a
        // shortcut is configured.  Non-interactive (aria-hidden); its purpose
        // is to remind the user of the panel toggle key without needing to look
        // at the external kbd-hint in the subbar.
        var kbdHintLabel = _shortcutLabel();
        if (kbdHintLabel) {
            var sep = document.createElement('hr');
            sep.className = 'ai-assistant-panel-hamburger-sep';
            sep.setAttribute('aria-hidden', 'true');
            pop.appendChild(sep);

            var kbdRow = document.createElement('div');
            kbdRow.className = 'ai-assistant-panel-hamburger-kbd-row';
            kbdRow.setAttribute('aria-hidden', 'true');

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
    function _buildInlineModelPicker() {
        var cfg = window.AI_ASSISTANT_CONFIG || {};
        if (cfg.panelInlineModelPicker === false) return null;
        var models = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
        if (models.length === 0) return null;

        var sel = document.createElement('select');
        sel.className =
            'ai-assistant-panel-footer-btn ai-assistant-panel-inline-model-picker';
        sel.setAttribute('aria-label', 'Active model');
        sel.title = 'Active model — affects the next reply only';

        var activeId = _getActiveModelId(models);
        models.forEach(function (m) {
            var opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = (m.label || m.id);
            if (m.id === activeId) opt.selected = true;
            sel.appendChild(opt);
        });

        sel.addEventListener('change', function () {
            var id = sel.value;
            _setActiveModelId(id);
            /* Sync the model-sheet radio buttons so opening the sheet after
               changing the inline picker always shows the correct selection. */
            _syncModelSheet(id);
            try {
                var m = _findModel(models, id);
                document.dispatchEvent(new CustomEvent(
                    'ai-assistant-model-change',
                    { detail: m ? { id: m.id, provider: m.provider, model: m.model }
                                : { id: id } }));
            } catch (_) {}
        });

        return sel;
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
        newChatBtn.addEventListener('click', clearConversation);

        // R4: export the conversation as a plain-text download.
        var exportBtn = _createIconBtn(
            'export', 'Export AI conversation as txt', ICONS.exportTxt);
        exportBtn.title = 'Export AI conversation as txt';
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
            leftCluster.appendChild(hint);
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
        rightCluster.appendChild(privacyLink);

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
            rightCluster.appendChild(termsLink);
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

            modelLink.setAttribute('aria-label', 'Choose a model');
            modelLink.title = 'Choose a model';
            rightCluster.appendChild(modelLink);
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
            rightCluster.appendChild(shareLink);
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
                _toggleSpeechRecognition();
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
            panel.dispatchEvent(new CustomEvent('ai-assistant-attach', {
                bubbles: true, cancelable: true,
            }));
        });
        footerActions.appendChild(attachBtn);

        // Right-side action cluster: model ▾ | mic | send
        var footerActionsRight = document.createElement('div');
        footerActionsRight.className = 'ai-assistant-panel-footer-actions-right';

        // Inline model picker (Claude-bar style): [model ▾?]
        // Returns null when no models are configured or panelInlineModelPicker=false.
        var inlinePicker = _buildInlineModelPicker();
        if (inlinePicker) footerActionsRight.appendChild(inlinePicker);

        // Microphone button (shown only when speech is supported): [🎤 mic?]
        var micBtnEl = null;
        if (hasSpeech) {
            micBtnEl = document.createElement('button');
            micBtnEl.className = 'ai-assistant-panel-footer-btn ai-assistant-panel-footer-btn--mic';
            micBtnEl.id = 'ai-assistant-panel-mic';
            micBtnEl.type = 'button';
            micBtnEl.setAttribute('aria-label', 'Speak your question');
            micBtnEl.setAttribute('title', 'Speak your question');
            micBtnEl.innerHTML = ICONS.mic;   // ICONS constant — safe.
            micBtnEl.addEventListener('click', function () {
                _toggleSpeechRecognition();
                _dismissSpeakBanner();
            });
            footerActionsRight.appendChild(micBtnEl);
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

        // R2: privacy/responsibility slide-over (absolute, covers panel).
        var privacySheet = _buildPrivacySheet();
        panel.appendChild(privacySheet);
        privacyLink.addEventListener('click', function () {
            privacySheet.setAttribute('data-open', 'true');
        });

        // ── Phase B: additional slide-over sheets + hamburger popover ──────
        //
        // Each sheet uses the exact same data-open contract as the privacy
        // sheet so the existing CSS animation and a11y semantics cover them
        // for free.  Only one sheet may be open at a time — opening one
        // closes the others, which keeps the panel readable on small screens.
        var modelSheet = _buildModelSheet();
        panel.appendChild(modelSheet);

        var termsSheet = (cfgRef.panelTerms !== false) ? _buildTermsSheet() : null;
        if (termsSheet) panel.appendChild(termsSheet);

        var shareSheet = (cfgRef.panelShare !== false) ? _buildShareSheet() : null;
        if (shareSheet) panel.appendChild(shareSheet);

        /**
         * Open exactly one sheet at a time.  Pass null to close all.
         * @param {HTMLElement|null} target
         */
        function _openSheet(target) {
            [privacySheet, modelSheet, termsSheet, shareSheet].forEach(function (s) {
                if (!s) return;
                s.setAttribute('data-open', (s === target) ? 'true' : 'false');
            });
        }

        // Wire the sub-bar buttons.  Each handler routes through _openSheet
        // so the "only one open at a time" invariant is honoured centrally.
        if (modelLink) {
            modelLink.addEventListener('click', function () { _openSheet(modelSheet); });
        }
        // Re-bind the privacy link through _openSheet so opening Privacy
        // closes any other sheet that may already be open.  (The earlier
        // direct binding above is harmless — both fire and converge on the
        // same final state — but the routed version is the source of truth.)
        privacyLink.addEventListener('click', function () { _openSheet(privacySheet); });
        if (termsLink && termsSheet) {
            termsLink.addEventListener('click', function () { _openSheet(termsSheet); });
        }
        if (shareLink && shareSheet) {
            shareLink.addEventListener('click', function () { _openSheet(shareSheet); });
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
                var lbl = modelLink.querySelector('.ai-assistant-panel-model-link-label');
                if (lbl) lbl.textContent = m ? (m.label || m.id) : d.id;
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
                onPrivacy: function () { _openSheet(privacySheet); },
                onTerms:   termsSheet  ? function () { _openSheet(termsSheet); }   : null,
                onShare:   shareSheet  ? function () { _openSheet(shareSheet); }   : null,
                onModel:   modelLink   ? function () { _openSheet(modelSheet); }   : null,
            });
            panel.appendChild(hamburgerMenuEl);

            // Left hamburger: anchor popover to the left edge.
            hamburgerBtn.addEventListener('click', function (e) {
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
            var _subbarRO = new ResizeObserver(function (entries) {
                var w = entries[0] && entries[0].contentRect && entries[0].contentRect.width;
                if (typeof w !== 'number') return;
                panel.setAttribute('data-narrow', w < 360 ? 'true' : 'false');
            });
            _subbarRO.observe(panel);
        }

        // ── Events ────────────────────────────────────────────────────────────

        closeBtn.addEventListener('click', closeAIPanel);

        minimizeBtn.addEventListener('click', function () { minimizeAIPanel(); });

        maximizeBtn.addEventListener('click', function () {
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
                openSheets.forEach(function (s) {
                    s.setAttribute('data-open', 'false');
                });
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

        trigger.addEventListener('click', function () { restoreAIPanel(); });

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

    // ── Speech recognition ────────────────────────────────────────────────────

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
            _speechRecognition.continuous     = false;
            _speechRecognition.interimResults = false;
            _speechRecognition.lang           = navigator.language || 'en-US';

            _speechRecognition.onresult = function (e) {
                var transcript = Array.from(e.results)
                    .map(function (r) { return r[0].transcript; })
                    .join(' ')
                    .trim();
                var input = document.getElementById('ai-assistant-panel-input');
                if (input && transcript) {
                    input.value = (input.value ? input.value + ' ' : '') + transcript;
                    _updateSendBtnState();
                    input.focus();
                }
            };

            _speechRecognition.onend = function () {
                _setMicActiveState(false);
                _isListening = false;
            };

            _speechRecognition.onerror = function (e) {
                _setMicActiveState(false);
                _isListening = false;
                if (e.error !== 'aborted' && e.error !== 'no-speech') {
                    showNotification('Speech recognition error: ' + e.error, true);
                }
            };
        }

        try {
            _speechRecognition.start();
            _isListening = true;
            _setMicActiveState(true);
        } catch (err) {
            console.error('AI Assistant: Speech recognition start error:', err);
            showNotification('Could not start microphone. Check browser permissions.', true);
        }
    }

    function _stopSpeechRecognition() {
        if (_speechRecognition && _isListening) {
            try { _speechRecognition.abort(); } catch (_) {}
        }
        _isListening = false;
        _setMicActiveState(false);
    }

    /**
     * Update the mic button and speak-banner visual state.
     *
     * @param {boolean} active  True → recording animation; false → idle.
     */
    function _setMicActiveState(active) {
        var micBtn    = document.getElementById('ai-assistant-panel-mic');
        var bannerBtn = document.getElementById('ai-assistant-panel-speak-banner');
        if (micBtn) {
            micBtn.classList.toggle('recording', active);
            micBtn.setAttribute('aria-label', active ? 'Stop recording' : 'Speak your question');
            micBtn.setAttribute('title',      active ? 'Stop recording' : 'Speak your question');
        }
        if (bannerBtn) {
            bannerBtn.classList.toggle('recording', active);
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
    function _renderBubble(body, text, role, question) {
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

        if (role === 'assistant' || role === 'error') {
            // ── R6: action row — Copy + Retry ─────────────────────────────────
            var actions = document.createElement('div');
            actions.className = 'ai-assistant-panel-bubble-actions';

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

            // Retry button — re-submits the paired user question.
            // Resolve the question to repeat: prefer the explicit param, then
            // walk _transcript backwards to find the last user turn.
            var retryQ = question || (function () {
                for (var i = _transcript.length - 1; i >= 0; i--) {
                    if (_transcript[i].role === 'user') return _transcript[i].text;
                }
                return null;
            }());
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
        _renderBubble(body, text, role);
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

        var systemPrompt = pageMarkdown
            ? 'You are a helpful documentation assistant. Answer questions ' +
              'about the following documentation page.\n\n---\n' +
              pageMarkdown.slice(0, 8000) + '\n---'
            : 'You are a helpful documentation assistant.';

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
                max_tokens: 1000,
                system:     systemPrompt,
                messages:   [{ role: 'user', content: question }],
            });
        } else {
            body = JSON.stringify({
                model:      modelName,
                max_tokens: 1000,
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

        if (panelBody && accumulated) {
            var acts = document.createElement('div');
            acts.className = 'ai-assistant-panel-bubble-actions';

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

            // Retry button — walk _transcript for the last user turn
            (function (answerText) {
                var retryQ2 = null;
                for (var ri = _transcript.length - 1; ri >= 0; ri--) {
                    if (_transcript[ri].role === 'user') { retryQ2 = _transcript[ri].text; break; }
                }
                if (retryQ2) {
                    var rb2 = document.createElement('button');
                    rb2.className = 'ai-assistant-panel-bubble-action';
                    rb2.type = 'button';
                    rb2.setAttribute('aria-label', 'Retry this answer');
                    rb2.title = 'Retry — re-send the same question';
                    rb2.innerHTML = ICONS.retry;
                    var rl2 = document.createElement('span'); rl2.textContent = 'Retry';
                    rb2.appendChild(rl2);
                    rb2.addEventListener('click', function () {
                        var pi = document.getElementById('ai-assistant-panel-input');
                        if (!pi) return;
                        pi.value = retryQ2;
                        _updateSendBtnState();
                        handleAIPanelSubmit();
                    });
                    acts.appendChild(rb2);
                }
            }(accumulated));

            // "⋯ More ▾" — extensible submenu (contains Listen + future actions)
            var moreW2 = _buildBubbleMore(accumulated);
            acts.appendChild(moreW2);

            panelBody.appendChild(acts);

            var fbIdx2 = panelBody.querySelectorAll('.ai-assistant-panel-feedback').length;
            var fb2 = _buildFeedbackBlock(fbIdx2, accumulated, null);
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
