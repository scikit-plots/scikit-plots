/**
 * Sphinx AI Assistant — ai-assistant.js
 *
 * Provides AI-powered features for Sphinx documentation pages:
 *   - Markdown export (clipboard copy + view as .md)
 *   - AI chat deep-links (Claude, ChatGPT, Gemini, …)
 *   - MCP tool integration (VS Code, Claude Desktop, …)
 *   - PDF export with URL-mode / Print-mode toggle
 *   - Floating AI panel (stub + optional Anthropic API)
 *
 * Theme compatibility:
 *   pydata_sphinx_theme  ·  furo  ·  sphinx-book-theme  ·  RTD  ·  Alabaster
 *
 * Configuration:
 *   All behaviour is driven by window.AI_ASSISTANT_CONFIG injected by the
 *   Python extension via add_ai_assistant_context().
 *
 * Security:
 *   - All user-facing HTML uses textContent / setAttribute, not innerHTML
 *     with raw user data.  The single exception (createAIPanel welcome msg)
 *     has been replaced with pure DOM construction in this revision.
 *   - window.open() calls pass 'noopener,noreferrer' on all outgoing links.
 *   - sessionStorage is used for PDF mode persistence (no cross-origin leak).
 *
 * Developer notes:
 *   - Every public function is at module scope but kept inside the IIFE to
 *     avoid polluting the global namespace.
 *   - The single document 'click' listener for dropdown-close is registered
 *     once only (guarded by _listenersAttached).
 *   - Turndown is loaded lazily from CDN on first use.
 *
 * Fixes applied (2026-05-18):
 *   FIX-1  data: / blob: URI icons no longer get _static/ prefix prepended.
 *   FIX-2  handleAIChat: null-guard on prompt_template / url_template.
 *   FIX-3  showInlineSuccessState: DOM mutation instead of innerHTML rebuild.
 *   FIX-4  createAIPanel welcome: pure DOM construction, no innerHTML.
 *   FIX-5  handleMCPInstall: explicit guard + message for invalid mcpb_url.
 *   FIX-6  console.log → console.debug for trace-level messages.
 *   FIX-7  createButton: aria-label + title on main copy button.
 *   FIX-8  createPdfSection toggle row: role="group" + aria-label.
 */

(function () {
    'use strict';

    // ── Module-level singletons ──────────────────────────────────────────────

    /**
     * Capture this script's own src immediately — document.currentScript is
     * only valid during synchronous execution of the script element.  Once
     * execution yields (e.g. inside a Turndown onload callback) the browser
     * resets it to null.  Capturing here, at IIFE top-scope, is the only
     * reliable way to get the value before any async boundary.
     *
     * @type {string|null}
     */
    var _selfSrc = (document.currentScript && document.currentScript.src) || null;

    /** True once the global document 'click' listener has been registered. */
    var _listenersAttached = false;

    /**
     * Lazy singleton: the floating AI panel element.
     *
     * Created on first call to toggleAIPanel(); null until then.
     *
     * CRITICAL: This MUST be declared at IIFE module scope.  The IIFE runs
     * with 'use strict', so reading an undeclared variable throws
     * ReferenceError before the assignment in toggleAIPanel() is ever
     * reached.  That was the root cause of "click button → nothing happens":
     *   1. createAIAssistantUI() → setupEventListeners() registers click handler
     *   2. User clicks AI panel button → toggleAIPanel() called
     *   3. `if (!_aiPanelEl)` → ReferenceError (undeclared) → handler aborts silently
     * Declaring it here with `null` makes the falsy check safe and correct.
     *
     * @type {HTMLElement|null}
     */
    var _aiPanelEl = null;

    /** sessionStorage key used to persist the user's chosen PDF mode. */
    var _PDF_MODE_KEY = 'ai-assistant-pdf-mode';

    /**
     * Feature flag defaults.
     *
     * Used as the base in Object.assign so that keys absent from the Python
     * extension's injected AI_ASSISTANT_CONFIG.features still resolve to a
     * sensible value instead of `undefined` (falsy).
     *
     * Design decision:
     *   - pdf_export defaults to true  — it is a core, always-useful feature
     *     that must not silently vanish when the Python extension omits it.
     *   - ai_panel  defaults to false  — requires API configuration; safe OFF.
     *   - mcp_integration defaults to false — opt-in.
     *
     * The Python extension can override any key by including it explicitly in
     * the features dict it passes to add_ai_assistant_context().
     */
    var FEATURE_DEFAULTS = {
        markdown_export: true,
        view_markdown:   true,
        ai_chat:         true,
        mcp_integration: false,
        pdf_export:      true,
        ai_panel:        false,
        theme_toggle:    false,
    };

    /**
     * Regex that matches any icon value that is already an absolute URI and
     * must NOT be prefixed with the _static path.
     *
     * Handles:
     *   http:// / https://  — remote CDN icons
     *   data:               — inline base64 SVG / PNG icons
     *   blob:               — object-URL icons generated at runtime
     *   /                   — root-relative paths
     *
     * Any string that does NOT match is a bare filename (e.g. 'claude.svg')
     * and must have the static path prepended.
     *
     * Developer note:
     *   The original check was `icon.startsWith('http')`.  That silently
     *   broke all data-URI icons (e.g. the Gemini and Cursor icons in the
     *   config) by prepending '_static/' to the base64 payload.  FIX-1.
     */
    var _ABSOLUTE_ICON_RE = /^(?:https?:|data:|blob:|\/)/;

    // ── Initialisation ───────────────────────────────────────────────────────

    /**
     * Load the Turndown Markdown converter from CDN.
     *
     * @param {Function} callback  Called once Turndown is available.
     */
    function loadTurndown(callback) {
        if (typeof TurndownService !== 'undefined') {
            callback();
            return;
        }
        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/turndown@7.1.2/dist/turndown.min.js';
        script.onload = callback;
        script.onerror = function () {
            console.error('AI Assistant: Failed to load Turndown library');
        };
        document.head.appendChild(script);
    }

    /** Entry point — called when DOM is ready. */
    function initAIAssistant() {
        loadTurndown(function () {
            createAIAssistantUI();
        });
    }

    // ── DOM construction ─────────────────────────────────────────────────────

    /** Build and insert the full AI assistant widget. */
    function createAIAssistantUI() {
        var container = createContainer();
        var button    = createButton();
        var dropdown  = createDropdown();

        container.appendChild(button);
        container.appendChild(dropdown);

        var position = (window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.position) || 'sidebar';
        insertContainer(container, position);
        setupEventListeners(button, dropdown);
    }

    /** @returns {HTMLElement} The root `.ai-assistant-container` div. */
    function createContainer() {
        var el = document.createElement('div');
        el.className = 'ai-assistant-container';
        el.id = 'ai-assistant-container';
        return el;
    }

    /**
     * Build the split-button trigger (main action + dropdown chevron).
     *
     * @returns {HTMLElement}
     */
    function createButton() {
        var container = document.createElement('div');
        container.className = 'ai-assistant-button';
        container.id = 'ai-assistant-button';

        var staticPath = getStaticPath();

        // Main copy button
        var mainBtn = document.createElement('button');
        mainBtn.className = 'ai-assistant-button-main';
        mainBtn.id = 'ai-assistant-button-main';
        mainBtn.type = 'button';
        // FIX-7: Descriptive aria-label so screen readers announce purpose, not
        // just the visible text ("Copy page"), which lacks format context.
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

        // Divider
        var divider = document.createElement('span');
        divider.className = 'ai-assistant-button-divider';

        // Dropdown chevron
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

    /**
     * Build the full dropdown menu.
     *
     * Sections (each separated by a divider when previous section is present):
     *   1. Markdown export  (copy-markdown, view-markdown)
     *   2. AI chat          (one item per enabled provider)
     *   3. MCP tools        (one item per enabled tool)
     *   4. PDF export       (centered button + URL/Print toggle)
     *   5. AI panel         (open floating chat panel)
     *
     * @returns {HTMLElement}
     */
    function createDropdown() {
        var dropdown = document.createElement('div');
        dropdown.className = 'ai-assistant-dropdown';
        dropdown.id = 'ai-assistant-dropdown';
        dropdown.setAttribute('role', 'menu');
        dropdown.style.display = 'none';

        var cfg        = window.AI_ASSISTANT_CONFIG || {};
        // Merge config features over defaults so absent keys resolve
        // to their safe values rather than undefined (falsy).
        var features   = Object.assign({}, FEATURE_DEFAULTS, cfg.features || {});
        var staticPath = getStaticPath();
        var hasItems   = false;  // tracks whether any section was added

        // ── 1. Markdown export ───────────────────────────────────────────────
        if (features.markdown_export) {
            var exportItem = createMenuItem(
                'copy-markdown',
                'Copy page',
                'Copy this page as Markdown for LLMs.',
                staticPath + '/copy-to-clipboard.svg'
            );
            dropdown.appendChild(exportItem);
            hasItems = true;
        }

        if (features.view_markdown) {
            var viewItem = createMenuItem(
                'view-markdown',
                'View as Markdown',
                'View this page as Markdown.',
                staticPath + '/markdown.svg'
            );
            dropdown.appendChild(viewItem);
            hasItems = true;
        }

        // ── 2. AI chat ───────────────────────────────────────────────────────
        if (features.ai_chat) {
            var providers = cfg.providers || {};
            var enabledProviders = Object.entries(providers).filter(function (kv) {
                return kv[1].enabled;
            });

            if (enabledProviders.length > 0) {
                if (hasItems) dropdown.appendChild(createSeparator());
                enabledProviders.forEach(function (kv) {
                    var key      = kv[0];
                    var provider = kv[1];
                    var icon     = provider.icon || 'comment-discussion.svg';
                    // FIX-1: use _ABSOLUTE_ICON_RE so data: / blob: / / prefixes
                    // are treated as absolute and never get _static/ prepended.
                    var iconPath = _ABSOLUTE_ICON_RE.test(icon) ? icon : (staticPath + '/' + icon);
                    var desc     = provider.description || 'Open AI chat with this page context.';
                    var item     = createMenuItem('ai-chat-' + key, provider.label, desc, iconPath);
                    item.dataset.provider = key;
                    dropdown.appendChild(item);
                });
                hasItems = true;
            }
        }

        // ── 3. MCP integration ───────────────────────────────────────────────
        if (features.mcp_integration) {
            var mcpTools = cfg.mcp_tools || {};
            var enabledTools = Object.entries(mcpTools).filter(function (kv) {
                return kv[1].enabled;
            });

            if (enabledTools.length > 0) {
                if (hasItems) dropdown.appendChild(createSeparator());
                enabledTools.forEach(function (kv) {
                    var key  = kv[0];
                    var tool = kv[1];
                    var icon     = tool.icon || 'ai-tools.svg';
                    // FIX-1: same absolute-URI guard as AI chat section above.
                    var iconPath = _ABSOLUTE_ICON_RE.test(icon) ? icon : (staticPath + '/' + icon);
                    var desc     = tool.description || 'Install MCP server.';
                    var item     = createMenuItem('mcp-' + key, tool.label, desc, iconPath);
                    item.dataset.mcpTool = key;
                    dropdown.appendChild(item);
                });
                hasItems = true;
            }
        }

        // ── 4. PDF export — centered button + URL/Print toggle ───────────────
        //
        // The PDF section differs from other sections: the main "Export as PDF"
        // button is horizontally centred, and below it sits an inline two-button
        // toggle that lets the user choose between:
        //   URL mode  → opens AI_ASSISTANT_CONFIG.pdfExportUrl in a new tab
        //   Print mode → calls window.print() (browser → Save as PDF)
        //
        // The chosen mode is persisted in sessionStorage so it survives page
        // navigations within the same Sphinx build, but resets on a new session.
        if (features.pdf_export) {
            if (hasItems) dropdown.appendChild(createSeparator());
            dropdown.appendChild(createPdfSection(staticPath, cfg));
            hasItems = true;
        }

        // ── 5. AI panel ──────────────────────────────────────────────────────
        if (features.ai_panel) {
            var panelTitle = cfg.panelTitle || 'AI Assistant';
            if (hasItems) dropdown.appendChild(createSeparator());
            var panelItem = createMenuItem(
                'ai-panel-open',
                panelTitle,
                'Ask ' + panelTitle + ' about this page',
                staticPath + '/ai-panel.svg'
            );
            dropdown.appendChild(panelItem);
        }

        return dropdown;
    }

    // ── PDF section ──────────────────────────────────────────────────────────

    /**
     * Build the PDF export section for the dropdown.
     *
     * Layout (inside the dropdown panel):
     *
     *   ┌────────────────────────────────────────────┐
     *   │        [ 📄  Export as PDF ]               │  ← centered button
     *   │  PDF mode:  [ URL ]  [ Print ]             │  ← toggle (optional)
     *   └────────────────────────────────────────────┘
     *
     * The toggle reads `pdfExportUrl` from the config:
     *   - Non-empty string → URL mode is the sensible default.
     *   - Empty string / falsy → Print mode is the sensible default.
     *   - User's last choice is persisted via sessionStorage.
     *
     * When `pdfUrlModeToggle` is false the toggle row is hidden; the
     * button always uses the config-driven behaviour.
     *
     * @param {string} staticPath  Path to the _static directory.
     * @param {Object} cfg         window.AI_ASSISTANT_CONFIG.
     * @returns {HTMLElement}
     */
    function createPdfSection(staticPath, cfg) {
        cfg = cfg || {};
        var pdfUrl        = (cfg.pdfExportUrl || '').trim();
        var showToggle    = cfg.pdfUrlModeToggle !== false; // default true

        // Determine initial mode: honour saved session preference, else infer
        // from whether a URL is configured.
        var savedMode = null;
        try {
            savedMode = sessionStorage.getItem(_PDF_MODE_KEY);
        } catch (_e) { /* private browsing / security policy */ }

        var initialMode;
        if (savedMode === 'url' || savedMode === 'print') {
            initialMode = savedMode;
        } else {
            initialMode = pdfUrl ? 'url' : 'print';
        }

        // ── Section wrapper ──────────────────────────────────────────────────
        var section = document.createElement('div');
        section.className = 'ai-assistant-pdf-section';

        // ── Centered export button ───────────────────────────────────────────
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
        // Description reflects current mode — updated by toggle
        btnDesc.textContent = _pdfModeDescription(initialMode, pdfUrl);

        btnContent.appendChild(btnTitle);
        btnContent.appendChild(btnDesc);
        btn.appendChild(btnContent);
        section.appendChild(btn);

        // ── URL / Print toggle ───────────────────────────────────────────────
        if (showToggle) {
            var toggleRow = document.createElement('div');
            toggleRow.className = 'ai-assistant-pdf-toggle';
            toggleRow.id = 'ai-assistant-pdf-toggle';
            // FIX-8: group role + label so screen readers announce the two
            // buttons as a cohesive "PDF export mode" selector, not in isolation.
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
            // Disable URL button when no URL is configured (would be a no-op)
            if (!pdfUrl) urlBtn.disabled = true;

            var printBtn = document.createElement('button');
            printBtn.className = 'ai-assistant-pdf-mode-btn' + (initialMode === 'print' ? ' active' : '');
            printBtn.id = 'ai-assistant-pdf-mode-print';
            printBtn.type = 'button';
            printBtn.textContent = 'Print';
            printBtn.title = 'Use browser print dialog (Save as PDF)';

            // Toggle click handlers — update active state + sessionStorage
            urlBtn.addEventListener('click', function (e) {
                e.stopPropagation();  // keep dropdown open during mode switch
                if (urlBtn.disabled) return;
                _setPdfMode('url');
            });
            printBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                _setPdfMode('print');
            });

            toggleRow.appendChild(toggleLabel);
            toggleRow.appendChild(urlBtn);
            toggleRow.appendChild(printBtn);
            section.appendChild(toggleRow);
        }

        return section;
    }

    /**
     * Return a short description string for the PDF button reflecting *mode*.
     *
     * @param {string} mode   'url' or 'print'.
     * @param {string} pdfUrl The configured PDF export URL (may be empty).
     * @returns {string}
     */
    function _pdfModeDescription(mode, pdfUrl) {
        if (mode === 'url' && pdfUrl) {
            return 'Opens PDF in a new tab.';
        }
        return 'Save as PDF via browser print dialog.';
    }

    /**
     * Persist the chosen PDF mode and update the toggle UI accordingly.
     *
     * @param {'url'|'print'} mode
     */
    function _setPdfMode(mode) {
        try {
            sessionStorage.setItem(_PDF_MODE_KEY, mode);
        } catch (_e) { /* private browsing */ }

        var urlBtn   = document.getElementById('ai-assistant-pdf-mode-url');
        var printBtn = document.getElementById('ai-assistant-pdf-mode-print');
        var descEl   = document.getElementById('ai-assistant-pdf-desc');
        var cfg      = window.AI_ASSISTANT_CONFIG || {};
        var pdfUrl   = (cfg.pdfExportUrl || '').trim();

        if (urlBtn) {
            urlBtn.classList.toggle('active', mode === 'url');
        }
        if (printBtn) {
            printBtn.classList.toggle('active', mode === 'print');
        }
        if (descEl) {
            descEl.textContent = _pdfModeDescription(mode, pdfUrl);
        }
    }

    /**
     * Read the current effective PDF mode.
     *
     * Precedence:
     *   1. sessionStorage preference set by toggle
     *   2. Inferred from whether pdfExportUrl is configured
     *
     * @returns {'url'|'print'}
     */
    function _getPdfMode() {
        var cfg    = window.AI_ASSISTANT_CONFIG || {};
        var pdfUrl = (cfg.pdfExportUrl || '').trim();

        try {
            var saved = sessionStorage.getItem(_PDF_MODE_KEY);
            if (saved === 'url' || saved === 'print') return saved;
        } catch (_e) { /* ignore */ }

        return pdfUrl ? 'url' : 'print';
    }

    // ── Menu helpers ─────────────────────────────────────────────────────────

    /**
     * Create a standard menu item button.
     *
     * @param {string} id       Base ID (prefixed with 'ai-assistant-').
     * @param {string} text     Button label.
     * @param {string} desc     Short description below the label.
     * @param {string} iconSrc  Absolute or relative URL of the icon.
     * @returns {HTMLButtonElement}
     */
    function createMenuItem(id, text, desc, iconSrc) {
        var item = document.createElement('button');
        item.className = 'ai-assistant-menu-item';
        item.id = 'ai-assistant-' + id;
        item.type = 'button';
        item.setAttribute('role', 'menuitem');

        var content = document.createElement('div');
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

    /**
     * Create a horizontal separator `<div>`.
     *
     * @returns {HTMLElement}
     */
    function createSeparator() {
        var sep = document.createElement('div');
        sep.className = 'ai-assistant-menu-separator';
        return sep;
    }

    // ── Static path detection ────────────────────────────────────────────────

    /**
     * Return the URL path to the Sphinx `_static/` directory.
     *
     * Detection priority:
     *   1. `_selfSrc` — captured synchronously at IIFE scope from
     *      `document.currentScript.src` before any async boundary.
     *      This is the most reliable source because this file lives
     *      inside `_static/`.
     *   2. Any `<script src="…_static…">` tag.
     *   3. Any `<link href="…_static…">` tag (CSS file).
     *   4. Fallback `'_static'` (relative; correct for root-level pages only).
     *
     * @returns {string}
     */
    function getStaticPath() {
        // 1 — best: captured synchronously at IIFE invocation time
        if (_selfSrc && _selfSrc.indexOf('_static') !== -1) {
            return _selfSrc.substring(0, _selfSrc.indexOf('_static') + 7);
        }

        // 2 — scan <script> tags
        var scripts = document.querySelectorAll('script[src]');
        for (var i = 0; i < scripts.length; i++) {
            var src = scripts[i].getAttribute('src') || '';
            if (src.indexOf('_static') !== -1) {
                return src.substring(0, src.indexOf('_static') + 7);
            }
        }

        // 3 — scan <link> tags
        var links = document.querySelectorAll('link[href]');
        for (var j = 0; j < links.length; j++) {
            var href = links[j].getAttribute('href') || '';
            if (href.indexOf('_static') !== -1) {
                return href.substring(0, href.indexOf('_static') + 7);
            }
        }

        // 4 — relative fallback (correct for root-level pages only)
        return '_static';
    }

    // ── Widget placement ─────────────────────────────────────────────────────

    /**
     * Insert the container into the appropriate DOM location.
     *
     * Sidebar selector priority order:
     *   PyData Sphinx Theme (≥ 0.13):  .bd-sidebar-secondary
     *   PyData Sphinx Theme (older):   .bd-toc
     *   Furo / PST shared:             .sidebar-secondary
     *   Furo TOC section:              aside.toc-sidebar
     *   Furo drawer wrapper:           .toc-drawer
     *   Generic ARIA:                  aside[role="complementary"]
     *
     * @param {HTMLElement} container  The widget container.
     * @param {string}      position   'sidebar' | 'title' | 'floating' | 'none'.
     */
    function insertContainer(container, position) {
        if (position === 'none') return;

        if (position === 'sidebar') {
            var sidebarSelectors = [
                // 🚫 More pretty without this for "pydata_sphinx_theme"
                // '.bd-sidebar-secondary',           // PyData Sphinx Theme ≥ 0.13
                // '.bd-toc',                         // PyData Sphinx Theme < 0.13
                '.sidebar-secondary',              // Furo right TOC sidebar / PST alias
                'aside.toc-sidebar',               // Furo toc sidebar section
                '.toc-drawer',                     // Furo drawer wrapper
                'aside[role="complementary"]',     // Generic ARIA secondary landmark
            ];

            for (var k = 0; k < sidebarSelectors.length; k++) {
                var sidebar = document.querySelector(sidebarSelectors[k]);
                if (sidebar) {
                    console.debug('AI Assistant: Inserting into sidebar:', sidebarSelectors[k]);
                    sidebar.insertBefore(container, sidebar.firstChild);
                    return;
                }
            }

            // No sidebar found → fall back gracefully to title position
            console.debug('AI Assistant: No sidebar found, falling back to title position');
            insertInTitlePosition(container);
            return;
        }

        if (position === 'title') {
            insertInTitlePosition(container);
            return;
        }

        // 'floating' and any unrecognised value → top of article
        var article = document.querySelector('article, [role="main"], .document, .body');
        if (article) {
            console.debug('AI Assistant: Inserting at top of article (floating / fallback)');
            article.insertBefore(container, article.firstChild);
        }
    }

    /**
     * Insert the container next to the page's `<h1>` heading.
     *
     * The heading and container are wrapped in a flex row so they sit
     * side-by-side without modifying the heading's DOM subtree.
     *
     * Guard: `data-ai-assistant-wrapped` prevents double-wrapping on themes
     * that call `insertInTitlePosition` more than once (e.g. MutationObserver
     * re-runs on SPAs or Sphinx live-reload setups).
     *
     * @param {HTMLElement} container
     */
    function insertInTitlePosition(container) {
        var article = document.querySelector('article, [role="main"]');
        var heading = article ? article.querySelector('h1') : null;

        if (!heading) {
            if (article) article.insertBefore(container, article.firstChild);
            return;
        }

        // Guard against double-wrapping
        if (heading.parentNode && heading.parentNode.dataset &&
                heading.parentNode.dataset.aiAssistantWrapped) {
            heading.parentNode.appendChild(container);
            return;
        }

        console.debug('AI Assistant: Inserting next to page title (h1)');
        var wrapper = document.createElement('div');
        wrapper.className = 'ai-assistant-title-wrapper';
        wrapper.dataset.aiAssistantWrapped = '1';

        heading.parentNode.insertBefore(wrapper, heading);
        wrapper.appendChild(heading);
        wrapper.appendChild(container);

        container.style.flexShrink = '0';
    }

    // ── Event wiring ─────────────────────────────────────────────────────────

    /**
     * Attach all event listeners for the widget.
     *
     * The global document 'click' (dropdown-close) listener is registered
     * only once, guarded by `_listenersAttached`.
     *
     * @param {HTMLElement} button    The split-button wrapper.
     * @param {HTMLElement} dropdown  The dropdown panel.
     */
    function setupEventListeners(button, dropdown) {
        var mainButton    = document.getElementById('ai-assistant-button-main');
        var dropdownButton = document.getElementById('ai-assistant-button-dropdown');

        // Main button: immediate copy action (inline success state)
        if (mainButton) {
            mainButton.addEventListener('click', function (e) {
                e.stopPropagation();
                handleCopyMarkdown(true);
            });
        }

        // Chevron button: toggle dropdown
        if (dropdownButton) {
            dropdownButton.addEventListener('click', function (e) {
                e.stopPropagation();
                var isOpen = dropdown.style.display !== 'none';
                dropdown.style.display = isOpen ? 'none' : 'block';
                dropdownButton.setAttribute('aria-expanded', String(!isOpen));
            });
        }

        // Global click → close dropdown (registered once only)
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

        // Keyboard: Escape closes dropdown
        if (button) {
            button.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') closeDropdown();
            });
        }

        // ── Individual menu items ────────────────────────────────────────────

        var copyMarkdownBtn = document.getElementById('ai-assistant-copy-markdown');
        if (copyMarkdownBtn) {
            copyMarkdownBtn.addEventListener('click', function () {
                // FIX: pass false so the toast notification is shown (not inline
                // success state) — the inline state belongs to the main button only.
                handleCopyMarkdown(false);
            });
        }

        var viewMarkdownBtn = document.getElementById('ai-assistant-view-markdown');
        if (viewMarkdownBtn) {
            viewMarkdownBtn.addEventListener('click', function () {
                handleViewMarkdown();
            });
        }

        // AI chat items (all buttons whose id starts with 'ai-assistant-ai-chat-')
        var aiChatBtns = dropdown.querySelectorAll('[id^="ai-assistant-ai-chat-"]');
        aiChatBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                handleAIChat(this.dataset.provider);
            });
        });

        // MCP tool items
        var mcpBtns = dropdown.querySelectorAll('[id^="ai-assistant-mcp-"]');
        mcpBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                handleMCPInstall(this.dataset.mcpTool);
            });
        });

        // PDF export button
        var pdfExportBtn = document.getElementById('ai-assistant-pdf-export');
        if (pdfExportBtn) {
            pdfExportBtn.addEventListener('click', function () {
                handlePdfExport();
            });
        }

        // AI panel open button
        var aiPanelOpenBtn = document.getElementById('ai-assistant-ai-panel-open');
        if (aiPanelOpenBtn) {
            aiPanelOpenBtn.addEventListener('click', function () {
                closeDropdown();
                toggleAIPanel();
            });
        }
    }

    // ── Markdown conversion ──────────────────────────────────────────────────

    /**
     * Convert the main page content to Markdown via Turndown.
     *
     * Uses the `content_selector` from the widget config to locate the
     * content element. Elements in `elementsToRemove` are stripped from a
     * clone before conversion to keep the original DOM untouched.
     *
     * @returns {Promise<string>}  Resolves to the Markdown string.
     */
    function convertToMarkdown() {
        var contentSelector = (window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.content_selector) || 'article';
        var content = document.querySelector(contentSelector);

        if (!content) {
            return Promise.reject(new Error('Could not find page content (selector: ' + contentSelector + ')'));
        }

        var cloned = content.cloneNode(true);

        var toRemove = [
            '.headerlink',
            '.ai-assistant-container',
            'script',
            'style',
            '.sidebar',
            'nav',
        ];
        toRemove.forEach(function (sel) {
            cloned.querySelectorAll(sel).forEach(function (el) { el.remove(); });
        });

        var ts = new TurndownService({
            headingStyle: 'atx',
            codeBlockStyle: 'fenced',
            emDelimiter: '*',
        });

        ts.addRule('preserveCodeBlocks', {
            filter: ['pre'],
            replacement: function (content, node) {
                var code = node.querySelector('code');
                if (code) {
                    var langMatch = code.className.match(/language-(\w+)/);
                    var lang = langMatch ? langMatch[1] : '';
                    return '\n\n```' + lang + '\n' + code.textContent + '\n```\n\n';
                }
                return '\n\n```\n' + content + '\n```\n\n';
            },
        });

        return Promise.resolve(ts.turndown(cloned.innerHTML));
    }

    /**
     * Derive the companion `.md` URL for the current page.
     *
     * Strips hash fragments, then replaces `.html` suffix with `.md`.
     * Directory-style URLs get `/index.md` appended.
     *
     * @returns {string}
     */
    function getMarkdownUrl() {
        var clean = window.location.href.split('#')[0];
        if (clean.endsWith('.html')) return clean.replace(/\.html$/, '.md');
        if (clean.endsWith('/'))     return clean + 'index.md';
        return clean + '.md';
    }

    // ── Action handlers ──────────────────────────────────────────────────────

    /**
     * Copy the current page as Markdown to the clipboard.
     *
     * @param {boolean} showInlineConfirmation
     *   When `true`, the main button shows a brief ✓ checkmark state.
     *   When `false`, a toast notification is shown instead.
     */
    function handleCopyMarkdown(showInlineConfirmation) {
        convertToMarkdown()
            .then(function (markdown) {
                // FIX (was always `true`): honour the caller's intent.
                copyToClipboard(markdown, showInlineConfirmation);
                closeDropdown();
            })
            .catch(function (err) {
                console.error('AI Assistant: Failed to convert to Markdown:', err);
                showNotification('Failed to convert page to Markdown.', true);
            });
    }

    /** Open the companion `.md` file for the current page in a new tab. */
    function handleViewMarkdown() {
        var url = getMarkdownUrl();
        console.debug('AI Assistant: Opening Markdown URL:', url);
        window.open(url, '_blank', 'noopener,noreferrer');
        closeDropdown();
    }

    /**
     * Open an AI provider's chat URL pre-filled with this page's context.
     *
     * @param {string} providerKey  Key from `AI_ASSISTANT_CONFIG.providers`.
     */
    function handleAIChat(providerKey) {
        try {
            var providers = (window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.providers) || {};
            var provider  = providers[providerKey];

            if (!provider) {
                showNotification('AI provider "' + providerKey + '" not configured.', true);
                return;
            }

            // FIX-2: Guard both template strings before calling .replace().
            // A missing or non-string template would throw TypeError with no
            // actionable feedback.  Fall back to a sensible default so the
            // feature degrades gracefully rather than crashing.
            var promptTpl = typeof provider.prompt_template === 'string'
                ? provider.prompt_template
                : 'Read this documentation page: {url}';
            var urlTpl = typeof provider.url_template === 'string'
                ? provider.url_template
                : null;

            if (!urlTpl) {
                showNotification('AI provider "' + providerKey + '" has no url_template.', true);
                return;
            }

            var mdUrl  = getMarkdownUrl();
            var prompt = promptTpl.replace('{url}', mdUrl);
            var aiUrl  = urlTpl.replace('{prompt}', encodeURIComponent(prompt));

            console.debug('AI Assistant: Opening AI chat:', providerKey, aiUrl);
            window.open(aiUrl, '_blank', 'noopener,noreferrer');
            closeDropdown();
        } catch (err) {
            console.error('AI Assistant: Failed to open AI chat:', err);
            showNotification('Failed to open AI chat. Please try again.', true);
        }
    }

    /**
     * Install or open an MCP tool integration.
     *
     * Supported tool types: `claude_desktop`, `vscode`, and generic (unknown).
     *
     * @param {string} toolKey  Key from `AI_ASSISTANT_CONFIG.mcp_tools`.
     */
    function handleMCPInstall(toolKey) {
        try {
            var mcpTools = (window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.mcp_tools) || {};
            var tool     = mcpTools[toolKey];

            if (!tool) {
                showNotification('MCP tool configuration not found.', true);
                return;
            }

            if (tool.type === 'claude_desktop') {
                var mcpbUrl  = tool.mcpb_url;
                // FIX-5: new URL() throws on empty string / relative / invalid
                // values.  Validate explicitly with a useful user-facing message
                // rather than letting the generic catch show a vague toast.
                var urlPath;
                try {
                    urlPath = new URL(mcpbUrl).pathname;
                } catch (_urlErr) {
                    showNotification(
                        'MCP tool "' + toolKey + '" has an invalid mcpb_url.',
                        true
                    );
                    return;
                }
                var filename = urlPath.split('/').pop() || (toolKey + '.zip');
                var a = document.createElement('a');
                a.href = mcpbUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                showNotification('MCP tool download started.');
                closeDropdown();
                return;
            }

            if (tool.type === 'vscode') {
                var mcpCfg = {
                    name: tool.server_name || toolKey,
                    type: tool.transport || 'sse',
                };
                if (tool.transport === 'stdio') {
                    mcpCfg.command = tool.command;
                    if (tool.args) mcpCfg.args = tool.args;
                } else {
                    mcpCfg.url = tool.server_url;
                }
                var installUrl = 'vscode:mcp/install?' + encodeURIComponent(JSON.stringify(mcpCfg));
                window.open(installUrl, '_self');
                closeDropdown();
                return;
            }

            // Unknown type — show a generic error
            console.warn('AI Assistant: Unknown MCP tool type:', tool.type);
            showNotification('Unknown MCP tool type: ' + tool.type, true);

        } catch (err) {
            console.error('AI Assistant: Failed to install MCP tool:', err);
            showNotification('Failed to install MCP tool. Please try again.', true);
        }
    }

    /**
     * Handle the "Export as PDF" button click.
     *
     * Reads the effective PDF mode from the toggle (persisted in sessionStorage)
     * and either:
     *   - URL mode  → opens `AI_ASSISTANT_CONFIG.pdfExportUrl` in a new tab.
     *   - Print mode → calls `window.print()` (browser → Save as PDF).
     *
     * Edge cases:
     *   - URL mode selected but `pdfExportUrl` is empty → falls back to print.
     *   - `window.open` blocked by popup blocker → silent (blocker notifies user).
     */
    function handlePdfExport() {
        var cfg    = window.AI_ASSISTANT_CONFIG || {};
        var pdfUrl = (cfg.pdfExportUrl || '').trim();
        var mode   = _getPdfMode();

        closeDropdown();

        if (mode === 'url' && pdfUrl) {
            console.debug('AI Assistant: PDF URL mode → opening:', pdfUrl);
            window.open(pdfUrl, '_blank', 'noopener,noreferrer');
        } else {
            console.debug('AI Assistant: PDF print mode → window.print()');
            window.print();
        }
    }

    // ── Clipboard ────────────────────────────────────────────────────────────

    /**
     * Copy *text* to the clipboard using the modern Clipboard API, with a
     * textarea fallback for older browsers.
     *
     * @param {string}  text                    Text to copy.
     * @param {boolean} showInlineConfirmation  When true, flash the main button.
     */
    function copyToClipboard(text, showInlineConfirmation) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function () {
                if (showInlineConfirmation) {
                    showInlineSuccessState();
                } else {
                    showNotification('Markdown copied to clipboard!');
                }
            }).catch(function (err) {
                console.error('AI Assistant: Clipboard API failed:', err);
                fallbackCopy(text, showInlineConfirmation);
            });
        } else {
            fallbackCopy(text, showInlineConfirmation);
        }
    }

    /**
     * Flash the main button with a checkmark for 2 seconds to confirm copy.
     *
     * Developer note (FIX-3):
     *   The original implementation rebuilt `mainButton.innerHTML` from a
     *   string, which:
     *     (a) violates the "no innerHTML with raw user data" policy even though
     *         staticPath is trusted — it's an inconsistent pattern.
     *     (b) destroys and recreates child DOM nodes on every call.
     *     (c) re-injects the saved innerHTML string back, which re-parses HTML
     *         unnecessarily and loses any future child event listeners.
     *
     *   The fix mutates only the two changing properties (.src and .textContent)
     *   on the existing child nodes, then restores them after 2 s.  This is
     *   correct, safe, and has zero DOM thrash.
     */
    function showInlineSuccessState() {
        var mainButton = document.getElementById('ai-assistant-button-main');
        if (!mainButton) return;

        // Locate existing children — avoid any DOM reconstruction.
        var iconEl   = mainButton.querySelector('.ai-assistant-icon');
        var textSpan = mainButton.querySelector('.ai-assistant-button-text');
        if (!iconEl || !textSpan) return;

        // Snapshot originals for restoration.
        var origSrc  = iconEl.src;
        var origText = textSpan.textContent;

        iconEl.src           = getStaticPath() + '/checked.svg';
        textSpan.textContent = 'Copied';
        mainButton.classList.add('ai-assistant-button-success');

        setTimeout(function () {
            iconEl.src           = origSrc;
            textSpan.textContent = origText;
            mainButton.classList.remove('ai-assistant-button-success');
        }, 2000);
    }

    /**
     * Textarea-based clipboard fallback for browsers without the Clipboard API.
     *
     * @param {string}  text
     * @param {boolean} showInlineConfirmation
     */
    function fallbackCopy(text, showInlineConfirmation) {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity  = '0';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            document.execCommand('copy');  // deprecated but still widely supported
            if (showInlineConfirmation) {
                showInlineSuccessState();
            } else {
                showNotification('Markdown copied to clipboard!');
            }
        } catch (err) {
            console.error('AI Assistant: Fallback copy failed:', err);
            showNotification('Failed to copy to clipboard.', true);
        }

        document.body.removeChild(textarea);
    }

    // ── Utility ──────────────────────────────────────────────────────────────

    /** Close the dropdown and reset the chevron aria-expanded state. */
    function closeDropdown() {
        var dropdown   = document.getElementById('ai-assistant-dropdown');
        var dropButton = document.getElementById('ai-assistant-button-dropdown');
        if (dropdown)   dropdown.style.display = 'none';
        if (dropButton) dropButton.setAttribute('aria-expanded', 'false');
    }

    /**
     * Show a brief toast notification in the viewport's top-right corner.
     *
     * @param {string}  message  Text to display.
     * @param {boolean} [isError=false]  When true the toast uses an error colour.
     */
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

    /**
     * Escape HTML special characters for safe inline insertion.
     *
     * Applied to user-supplied config strings (panelTitle, panelPlaceholder)
     * before they are embedded in panel markup.
     *
     * @param {string} str
     * @returns {string}
     */
    function _escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str
            .replace(/&/g,  '&amp;')
            .replace(/</g,  '&lt;')
            .replace(/>/g,  '&gt;')
            .replace(/"/g,  '&quot;')
            .replace(/'/g,  '&#039;');
    }

    // ── AI Panel ─────────────────────────────────────────────────────────────

    /**
     * Create the floating AI assistant panel DOM element (lazy singleton).
     *
     * The panel is a slide-in drawer anchored to the bottom-right viewport
     * edge. It uses the same three-layer CSS variable chain as the rest of the
     * widget (PST → Furo → hardcoded fallback).
     *
     * Accessibility:
     *   - `role="dialog"` + `aria-modal="true"`
     *   - Close button has a keyboard-visible focus ring
     *   - Enter (without Shift) submits; Shift+Enter inserts newline
     *   - Escape closes the panel
     *
     * @returns {HTMLElement}
     */
    function createAIPanel() {
        var cfg         = window.AI_ASSISTANT_CONFIG || {};
        var title       = cfg.panelTitle || 'AI Assistant';
        var placeholder = cfg.panelPlaceholder || 'Ask a question about this page\u2026';
        var staticPath  = getStaticPath();

        var panel = document.createElement('div');
        panel.id = 'ai-assistant-panel';
        panel.className = 'ai-assistant-panel';
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-label', title);
        panel.style.display = 'none';

        // Header
        var header = document.createElement('div');
        header.className = 'ai-assistant-panel-header';

        var headerTitle = document.createElement('div');
        headerTitle.className = 'ai-assistant-panel-header-title';

        var logo = document.createElement('img');
        logo.src = staticPath + '/ai-panel.svg';
        logo.className = 'ai-assistant-panel-logo';
        logo.setAttribute('aria-hidden', 'true');
        logo.alt = '';

        var titleSpan = document.createElement('span');
        titleSpan.textContent = title;

        headerTitle.appendChild(logo);
        headerTitle.appendChild(titleSpan);

        var closeBtn = document.createElement('button');
        closeBtn.className = 'ai-assistant-panel-close';
        closeBtn.id = 'ai-assistant-panel-close';
        closeBtn.type = 'button';
        closeBtn.setAttribute('aria-label', 'Close ' + _escapeHtml(title));
        closeBtn.textContent = '\u2715';

        header.appendChild(headerTitle);
        header.appendChild(closeBtn);

        // Body
        var body = document.createElement('div');
        body.className = 'ai-assistant-panel-body';
        body.id = 'ai-assistant-panel-body';

        var welcome = document.createElement('div');
        welcome.className = 'ai-assistant-panel-welcome';
        // FIX-4: Build the welcome message with pure DOM construction.
        // The previous implementation used innerHTML with _escapeHtml(title),
        // which was safe but inconsistent with the rest of the codebase and
        // incompatible with strict Content-Security-Policy configurations that
        // disallow 'unsafe-inline'.  textContent auto-escapes all special chars.
        var p1     = document.createElement('p');
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

        // Footer
        var footer = document.createElement('div');
        footer.className = 'ai-assistant-panel-footer';

        var input = document.createElement('textarea');
        input.id = 'ai-assistant-panel-input';
        input.className = 'ai-assistant-panel-input';
        input.rows = 2;
        input.placeholder = placeholder;
        input.setAttribute('aria-label', 'Your question');

        var sendBtn = document.createElement('button');
        sendBtn.className = 'ai-assistant-panel-send';
        sendBtn.id = 'ai-assistant-panel-send';
        sendBtn.type = 'button';
        sendBtn.setAttribute('aria-label', 'Send question');
        sendBtn.textContent = 'Send';

        footer.appendChild(input);
        footer.appendChild(sendBtn);

        panel.appendChild(header);
        panel.appendChild(body);
        panel.appendChild(footer);

        // Events
        closeBtn.addEventListener('click', closeAIPanel);
        sendBtn.addEventListener('click', handleAIPanelSubmit);
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleAIPanelSubmit();
            }
        });
        panel.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') closeAIPanel();
        });

        document.body.appendChild(panel);
        return panel;
    }

    /** Toggle the AI panel open/closed. Creates it on first call. */
    function toggleAIPanel() {
        if (!_aiPanelEl) _aiPanelEl = createAIPanel();

        var isVisible = _aiPanelEl.style.display !== 'none';
        if (isVisible) {
            closeAIPanel();
        } else {
            _aiPanelEl.style.display = 'flex';
            requestAnimationFrame(function () {
                _aiPanelEl.classList.add('ai-assistant-panel--open');
            });
            var inp = document.getElementById('ai-assistant-panel-input');
            if (inp) setTimeout(function () { inp.focus(); }, 100);
        }
    }

    /** Close the AI panel with a slide-out animation. */
    function closeAIPanel() {
        if (!_aiPanelEl) return;
        _aiPanelEl.classList.remove('ai-assistant-panel--open');
        setTimeout(function () {
            if (_aiPanelEl) _aiPanelEl.style.display = 'none';
        }, 300);
        var dropBtn = document.getElementById('ai-assistant-button-dropdown');
        if (dropBtn) dropBtn.focus();
    }

    /**
     * Append a message bubble to the panel body.
     *
     * @param {string} text  Message text (plain, set via textContent).
     * @param {'user'|'assistant'|'error'} role
     */
    function _appendPanelMessage(text, role) {
        var body = document.getElementById('ai-assistant-panel-body');
        if (!body) return;

        var welcome = body.querySelector('.ai-assistant-panel-welcome');
        if (welcome) welcome.remove();

        var bubble = document.createElement('div');
        bubble.className = 'ai-assistant-panel-bubble ai-assistant-panel-bubble--' + role;
        bubble.textContent = text;
        body.appendChild(bubble);
        body.scrollTop = body.scrollHeight;
    }

    /**
     * Read the panel input, display the user message, then call the API
     * (when `panelApiEnabled` is true) or show a stub response.
     *
     * Edge cases:
     *   - Empty input → no-op.
     *   - Input > 4 000 chars → truncated with a warning suffix.
     *   - Network failure → error bubble.
     */
    async function handleAIPanelSubmit() {
        var input   = document.getElementById('ai-assistant-panel-input');
        var sendBtn = document.getElementById('ai-assistant-panel-send');
        if (!input) return;

        var rawText = input.value.trim();
        if (!rawText) return;

        var MAX_CHARS    = 4000;
        var questionText = rawText.length > MAX_CHARS
            ? rawText.slice(0, MAX_CHARS) + '\u2026 [truncated]'
            : rawText;

        _appendPanelMessage(questionText, 'user');
        input.value = '';
        input.disabled = true;
        if (sendBtn) sendBtn.disabled = true;

        var cfg = window.AI_ASSISTANT_CONFIG || {};
        try {
            if (cfg.panelApiEnabled) {
                await _panelApiCall(questionText, cfg);
            } else {
                await _panelStubReply(questionText);
            }
        } catch (err) {
            console.error('AI Assistant panel error:', err);
            _appendPanelMessage('Sorry, something went wrong: ' + err.message, 'error');
        } finally {
            input.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            input.focus();
        }
    }

    /**
     * Live Anthropic API call for the AI panel.
     *
     * POSTs to the Anthropic `/v1/messages` endpoint. The API key must be
     * handled by the deployment environment (reverse proxy / CORS header).
     *
     * @param {string} question  User's question text.
     * @param {Object} cfg       `window.AI_ASSISTANT_CONFIG`.
     */
    async function _panelApiCall(question, cfg) {
        var pageMarkdown = '';
        try {
            pageMarkdown = await convertToMarkdown();
        } catch (_) { /* non-fatal: continue without page context */ }

        var systemPrompt = pageMarkdown
            ? 'You are a helpful documentation assistant. Answer questions about the following documentation page.\n\n---\n' + pageMarkdown.slice(0, 8000) + '\n---'
            : 'You are a helpful documentation assistant.';

        var response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model:      'claude-sonnet-4-20250514',
                max_tokens: 1000,
                system:     systemPrompt,
                messages:   [{ role: 'user', content: question }],
            }),
        });

        if (!response.ok) {
            var errBody = await response.text().catch(function () { return ''; });
            throw new Error('API ' + response.status + ': ' + errBody.slice(0, 120));
        }

        var data  = await response.json();
        var reply = (data.content || [])
            .filter(function (b) { return b.type === 'text'; })
            .map(function (b) { return b.text; })
            .join('\n')
            .trim();

        _appendPanelMessage(reply || '(no response)', 'assistant');
    }

    /**
     * Stub reply for when `panelApiEnabled` is `false`.
     *
     * Simulates a 400 ms delay so the loading state is visible during demos.
     */
    async function _panelStubReply(_question) {
        await new Promise(function (resolve) { setTimeout(resolve, 400); });
        _appendPanelMessage(
            'This AI assistant panel is running in stub mode. ' +
            'Set ai_assistant_panel_api_enabled = True in conf.py ' +
            'to enable live responses.',
            'assistant'
        );
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAIAssistant);
    } else {
        initAIAssistant();
    }

})();
