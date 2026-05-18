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
     * Feature flag defaults — used as base in Object.assign so absent keys
     * from config resolve to sensible values rather than undefined.
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
     * Regex: matches any icon value that is already an absolute URI — must
     * NOT be prefixed with the _static path.
     */
    var _ABSOLUTE_ICON_RE = /^(?:https?:|data:|blob:|\\/)/;

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
    };

    // ── Initialisation ────────────────────────────────────────────────────────

    function loadTurndown(callback) {
        if (typeof TurndownService !== 'undefined') { callback(); return; }
        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/turndown@7.1.2/dist/turndown.min.js';
        script.onload  = callback;
        script.onerror = function () { console.error('AI Assistant: Failed to load Turndown'); };
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
        dropdown.style.display = 'none';

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

        if (heading.parentNode && heading.parentNode.dataset && heading.parentNode.dataset.aiAssistantWrapped) {
            heading.parentNode.appendChild(container);
            return;
        }

        var wrapper = document.createElement('div');
        wrapper.className = 'ai-assistant-title-wrapper';
        wrapper.dataset.aiAssistantWrapped = '1';
        heading.parentNode.insertBefore(wrapper, heading);
        wrapper.appendChild(heading);
        wrapper.appendChild(container);
        container.style.flexShrink = '0';
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
                dropdown.style.display = isOpen ? 'none' : 'block';
                dropdownButton.setAttribute('aria-expanded', String(!isOpen));
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
        var clean = window.location.href.split('#')[0];
        if (clean.endsWith('.html')) return clean.replace(/\.html$/, '.md');
        if (clean.endsWith('/'))     return clean + 'index.md';
        return clean + '.md';
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
                var urlPath;
                try { urlPath = new URL(tool.mcpb_url).pathname; }
                catch (_urlErr) { showNotification('MCP tool "' + toolKey + '" has an invalid mcpb_url.', true); return; }
                var a = document.createElement('a');
                a.href = tool.mcpb_url;
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
                window.open('vscode:mcp/install?' + encodeURIComponent(JSON.stringify(mcpCfg)), '_self');
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
        var quickQs     = Array.isArray(cfg.panelQuickQuestions) ? cfg.panelQuickQuestions.slice(0, 5) : [];
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

        headerActions.appendChild(minimizeBtn);
        headerActions.appendChild(maximizeBtn);
        headerActions.appendChild(closeBtn);

        header.appendChild(headerTitle);
        header.appendChild(headerActions);

        // ── Body ─────────────────────────────────────────────────────────────
        var body = document.createElement('div');
        body.className = 'ai-assistant-panel-body';
        body.id = 'ai-assistant-panel-body';

        // Welcome message
        var welcome = document.createElement('div');
        welcome.className = 'ai-assistant-panel-welcome';

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

        // Quick suggestion chips
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
                    if (input) {
                        input.value = q;
                        _updateSendBtnState();
                        input.focus();
                    }
                });
                suggestionsEl.appendChild(chip);
            });

            body.appendChild(suggestionsEl);
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

        // Unified input group: [textarea] [mic?] [send]
        var inputGroup = document.createElement('div');
        inputGroup.className = 'ai-assistant-panel-input-group';

        var input = document.createElement('textarea');
        input.id = 'ai-assistant-panel-input';
        input.className = 'ai-assistant-panel-input';
        input.rows = 2;
        input.placeholder = placeholder;
        input.setAttribute('aria-label', 'Your question');

        inputGroup.appendChild(input);

        // Microphone button (shown only when speech is supported)
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
            inputGroup.appendChild(micBtnEl);
        }

        // Send icon button
        var sendBtn = document.createElement('button');
        sendBtn.className = 'ai-assistant-panel-footer-btn ai-assistant-panel-footer-btn--send';
        sendBtn.id = 'ai-assistant-panel-send';
        sendBtn.type = 'button';
        sendBtn.setAttribute('aria-label', 'Send question');
        sendBtn.setAttribute('title', 'Send (Enter)');
        sendBtn.innerHTML = ICONS.send;   // ICONS constant — safe.

        inputGroup.appendChild(sendBtn);
        footer.appendChild(inputGroup);

        // ── Assemble panel ────────────────────────────────────────────────────
        panel.appendChild(header);
        panel.appendChild(body);

        // Speak banner goes between body and footer, attached to the panel
        // so it is always visible above the input regardless of scroll.
        if (speakBannerEl) panel.appendChild(speakBannerEl);

        panel.appendChild(footer);

        // ── Events ────────────────────────────────────────────────────────────

        closeBtn.addEventListener('click', closeAIPanel);

        minimizeBtn.addEventListener('click', function () { minimizeAIPanel(); });

        maximizeBtn.addEventListener('click', function () {
            var isMax = panel.getAttribute('data-maximized') === 'true';
            if (isMax) {
                panel.removeAttribute('data-maximized');
                maximizeBtn.setAttribute('aria-label', 'Maximize panel');
                maximizeBtn.innerHTML = ICONS.maximize;
            } else {
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
            if (e.key === 'Escape') closeAIPanel();
        });

        document.body.appendChild(panel);

        // ── Floating trigger pill (shown when minimized) ──────────────────────
        _aiTriggerEl = _createTriggerPill(title);
        document.body.appendChild(_aiTriggerEl);

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
        label.textContent = 'Ask Us';

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
        if (inp) setTimeout(function () { inp.focus(); }, 100);
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
     * Update the send icon button's accent state based on textarea content.
     * The send icon gets a brand-colour accent when there is text to submit.
     */
    function _updateSendBtnState() {
        var input   = document.getElementById('ai-assistant-panel-input');
        var sendBtn = document.getElementById('ai-assistant-panel-send');
        if (!sendBtn || !input) return;
        var hasText = input.value.trim().length > 0;
        sendBtn.classList.toggle('has-text', hasText);
    }

    // ── Message bubbles ───────────────────────────────────────────────────────

    function _appendPanelMessage(text, role) {
        var body = document.getElementById('ai-assistant-panel-body');
        if (!body) return;

        // Remove welcome + suggestions on first real message
        var welcome = body.querySelector('.ai-assistant-panel-welcome');
        if (welcome) welcome.remove();
        var suggestions = body.querySelector('.ai-assistant-panel-suggestions');
        if (suggestions) suggestions.remove();

        var bubble = document.createElement('div');
        bubble.className = 'ai-assistant-panel-bubble ai-assistant-panel-bubble--' + role;
        bubble.textContent = text;
        body.appendChild(bubble);
        body.scrollTop = body.scrollHeight;
    }

    // ── Panel submit ──────────────────────────────────────────────────────────

    async function handleAIPanelSubmit() {
        var input   = document.getElementById('ai-assistant-panel-input');
        var sendBtn = document.getElementById('ai-assistant-panel-send');
        if (!input) return;

        var rawText = input.value.trim();
        if (!rawText) return;

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
            _updateSendBtnState();
            input.focus();
        }
    }

    // ── API + stub ────────────────────────────────────────────────────────────

    async function _panelApiCall(question, cfg) {
        var pageMarkdown = '';
        try { pageMarkdown = await convertToMarkdown(); } catch (_) {}

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

    async function _panelStubReply(_question) {
        await new Promise(function (resolve) { setTimeout(resolve, 400); });
        _appendPanelMessage(
            'This AI assistant panel is running in stub mode. ' +
            'Set ai_assistant_panel_api_enabled = True in conf.py to enable live responses.',
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
