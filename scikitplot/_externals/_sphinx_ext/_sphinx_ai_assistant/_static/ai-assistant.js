/**
 * Sphinx AI Assistant
 *
 * Provides AI-powered features for Sphinx documentation pages.
 * - Markdown export functionality
 * - AI chat integration with pre-filled context
 * - MCP tool integration
 */

(function() {
    'use strict';

    // Load Turndown library from CDN
    function loadTurndown(callback) {
        if (typeof TurndownService !== 'undefined') {
            callback();
            return;
        }

        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/turndown@7.1.2/dist/turndown.min.js';
        script.onload = callback;
        script.onerror = function() {
            console.error('Failed to load Turndown library');
        };
        document.head.appendChild(script);
    }

    // Initialize the AI assistant when DOM is ready
    function initAIAssistant() {
        loadTurndown(function() {
            createAIAssistantUI();
        });
    }

    // Create the AI assistant UI
    function createAIAssistantUI() {
        const container = createContainer();
        const button = createButton();
        const dropdown = createDropdown();

        container.appendChild(button);
        container.appendChild(dropdown);

        // Insert into the appropriate location based on configuration
        const position = window.AI_ASSISTANT_CONFIG?.position || 'sidebar';
        insertContainer(container, position);

        // Setup event listeners
        setupEventListeners(button, dropdown);
    }

    // Create the main container
    function createContainer() {
        const container = document.createElement('div');
        container.className = 'ai-assistant-container';
        container.id = 'ai-assistant-container';
        return container;
    }

    // Create the trigger button (actually a container with two buttons)
    function createButton() {
        const container = document.createElement('div');
        container.className = 'ai-assistant-button';
        container.id = 'ai-assistant-button';

        // Get the static path from the page's _static directory
        const staticPath = getStaticPath();

        container.innerHTML = `
            <button class="ai-assistant-button-main" id="ai-assistant-button-main" type="button">
                <img src="${staticPath}/copy-to-clipboard.svg" class="ai-assistant-icon" aria-hidden="true" alt="">
                <span class="ai-assistant-button-text">Copy page</span>
            </button>
            <span class="ai-assistant-button-divider"></span>
            <button class="ai-assistant-button-dropdown" id="ai-assistant-button-dropdown" type="button" aria-label="More options" aria-expanded="false" aria-haspopup="true">
                <img src="${staticPath}/arrow-down.svg" class="ai-assistant-dropdown-icon" aria-hidden="true" alt="">
            </button>
        `;

        return container;
    }

    // Get the path to _static directory
    function getStaticPath() {
        // Try to find the _static path from existing script tags
        const scripts = document.querySelectorAll('script[src*="_static"]');
        if (scripts.length > 0) {
            const src = scripts[0].getAttribute('src');
            const staticPath = src.substring(0, src.indexOf('_static') + 7);
            return staticPath;
        }
        // Fallback to common path
        return '_static';
    }

    // Create the dropdown menu
    function createDropdown() {
        const dropdown = document.createElement('div');
        dropdown.className = 'ai-assistant-dropdown';
        dropdown.id = 'ai-assistant-dropdown';
        dropdown.setAttribute('role', 'menu');
        dropdown.style.display = 'none';

        const features = window.AI_ASSISTANT_CONFIG?.features || { markdown_export: true };
        const staticPath = getStaticPath();

        // Markdown export
        if (features.markdown_export) {
            const exportItem = createMenuItem(
                'copy-markdown',
                'Copy page',
                'Copy this page as Markdown for LLMs.',
                `${staticPath}/copy-to-clipboard.svg`
            );
            dropdown.appendChild(exportItem);
        }

        // View as Markdown
        if (features.view_markdown) {
            const viewItem = createMenuItem(
                'view-markdown',
                'View as Markdown',
                'View this page as Markdown.',
                `${staticPath}/markdown.svg`
            );
            dropdown.appendChild(viewItem);
        }

        // AI chat integration
        if (features.ai_chat) {
            const providers = window.AI_ASSISTANT_CONFIG?.providers || {};

            // Add a separator, if we have markdown or view features
            if (features.markdown_export || features.view_markdown) {
                const separator = document.createElement('div');
                separator.className = 'ai-assistant-menu-separator';
                dropdown.appendChild(separator);
            }

            // Add AI provider menu items
            Object.entries(providers).forEach(([key, provider]) => {
                if (provider.enabled) {
                    const description = provider.description || 'Open AI chat with this page context.';
                    const icon = provider.icon || 'comment-discussion.svg';
                    const iconPath = icon.startsWith('http') ? icon : `${staticPath}/${icon}`;

                    const aiItem = createMenuItem(
                        `ai-chat-${key}`,
                        provider.label,
                        description,
                        iconPath
                    );
                    aiItem.dataset.provider = key;
                    dropdown.appendChild(aiItem);
                }
            });
        }

        // MCP integration
        if (features.mcp_integration) {
            const mcpTools = window.AI_ASSISTANT_CONFIG?.mcp_tools || {};

            // Add separator if we have AI chat or markdown features
            if (features.ai_chat || features.markdown_export || features.view_markdown) {
                const separator = document.createElement('div');
                separator.className = 'ai-assistant-menu-separator';
                dropdown.appendChild(separator);
            }

            // Add MCP tool menu items
            Object.entries(mcpTools).forEach(([key, tool]) => {
                if (tool.enabled) {
                    const description = tool.description || 'Install MCP server';
                    const icon = tool.icon || 'ai-tools.svg';
                    const iconPath = icon.startsWith('http') ? icon : `${staticPath}/${icon}`;

                    const mcpItem = createMenuItem(
                        `mcp-${key}`,
                        tool.label,
                        description,
                        iconPath
                    );
                    mcpItem.dataset.mcpTool = key;
                    dropdown.appendChild(mcpItem);
                }
            });
        }

        // ---- PDF Export section -------------------------------------------
        // Rendered last; always gets its own separator so it is visually
        // distinct from the MCP tools section (or the AI chat section when MCP
        // integration is disabled).
        if (features.pdf_export) {
            const pdfSep = document.createElement('div');
            pdfSep.className = 'ai-assistant-menu-separator';
            dropdown.appendChild(pdfSep);

            const pdfItem = createMenuItem(
                'pdf-export',
                'Export as PDF',
                'Save this page as a PDF file.',
                `${staticPath}/file-pdf.svg`
            );
            dropdown.appendChild(pdfItem);
        }

        // ---- AI Panel section (floating chat stub / API) ------------------
        // Appears as the very last dropdown entry so it is never confused with
        // the provider-level "Ask Claude / Ask ChatGPT" links above.
        if (features.ai_panel) {
            const panelSep = document.createElement('div');
            panelSep.className = 'ai-assistant-menu-separator';
            dropdown.appendChild(panelSep);

            const panelTitle = window.AI_ASSISTANT_CONFIG?.panelTitle || 'AI Assistant';
            const panelItem = createMenuItem(
                'ai-panel-open',
                panelTitle,
                `Ask ${panelTitle} about this page`,
                `${staticPath}/ai-panel.svg`
            );
            dropdown.appendChild(panelItem);
        }

        return dropdown;
    }

    // Create a menu item
    function createMenuItem(id, text, description, iconSrc) {
        const item = document.createElement('button');
        item.className = 'ai-assistant-menu-item';
        item.id = `ai-assistant-${id}`;
        item.setAttribute('role', 'menuitem');

        item.innerHTML = `
            <div class="ai-assistant-menu-item-content">
                <div class="ai-assistant-menu-item-title">
                    <img src="${iconSrc}" class="ai-assistant-menu-icon" aria-hidden="true" alt="">
                    <span>${text}</span>
                </div>
                <div class="ai-assistant-menu-item-description">${description}</div>
            </div>
        `;

        return item;
    }

    // Insert container into the appropriate location
    function insertContainer(container, position) {
        if (position === 'sidebar') {
            // For Furo theme, target the right sidebar (page TOC)
            // Try multiple selectors in order of preference
            const selectors = [
                '.toc-drawer',                    // Furo's right sidebar drawer
                'aside.toc-sidebar',              // Furo's right sidebar
                '.sidebar-secondary',             // Generic right sidebar
                'aside[role="complementary"]',    // ARIA role for secondary sidebar
            ];

            for (const selector of selectors) {
                const sidebar = document.querySelector(selector);
                if (sidebar) {
                    console.log('AI Assistant: Inserting into sidebar:', selector);
                    sidebar.insertBefore(container, sidebar.firstChild);
                    return;
                }
            }

            console.log('AI Assistant: No sidebar found, falling back to title position');
            // If no sidebar found, fall back to title position
            insertInTitlePosition(container);
            return;
        }

        if (position === 'title') {
            insertInTitlePosition(container);
            return;
        }

        // Final fallback: insert at the top of the article container
        const article = document.querySelector('article, .document, .body');
        if (article) {
            console.log('AI Assistant: Using fallback position at top of article');
            article.insertBefore(container, article.firstChild);
        }
    }

    // Helper function to insert in title position
    function insertInTitlePosition(container) {
        const article = document.querySelector('article');
        const heading = article?.querySelector('h1');

        if (heading) {
            console.log('AI Assistant: Inserting next to title');

            // Create a wrapper to position button on the same line as title
            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.alignItems = 'flex-start';
            wrapper.style.justifyContent = 'space-between';
            wrapper.style.gap = '1rem';
            wrapper.style.marginBottom = '1rem';

            // Move the heading into the wrapper
            heading.parentNode.insertBefore(wrapper, heading);
            wrapper.appendChild(heading);
            wrapper.appendChild(container);

            // Adjust container styles for title position
            container.style.flexShrink = '0';
        } else {
            // If no heading found, insert at top of article
            if (article) {
                article.insertBefore(container, article.firstChild);
            }
        }
    }

    // Setup event listeners
    function setupEventListeners(button, dropdown) {
        const mainButton = document.getElementById('ai-assistant-button-main');
        const dropdownButton = document.getElementById('ai-assistant-button-dropdown');

        // Main button - direct copy action
        mainButton.addEventListener('click', function(e) {
            e.stopPropagation();
            handleCopyMarkdown(true); // Pass true to show inline confirmation
        });

        // Dropdown button - toggle menu
        dropdownButton.addEventListener('click', function(e) {
            e.stopPropagation();
            const isOpen = dropdown.style.display !== 'none';
            dropdown.style.display = isOpen ? 'none' : 'block';
            dropdownButton.setAttribute('aria-expanded', !isOpen);
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!button.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.style.display = 'none';
                dropdownButton.setAttribute('aria-expanded', 'false');
            }
        });

        // Handle menu item clicks
        const copyMarkdownBtn = document.getElementById('ai-assistant-copy-markdown');
        if (copyMarkdownBtn) {
            copyMarkdownBtn.addEventListener('click', function() {
                handleCopyMarkdown(false); // Regular notification
            });
        }

        // Handle view markdown button
        const viewMarkdownBtn = document.getElementById('ai-assistant-view-markdown');
        if (viewMarkdownBtn) {
            viewMarkdownBtn.addEventListener('click', function() {
                handleViewMarkdown();
            });
        }

        // Handle AI chat menu items
        const aiChatButtons = dropdown.querySelectorAll('[id^="ai-assistant-ai-chat-"]');
        aiChatButtons.forEach(button => {
            button.addEventListener('click', function() {
                const provider = this.dataset.provider;
                handleAIChat(provider);
            });
        });

        // Handle MCP tool menu items
        const mcpButtons = dropdown.querySelectorAll('[id^="ai-assistant-mcp-"]');
        mcpButtons.forEach(button => {
            button.addEventListener('click', function() {
                const toolKey = this.dataset.mcpTool;
                handleMCPInstall(toolKey);
            });
        });

        // Handle PDF export button
        const pdfExportBtn = document.getElementById('ai-assistant-pdf-export');
        if (pdfExportBtn) {
            pdfExportBtn.addEventListener('click', function() {
                handlePdfExport();
            });
        }

        // Handle AI panel open button
        const aiPanelOpenBtn = document.getElementById('ai-assistant-ai-panel-open');
        if (aiPanelOpenBtn) {
            aiPanelOpenBtn.addEventListener('click', function() {
                closeDropdown();
                toggleAIPanel();
            });
        }
    }

    // Convert HTML to Markdown
    async function convertToMarkdown() {
        console.log('AI Assistant: Starting markdown conversion');

        const contentSelector = window.AI_ASSISTANT_CONFIG?.content_selector || 'article';
        console.log('AI Assistant: Looking for content with selector:', contentSelector);

        const content = document.querySelector(contentSelector);

        if (!content) {
            console.error('AI Assistant: Could not find content element');
            throw new Error('Could not find page content to convert');
        }

        console.log('AI Assistant: Found content element:', content);

        // Clone the content to avoid modifying the original
        const clonedContent = content.cloneNode(true);

        // Remove elements we don't want in the markdown
        const elementsToRemove = [
            '.headerlink',
            '.ai-assistant-container',
            'script',
            'style',
            '.sidebar',
            'nav'
        ];

        elementsToRemove.forEach(selector => {
            clonedContent.querySelectorAll(selector).forEach(el => el.remove());
        });

        console.log('AI Assistant: Cleaned content, starting Turndown conversion');

        // Convert to markdown using Turndown
        const turndownService = new TurndownService({
            headingStyle: 'atx',
            codeBlockStyle: 'fenced',
            emDelimiter: '*',
        });

        // Add custom rules for code blocks
        turndownService.addRule('preserveCodeBlocks', {
            filter: ['pre'],
            replacement: function(content, node) {
                const code = node.querySelector('code');
                if (code) {
                    const language = code.className.match(/language-(\w+)/);
                    const lang = language ? language[1] : '';
                    return '\n\n```' + lang + '\n' + code.textContent + '\n```\n\n';
                }
                return '\n\n```\n' + content + '\n```\n\n';
            }
        });

        const markdown = turndownService.turndown(clonedContent.innerHTML);

        console.log('AI Assistant: Markdown generated, length:', markdown.length);

        return markdown;
    }

    // Get markdown URL for the current page
    function getMarkdownUrl() {
        const currentUrl = window.location.href;

        // Remove anchor/hash from URL first
        const urlWithoutHash = currentUrl.split('#')[0];

        // Replace .html with .md in the current URL
        if (urlWithoutHash.endsWith('.html')) {
            return urlWithoutHash.replace(/\.html$/, '.md');
        } else if (urlWithoutHash.endsWith('/')) {
            // For directory-style URLs, look for index.md
            return urlWithoutHash + 'index.md';
        } else {
            // Assume it's a page without extension
            return urlWithoutHash + '.md';
        }
    }

    // Handle MCP tool installation
    function handleMCPInstall(toolKey) {
        console.log('AI Assistant: Installing MCP tool:', toolKey);

        try {
            const mcpTools = window.AI_ASSISTANT_CONFIG?.mcp_tools || {};
            const tool = mcpTools[toolKey];

            if (!tool) {
                console.error('AI Assistant: MCP tool not found:', toolKey);
                showNotification('MCP tool configuration not found.', true);
                return;
            }

            // Handle Claude Desktop with .mcpb file
            if (tool.type === 'claude_desktop') {
                const mcpbUrl = tool.mcpb_url;

                // Extract filename from URL
                const urlPath = new URL(mcpbUrl).pathname;
                const filename = urlPath.split('/').pop();

                // Direct download - no fetch needed
                const downloadLink = document.createElement('a');
                downloadLink.href = mcpbUrl;
                downloadLink.download = filename;
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);

                showNotification('MCP tool download started.');
                closeDropdown();
                return;
            }

            // Handle VS Code with vscode: URL
            if (tool.type === 'vscode') {
                // Build MCP configuration JSON
                const mcpConfig = {
                    name: tool.server_name || toolKey,
                    type: tool.transport || 'sse',
                };

                // Add URL or command based on transport type
                if (tool.transport === 'stdio') {
                    mcpConfig.command = tool.command;
                    if (tool.args) {
                        mcpConfig.args = tool.args;
                    }
                } else {
                    // Default to SSE/HTTP
                    mcpConfig.url = tool.server_url;
                }

                // Generate VS Code installation URL
                const jsonString = JSON.stringify(mcpConfig);
                const encoded = encodeURIComponent(jsonString);
                const installUrl = `vscode:mcp/install?${encoded}`;

                console.log('AI Assistant: Opening installation URL:', installUrl);

                // Open the installation URL
                window.open(installUrl, '_self');

                // Close dropdown
                closeDropdown();
                return;
            }

            // Unknown tool type
            console.error('AI Assistant: Unknown MCP tool type:', tool.type);
            showNotification('Unknown MCP tool type.', true);

        } catch (error) {
            console.error('AI Assistant: Failed to install MCP tool:', error);
            showNotification('Failed to install MCP tool. Please try again.', true);
        }
    }

    // Handle AI chat integration
    async function handleAIChat(providerKey) {
        console.log('AI Assistant: Opening AI chat with provider:', providerKey);

        try {
            const providers = window.AI_ASSISTANT_CONFIG?.providers || {};
            const provider = providers[providerKey];

            if (!provider) {
                console.error('AI Assistant: Provider not found:', providerKey);
                showNotification('AI provider configuration not found.', true);
                return;
            }

            // Get the markdown URL for this page
            const markdownUrl = getMarkdownUrl();

            // Use the provider's prompt template with the URL
            const prompt = provider.prompt_template.replace('{url}', markdownUrl);
            const encodedPrompt = encodeURIComponent(prompt);
            const aiUrl = provider.url_template.replace('{prompt}', encodedPrompt);

            console.log('AI Assistant: Opening URL:', aiUrl);

            // Open in new tab
            window.open(aiUrl, '_blank');

            // Close dropdown
            closeDropdown();

        } catch (error) {
            console.error('AI Assistant: Failed to open AI chat:', error);
            showNotification('Failed to open AI chat. Please try again.', true);
        }
    }

    // Handle view as markdown
    function handleViewMarkdown() {
        const markdownUrl = getMarkdownUrl();
        console.log('AI Assistant: Opening markdown URL:', markdownUrl);

        // Open in new tab
        window.open(markdownUrl, '_blank');

        // Close dropdown
        closeDropdown();
    }

    // Handle copy as markdown
    function handleCopyMarkdown(showInlineConfirmation) {
        convertToMarkdown()
            .then(markdown => {
                console.log('AI Assistant: First 200 chars:', markdown.substring(0, 200));
                copyToClipboard(markdown, true);
                closeDropdown();
            })
            .catch(error => {
                console.error('AI Assistant: Failed to convert to markdown:', error);
                showNotification('Failed to convert page to markdown.', true);
            });
    }

    // Copy text to clipboard
    function copyToClipboard(text, showInlineConfirmation) {
        console.log('AI Assistant: Attempting to copy to clipboard');

        if (navigator.clipboard && navigator.clipboard.writeText) {
            console.log('AI Assistant: Using Clipboard API');
            navigator.clipboard.writeText(text).then(function() {
                console.log('AI Assistant: Successfully copied to clipboard');
                if (showInlineConfirmation) {
                    showInlineSuccessState();
                } else {
                    showNotification('Markdown copied to clipboard!');
                }
            }).catch(function(err) {
                console.error('AI Assistant: Clipboard API failed:', err);
                fallbackCopy(text, showInlineConfirmation);
            });
        } else {
            console.log('AI Assistant: Clipboard API not available, using fallback');
            fallbackCopy(text, showInlineConfirmation);
        }
    }

    // Show success state inline in the button
    function showInlineSuccessState() {
        const mainButton = document.getElementById('ai-assistant-button-main');
        if (!mainButton) return;

        const staticPath = getStaticPath();
        const originalContent = mainButton.innerHTML;

        // Change button to show checkmark and "Copied"
        mainButton.innerHTML = `
            <img src="${staticPath}/checked.svg" class="ai-assistant-icon" aria-hidden="true" alt="">
            <span class="ai-assistant-button-text">Copied</span>
        `;
        mainButton.classList.add('ai-assistant-button-success');

        // Revert after 2 seconds
        setTimeout(function() {
            mainButton.innerHTML = originalContent;
            mainButton.classList.remove('ai-assistant-button-success');
        }, 2000);
    }

    // Fallback copy method for older browsers
    function fallbackCopy(text, showInlineConfirmation) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            document.execCommand('copy');
            if (showInlineConfirmation) {
                showInlineSuccessState();
            } else {
                showNotification('Markdown copied to clipboard!');
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
            showNotification('Failed to copy to clipboard.', true);
        }

        document.body.removeChild(textarea);
    }

    // Close dropdown helper
    function closeDropdown() {
        const dropdown = document.getElementById('ai-assistant-dropdown');
        const dropdownButton = document.getElementById('ai-assistant-button-dropdown');
        if (dropdown && dropdownButton) {
            dropdown.style.display = 'none';
            dropdownButton.setAttribute('aria-expanded', 'false');
        }
    }

    // Show notification
    function showNotification(message, isError = false) {
        const notification = document.createElement('div');
        notification.className = 'ai-assistant-notification' + (isError ? ' error' : '');
        notification.textContent = message;
        document.body.appendChild(notification);

        // Trigger animation
        setTimeout(() => notification.classList.add('show'), 10);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // ---------------------------------------------------------------------------
    // PDF export
    // ---------------------------------------------------------------------------

    /**
     * Handle "Export as PDF" button click.
     *
     * Behaviour (in priority order):
     *   1. If ``AI_ASSISTANT_CONFIG.pdfExportUrl`` is a non-empty string, open
     *      it in a new browser tab.  This supports server-side PDF endpoints
     *      (e.g. WeasyPrint, GitBook-style ``~gitbook/pdf?page=…``).
     *   2. Otherwise call ``window.print()`` so the browser renders the page
     *      as a print document — the user can then save it as PDF via the
     *      browser's built-in print-to-PDF driver.
     *
     * Edge cases:
     *   - ``pdfExportUrl`` is ``null``, ``undefined``, or ``""`` → print dialog.
     *   - URL is provided but ``window.open`` is blocked by a popup blocker
     *     → silent (no notification spam; the blocker itself informs the user).
     */
    function handlePdfExport() {
        const pdfUrl = window.AI_ASSISTANT_CONFIG?.pdfExportUrl;
        closeDropdown();
        if (pdfUrl && typeof pdfUrl === 'string' && pdfUrl.trim() !== '') {
            console.log('AI Assistant: Opening PDF export URL:', pdfUrl);
            window.open(pdfUrl.trim(), '_blank', 'noopener,noreferrer');
        } else {
            console.log('AI Assistant: Triggering browser print dialog (PDF export)');
            window.print();
        }
    }

    // ---------------------------------------------------------------------------
    // AI Panel — floating chat stub / API-backed panel
    // ---------------------------------------------------------------------------

    /** Singleton panel element; created once, toggled on/off thereafter. */
    let _aiPanelEl = null;

    /**
     * Create the floating AI assistant panel DOM element.
     *
     * The panel is a slide-in drawer anchored to the bottom-right viewport
     * edge.  It is intentionally theme-agnostic: all colours are resolved
     * through the same three-layer CSS variable chain used by the rest of the
     * widget (PST → Furo → hardcoded fallback).
     *
     * When ``AI_ASSISTANT_CONFIG.panelApiEnabled`` is ``false`` the send
     * button shows a stub response, making the panel safe to ship with any
     * Sphinx build that does not have API credentials configured.
     *
     * Accessibility:
     *   - Panel has ``role="dialog"`` and ``aria-modal="true"``.
     *   - Close button is keyboard-focusable with a visible focus ring.
     *   - Input and send button are both keyboard-accessible.
     *
     * @returns {HTMLElement} The fully constructed (but not yet inserted) panel.
     */
    function createAIPanel() {
        const cfg = window.AI_ASSISTANT_CONFIG || {};
        const title = cfg.panelTitle || 'AI Assistant';
        const placeholder = cfg.panelPlaceholder || 'Ask a question about this page\u2026';
        const staticPath = getStaticPath();

        const panel = document.createElement('div');
        panel.id = 'ai-assistant-panel';
        panel.className = 'ai-assistant-panel';
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-label', title);
        panel.style.display = 'none';

        panel.innerHTML = `
            <div class="ai-assistant-panel-header">
                <div class="ai-assistant-panel-header-title">
                    <img src="${staticPath}/ai-panel.svg"
                         class="ai-assistant-panel-logo"
                         aria-hidden="true" alt="">
                    <span>${_escapeHtml(title)}</span>
                </div>
                <button class="ai-assistant-panel-close"
                        id="ai-assistant-panel-close"
                        type="button"
                        aria-label="Close ${_escapeHtml(title)}">
                    &#x2715;
                </button>
            </div>
            <div class="ai-assistant-panel-body" id="ai-assistant-panel-body">
                <div class="ai-assistant-panel-welcome">
                    <p>Hi! I\u2019m <strong>${_escapeHtml(title)}</strong>.</p>
                    <p>Ask me anything about this documentation page.</p>
                </div>
            </div>
            <div class="ai-assistant-panel-footer">
                <textarea
                    id="ai-assistant-panel-input"
                    class="ai-assistant-panel-input"
                    rows="2"
                    placeholder="${_escapeHtml(placeholder)}"
                    aria-label="Your question"
                ></textarea>
                <button class="ai-assistant-panel-send"
                        id="ai-assistant-panel-send"
                        type="button"
                        aria-label="Send question">
                    Send
                </button>
            </div>
        `;

        // Close button
        panel.querySelector('#ai-assistant-panel-close').addEventListener('click', function() {
            closeAIPanel();
        });

        // Send button
        panel.querySelector('#ai-assistant-panel-send').addEventListener('click', function() {
            handleAIPanelSubmit();
        });

        // Keyboard: Enter (without Shift) submits; Shift+Enter inserts newline.
        panel.querySelector('#ai-assistant-panel-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleAIPanelSubmit();
            }
        });

        // Escape key closes the panel.
        panel.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeAIPanel();
            }
        });

        document.body.appendChild(panel);
        return panel;
    }

    /**
     * Escape HTML special characters for safe innerHTML interpolation.
     *
     * Used exclusively for user-supplied config strings (``panelTitle``,
     * ``panelPlaceholder``) that are embedded in panel markup.  Prevents
     * XSS if a conf.py author inadvertently includes ``<`` or ``>``
     * characters in the title string.
     *
     * @param {string} str - Raw string to escape.
     * @returns {string} HTML-safe string.
     */
    function _escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Toggle the AI panel open/closed.
     *
     * Creates the panel DOM on first call (lazy singleton).  Subsequent calls
     * simply flip ``display`` between ``'flex'`` and ``'none'``.  On open,
     * focus is moved to the text input for immediate keyboard interaction.
     */
    function toggleAIPanel() {
        if (!_aiPanelEl) {
            _aiPanelEl = createAIPanel();
        }
        const isVisible = _aiPanelEl.style.display !== 'none';
        if (isVisible) {
            closeAIPanel();
        } else {
            _aiPanelEl.style.display = 'flex';
            // Slide-in animation driven by CSS transition on opacity/transform.
            requestAnimationFrame(function() {
                _aiPanelEl.classList.add('ai-assistant-panel--open');
            });
            const input = document.getElementById('ai-assistant-panel-input');
            if (input) {
                setTimeout(function() { input.focus(); }, 100);
            }
        }
    }

    /**
     * Close the AI panel with a slide-out animation.
     *
     * Removes the ``--open`` modifier class (triggering the CSS transition)
     * then hides the element once the transition ends.  Focus is returned to
     * the dropdown trigger button so keyboard users are not stranded.
     */
    function closeAIPanel() {
        if (!_aiPanelEl) return;
        _aiPanelEl.classList.remove('ai-assistant-panel--open');
        // Wait for CSS transition (300 ms) before hiding so the slide-out
        // animation is fully visible.
        setTimeout(function() {
            if (_aiPanelEl) {
                _aiPanelEl.style.display = 'none';
            }
        }, 300);
        // Return focus to the dropdown toggle.
        const dropBtn = document.getElementById('ai-assistant-button-dropdown');
        if (dropBtn) dropBtn.focus();
    }

    /**
     * Append a message bubble to the panel body.
     *
     * @param {string} text    - Message content (plain text; HTML-escaped).
     * @param {'user'|'assistant'|'error'} role - Determines CSS class and alignment.
     */
    function _appendPanelMessage(text, role) {
        const body = document.getElementById('ai-assistant-panel-body');
        if (!body) return;

        // Remove the welcome banner on first real message exchange.
        const welcome = body.querySelector('.ai-assistant-panel-welcome');
        if (welcome) welcome.remove();

        const bubble = document.createElement('div');
        bubble.className = `ai-assistant-panel-bubble ai-assistant-panel-bubble--${role}`;
        bubble.textContent = text;
        body.appendChild(bubble);
        // Auto-scroll to newest message.
        body.scrollTop = body.scrollHeight;
    }

    /**
     * Read the panel input, display the user message, then either call the
     * Anthropic API (when ``panelApiEnabled`` is ``true``) or show a polite
     * stub response.
     *
     * **API mode** (``panelApiEnabled: true``):
     *   POSTs ``{model, max_tokens, messages}`` to the Anthropic
     *   ``/v1/messages`` endpoint using the same pattern documented in the
     *   Anthropic API-in-Artifacts spec.  Page Markdown is prepended to the
     *   conversation as a system message so the model has full page context.
     *
     * **Stub mode** (``panelApiEnabled: false``, default):
     *   Returns a static placeholder reply so the panel UI is fully usable
     *   and testable without any API credentials or network calls.
     *
     * Edge cases:
     *   - Empty input → silently returns; no message appended.
     *   - Input longer than 4000 characters → truncated with a warning note.
     *   - Network failure → shows an error bubble with the reason.
     *
     * @returns {Promise<void>}
     */
    async function handleAIPanelSubmit() {
        const input = document.getElementById('ai-assistant-panel-input');
        if (!input) return;

        const rawText = input.value.trim();
        if (!rawText) return;

        // Enforce a reasonable per-message length guard.
        const MAX_INPUT_CHARS = 4000;
        const questionText = rawText.length > MAX_INPUT_CHARS
            ? rawText.slice(0, MAX_INPUT_CHARS) + '\u2026 [truncated]'
            : rawText;

        _appendPanelMessage(questionText, 'user');
        input.value = '';
        input.disabled = true;

        const sendBtn = document.getElementById('ai-assistant-panel-send');
        if (sendBtn) sendBtn.disabled = true;

        const cfg = window.AI_ASSISTANT_CONFIG || {};

        try {
            if (cfg.panelApiEnabled) {
                await _panelApiCall(questionText, cfg);
            } else {
                await _panelStubReply(questionText);
            }
        } catch (err) {
            console.error('AI Assistant panel error:', err);
            _appendPanelMessage(
                `Sorry, something went wrong: ${err.message}`,
                'error'
            );
        } finally {
            input.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            input.focus();
        }
    }

    /**
     * Live Anthropic API call for the AI panel.
     *
     * Collects the current page Markdown (via ``convertToMarkdown``), builds
     * a system prompt that grounds the model in the page content, then posts
     * to ``/v1/messages``.  The API key is intentionally NOT embedded here;
     * this function assumes the deployment environment has configured the
     * Anthropic reverse-proxy or CORS settings so the browser fetch succeeds.
     *
     * @param {string} question - User's trimmed question text.
     * @param {Object} cfg      - ``window.AI_ASSISTANT_CONFIG``.
     * @returns {Promise<void>}
     */
    async function _panelApiCall(question, cfg) {
        let pageMarkdown = '';
        try {
            pageMarkdown = await convertToMarkdown();
        } catch (_) {
            // Non-fatal: continue without page context if conversion fails.
        }

        const systemPrompt = pageMarkdown
            ? `You are a helpful documentation assistant. Answer questions about the following documentation page.\n\n---\n${pageMarkdown.slice(0, 8000)}\n---`
            : 'You are a helpful documentation assistant.';

        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'claude-sonnet-4-20250514',
                max_tokens: 1000,
                system: systemPrompt,
                messages: [{ role: 'user', content: question }],
            }),
        });

        if (!response.ok) {
            const errBody = await response.text().catch(() => '');
            throw new Error(`API ${response.status}: ${errBody.slice(0, 120)}`);
        }

        const data = await response.json();
        const reply = (data.content || [])
            .filter(b => b.type === 'text')
            .map(b => b.text)
            .join('\n')
            .trim();

        _appendPanelMessage(reply || '(no response)', 'assistant');
    }

    /**
     * Stub reply for when ``panelApiEnabled`` is ``false``.
     *
     * Simulates a realistic async delay (400 ms) so the loading state is
     * visible during development / demo and the UX is not jarring.
     *
     * @param {string} _question - User question (unused in stub mode).
     * @returns {Promise<void>}
     */
    async function _panelStubReply(_question) {
        await new Promise(resolve => setTimeout(resolve, 400));
        _appendPanelMessage(
            'This AI assistant panel is running in stub mode. '
            + 'Set ai_assistant_panel_api_enabled = True in conf.py '
            + 'to enable live responses.',
            'assistant'
        );
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAIAssistant);
    } else {
        initAIAssistant();
    }

})();
