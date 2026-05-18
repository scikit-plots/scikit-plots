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

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAIAssistant);
    } else {
        initAIAssistant();
    }

})();
