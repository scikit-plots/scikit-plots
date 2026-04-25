/**
 * ============================================================================
 * ai-assistant.js  —  Sphinx AI Assistant Widget
 * ============================================================================
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                      MIND MAP — JS OVERVIEW                            │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 *  ai-assistant.js
 *  │
 *  ├── A. IIFE WRAPPER  "(function() { ... })()"
 *  │       → Everything lives inside this function to avoid polluting the
 *  │         global scope (window object).  No variables "leak" out.
 *  │
 *  ├── B. THEME SYSTEM  (dark / light / auto)
 *  │       ├── _THEME_KEY       → localStorage key name (namespace collision guard)
 *  │       ├── _THEME_CYCLE     → ['auto', 'light', 'dark'] — click order
 *  │       ├── _getThemeMode()  → reads localStorage safely (try/catch)
 *  │       ├── _resolveTheme()  → converts 'auto' → 'light'|'dark' via OS query
 *  │       ├── applyThemeMode() → sets 3 HTML attributes + persists intent
 *  │       ├── handleThemeToggle() → advances cycle index, calls applyThemeMode
 *  │       ├── _updateThemeMenuItem() → keeps dropdown label in sync
 *  │       ├── _themeIcon()     → returns inline SVG string (moon/sun/half-circle)
 *  │       └── initThemeMode()  → called FIRST to prevent flash-of-wrong-theme
 *  │
 *  ├── C. LIBRARY LOADING
 *  │       └── loadTurndown()   → loads Turndown from CDN if not already loaded
 *  │
 *  ├── D. UI CONSTRUCTION
 *  │       ├── initAIAssistant()    → entry point, calls theme init + Turndown + UI build
 *  │       ├── createAIAssistantUI() → orchestrates container + button + dropdown
 *  │       ├── createContainer()    → outer <div> wrapper
 *  │       ├── createButton()       → split-button HTML (main + divider + arrow)
 *  │       ├── createDropdown()     → builds menu items from CONFIG
 *  │       │       ├── markdown export item
 *  │       │       ├── view markdown item
 *  │       │       ├── AI provider items (from CONFIG.providers)
 *  │       │       ├── MCP tool items (from CONFIG.mcp_tools)
 *  │       │       └── theme toggle item
 *  │       ├── createMenuItem()     → generic menu row (icon + label + description)
 *  │       ├── createThemeToggleItem() → special menu row for the theme cycle button
 *  │       ├── insertContainer()    → inserts widget into sidebar OR title area
 *  │       └── insertInTitlePosition() → wraps h1 + widget in a flex row
 *  │
 *  ├── E. EVENT LISTENERS
 *  │       └── setupEventListeners() → wires all click handlers
 *  │               ├── mainButton click   → handleCopyMarkdown(inline=true)
 *  │               ├── dropdownButton click → toggle menu open/close
 *  │               ├── document click   → close menu when clicking outside
 *  │               ├── copy-markdown click → handleCopyMarkdown(inline=false)
 *  │               ├── view-markdown click → handleViewMarkdown()
 *  │               ├── ai-chat-* clicks → handleAIChat(providerKey)
 *  │               ├── mcp-* clicks     → handleMCPInstall(toolKey)
 *  │               └── theme-toggle click → handleThemeToggle()
 *  │
 *  ├── F. ACTIONS
 *  │       ├── convertToMarkdown()  → clones DOM, strips noise, runs Turndown
 *  │       ├── getMarkdownUrl()     → converts .html URL → .md URL
 *  │       ├── handleAIChat()       → builds prompt URL and opens new tab
 *  │       ├── handleViewMarkdown() → opens .md URL in new tab
 *  │       ├── handleCopyMarkdown() → calls convertToMarkdown then clipboard
 *  │       ├── copyToClipboard()    → modern Clipboard API with fallback
 *  │       ├── fallbackCopy()       → legacy execCommand('copy') for old browsers
 *  │       └── handleMCPInstall()   → handles claude_desktop (.mcpb) or vscode (URL)
 *  │
 *  └── G. HELPERS
 *          ├── getStaticPath()         → finds _static directory from script tags
 *          ├── closeDropdown()         → hides dropdown + resets aria-expanded
 *          ├── showNotification()      → creates/animates/removes toast element
 *          └── showInlineSuccessState() → temporarily shows "Copied!" on main button
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * KEY JAVASCRIPT CONCEPTS FOR NEWBIES
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  const vs let vs var
 *  ───────────────────
 *  const  → value never reassigned; block-scoped (preferred for most things)
 *  let    → value CAN be reassigned; block-scoped
 *  var    → value CAN be reassigned; function-scoped (older style; used here for
 *           compatibility with older Sphinx theme scripts)
 *
 *  Optional chaining  (?.)
 *  ─────────────────────────
 *  window.AI_ASSISTANT_CONFIG?.providers
 *  ↑ if AI_ASSISTANT_CONFIG is null/undefined, return undefined instead of crashing.
 *  Same as: AI_ASSISTANT_CONFIG && AI_ASSISTANT_CONFIG.providers
 *
 *  Nullish coalescing  (??)  and  Logical OR  (||)
 *  ─────────────────────────────────────────────────
 *  a ?? b  → use b only if a is null or undefined
 *  a || b  → use b if a is any falsy value (null, undefined, 0, '', false)
 *
 *  Template literals  (`...`)
 *  ──────────────────────────
 *  `Hello ${name}!`  → embed expressions inside a string using ${...}
 *  Multi-line strings without \n concatenation.
 *
 *  Arrow functions  (=>)
 *  ─────────────────────
 *  const add = (a, b) => a + b;   ← concise, single-expression
 *  setTimeout(() => { ... }, ms)  ← common pattern for callbacks
 *
 *  Promises and async/await
 *  ────────────────────────
 *  A Promise represents a value that will be available later (asynchronously).
 *  async function foo() { const result = await somePromise(); }
 *  await pauses execution INSIDE the async function until the promise resolves.
 *  .then(callback) is the older promise chaining style (also used here).
 *
 *  DOM manipulation essentials
 *  ──────────────────────────
 *  document.createElement('div')      → create a new element
 *  element.className = 'my-class'     → set CSS class
 *  element.setAttribute('role', 'x') → set any HTML attribute
 *  parent.appendChild(child)          → add child at the END of parent
 *  parent.insertBefore(child, ref)    → add child BEFORE ref element
 *  element.querySelector('.sel')      → find FIRST matching descendant
 *  element.querySelectorAll('.sel')   → find ALL matching descendants
 *  element.remove()                   → remove element from the DOM
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */


/* =============================================================================
 * A. IIFE WRAPPER  — Immediately Invoked Function Expression
 * =============================================================================
 *
 * Pattern:  (function() { 'use strict'; /* all code here * / })();
 *                           ↑                                    ↑
 *           define the function                   immediately call it
 *
 * WHY WRAP EVERYTHING IN AN IIFE?
 * ─────────────────────────────────
 * Without a wrapper, every variable and function declared with "var" or "function"
 * would become a property of the global "window" object.  On a documentation page
 * with many scripts, this risks:
 *   - Name collisions: two scripts both define "function handleClick()" → one overwrites the other
 *   - Security: other scripts can read/modify your private variables
 *
 * The IIFE creates a private function scope.  Nothing inside is visible outside.
 *
 * 'use strict'
 * ─────────────
 * Opt-in to JavaScript's "strict mode":
 *   - Prevents silent errors (e.g. typos in variable names become ReferenceErrors)
 *   - Disallows dangerous features (e.g. with statements, duplicate parameters)
 *   - Makes code easier to optimise for the JS engine
 * ============================================================================= */
(function() {
    'use strict';

    /* =========================================================================
     * B. THEME SYSTEM
     * =========================================================================
     *
     * ┌──────────────────────────────────────────────────────────────────────┐
     * │  MIND MAP — Theme System Flow                                        │
     * └──────────────────────────────────────────────────────────────────────┘
     *
     *  User clicks "Color scheme" in dropdown
     *       │
     *       ▼
     *  handleThemeToggle()
     *       │ reads current mode from localStorage
     *       │ advances index in _THEME_CYCLE array
     *       │ gets next mode ('auto' → 'light' → 'dark' → 'auto' → ...)
     *       ▼
     *  applyThemeMode(mode)
     *       │ _resolveTheme(mode) converts 'auto' → OS system preference
     *       │ sets data-bs-theme on <html>   ← Bootstrap / pydata theme reads this
     *       │ sets data-theme on <html>       ← Furo reads this
     *       │ sets data-ai-theme on <html>    ← our CSS reads this
     *       │ saves INTENT to localStorage    ← 'auto', not resolved 'light'/'dark'
     *       ▼
     *  _updateThemeMenuItem(mode)
     *       → updates dropdown label + SVG icon to reflect new mode
     *
     * PAGE LOAD (initThemeMode)
     *  ─────────────────────────
     *  Called synchronously BEFORE any asynchronous network request (Turndown CDN).
     *  Applying the saved theme this early means <html> attributes are set before
     *  the browser paints any visible content — eliminating the "white flash then
     *  dark flip" problem common in late-initialised toggles.
     *
     * DESIGN DECISION — store INTENT, not RESOLVED VALUE
     * ─────────────────────────────────────────────────────
     *  localStorage stores 'auto', not 'light' or 'dark'.
     *  On the next page load, 'auto' is re-evaluated against the CURRENT system
     *  preference.  A user who scheduled dark mode at sunset always sees dark
     *  after sunset, without any extra interaction.
     * ========================================================================= */

    /**
     * localStorage key for persisting the color-scheme preference.
     * Prefixed with 'ai-assistant-' to avoid naming collisions with other
     * scripts on the host page that might also use localStorage.
     */
    var _THEME_KEY = 'ai-assistant-color-scheme';

    /**
     * The ordered list of modes.  indexOf + modulo arithmetic provides
     * clean cycling without bounds checks:
     *   cycle = ['auto', 'light', 'dark']
     *   'auto'  → index 0 → next index = (0+1) % 3 = 1 → 'light'
     *   'light' → index 1 → next index = (1+1) % 3 = 2 → 'dark'
     *   'dark'  → index 2 → next index = (2+1) % 3 = 0 → 'auto'  ← wraps!
     */
    var _THEME_CYCLE = ['auto', 'light', 'dark'];

    /**
     * Read the persisted color-scheme preference from localStorage.
     *
     * Returns 'auto' when:
     *   - No preference has been saved yet (first-time visitor)
     *   - localStorage throws SecurityError (private browsing, cross-origin iframe)
     *   - localStorage is full (quota exceeded — rare but real)
     *
     * @returns {'auto'|'light'|'dark'}
     */
    function _getThemeMode() {
        /*
         * try/catch is MANDATORY around localStorage access.
         * Browsers throw a SecurityError in:
         *   - Cross-origin iframes with cookies blocked
         *   - Strict private browsing modes (e.g. Firefox private window)
         *   - Storage quota exceeded
         * Without the try/catch, these environments would crash the entire script.
         * With it, we gracefully fall back to 'auto'.
         */
        try {
            return localStorage.getItem(_THEME_KEY) || 'auto';
            /*
             * localStorage.getItem returns NULL when the key doesn't exist.
             * The || 'auto' converts null → 'auto' (our safe default).
             */
        } catch (e) {
            return 'auto'; // storage unavailable — treat as first-time visitor
        }
    }

    /**
     * Convert a stored intent into a concrete DOM value.
     *
     * 'auto' means "match the OS/browser preference RIGHT NOW".
     * We use the CSS Media Queries Level 5 API: window.matchMedia().
     * This API lets JavaScript READ the same media queries that CSS uses.
     *
     * 'light' and 'dark' are explicit user overrides — they pass straight through.
     *
     * @param {'auto'|'light'|'dark'} mode - The stored intent.
     * @returns {'light'|'dark'}           - The concrete value to write to the DOM.
     */
    function _resolveTheme(mode) {
        if (mode !== 'auto') return mode; // explicit override — honour it as-is

        /*
         * window.matchMedia('(prefers-color-scheme: dark)').matches
         *   ↑ same as @media (prefers-color-scheme: dark) in CSS, but in JS
         *
         * .matches returns true  → OS/browser is in dark mode
         * .matches returns false → OS/browser is in light mode
         *
         * Guard with "window.matchMedia &&" because:
         *   - Some old browsers don't have matchMedia
         *   - Non-browser environments (Node.js, jsdom) may lack it
         *   If matchMedia doesn't exist, fall back to 'light' (safe default).
         */
        return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
            ? 'dark'
            : 'light';
    }

    /**
     * Apply a color-scheme mode to the document and persist the user's choice.
     *
     * This is the SINGLE SOURCE OF TRUTH for all theme changes.  Every code path
     * that needs to change the theme calls this function.
     *
     * Steps performed:
     *   1. Resolve 'auto' to 'light' or 'dark' based on OS preference
     *   2. Write the resolved value to THREE HTML attributes simultaneously
     *      (covers every Sphinx theme variant — see comments inline)
     *   3. Persist the INTENT (not the resolved value) to localStorage
     *   4. Update the dropdown menu label + icon
     *
     * @param {'auto'|'light'|'dark'} mode - The mode to apply.
     */
    function applyThemeMode(mode) {
        var resolved = _resolveTheme(mode); // 'auto' → 'light' or 'dark'
        var html = document.documentElement; // <html> element — root of the DOM

        /*
         * WHY SET THREE ATTRIBUTES?
         * ─────────────────────────
         * Different Sphinx themes "watch" different attributes on <html>
         * to activate dark mode.  By setting ALL THREE, we cover every theme
         * variant without needing to know which one is installed.
         *
         * Setting an attribute that a theme ignores is harmless — the browser
         * simply doesn't use it.
         */

        // Bootstrap 5 attribute → pydata-sphinx-theme (scikit-learn, NumPy, pandas, etc.)
        // When set to "dark", Bootstrap activates its complete dark palette automatically.
        html.setAttribute('data-bs-theme', resolved);

        // Furo attribute → Furo theme and several generic Sphinx themes.
        // Furo ships two CSS custom-property sets and switches between them via this.
        html.setAttribute('data-theme', resolved);

        // Our own attribute → consumed exclusively by html[data-ai-theme="dark"] in CSS.
        // Provides the fallback dark mode for themes that respond to neither of the above.
        html.setAttribute('data-ai-theme', resolved);

        /*
         * PERSIST THE INTENT, NOT THE RESOLVED VALUE.
         * ─────────────────────────────────────────────
         * We save 'auto', NOT 'dark' (even if 'auto' resolved to 'dark').
         *
         * Why? On the NEXT page load, 'auto' will be re-evaluated against the
         * THEN-current OS preference.  If the user toggled their OS to light mode
         * overnight, 'auto' will resolve to 'light' correctly.
         *
         * If we saved 'dark' (the resolved value), the user's "follow the OS"
         * intent would be lost forever.
         */
        try {
            localStorage.setItem(_THEME_KEY, mode);
        } catch (e) {
            /* Quota exceeded or private browsing — silently ignore.
             * The theme change still applies for this page session;
             * it just won't persist across page loads. */
        }

        // Keep the dropdown menu item in sync with the new mode
        _updateThemeMenuItem(mode);
    }

    /**
     * Advance the theme mode to the next step in the cycle and apply it.
     *
     * Modulo arithmetic explains the cycling:
     *   idx = _THEME_CYCLE.indexOf('dark')    // = 2
     *   next_idx = (2 + 1) % 3                // = 0  (wraps back to start)
     *   next_mode = _THEME_CYCLE[0]           // = 'auto'
     *
     * This always wraps cleanly at the end of the array with no if/else.
     */
    function handleThemeToggle() {
        var current = _getThemeMode();
        var idx = _THEME_CYCLE.indexOf(current);
        var next = _THEME_CYCLE[(idx + 1) % _THEME_CYCLE.length];
        applyThemeMode(next);
        closeDropdown(); // consistent with every other dropdown menu action
    }

    /**
     * Update the theme toggle menu item's label text and SVG icon.
     *
     * Called by applyThemeMode() after every mode change.
     * DOM lookups are guarded with null checks because:
     * - initThemeMode() fires BEFORE the UI is built (Turndown loads asynchronously)
     * - In that case, the menu item doesn't exist yet — safe to skip
     * - When the menu IS built, createThemeToggleItem() calls _getThemeMode()
     *   directly to render the correct initial state
     *
     * @param {'auto'|'light'|'dark'} mode - The mode that was just applied.
     */
    function _updateThemeMenuItem(mode) {
        var el = document.getElementById('ai-assistant-theme-toggle');
        if (!el) return; // UI not built yet — safe to skip (see comment above)

        var span = el.querySelector('.ai-assistant-theme-label'); // text label
        var icon = el.querySelector('.ai-assistant-theme-icon');  // SVG wrapper

        var labels = {
            auto:  'Color scheme: System',  // follow OS / browser preference
            light: 'Color scheme: Light',   // forced light
            dark:  'Color scheme: Dark',    // forced dark
        };

        if (span) span.textContent = labels[mode] || labels['auto'];
        if (icon) icon.innerHTML = _themeIcon(mode); // swap in new SVG
    }

    /**
     * Return an inline SVG string for the given color-scheme mode.
     *
     * WHY INLINE SVG (not <img> src)?
     * ─────────────────────────────────
     * 1. No extra files to bundle with the extension.
     * 2. stroke="currentColor" means the icon automatically inherits the
     *    text colour of its parent — correct in both light and dark mode
     *    without any CSS filter rules.
     * 3. The icon is always in sync with the current mode.
     *
     * ICON MEANINGS:
     *   dark  → Crescent moon (🌙) — universal symbol for night / dark mode
     *   light → Radiating sun (☀️) — universal symbol for daytime / light mode
     *   auto  → Half-filled circle — "half and half"; the OS decides.
     *           Same symbol used by GitHub and pydata-sphinx-theme.
     *
     * HOW THE HALF-CIRCLE IS DRAWN:
     *   <circle cx="12" cy="12" r="10"/>     ← draws the OUTLINE ring
     *   <path d="M12 2a10 10 0 0 1 0 20z" fill="currentColor" stroke="none"/>
     *         ↑ arc from top-center (12,2), clockwise around the LEFT half,
     *           to bottom-center (12,22), then "z" closes back to start.
     *           fill="currentColor" fills the left half solid.
     *   Result: left half solid, right half hollow = half-circle.
     *
     * @param {'auto'|'light'|'dark'} mode
     * @returns {string} Raw SVG markup.
     */
    function _themeIcon(mode) {
        if (mode === 'dark') {
            // Crescent moon — Feather Icons "moon" shape
            return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
        }
        if (mode === 'light') {
            // Radiating sun — Feather Icons "sun" (circle + 8 evenly spaced rays)
            return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
        }
        // 'auto' — half-filled circle (left half solid, right half outline only)
        return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20z" fill="currentColor" stroke="none"/></svg>';
    }

    /**
     * Apply the saved theme IMMEDIATELY on page load to prevent visual flash.
     *
     * IMPORTANT: Called BEFORE loadTurndown() because Turndown is a network
     * request (CDN).  If we waited, users would see a "white flash" while the
     * CDN request is in flight.  Calling this synchronously ensures the correct
     * theme attributes are on <html> before the browser paints visible content.
     *
     * SYSTEM-PREFERENCE LIVE LISTENER:
     *   Registers a listener on the OS dark-mode media query.
     *   When the user switches their OS to dark mode (e.g. scheduled sunset),
     *   this fires WHILE the page is open and re-applies the theme if mode is 'auto'.
     *   If the user chose an explicit 'light' or 'dark' override, the listener
     *   does nothing — their explicit choice must not be undone by the OS.
     */
    function initThemeMode() {
        // Apply saved preference immediately — synchronous, no network round-trip
        applyThemeMode(_getThemeMode());

        /*
         * Guard with "if (window.matchMedia)" because:
         *   - Some old browsers don't support matchMedia
         *   - Non-browser environments (Node.js, jsdom) may not have it
         *   Without the guard, we'd get a TypeError in those environments.
         */
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
                // Only react to OS changes when the user chose "follow the OS"
                if (_getThemeMode() === 'auto') {
                    applyThemeMode('auto');
                }
            });
        }
    }

    /* =========================================================================
     * C. LIBRARY LOADING — Turndown (HTML → Markdown converter)
     * =========================================================================
     *
     * Turndown is a third-party library that converts HTML to Markdown.
     * We load it from a CDN (Content Delivery Network) on demand, instead of
     * bundling it, to keep the extension package small.
     *
     * LOADING PATTERN:
     *   1. Check if TurndownService is already defined (maybe another script
     *      on the page already loaded it).  If so, call the callback immediately.
     *   2. Otherwise, create a <script> element pointing to the CDN URL.
     *      The browser will download and execute it automatically once appended.
     *   3. script.onload  → fires when the script finishes loading successfully.
     *   4. script.onerror → fires if the CDN is unreachable.  Log an error so
     *      developers can diagnose the problem.
     * ========================================================================= */

    function loadTurndown(callback) {
        // If TurndownService already exists (another script loaded it), skip the CDN request
        if (typeof TurndownService !== 'undefined') {
            callback();
            return; // early exit — no need to load the script again
        }

        /*
         * Create a new <script> element and inject it into <head>.
         * The browser starts downloading the script immediately when it's appended.
         * This is the standard "dynamic script loading" pattern.
         */
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/turndown@7.1.2/dist/turndown.min.js';
        // Pinned to version 7.1.2 — exact version prevents unexpected breakage
        // from future library updates.

        script.onload = callback; // when script loads successfully, continue with UI setup
        script.onerror = function() {
            // If the CDN is down, warn the developer — the widget's markdown
            // features won't work, but the page itself is unaffected.
            console.error('Failed to load Turndown library');
        };

        document.head.appendChild(script); // trigger the download
    }

    /* =========================================================================
     * D. UI CONSTRUCTION — Entry Point
     * =========================================================================
     *
     * Initialization flow:
     *
     *  initAIAssistant()
     *       │
     *       ├── initThemeMode()          ← SYNCHRONOUS (no network)
     *       │       → sets theme before anything is painted
     *       │
     *       └── loadTurndown(callback)   ← ASYNCHRONOUS (CDN request)
     *               → when script loads:
     *                       └── createAIAssistantUI()
     *                               ├── createContainer()
     *                               ├── createButton()
     *                               ├── createDropdown()
     *                               ├── container.appendChild(button + dropdown)
     *                               ├── insertContainer(container, position)
     *                               └── setupEventListeners(button, dropdown)
     * ========================================================================= */

    function initAIAssistant() {
        initThemeMode(); // apply saved/system theme NOW to prevent flash
        loadTurndown(function() {
            // This callback runs AFTER Turndown finishes loading
            createAIAssistantUI();
        });
    }

    /**
     * Orchestrate the widget's DOM construction.
     * Reads the configured position ('sidebar' or 'title') from
     * window.AI_ASSISTANT_CONFIG, which is injected by the Sphinx extension.
     */
    function createAIAssistantUI() {
        const container = createContainer(); // outer wrapper div
        const button    = createButton();    // split-button bar
        const dropdown  = createDropdown();  // floating menu panel

        // Assemble: put button and dropdown inside container
        container.appendChild(button);
        container.appendChild(dropdown);

        /*
         * Optional chaining (?.) safely reads a nested property:
         *   window.AI_ASSISTANT_CONFIG?.position
         *   ↑ if CONFIG is undefined, return undefined (not crash)
         *   || 'sidebar'  → fall back to 'sidebar' if undefined
         */
        const position = window.AI_ASSISTANT_CONFIG?.position || 'sidebar';
        insertContainer(container, position);

        // Wire up all click handlers
        setupEventListeners(button, dropdown);
    }

    /* ── Container ──────────────────────────────────────────────────────────── */
    function createContainer() {
        const container = document.createElement('div');
        container.className = 'ai-assistant-container'; // CSS styling hook
        container.id = 'ai-assistant-container';         // unique ID for JS lookups
        return container;
    }

    /* ── Split-Button ───────────────────────────────────────────────────────── */
    /**
     * Create the trigger button container (two buttons + a divider).
     *
     * STRUCTURE:
     *   <div class="ai-assistant-button">          ← outer container (for border/radius)
     *     <button class="...button-main">          ← "Copy page" — direct action
     *       <img /> + <span>Copy page</span>
     *     </button>
     *     <span class="...divider"></span>         ← 1px vertical line
     *     <button class="...button-dropdown">      ← ▼ arrow — opens the menu
     *       <img />
     *     </button>
     *   </div>
     *
     * WHY TWO SEPARATE BUTTONS?
     * ─────────────────────────
     * One click area = immediate "Copy page" action (no menu needed for the most
     * common task).  The second, smaller click area = reveal more options.
     * This "split button" pattern is common in toolbar UIs (e.g. Git clients).
     */
    function createButton() {
        const container = document.createElement('div');
        container.className = 'ai-assistant-button';
        container.id = 'ai-assistant-button';

        const staticPath = getStaticPath(); // path to the _static directory for icons

        /*
         * Template literal (backtick string) with ${...} interpolation.
         * innerHTML sets the ENTIRE inner HTML at once — more efficient than
         * multiple createElement calls for static content.
         *
         * aria-label / aria-expanded / aria-haspopup:
         *   These ARIA attributes are read by screen readers to announce
         *   "this button opens a popup menu" and track whether it's open.
         *   They are essential for accessibility.
         */
        container.innerHTML = `
            <button class="ai-assistant-button-main" id="ai-assistant-button-main" type="button">
                <img src="${staticPath}/copy-to-clipboard.svg" class="ai-assistant-icon" aria-hidden="true" alt="">
                <span class="ai-assistant-button-text">Copy page</span>
            </button>
            <span class="ai-assistant-button-divider"></span>
            <button class="ai-assistant-button-dropdown" id="ai-assistant-button-dropdown" type="button"
                    aria-label="More options"
                    aria-expanded="false"
                    aria-haspopup="true">
                <img src="${staticPath}/arrow-down.svg" class="ai-assistant-dropdown-icon" aria-hidden="true" alt="">
            </button>
        `;
        return container;
    }

    /* ── Static Path Resolver ───────────────────────────────────────────────── */
    /**
     * Find the URL path to the Sphinx _static directory.
     *
     * WHY NOT HARDCODE "_static"?
     * ─────────────────────────────
     * Documentation sites can be served from subdirectories
     * (e.g. /docs/project/v2/_static/...), so the path to _static is
     * always relative and never known at development time.
     *
     * STRATEGY:
     *   1. Find any <script> tag whose src contains "_static" — there will always
     *      be at least one, since this very script is loaded from _static.
     *   2. Slice the URL up to and including "_static" to get the directory path.
     *   3. Fall back to the simple string "_static" for local development.
     */
    function getStaticPath() {
        const scripts = document.querySelectorAll('script[src*="_static"]');
        if (scripts.length > 0) {
            const src = scripts[0].getAttribute('src');
            // indexOf('_static') returns the character position of "_static" in the URL.
            // + 7 includes the "_static" string itself (7 characters).
            const staticPath = src.substring(0, src.indexOf('_static') + 7);
            return staticPath;
        }
        return '_static'; // fallback for local development / testing
    }

    /* ── Dropdown Panel ─────────────────────────────────────────────────────── */
    /**
     * Build the dropdown menu panel from window.AI_ASSISTANT_CONFIG.
     *
     * The config object controls which sections appear:
     *   features.markdown_export  → "Copy page" menu item
     *   features.view_markdown    → "View as Markdown" menu item
     *   features.ai_chat          → one item per enabled provider
     *   features.mcp_integration  → one item per enabled MCP tool
     *   features.theme_toggle     → color-scheme cycle item (default: true)
     *
     * Separators (<div class="separator">) are inserted between sections
     * only when BOTH adjacent sections are present.
     */
    function createDropdown() {
        const dropdown = document.createElement('div');
        dropdown.className = 'ai-assistant-dropdown';
        dropdown.id = 'ai-assistant-dropdown';
        dropdown.setAttribute('role', 'menu'); // ARIA: screen readers read this as a menu
        dropdown.style.display = 'none';       // hidden by default; JS toggles it

        /*
         * Optional chaining + fallback:
         *   window.AI_ASSISTANT_CONFIG?.features || { markdown_export: true }
         *   → if CONFIG is missing, default to showing only the markdown export feature
         */
        const features   = window.AI_ASSISTANT_CONFIG?.features || { markdown_export: true };
        const staticPath = getStaticPath();

        /* ── Markdown export item ─────────────────────────────────────────── */
        if (features.markdown_export) {
            const exportItem = createMenuItem(
                'copy-markdown',            // ID suffix → "ai-assistant-copy-markdown"
                'Copy page',
                'Copy this page as Markdown for LLMs.',
                `${staticPath}/copy-to-clipboard.svg`
            );
            dropdown.appendChild(exportItem);
        }

        /* ── View as Markdown item ────────────────────────────────────────── */
        if (features.view_markdown) {
            const viewItem = createMenuItem(
                'view-markdown',
                'View as Markdown',
                'View this page as Markdown.',
                `${staticPath}/markdown.svg`
            );
            dropdown.appendChild(viewItem);
        }

        /* ── AI Chat provider items ───────────────────────────────────────── */
        if (features.ai_chat) {
            const providers = window.AI_ASSISTANT_CONFIG?.providers || {};

            // Add a separator ONLY if there are items above this section
            if (features.markdown_export || features.view_markdown) {
                const separator = document.createElement('div');
                separator.className = 'ai-assistant-menu-separator';
                dropdown.appendChild(separator);
            }

            /*
             * Object.entries(obj) returns an array of [key, value] pairs.
             * Example: { claude: { enabled: true, label: 'Claude' }, ... }
             * → [['claude', { enabled: true, label: 'Claude' }], ...]
             *
             * forEach destructures each pair as ([key, provider]).
             */
            Object.entries(providers).forEach(([key, provider]) => {
                if (provider.enabled) {
                    // Icon can be a full URL (e.g. provider logo) or a filename in _static
                    const icon     = provider.icon || 'comment-discussion.svg';
                    const iconPath = icon.startsWith('http') ? icon : `${staticPath}/${icon}`;

                    const aiItem = createMenuItem(
                        `ai-chat-${key}`,
                        provider.label,
                        provider.description || 'Open AI chat with this page context.',
                        iconPath
                    );
                    /*
                     * dataset is a special DOM property that maps to data-* HTML attributes.
                     * Setting aiItem.dataset.provider = key
                     * is equivalent to: aiItem.setAttribute('data-provider', key)
                     * We read it later in the click handler: this.dataset.provider
                     */
                    aiItem.dataset.provider = key;
                    dropdown.appendChild(aiItem);
                }
            });
        }

        /* ── MCP tool items ──────────────────────────────────────────────── */
        if (features.mcp_integration) {
            const mcpTools = window.AI_ASSISTANT_CONFIG?.mcp_tools || {};

            // Separator if there are any items above
            if (features.ai_chat || features.markdown_export || features.view_markdown) {
                const separator = document.createElement('div');
                separator.className = 'ai-assistant-menu-separator';
                dropdown.appendChild(separator);
            }

            Object.entries(mcpTools).forEach(([key, tool]) => {
                if (tool.enabled) {
                    const icon     = tool.icon || 'ai-tools.svg';
                    const iconPath = icon.startsWith('http') ? icon : `${staticPath}/${icon}`;

                    const mcpItem = createMenuItem(
                        `mcp-${key}`,
                        tool.label,
                        tool.description || 'Install MCP server',
                        iconPath
                    );
                    mcpItem.dataset.mcpTool = key; // store tool key for click handler
                    dropdown.appendChild(mcpItem);
                }
            });
        }

        /* ── Theme toggle item ───────────────────────────────────────────── */
        /*
         * theme_toggle defaults to TRUE (shown unless explicitly disabled).
         * !== false checks for explicit false; undefined / missing → show it.
         */
        if (features.theme_toggle !== false) {
            var hasOtherItems = (
                features.markdown_export ||
                features.view_markdown   ||
                features.ai_chat         ||
                features.mcp_integration
            );
            if (hasOtherItems) {
                var sep = document.createElement('div');
                sep.className = 'ai-assistant-menu-separator';
                dropdown.appendChild(sep);
            }
            dropdown.appendChild(createThemeToggleItem());
        }

        return dropdown;
    }

    /* ── Theme Toggle Menu Item ─────────────────────────────────────────────── */
    /**
     * Build the special "Color scheme: System/Light/Dark" menu item.
     *
     * Unlike regular menu items (which use createMenuItem), this item:
     *   - Has a dynamic SVG icon (not an <img> file)
     *   - Has special class names so _updateThemeMenuItem() can target it
     *   - Reads the current mode from localStorage at render time
     */
    function createThemeToggleItem() {
        var mode   = _getThemeMode(); // read current preference
        var labels = {
            auto:  'Color scheme: System',
            light: 'Color scheme: Light',
            dark:  'Color scheme: Dark',
        };

        var item = document.createElement('button');
        item.className = 'ai-assistant-menu-item ai-assistant-theme-toggle-item';
        item.id = 'ai-assistant-theme-toggle';
        item.setAttribute('role', 'menuitem');
        item.setAttribute('type', 'button');

        /*
         * Array.join('') is an old-style way to build multi-line HTML strings
         * that works in ALL JavaScript versions, including ES3/ES5 environments
         * used by some Sphinx themes.
         *
         * [line1, line2, line3].join('') === 'line1line2line3'
         */
        item.innerHTML = [
            '<div class="ai-assistant-menu-item-content">',
            '  <div class="ai-assistant-menu-item-title">',
            /* ai-assistant-theme-icon: JS + CSS both know this class (SVG gets special treatment) */
            '    <span class="ai-assistant-theme-icon ai-assistant-menu-icon">' + _themeIcon(mode) + '</span>',
            '    <span class="ai-assistant-theme-label">' + (labels[mode] || labels['auto']) + '</span>',
            '  </div>',
            '  <div class="ai-assistant-menu-item-description">Click to cycle: System → Light → Dark.</div>',
            '</div>',
        ].join('');

        return item;
    }

    /* ── Generic Menu Item ──────────────────────────────────────────────────── */
    /**
     * Create a standard menu row: icon, title label, and description text.
     *
     * @param {string} id          - ID suffix (full ID = "ai-assistant-{id}")
     * @param {string} text        - Title label shown in bold
     * @param {string} description - Helper text shown below the label
     * @param {string} iconSrc     - URL to the icon image
     * @returns {HTMLButtonElement}
     */
    function createMenuItem(id, text, description, iconSrc) {
        const item = document.createElement('button');
        item.className = 'ai-assistant-menu-item';
        item.id = `ai-assistant-${id}`;
        item.setAttribute('role', 'menuitem'); // ARIA: screen reader reads "menuitem"

        /*
         * aria-hidden="true" on the icon tells screen readers to IGNORE it.
         * The icon is decorative — the label text already conveys the meaning.
         * alt="" on the <img> also marks it as decorative.
         * Together, these prevent screen readers from saying "image [filename]".
         */
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

    /* ── Container Insertion ────────────────────────────────────────────────── */
    /**
     * Insert the assembled widget container into the correct location on the page.
     *
     * POSITION OPTIONS:
     *   'sidebar' → try multiple sidebar selectors in order of preference;
     *               fall back to title position if no sidebar is found.
     *   'title'   → inline next to the <h1> heading.
     *   default   → top of the main article/content element.
     */
    function insertContainer(container, position) {
        if (position === 'sidebar') {
            /*
             * Try selectors from most to least specific.
             * Different Sphinx themes use different class names for their
             * right sidebar (table-of-contents panel).  We try them in order
             * and stop at the first one we find.
             */
            const selectors = [
                '.toc-drawer',              // Furo's right sidebar drawer wrapper
                'aside.toc-sidebar',        // Furo's right sidebar <aside>
                '.sidebar-secondary',       // generic secondary sidebar
                'aside[role="complementary"]', // ARIA fallback for any sidebar
            ];

            for (const selector of selectors) {
                const sidebar = document.querySelector(selector);
                if (sidebar) {
                    console.log('AI Assistant: Inserting into sidebar:', selector);
                    /*
                     * insertBefore(newChild, referenceChild) inserts newChild
                     * BEFORE referenceChild.  sidebar.firstChild means
                     * "before any existing content" → widget appears at the top.
                     */
                    sidebar.insertBefore(container, sidebar.firstChild);
                    return; // found it — stop trying other selectors
                }
            }

            // No sidebar found on this theme — fall through to title position
            console.log('AI Assistant: No sidebar found, falling back to title position');
            insertInTitlePosition(container);
            return;
        }

        if (position === 'title') {
            insertInTitlePosition(container);
            return;
        }

        // Final fallback: insert at the very top of the main article element
        const article = document.querySelector('article, .document, .body');
        if (article) {
            console.log('AI Assistant: Using fallback position at top of article');
            article.insertBefore(container, article.firstChild);
        }
    }

    /**
     * Insert the widget inline next to the <h1> page title.
     *
     * Creates a flex wrapper to put the title and widget on the same line:
     *
     *  ┌──────────────────────────────────────────────────────────────┐
     *  │  <div style="display:flex; ...">                             │
     *  │    <h1>Page Title</h1>         [Copy page ▼]                │
     *  └──────────────────────────────────────────────────────────────┘
     *
     * HOW IT WORKS:
     *   - Find the <h1> inside <article>
     *   - Create a new <div> wrapper with flex layout
     *   - Insert the wrapper WHERE the <h1> currently is
     *     (insertBefore puts it in the exact same position in the DOM)
     *   - Move the <h1> INTO the wrapper (appendChild)
     *   - Add our container next to the <h1> inside the wrapper
     */
    function insertInTitlePosition(container) {
        const article = document.querySelector('article');
        const heading = article?.querySelector('h1'); // optional chaining

        if (heading) {
            console.log('AI Assistant: Inserting next to title');

            // Build the flex wrapper div using inline styles (avoids CSS class conflicts)
            const wrapper = document.createElement('div');
            wrapper.style.display        = 'flex';
            wrapper.style.alignItems     = 'flex-start';   // align tops of h1 and button
            wrapper.style.justifyContent = 'space-between'; // push h1 left, button right
            wrapper.style.gap            = '1rem';          // spacing between them
            wrapper.style.marginBottom   = '1rem';

            // 1. Insert the empty wrapper where the <h1> currently is
            heading.parentNode.insertBefore(wrapper, heading);
            // 2. Move the <h1> inside the wrapper
            wrapper.appendChild(heading);
            // 3. Add our widget after the <h1> inside the wrapper
            wrapper.appendChild(container);

            // flex-shrink: 0 ensures the button never gets compressed by a long title
            container.style.flexShrink = '0';

        } else {
            // No <h1> found — fall back to top of article
            if (article) {
                article.insertBefore(container, article.firstChild);
            }
        }
    }

    /* =========================================================================
     * E. EVENT LISTENERS
     * =========================================================================
     *
     * All user interactions are wired here.  Separating event registration
     * from UI construction keeps each function focused on one responsibility.
     *
     * KEY PATTERN — e.stopPropagation()
     * ───────────────────────────────────
     * Click events "bubble" up the DOM tree.  If you click the main button,
     * the event fires on the button AND propagates up to the container AND
     * the document.
     *
     * We have a document-level click listener that CLOSES the dropdown when
     * you click anywhere outside.  Without stopPropagation(), clicking the
     * dropdown button would:
     *   1. Open the dropdown (button click handler)
     *   2. Immediately close it again (document click handler)
     *
     * stopPropagation() prevents the event from reaching the document handler
     * when the click originates from our own buttons.
     * ========================================================================= */
    function setupEventListeners(button, dropdown) {
        const mainButton     = document.getElementById('ai-assistant-button-main');
        const dropdownButton = document.getElementById('ai-assistant-button-dropdown');

        /* ── Main button: copy page immediately ─────────────────────────── */
        mainButton.addEventListener('click', function(e) {
            e.stopPropagation(); // don't bubble up to the document click handler
            handleCopyMarkdown(true); // true = show inline "Copied!" on button
        });

        /* ── Dropdown arrow: toggle menu open / closed ──────────────────── */
        dropdownButton.addEventListener('click', function(e) {
            e.stopPropagation();
            const isOpen = dropdown.style.display !== 'none';
            dropdown.style.display = isOpen ? 'none' : 'block';
            /*
             * aria-expanded must be a string "true" or "false", not a boolean.
             * !isOpen is a boolean, but setAttribute converts it to a string automatically.
             */
            dropdownButton.setAttribute('aria-expanded', !isOpen);
        });

        /* ── Close dropdown when clicking OUTSIDE the widget ────────────── */
        /*
         * document.addEventListener catches EVERY click on the page.
         * We check if the click target is INSIDE our button or dropdown.
         * If not, close the dropdown.
         *
         * button.contains(e.target):
         *   Returns true if e.target is the button itself OR any descendant.
         *   This correctly handles clicking on the icon INSIDE the button.
         */
        document.addEventListener('click', function(e) {
            if (!button.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.style.display = 'none';
                dropdownButton.setAttribute('aria-expanded', 'false');
            }
        });

        /* ── Dropdown menu items ─────────────────────────────────────────── */

        const copyMarkdownBtn = document.getElementById('ai-assistant-copy-markdown');
        if (copyMarkdownBtn) {
            copyMarkdownBtn.addEventListener('click', function() {
                handleCopyMarkdown(false); // false = show toast notification instead
            });
        }

        const viewMarkdownBtn = document.getElementById('ai-assistant-view-markdown');
        if (viewMarkdownBtn) {
            viewMarkdownBtn.addEventListener('click', function() {
                handleViewMarkdown();
            });
        }

        /*
         * AI chat buttons: select all buttons whose ID starts with "ai-assistant-ai-chat-"
         * The CSS attribute selector [id^="prefix"] matches IDs that START WITH the value.
         * querySelectorAll returns a NodeList — we forEach over it like an array.
         *
         * Inside the listener, "this" refers to the button that was clicked.
         * this.dataset.provider was set in createDropdown() when we built the button.
         */
        const aiChatButtons = dropdown.querySelectorAll('[id^="ai-assistant-ai-chat-"]');
        aiChatButtons.forEach(button => {
            button.addEventListener('click', function() {
                const provider = this.dataset.provider;
                handleAIChat(provider);
            });
        });

        /* MCP tool buttons — same pattern as AI chat buttons */
        const mcpButtons = dropdown.querySelectorAll('[id^="ai-assistant-mcp-"]');
        mcpButtons.forEach(button => {
            button.addEventListener('click', function() {
                const toolKey = this.dataset.mcpTool;
                handleMCPInstall(toolKey);
            });
        });

        /* Theme toggle button */
        var themeToggleBtn = document.getElementById('ai-assistant-theme-toggle');
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', function() {
                handleThemeToggle();
            });
        }
    }

    /* =========================================================================
     * F. ACTIONS
     * ========================================================================= */

    /* ── HTML → Markdown Conversion ─────────────────────────────────────────── */
    /**
     * Convert the current page's HTML content to Markdown format.
     *
     * WHY CLONE THE DOM?
     * ───────────────────
     * We need to remove navigation elements, scripts, and our own widget from
     * the Markdown output.  Removing them from the LIVE DOM would destroy the
     * user's page.  By cloning the content first, we operate on a detached
     * copy that's never visible to the user.
     *
     * WHY USE TURNDOWN?
     * ──────────────────
     * HTML is not a good format for pasting into AI assistants — it's full
     * of tags, styles, and metadata.  Markdown is clean, human-readable,
     * and well-understood by all modern AI models.
     *
     * @returns {Promise<string>} The Markdown content.
     */
    async function convertToMarkdown() {
        console.log('AI Assistant: Starting markdown conversion');

        // CONFIG can specify which element contains the main page content.
        // Default: 'article' — the standard semantic HTML element for content.
        const contentSelector = window.AI_ASSISTANT_CONFIG?.content_selector || 'article';
        const content = document.querySelector(contentSelector);

        if (!content) {
            // Throw an Error (not just console.error) so the caller's .catch()
            // handler can show the user a helpful notification.
            throw new Error('Could not find page content to convert');
        }

        /*
         * cloneNode(true) — deep clone.
         * true = clone the element AND all its children (recursive).
         * false would clone only the element itself (empty container).
         */
        const clonedContent = content.cloneNode(true);

        // Remove elements that clutter the Markdown output:
        //   .headerlink  → Sphinx's ¶ anchor links on headings
        //   .ai-assistant-container → our own widget
        //   script, style → executable/style content (not readable)
        //   .sidebar, nav → page navigation (not part of the article)
        const elementsToRemove = [
            '.headerlink',
            '.ai-assistant-container',
            'script',
            'style',
            '.sidebar',
            'nav',
        ];

        elementsToRemove.forEach(selector => {
            /*
             * querySelectorAll returns a NodeList (not an Array, but iterable).
             * forEach works on it directly in modern browsers.
             * el.remove() detaches the element from its parent.
             */
            clonedContent.querySelectorAll(selector).forEach(el => el.remove());
        });

        // Configure Turndown:
        const turndownService = new TurndownService({
            headingStyle:   'atx',    // ATX style: # Heading 1, ## Heading 2
            codeBlockStyle: 'fenced', // fenced: ```python ... ``` (vs indented)
            emDelimiter:    '*',      // *italic* (vs _italic_)
        });

        /*
         * Custom Turndown RULE for <pre> code blocks.
         * Without this, Turndown may miss the language class and produce
         * plain ``` blocks.  This rule explicitly extracts the language
         * from the <code> element's className (e.g. "language-python").
         *
         * Turndown rules: { filter: selector, replacement: fn(content, node) }
         *   filter      → which HTML elements this rule applies to
         *   replacement → function that returns the Markdown equivalent
         */
        turndownService.addRule('preserveCodeBlocks', {
            filter: ['pre'],
            replacement: function(content, node) {
                const code = node.querySelector('code');
                if (code) {
                    // Extract language from class name: "language-python" → "python"
                    const language = code.className.match(/language-(\w+)/);
                    const lang     = language ? language[1] : '';
                    return '\n\n```' + lang + '\n' + code.textContent + '\n```\n\n';
                }
                // No <code> child — return as plain fenced block
                return '\n\n```\n' + content + '\n```\n\n';
            }
        });

        // Run the conversion: clonedContent.innerHTML → Markdown string
        const markdown = turndownService.turndown(clonedContent.innerHTML);
        console.log('AI Assistant: Markdown generated, length:', markdown.length);

        return markdown;
    }

    /* ── Markdown URL Builder ───────────────────────────────────────────────── */
    /**
     * Derive the .md URL for the current page by transforming the .html URL.
     *
     * Sphinx documentation sites can serve Markdown files alongside HTML
     * (generated by the myst-parser extension).  The URL mapping is:
     *
     *   /docs/api.html  → /docs/api.md
     *   /docs/guide/    → /docs/guide/index.md
     *   /docs/guide     → /docs/guide.md
     *
     * WHY STRIP THE HASH (#anchor) FIRST?
     * ─────────────────────────────────────
     * window.location.href includes the anchor fragment:
     *   https://example.com/docs/api.html#some-section
     * We replace only the .html → .md at the path level, not the fragment.
     * split('#')[0] safely removes everything after # (if any).
     */
    function getMarkdownUrl() {
        const currentUrl     = window.location.href;
        const urlWithoutHash = currentUrl.split('#')[0]; // remove #anchor

        if (urlWithoutHash.endsWith('.html')) {
            // Replace trailing .html with .md
            return urlWithoutHash.replace(/\.html$/, '.md');
        } else if (urlWithoutHash.endsWith('/')) {
            // Directory-style URL — append index.md
            return urlWithoutHash + 'index.md';
        } else {
            // No extension — append .md
            return urlWithoutHash + '.md';
        }
    }

    /* ── Open AI Chat ───────────────────────────────────────────────────────── */
    /**
     * Build a prompt URL for the selected AI provider and open it in a new tab.
     *
     * The CONFIG.providers object provides:
     *   prompt_template: "Read this page: {url}\n\nHelp me understand it."
     *   url_template:    "https://claude.ai/new?q={prompt}"
     *
     * We substitute {url} and {prompt} placeholders, URL-encode the prompt,
     * and open the result.  The AI assistant receives the Markdown URL as
     * context — many AI chat interfaces will automatically fetch and read it.
     */
    async function handleAIChat(providerKey) {
        try {
            const providers = window.AI_ASSISTANT_CONFIG?.providers || {};
            const provider  = providers[providerKey];

            if (!provider) {
                showNotification('AI provider configuration not found.', true);
                return;
            }

            const markdownUrl    = getMarkdownUrl();
            const prompt         = provider.prompt_template.replace('{url}', markdownUrl);
            /*
             * encodeURIComponent converts special characters to %-encoding:
             *   "Hello World!" → "Hello%20World%21"
             * This is required before embedding the prompt in a URL parameter.
             */
            const encodedPrompt  = encodeURIComponent(prompt);
            const aiUrl          = provider.url_template.replace('{prompt}', encodedPrompt);

            window.open(aiUrl, '_blank'); // '_blank' opens in a new browser tab
            closeDropdown();

        } catch (error) {
            console.error('AI Assistant: Failed to open AI chat:', error);
            showNotification('Failed to open AI chat. Please try again.', true);
        }
    }

    /* ── View as Markdown ───────────────────────────────────────────────────── */
    function handleViewMarkdown() {
        const markdownUrl = getMarkdownUrl();
        window.open(markdownUrl, '_blank'); // open .md URL in new tab
        closeDropdown();
    }

    /* ── Copy Markdown ──────────────────────────────────────────────────────── */
    /**
     * Convert the page to Markdown and copy it to the clipboard.
     *
     * @param {boolean} showInlineConfirmation
     *   true  → show "Copied!" directly on the main button (for direct-click)
     *   false → show a toast notification (for menu item click)
     */
    function handleCopyMarkdown(showInlineConfirmation) {
        convertToMarkdown()
            .then(markdown => {
                copyToClipboard(markdown, showInlineConfirmation);
                closeDropdown();
            })
            .catch(error => {
                console.error('AI Assistant: Failed to convert to markdown:', error);
                showNotification('Failed to convert page to markdown.', true);
            });
    }

    /* ── Clipboard API ──────────────────────────────────────────────────────── */
    /**
     * Copy text to the clipboard using the modern Clipboard API,
     * with an automatic fallback to the legacy execCommand method.
     *
     * WHY TWO METHODS?
     * ─────────────────
     * navigator.clipboard.writeText() is the modern standard (Chrome 66+, Firefox 63+).
     * It requires HTTPS or localhost and may require user permission.
     *
     * document.execCommand('copy') is the legacy method (IE, old browsers).
     * It works over HTTP and without permissions, but is deprecated.
     * We keep it as a fallback so the feature works on old browsers and
     * non-HTTPS documentation mirrors.
     */
    function copyToClipboard(text, showInlineConfirmation) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text)
                .then(function() {
                    // Success callback
                    if (showInlineConfirmation) {
                        showInlineSuccessState(); // change button to show "Copied!"
                    } else {
                        showNotification('Markdown copied to clipboard!');
                    }
                })
                .catch(function(err) {
                    // Clipboard API failed — try the legacy method
                    console.error('AI Assistant: Clipboard API failed:', err);
                    fallbackCopy(text, showInlineConfirmation);
                });
        } else {
            // Clipboard API not available — use legacy method directly
            fallbackCopy(text, showInlineConfirmation);
        }
    }

    /**
     * Temporarily change the main button to show a success state.
     *
     * Saves the original HTML, swaps in a checkmark + "Copied!" text,
     * then restores the original after 2 seconds.
     *
     * setTimeout(fn, ms) — calls fn after ms milliseconds.
     * This is non-blocking: code after setTimeout runs immediately;
     * fn runs later when the timer expires.
     */
    function showInlineSuccessState() {
        const mainButton = document.getElementById('ai-assistant-button-main');
        if (!mainButton) return; // safety check

        const staticPath = getStaticPath();
        const originalContent = mainButton.innerHTML; // save original HTML

        // Swap button content to show checkmark + "Copied!"
        mainButton.innerHTML = `
            <img src="${staticPath}/checked.svg" class="ai-assistant-icon" aria-hidden="true" alt="">
            <span class="ai-assistant-button-text">Copied</span>
        `;
        mainButton.classList.add('ai-assistant-button-success'); // green tint via CSS

        // After 2 seconds, restore original button state
        setTimeout(function() {
            mainButton.innerHTML = originalContent;
            mainButton.classList.remove('ai-assistant-button-success');
        }, 2000); // 2000 milliseconds = 2 seconds
    }

    /**
     * Legacy fallback clipboard copy using document.execCommand('copy').
     *
     * TECHNIQUE:
     *   1. Create an invisible <textarea> with the text to copy.
     *   2. Append it to <body> so it exists in the DOM.
     *   3. .select() programmatically selects all its text.
     *   4. execCommand('copy') copies the selection to the clipboard.
     *   5. Remove the <textarea> from the DOM.
     *
     * The textarea is made invisible with position:fixed + opacity:0 so the
     * user never sees it appear on screen.
     */
    function fallbackCopy(text, showInlineConfirmation) {
        const textarea      = document.createElement('textarea');
        textarea.value      = text;
        textarea.style.position = 'fixed';  // out of normal flow
        textarea.style.opacity  = '0';      // invisible
        document.body.appendChild(textarea);
        textarea.select(); // select all text in the textarea

        try {
            document.execCommand('copy'); // copy selected text to clipboard
            if (showInlineConfirmation) {
                showInlineSuccessState();
            } else {
                showNotification('Markdown copied to clipboard!');
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
            showNotification('Failed to copy to clipboard.', true);
        } finally {
            /*
             * The finally block always runs, even if an error was thrown above.
             * This guarantees the textarea is removed from the DOM no matter what.
             * Leaving DOM elements behind is a memory/layout leak.
             */
            document.body.removeChild(textarea);
        }
    }

    /* ── MCP Tool Installation ──────────────────────────────────────────────── */
    /**
     * Install or open an MCP (Model Context Protocol) tool.
     *
     * TWO INSTALLATION FLOWS:
     *
     *  'claude_desktop':
     *    Downloads a .mcpb file — a bundle format that Claude Desktop
     *    understands natively.  Creates an invisible <a download> element,
     *    simulates a click, then removes it.
     *
     *  'vscode':
     *    Builds a vscode:mcp/install?... URL that VS Code's deep-link handler
     *    understands.  window.open() with the URL triggers the protocol handler,
     *    which VS Code intercepts and uses to install the MCP server.
     *
     * @param {string} toolKey - Key in CONFIG.mcp_tools (e.g. 'python-docs')
     */
    function handleMCPInstall(toolKey) {
        try {
            const mcpTools = window.AI_ASSISTANT_CONFIG?.mcp_tools || {};
            const tool     = mcpTools[toolKey];

            if (!tool) {
                showNotification('MCP tool configuration not found.', true);
                return;
            }

            /* ── Claude Desktop: download .mcpb file ─────────────────────── */
            if (tool.type === 'claude_desktop') {
                const mcpbUrl = tool.mcpb_url;

                // Extract filename from the URL path: "/files/python.mcpb" → "python.mcpb"
                const urlPath  = new URL(mcpbUrl).pathname; // get the path part only
                const filename = urlPath.split('/').pop();   // last segment after /

                /*
                 * Programmatic download trick:
                 * Create an <a href="..." download="filename"> element, click it,
                 * then immediately remove it.  The browser starts a file download.
                 * The 'download' attribute tells the browser to download instead of navigate.
                 */
                const downloadLink      = document.createElement('a');
                downloadLink.href       = mcpbUrl;
                downloadLink.download   = filename;
                document.body.appendChild(downloadLink);
                downloadLink.click();    // trigger the download
                document.body.removeChild(downloadLink); // clean up

                showNotification('MCP tool download started.');
                closeDropdown();
                return;
            }

            /* ── VS Code: open vscode: deep-link URL ─────────────────────── */
            if (tool.type === 'vscode') {
                // Build the MCP configuration object that VS Code expects
                const mcpConfig = {
                    name: tool.server_name || toolKey,
                    type: tool.transport || 'sse', // Server-Sent Events (default) or stdio
                };

                if (tool.transport === 'stdio') {
                    // stdio transport: VS Code launches a local process
                    mcpConfig.command = tool.command;
                    if (tool.args) mcpConfig.args = tool.args;
                } else {
                    // SSE/HTTP transport: VS Code connects to a URL
                    mcpConfig.url = tool.server_url;
                }

                /*
                 * JSON.stringify converts a JS object to a JSON string:
                 * { name: "python-docs", type: "sse", url: "https://..." }
                 * → '{"name":"python-docs","type":"sse","url":"https://..."}'
                 *
                 * encodeURIComponent makes it safe to embed in a URL.
                 *
                 * vscode:mcp/install?... is VS Code's custom URL scheme.
                 * When the OS sees this URL, it delegates to VS Code.
                 */
                const jsonString = JSON.stringify(mcpConfig);
                const encoded    = encodeURIComponent(jsonString);
                const installUrl = `vscode:mcp/install?${encoded}`;

                window.open(installUrl, '_self'); // '_self' stays in same tab (custom scheme)
                closeDropdown();
                return;
            }

            // Unknown tool type — should never happen if CONFIG is well-formed
            console.error('AI Assistant: Unknown MCP tool type:', tool.type);
            showNotification('Unknown MCP tool type.', true);

        } catch (error) {
            console.error('AI Assistant: Failed to install MCP tool:', error);
            showNotification('Failed to install MCP tool. Please try again.', true);
        }
    }


    /* =========================================================================
     * G. HELPERS
     * ========================================================================= */

    /* ── Close Dropdown ─────────────────────────────────────────────────────── */
    /**
     * Hide the dropdown panel and reset its ARIA state.
     *
     * Called by every menu item action so the menu always closes after an action.
     * Centralising this in one function ensures consistency — no action forgets
     * to close the dropdown.
     */
    function closeDropdown() {
        const dropdown       = document.getElementById('ai-assistant-dropdown');
        const dropdownButton = document.getElementById('ai-assistant-button-dropdown');
        if (dropdown && dropdownButton) {
            dropdown.style.display = 'none';
            dropdownButton.setAttribute('aria-expanded', 'false');
            /*
             * Always set string 'false', not boolean false.
             * aria-expanded is an HTML attribute — its value is always a string.
             * 'false' tells screen readers "this button's popup is now closed".
             */
        }
    }

    /* ── Toast Notification ─────────────────────────────────────────────────── */
    /**
     * Show a temporary status message in the top-right corner of the screen.
     *
     * ANIMATION FLOW:
     *   1. Create element with opacity:0 + translateY(-1rem) (via CSS)
     *   2. Append to <body>
     *   3. After 10ms (one "tick"), add .show class → CSS transition plays
     *      → fades in + slides down
     *   4. After 3 seconds, remove .show → transition plays in reverse
     *      → fades out + slides up
     *   5. After animation finishes (300ms), remove element from DOM
     *
     * WHY THE 10ms DELAY BEFORE ADDING .show?
     * ─────────────────────────────────────────
     * The browser optimises layout: if you append AND add .show in the same
     * synchronous block, the browser may skip the "initial hidden state" and
     * jump straight to the final state — no animation.
     * The 10ms delay forces a "paint frame" between the two operations,
     * ensuring the transition actually plays.
     *
     * @param {string}  message - The text to display.
     * @param {boolean} isError - If true, uses red background instead of blue.
     */
    function showNotification(message, isError = false) {
        const notification = document.createElement('div');
        /*
         * String concatenation builds the class name:
         *   'ai-assistant-notification' + (isError ? ' error' : '')
         *   → 'ai-assistant-notification error'  (if isError)
         *   → 'ai-assistant-notification'         (if not isError)
         */
        notification.className = 'ai-assistant-notification' + (isError ? ' error' : '');
        notification.textContent = message; // textContent is safer than innerHTML (no XSS)
        document.body.appendChild(notification);

        // Step 3: trigger the CSS entrance transition after one paint frame
        setTimeout(() => notification.classList.add('show'), 10);

        // Step 4 & 5: remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show'); // trigger exit transition
            setTimeout(() => notification.remove(), 300); // remove after transition completes
        }, 3000); // 3000ms = 3 seconds visible
    }


    /* =========================================================================
     * INITIALIZATION — DOM Ready Check
     * =========================================================================
     *
     * We must not call initAIAssistant() before the DOM is fully parsed,
     * because we query for <article>, <h1>, sidebar elements, etc.
     * Accessing DOM elements before they exist returns null.
     *
     * document.readyState:
     *   'loading'      → HTML is still being parsed (DOM not complete)
     *   'interactive'  → HTML parsed, but resources (images, CSS) may still load
     *   'complete'     → everything loaded
     *
     * DOMContentLoaded fires at 'interactive' — DOM is ready, safe to query.
     *
     * TWO CASES:
     *   1. Script loads WHILE the page is still parsing (readyState === 'loading')
     *      → add listener for DOMContentLoaded
     *   2. Script loads AFTER the page is already parsed (e.g. async script attribute)
     *      → call initAIAssistant() directly (DOMContentLoaded already fired)
     * ========================================================================= */
    if (document.readyState === 'loading') {
        // DOM not ready yet — wait for it
        document.addEventListener('DOMContentLoaded', initAIAssistant);
    } else {
        // DOM already ready (script loaded async or page already parsed)
        initAIAssistant();
    }

})(); // ← immediately invoke the IIFE
