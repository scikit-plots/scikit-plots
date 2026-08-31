(() => {
    "use strict";

    /*
     * Generic automatic code-copy enhancer.
     *
     * Usage:
     *
     *   <div data-auto-copy>
     *     <pre><code class="language-python">...</code></pre>
     *     <pre><code class="language-bash">...</code></pre>
     *   </div>
     *
     * No IDs, no data-copy-target mappings, and no manually-created buttons.
     *
     * Scope is opt-in through [data-auto-copy], which avoids conflicting with
     * Sphinx/PyData's own copy-button behavior elsewhere in the documentation.
     */

    const ROOT_SELECTOR = "[data-auto-copy]";
    const CODE_SELECTOR = "pre > code";
    const BUTTON_SELECTOR = ".copy-btn[data-copy-auto='true']";

    const COPY_LABEL = "📋 Copy";
    const SUCCESS_LABEL = "✓ Copied";
    const ERROR_LABEL = "Copy failed";

    const RESET_DELAY_MS = 2000;

    const sourceByButton = new WeakMap();
    const resetTimerByButton = new WeakMap();


    function normalizeCodeText(code) {
        let text = code.textContent ?? "";

        /*
         * Remove only one template-formatting newline at either edge.
         * Do not use trim(): leading whitespace can be meaningful in Python,
         * YAML, shell scripts, and other indentation-sensitive formats.
         */
        if (text.startsWith("\n")) {
            text = text.slice(1);
        }

        if (text.endsWith("\n")) {
            text = text.slice(0, -1);
        }

        return text;
    }


    function getLanguageLabel(code) {
        const languageClass = Array.from(code.classList).find(
            (name) => name.startsWith("language-")
        );

        if (!languageClass) {
            return "";
        }

        const language = languageClass.slice("language-".length).trim();

        if (!language) {
            return "";
        }

        return language.charAt(0).toUpperCase() + language.slice(1);
    }


    function getButtonHost(code) {
        const pre = code.closest("pre");

        if (!pre) {
            return null;
        }

        /*
         * Prefer the existing project wrapper so positioning remains local to
         * each code example. Fall back to <pre> for generic HTML.
         */
        return code.closest(".code-content") || pre;
    }


    function createCopyButton(code) {
        const button = document.createElement("button");
        const language = getLanguageLabel(code);

        button.type = "button";
        button.className = "copy-btn";
        button.dataset.copyAuto = "true";
        button.textContent = COPY_LABEL;

        button.setAttribute(
            "aria-label",
            language ? `Copy ${language} code` : "Copy code"
        );

        button.setAttribute(
            "title",
            "Copy code to clipboard"
        );

        sourceByButton.set(button, code);

        return button;
    }


    function enhanceCodeBlock(code) {
        if (!(code instanceof HTMLElement)) {
            return;
        }

        if (
            code.matches("[data-copy-ignore]") ||
            code.closest("[data-copy-ignore]")
        ) {
            return;
        }

        if (code.dataset.copyEnhanced === "true") {
            return;
        }

        const host = getButtonHost(code);

        if (!host) {
            return;
        }

        host.classList.add("copy-btn-host");

        const button = createCopyButton(code);

        host.appendChild(button);

        code.dataset.copyEnhanced = "true";
    }


    function enhanceRoot(root) {
        if (!(root instanceof Element)) {
            return;
        }

        /*
         * If the root itself is a <pre><code> structure contained in an
         * auto-copy region, also allow it to be enhanced.
         */
        if (root.matches(CODE_SELECTOR)) {
            enhanceCodeBlock(root);
        }

        root.querySelectorAll(CODE_SELECTOR).forEach(enhanceCodeBlock);
    }


    function enhancePage() {
        document.querySelectorAll(ROOT_SELECTOR).forEach(enhanceRoot);
    }


    async function fallbackCopyText(text) {
        const textarea = document.createElement("textarea");

        textarea.value = text;
        textarea.setAttribute("readonly", "");

        Object.assign(textarea.style, {
            position: "fixed",
            inset: "0 auto auto 0",
            width: "1px",
            height: "1px",
            opacity: "0",
            pointerEvents: "none",
        });

        document.body.appendChild(textarea);

        textarea.select();
        textarea.setSelectionRange(0, textarea.value.length);

        let copied = false;

        try {
            copied = document.execCommand("copy");
        } finally {
            textarea.remove();
        }

        if (!copied) {
            throw new Error("Clipboard copy is unavailable.");
        }
    }


    async function writeClipboard(text) {
        if (
            navigator.clipboard &&
            typeof navigator.clipboard.writeText === "function"
        ) {
            try {
                await navigator.clipboard.writeText(text);
                return;
            } catch {
                /*
                 * Clipboard API may exist but be blocked by permissions,
                 * protocol, iframe policy, or browser settings.
                 */
            }
        }

        await fallbackCopyText(text);
    }


    function setButtonState(button, label, state) {
        const oldTimer = resetTimerByButton.get(button);

        if (oldTimer) {
            window.clearTimeout(oldTimer);
        }

        button.textContent = label;
        button.dataset.copyState = state;

        const timer = window.setTimeout(() => {
            button.textContent = COPY_LABEL;
            delete button.dataset.copyState;
            resetTimerByButton.delete(button);
        }, RESET_DELAY_MS);

        resetTimerByButton.set(button, timer);
    }


    async function handleCopy(button) {
        const code = sourceByButton.get(button);

        if (!(code instanceof HTMLElement)) {
            setButtonState(button, ERROR_LABEL, "error");
            return;
        }

        const text = normalizeCodeText(code);

        try {
            await writeClipboard(text);
            setButtonState(button, SUCCESS_LABEL, "success");
        } catch (error) {
            console.error("Unable to copy code:", error);
            setButtonState(button, ERROR_LABEL, "error");
        }
    }


    /*
     * Event delegation:
     * one listener supports every current and future generated button.
     */
    document.addEventListener("click", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const button = target.closest(BUTTON_SELECTOR);

        if (!(button instanceof HTMLButtonElement)) {
            return;
        }

        void handleCopy(button);
    });


    function observeDynamicContent() {
        if (!("MutationObserver" in window)) {
            return;
        }

        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (!(node instanceof Element)) {
                        continue;
                    }

                    if (node.matches(ROOT_SELECTOR)) {
                        enhanceRoot(node);
                    }

                    node.querySelectorAll?.(ROOT_SELECTOR).forEach(enhanceRoot);

                    const owningRoot = node.closest?.(ROOT_SELECTOR);

                    if (owningRoot) {
                        enhanceRoot(owningRoot);
                    }

                    if (node.matches?.(WORKSPACE_SELECTOR)) {
                        initializeWorkspace(node);
                    }

                    node
                        .querySelectorAll?.(WORKSPACE_SELECTOR)
                        .forEach(initializeWorkspace);
                }
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
        });
    }



    /* ======================================================================
       Interactive workspace layout controller
       ====================================================================== */

    const WORKSPACE_SELECTOR = "[data-shell-workspace]";
    const VIEW_BUTTON_SELECTOR = "[data-shell-view-button]";
    const PANEL_SELECTOR = "[data-shell-panel]";
    const SPLITTER_SELECTOR = "[data-shell-splitter]";
    const SPLIT_RESET_SELECTOR = "[data-shell-split-reset]";
    const SPLIT_DESCRIPTION_SELECTOR = "[data-shell-split-description]";
    const VALID_WORKSPACE_VIEWS = new Set(["split", "code", "shell"]);

    const DEFAULT_SPLIT_RATIO = 33.33333333;
    const DEFAULT_SPLIT_MIN = 20;
    const DEFAULT_SPLIT_MAX = 80;
    const SPLIT_KEYBOARD_STEP = 2;
    const SPLIT_KEYBOARD_LARGE_STEP = 10;


    function numberFromDataset(value, fallback) {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }


    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }


    function getSplitConfig(workspace) {
        const min = clamp(
            numberFromDataset(
                workspace.dataset.shellSplitMin,
                DEFAULT_SPLIT_MIN
            ),
            5,
            45
        );

        const max = clamp(
            numberFromDataset(
                workspace.dataset.shellSplitMax,
                DEFAULT_SPLIT_MAX
            ),
            55,
            95
        );

        const defaultRatio = clamp(
            numberFromDataset(
                workspace.dataset.shellSplitDefault,
                DEFAULT_SPLIT_RATIO
            ),
            min,
            max
        );

        return { min, max, defaultRatio };
    }


    function readStoredSplitRatio(workspace) {
        if (workspace.dataset.shellSplitPersist !== "true") {
            return null;
        }

        const key = workspace.dataset.shellStorageKey;

        if (!key) {
            return null;
        }

        try {
            const stored = window.localStorage.getItem(key);
            const value = Number.parseFloat(stored);

            return stored !== null && Number.isFinite(value)
                ? value
                : null;
        } catch {
            return null;
        }
    }


    function storeSplitRatio(workspace, ratio) {
        if (workspace.dataset.shellSplitPersist !== "true") {
            return;
        }

        const key = workspace.dataset.shellStorageKey;

        if (!key) {
            return;
        }

        try {
            window.localStorage.setItem(key, String(ratio));
        } catch {
            /* Persistence is optional; current-page resizing still works. */
        }
    }


    function updateSplitDescription(workspace, ratio) {
        const description = workspace.querySelector(
            SPLIT_DESCRIPTION_SELECTOR
        );

        if (!(description instanceof HTMLElement)) {
            return;
        }

        const code = Math.round(ratio);
        const shell = 100 - code;

        description.textContent = `Code ${code}% · Shell ${shell}%`;
    }


    function setSplitRatio(
        workspace,
        requestedRatio,
        { persist = true, announce = true } = {}
    ) {
        if (!(workspace instanceof HTMLElement)) {
            return;
        }

        const config = getSplitConfig(workspace);

        const ratio = clamp(
            numberFromDataset(requestedRatio, config.defaultRatio),
            config.min,
            config.max
        );

        workspace.style.setProperty(
            "--shell-code-percent",
            `${ratio}%`
        );

        workspace.dataset.shellSplitRatio = String(ratio);

        const splitter = workspace.querySelector(SPLITTER_SELECTOR);

        if (splitter instanceof HTMLElement) {
            splitter.setAttribute(
                "aria-valuemin",
                String(Math.round(config.min))
            );
            splitter.setAttribute(
                "aria-valuemax",
                String(Math.round(config.max))
            );
            splitter.setAttribute(
                "aria-valuenow",
                String(Math.round(ratio))
            );
            splitter.setAttribute(
                "aria-valuetext",
                `Code ${Math.round(ratio)} percent, shell ${100 - Math.round(ratio)} percent`
            );
        }

        updateSplitDescription(workspace, ratio);

        if (persist) {
            storeSplitRatio(workspace, ratio);
        }

        if (announce) {
            workspace.dispatchEvent(
                new CustomEvent("shellsplitresize", {
                    bubbles: false,
                    detail: {
                        codePercent: ratio,
                        shellPercent: 100 - ratio,
                    },
                })
            );
        }
    }


    function resetSplitRatio(workspace) {
        setSplitRatio(
            workspace,
            getSplitConfig(workspace).defaultRatio
        );
    }


    function ratioFromPointer(workspace, clientX) {
        const container = workspace.querySelector(
            ".numpy-shell-container"
        );

        if (!(container instanceof HTMLElement)) {
            return null;
        }

        const rect = container.getBoundingClientRect();

        if (rect.width <= 0) {
            return null;
        }

        const isRtl =
            window.getComputedStyle(container).direction === "rtl";

        const offset = isRtl
            ? rect.right - clientX
            : clientX - rect.left;

        return (offset / rect.width) * 100;
    }


    function canResizeSplit(workspace) {
        return (
            workspace.dataset.shellView === "split" &&
            window.matchMedia("(min-width: 720px)").matches
        );
    }


    function startSplitterDrag(workspace, splitter, event) {
        if (
            !canResizeSplit(workspace) ||
            event.button !== 0
        ) {
            return;
        }

        event.preventDefault();

        workspace.dataset.shellResizing = "true";
        splitter.setPointerCapture?.(event.pointerId);

        const apply = (pointerEvent) => {
            const ratio = ratioFromPointer(
                workspace,
                pointerEvent.clientX
            );

            if (ratio === null) {
                return;
            }

            setSplitRatio(workspace, ratio, {
                persist: false,
                announce: false,
            });
        };

        const finish = (pointerEvent) => {
            apply(pointerEvent);

            delete workspace.dataset.shellResizing;

            try {
                splitter.releasePointerCapture?.(
                    pointerEvent.pointerId
                );
            } catch {
                /* Capture may already be released. */
            }

            splitter.removeEventListener("pointermove", apply);
            splitter.removeEventListener("pointerup", finish);
            splitter.removeEventListener("pointercancel", finish);

            setSplitRatio(
                workspace,
                numberFromDataset(
                    workspace.dataset.shellSplitRatio,
                    getSplitConfig(workspace).defaultRatio
                )
            );
        };

        splitter.addEventListener("pointermove", apply);
        splitter.addEventListener("pointerup", finish);
        splitter.addEventListener("pointercancel", finish);

        apply(event);
    }


    function handleSplitterKeydown(workspace, event) {
        if (!canResizeSplit(workspace)) {
            return;
        }

        const config = getSplitConfig(workspace);
        const current = numberFromDataset(
            workspace.dataset.shellSplitRatio,
            config.defaultRatio
        );

        const step = event.shiftKey
            ? SPLIT_KEYBOARD_LARGE_STEP
            : SPLIT_KEYBOARD_STEP;

        const isRtl =
            window.getComputedStyle(workspace).direction === "rtl";

        let next = null;

        switch (event.key) {
            case "ArrowLeft":
                next = current + (isRtl ? step : -step);
                break;
            case "ArrowRight":
                next = current + (isRtl ? -step : step);
                break;
            case "Home":
                next = config.min;
                break;
            case "End":
                next = config.max;
                break;
            default:
                return;
        }

        event.preventDefault();
        setSplitRatio(workspace, next);
    }


    function setPanelVisibility(panel, visible) {
        if (!(panel instanceof HTMLElement)) {
            return;
        }

        panel.hidden = !visible;
        panel.setAttribute("aria-hidden", String(!visible));

        /*
         * inert prevents hidden panel descendants (including iframe content
         * entry points) from remaining keyboard-focusable where supported.
         */
        if ("inert" in panel) {
            panel.inert = !visible;
        }
    }


    function applyWorkspaceView(workspace, requestedView) {
        if (!(workspace instanceof HTMLElement)) {
            return;
        }

        const view = VALID_WORKSPACE_VIEWS.has(requestedView)
            ? requestedView
            : "split";

        const codePanel = workspace.querySelector(
            `${PANEL_SELECTOR}[data-shell-panel="code"]`
        );

        const shellPanel = workspace.querySelector(
            `${PANEL_SELECTOR}[data-shell-panel="shell"]`
        );

        /*
         * Never allow a state where both panels disappear.
         */
        setPanelVisibility(codePanel, view !== "shell");
        setPanelVisibility(shellPanel, view !== "code");

        workspace.dataset.shellView = view;

        const viewMenu = workspace.querySelector("[data-shell-view-menu]");
        const viewTrigger = viewMenu?.querySelector("summary");

        if (viewTrigger instanceof HTMLElement) {
            const currentLabel =
                view.charAt(0).toUpperCase() + view.slice(1);

            viewTrigger.setAttribute(
                "aria-label",
                `Change workspace view. Current view: ${currentLabel}`
            );

            viewTrigger.setAttribute(
                "title",
                `Workspace view: ${currentLabel}`
            );
        }

        workspace.querySelectorAll(VIEW_BUTTON_SELECTOR).forEach((button) => {
            const active = button.dataset.shellViewButton === view;

            button.setAttribute(
                "aria-pressed",
                String(active)
            );
        });

        /*
         * Small public extension hook for future integrations.
         * Example:
         *
         * workspace.addEventListener("shellworkspacechange", (event) => {
         *   console.log(event.detail.view);
         * });
         */
        workspace.dispatchEvent(
            new CustomEvent("shellworkspacechange", {
                bubbles: false,
                detail: { view },
            })
        );
    }


    function initializeWorkspace(workspace) {
        if (!(workspace instanceof HTMLElement)) {
            return;
        }

        if (workspace.dataset.shellWorkspaceReady === "true") {
            return;
        }

        const initialView = workspace.dataset.shellView || "split";
        const config = getSplitConfig(workspace);
        const storedRatio = readStoredSplitRatio(workspace);

        setSplitRatio(
            workspace,
            storedRatio ?? config.defaultRatio,
            {
                persist: false,
                announce: false,
            }
        );

        applyWorkspaceView(workspace, initialView);

        workspace.dataset.shellWorkspaceReady = "true";
        workspace.dataset.shellReady = "true";
    }


    function initializeWorkspaces() {
        document
            .querySelectorAll(WORKSPACE_SELECTOR)
            .forEach(initializeWorkspace);
    }


    /*
     * Event delegation supports any current/future workspace controls.
     */
    document.addEventListener("click", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const button = target.closest(VIEW_BUTTON_SELECTOR);

        if (!(button instanceof HTMLButtonElement)) {
            return;
        }

        const workspace = button.closest(WORKSPACE_SELECTOR);

        if (!(workspace instanceof HTMLElement)) {
            return;
        }

        applyWorkspaceView(
            workspace,
            button.dataset.shellViewButton
        );

        const menu = button.closest("[data-shell-view-menu]");

        if (menu instanceof HTMLDetailsElement) {
            menu.open = false;
        }
    });


    document.addEventListener("pointerdown", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const splitter = target.closest(SPLITTER_SELECTOR);
        const workspace = splitter?.closest(WORKSPACE_SELECTOR);

        if (
            splitter instanceof HTMLElement &&
            workspace instanceof HTMLElement
        ) {
            startSplitterDrag(workspace, splitter, event);
        }
    });


    document.addEventListener("keydown", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const splitter = target.closest(SPLITTER_SELECTOR);
        const workspace = splitter?.closest(WORKSPACE_SELECTOR);

        if (
            splitter instanceof HTMLElement &&
            workspace instanceof HTMLElement
        ) {
            handleSplitterKeydown(workspace, event);
        }
    });


    document.addEventListener("dblclick", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const splitter = target.closest(SPLITTER_SELECTOR);
        const workspace = splitter?.closest(WORKSPACE_SELECTOR);

        if (workspace instanceof HTMLElement) {
            resetSplitRatio(workspace);
        }
    });


    document.addEventListener("click", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const resetButton = target.closest(SPLIT_RESET_SELECTOR);

        if (!(resetButton instanceof HTMLButtonElement)) {
            return;
        }

        const workspace = resetButton.closest(WORKSPACE_SELECTOR);

        if (!(workspace instanceof HTMLElement)) {
            return;
        }

        resetSplitRatio(workspace);

        const menu = resetButton.closest("[data-shell-view-menu]");

        if (menu instanceof HTMLDetailsElement) {
            menu.open = false;
        }
    });


    document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") {
            return;
        }

        document
            .querySelectorAll("[data-shell-view-menu][open]")
            .forEach((menu) => {
                if (menu instanceof HTMLDetailsElement) {
                    menu.open = false;
                }
            });
    });


    function initialize() {
        enhancePage();
        initializeWorkspaces();
        observeDynamicContent();
    }


    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, {
            once: true,
        });
    } else {
        initialize();
    }
})();
