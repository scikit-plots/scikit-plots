"""
Mutant catalogue — the regressions this suite must never stop catching.

Why this file exists
--------------------
A test that passes against a deliberately broken source proves nothing. Over
this project's history that has not been hypothetical: harnesses shipped green
while asserting nothing about the defect they were written for, more than once.

Each entry below is a real defect, either one that was found and fixed or one
that a mutation pass showed the tests would have missed. Applying it must make
a named harness fail. If it does not, the harness has decayed into decoration
and the mutant says so.

Format
------
Each mutant is a dict::

    {
        "id":       short stable identifier,
        "why":      what breaks in the real world if this ships,
        "find":     exact substring in _static/ai-assistant.js (must be unique),
        "replace":  what to put there instead,
        "harness":  the test_*.mjs that must fail as a result,
    }

Scope, and why
--------------
JavaScript only. The harnesses take their target path as ``argv[2]``, so a
mutant can be written to a temp file and checked without touching the working
tree — no copying, no restore step, nothing to leave behind if a run dies
halfway. Python-side mutation would need an importable copy of the package on
``sys.path``; that is doable but it is a different mechanism, and mixing the two
here would make the cheap case pay for the expensive one.

The Python side is not unguarded: ``TestSecretPatternParity`` reads the shipped
JS and fails on drift, which is the cross-language defect that actually
occurred.

Adding a mutant
---------------
When a bug is fixed, add the mutant that reintroduces it. That is the whole
discipline: a fix without a mutant is a fix that can be silently undone.

SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

MUTANTS: list[dict[str, str]] = [
    # ── Containment ───────────────────────────────────────────────────────
    {
        "id": "fence-constant-nonce",
        "why": (
            "A fixed delimiter is guessable by page content authored at any "
            "time, which is exactly what the nonce exists to prevent."
        ),
        "find": "        var nonce = _untrustedNonce();",
        "replace": "        var nonce = 'CTX-fixed';",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "fence-literal-dashes",
        "why": (
            "The original defect: every page with a horizontal rule or a YAML "
            "front-matter example closed the fence early, putting its own text "
            "outside it where it read as instructions."
        ),
        "find": "            '<<<' + nonce + '>>>',\n            body,",
        "replace": "            '---',\n            body,",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "fence-ignores-limit",
        "why": "An unbounded context blows the model's window and the bill.",
        "find": "        var body = text.slice(0, max);",
        "replace": "        var body = text;",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    # ── Neutralisation ────────────────────────────────────────────────────
    {
        "id": "invisible-chars-narrowed",
        "why": (
            "Bidi overrides and joiners are invisible to the reader and plain "
            "text to the model. Covering only U+200B leaves the rest of the "
            "carrier set intact."
        ),
        "find": (
            "/[\\u200B-\\u200F\\u202A-\\u202E\\u2060-\\u2064"
            "\\u2066-\\u2069\\uFEFF]/g"
        ),
        "replace": "/[\\u200B]/g",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "invisible-nodes-not-stripped",
        "why": (
            "Text hidden by CSS or aria-hidden is invisible to the human "
            "reviewing the page and fully visible to the model. That asymmetry "
            "is the attack."
        ),
        "find": "        _stripInvisibleNodes(cloned);\n",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "custom-prompt-unfenced",
        "why": (
            "Substituting raw text into {context} would make the safer path "
            "the one nobody takes."
        ),
        "find": "            ? cfg.panelSystemPrompt.replace('{context}', _fenced)",
        "replace": "            ? cfg.panelSystemPrompt.replace('{context}', _cleaned.text)",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    # ── Egress redaction ──────────────────────────────────────────────────
    {
        "id": "redaction-removed",
        "why": "A key published in a docstring is sent to a third-party proxy.",
        "find": "        var _redacted = _redactSecrets(_cleaned.text);",
        "replace": "        var _redacted = { text: _cleaned.text, findings: [] };",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "redaction-silent",
        "why": (
            "A silent redaction protects the secret and leaves the reader "
            "never learning their key is published on the page."
        ),
        "find": "        _announceRedaction(_redacted.findings);\n",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "redaction-loses-kind",
        "why": (
            "The placeholder tells the model what sort of thing was removed so "
            "it can answer sensibly; a bare marker tells it nothing."
        ),
        "find": "                return '[redacted:' + spec.name + ']';",
        "replace": "                return '[redacted]';",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "secret-pattern-too-loose",
        "why": (
            "A pattern that fires on prose mangles the documentation it is "
            "protecting, and a redactor that does that gets switched off."
        ),
        "find": "{ name: 'openai_key',         re: /\\bsk-[A-Za-z0-9]{20,}\\b/g },",
        "replace": "{ name: 'openai_key',         re: /\\bsk-[A-Za-z0-9]{2,}\\b/g },",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    # ── Detection ─────────────────────────────────────────────────────────
    {
        "id": "injection-threshold-one",
        "why": (
            "One instruction-shaped phrase is ordinary on a page about LLM "
            "security. Flagging on one trains readers to ignore the notice."
        ),
        "find": "    var _INJECTION_THRESHOLD = 3;",
        "replace": "    var _INJECTION_THRESHOLD = 1;",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "injection-override-too-loose",
        "why": (
            "Bare verbs match ordinary API prose. This one escaped the first "
            "mutation pass because the threshold hid it -- which is why the "
            "corpus asserts zero kinds, not merely 'below threshold'."
        ),
        "find": (
            "          re: /\\b(?:ignore|disregard|forget)\\s+(?:all\\s+|any\\s+)?"
            "(?:your\\s+|the\\s+|previous\\s+|prior\\s+|above\\s+)+"
            "(?:previous\\s+|prior\\s+)?(?:instructions?|rules?|prompts?|directions?)\\b/i },"
        ),
        "replace": "          re: /\\b(?:ignore|disregard|forget)\\b/i },",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "injection-exfiltration-rebroadened",
        "why": (
            "A real false positive that shipped briefly: 'print the "
            "instructions for each fold' is ordinary API documentation. The "
            "possessive is what carries the address."
        ),
        "find": "(?:your\\s+(?:system\\s+)?(?:prompt|instructions?|rules?)|the\\s+system\\s+prompt)",
        "replace": "(?:your|the)\\s+(?:system\\s+)?(?:prompt|instructions?|rules?)",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "injection-bypass-rebroadened",
        "why": (
            "The other real false positive: 'Debug mode is enabled with "
            "SKPLT_DEBUG=1' is a sentence every software project contains."
        ),
        "find": (
            "\\b(?:enter|enable|activate|switch\\s+to|go\\s+into)\\s+"
            "(?:developer|debug|god|dan)\\s+mode\\b"
        ),
        "replace": "\\b(?:developer|debug|god)\\s+mode\\b",
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    {
        "id": "injection-summary-alarmist",
        "why": (
            "Telling a reader a tutorial is malicious, and claiming a block "
            "that never happened, teaches them to dismiss the next notice."
        ),
        "find": (
            "             + 'sent as data, not as instructions, and nothing "
            "was removed \\u2014 '"
        ),
        "replace": (
            "             + 'This page may be malicious and the attack was "
            "blocked \\u2014 '"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__untrusted_context.mjs",
    },
    # ── Panel trigger visibility ──────────────────────────────────────────
    {
        "id": "trigger-switch-desync",
        "why": (
            "The reported bug: minimizing forced the pill visible while the "
            "switch still read 'Hidden'."
        ),
        "find": "        _syncPanelTriggerUI(info);\n        return info.pill;",
        "replace": "        return info.pill;",
        "harness": "_static/ai_assistant/test_ai_assistant__panel_trigger.mjs",
    },
    {
        "id": "trigger-minimize-strands",
        "why": (
            "Honouring a hidden preference while minimized strands a live "
            "transcript behind two dropdown clicks."
        ),
        "find": "            pill:       minimized || (panel !== 'open' && preference),",
        "replace": "            pill:       (panel !== 'open' && preference),",
        "harness": "_static/ai_assistant/test_ai_assistant__panel_trigger.mjs",
    },
    # ── Export surfaces ───────────────────────────────────────────────────
    {'id': 'export-duplicate-preview',
     'why': 'The live format-card list must contain each registry format exactly once; duplicating the live '
            'registry recreates duplicate data-fmt cards.',
     'find': '    var _EXPORT_CARD_FORMATS = _EXPORT_FORMATS.concat(_EXPORT_STUB_FORMATS);',
     'replace': '    var _EXPORT_CARD_FORMATS = _EXPORT_FORMATS.concat(_EXPORT_FORMATS);',
     'harness': '_static/ai_assistant/test_ai_assistant__export_formats.mjs'},
    {
        "id": "export-previews-unreachable",
        "why": (
            "disabled + tabindex=-1 removed previews from the reachable "
            "accessibility tree, so the roadmap did not exist for keyboard or "
            "screen-reader users."
        ),
        "find": "        el.setAttribute('aria-disabled', 'true');\n        el.setAttribute('title',",
        "replace": (
            "        el.disabled = true;\n"
            "        el.setAttribute('aria-disabled', 'true');\n"
            "        el.setAttribute('title',"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__export_formats.mjs",
    },
    # ── Reasoning capability ──────────────────────────────────────────────
    {
        "id": "reasoning-default-on",
        "why": (
            "The dangerous direction. A strict endpoint answers an unknown "
            "top-level field with a 400, so guessing 'supported' breaks chat "
            "for every deployment that never opted in."
        ),
        "find": "        if (decl === undefined || decl === null || decl === false) return off;",
        "replace": (
            "        if (decl === false) return off;\n"
            "        if (decl === undefined || decl === null) decl = true;"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__reasoning_support.mjs",
    },
    {
        "id": "discovery-reserved-names-allowed",
        "why": (
            "Without the denylist a compromised proxy can name 'messages' as "
            "its effort parameter and rewrite every request body."
        ),
        "find": "        if (_CAPS_RESERVED_PARAMS.indexOf(name) !== -1) return null;\n",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__reasoning_support.mjs",
    },
    {
        "id": "budget-slider-ignores-support",
        "why": (
            "The stored 'thinking on' preference survives a model switch, so "
            "an unsupported endpoint gets a live slider controlling nothing."
        ),
        "find": (
            "            var live = thinkingOn && _support.thinking && budgetMode;"
        ),
        "replace": "            var live = thinkingOn && budgetMode;",
        "harness": "_static/ai_assistant/test_ai_assistant__reasoning_support.mjs",
    },
    # ── Per-model live resolution ─────────────────────────────────────────
    {
        "id": "sheet-support-resolved-once",
        "why": (
            "Support is a property of the active model, which can change while "
            "the sheet is open. Dropping the model-change listener leaves "
            "Effort and Thinking showing the previous model's state: a model "
            "that accepts these settings looks inert, and one that does not "
            "looks live and silently discards them."
        ),
        "find": (
            "        (typeof _assistantEvents !== 'undefined' ? _assistantEvents : document).addEventListener('ai-assistant-model-change', "
            "_applyReasoningUI);"
        ),
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__reasoning_support.mjs",
    },
    {
        "id": "support-flag-is-a-latch",
        "why": (
            "A one-way write sets the inert flag but never clears it, so "
            "switching from an unsupporting model to a supporting one leaves "
            "the control greyed out forever."
        ),
        "find": "            effortSeg.dataset.unsupported = support.effort ? 'false' : 'true';",
        "replace": "            if (!support.effort) effortSeg.dataset.unsupported = 'true';",
        "harness": "_static/ai_assistant/test_ai_assistant__reasoning_support.mjs",
    },

    # ── Effort levels ─────────────────────────────────────────────────────
    {
        "id": "effort-id-unvalidated",
        "why": (
            "An unknown stored id left the segmented control with no radio "
            "checked and a blank description, with no way back short of "
            "clearing storage."
        ),
        "find": "        return _effortById(raw).id;",
        "replace": "        return raw || _EFFORT_DEFAULT;",
        "harness": "_static/ai_assistant/test_ai_assistant__effort_levels.mjs",
    },
    {
        "id": "effort-truthiness-report",
        "why": (
            "Membership, not truthiness: a field sent as '' or 0 WAS sent, and "
            "reporting it absent sends a maintainer hunting a client bug that "
            "does not exist."
        ),
        "find": "    var _EFFORT_DEFAULT = 'high';",
        "replace": "    var _EFFORT_DEFAULT = 'medium';",
        "harness": "_static/ai_assistant/test_ai_assistant__effort_levels.mjs",
    },
    # ── Model overrides ───────────────────────────────────────────────────
    {
        "id": "override-rejection-clears-field",
        "why": (
            "A typo'd endpoint such as 'javascript:alert(1)' sanitises to '' "
            "and would be stored as an override that CLEARS the endpoint, "
            "leaving the model pointing nowhere -- worse than the typo, and "
            "invisible to the reader."
        ),
        "find": (
            "                if (typeof supplied === 'string' && supplied.trim() !== '' &&\n"
            "                    full[key] === '') {\n"
            "                    continue;\n"
            "                }"
        ),
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "override-replaces-instead-of-diffing",
        "why": (
            "Storing the whole model instead of a diff freezes it at the "
            "version it was overridden from, so later conf.py fixes never "
            "reach the reader."
        ),
        "find": "                if (!Object.prototype.hasOwnProperty.call(src, key)) continue;",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "override-mutates-the-source",
        "why": (
            "Mutating the build-time entry makes the diff unshowable and the "
            "original unrecoverable, so 'reset' cannot restore anything."
        ),
        "find": "                var merged = {};",
        "replace": "                var merged = m;",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "active-model-skips-overrides",
        "why": (
            "The request would go to the endpoint the reader just corrected "
            "away from -- the bug the feature exists to fix, one layer down."
        ),
        "find": "        var models = _MODEL_STORE.applyOverrides(builtins).filter(function (m) {",
        "replace": "        var models = builtins.filter(function (m) {",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "reasoning-decl-reserved-names",
        "why": (
            "Without the denylist a per-model declaration typed into a text "
            "field can name 'messages' as its effort parameter and rewrite "
            "every request body that model sends."
        ),
        "find": "                if (reserved.indexOf(name) !== -1) return null;",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },

    {
        "id": "edit-builtin-rewrites-instead-of-diffing",
        "why": (
            "Rewriting a build-time model as a custom entry loses the link to "
            "conf.py: later upstream fixes never reach the reader, and reset "
            "has nothing to restore."
        ),
        "find": "                    ? _MODEL_STORE.setOverride(_editingId, patch)",
        "replace": "                    ? _MODEL_STORE.addModel(_editingId, patch)",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "edit-does-not-announce",
        "why": (
            "Without the event the correction sits in storage while every "
            "surface keeps showing the old value until the sheet is reopened "
            "-- which is the rebuild-to-see-it problem this feature removes."
        ),
        "find": "                        { detail: { reason: 'model-edited', id: editedId } }));",
        "replace": "                        { detail: {} }));",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "edit-id-stays-editable",
        "why": (
            "The id keys the override and matches the radio row. Editing it "
            "silently creates a second entry instead of correcting the first."
        ),
        "find": "            idInp.disabled = true;",
        "replace": "            idInp.disabled = false;",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },
    {
        "id": "edit-click-selects-the-model",
        "why": (
            "The button lives inside a <label>; without preventDefault a click "
            "on 'edit' also switches the active model."
        ),
        "find": "            function _requestModelEdit(ev) {\n                if (ev) {\n                    ev.preventDefault();\n                    ev.stopPropagation();",
        "replace": "            function _requestModelEdit(ev) {\n                if (ev) {\n                    ev.stopPropagation();",
        "harness": "_static/ai_assistant/test_ai_assistant__model_overrides.mjs",
    },

    {
        "id": "custom-section-use-before-create",
        "why": (
            "The reported crash: `var` hoists the binding but not the "
            "assignment, so appending an element declared further down passes "
            "`undefined` to appendChild and the whole model sheet fails to "
            "build. node --check passes such a file and every source-text "
            "assertion passes too -- only executing the function catches it."
        ),
        "find": (
            "        var cancelBtn = document.createElement('button');\n"
            "        cancelBtn.type = 'button';"
        ),
        "replace": "        var cancelBtn;\n        void 0;",
        "harness": "_static/ai_assistant/test_ai_assistant__custom_section_dom.mjs",
    },

    # ── Composition wiring ────────────────────────────────────────────────
    # These are caught by tests/test_context_roundtrip.py rather than a .mjs
    # harness, so they are recorded there rather than here; see the docstring
    # of TestCompositionMatchesProduction for why a fixture alone missed them.

    # ── Menu shortcuts ────────────────────────────────────────────────────
    {
        "id": "shortcuts-menu-scoped",
        "why": (
            "The reported bug: a popover-scoped listener died the moment an "
            "item opened a sheet, while its keycaps stayed on screen."
        ),
        "find": (
            "        if (!panel || !pop) return;\n"
            "        panel.addEventListener('keydown', function (e) {"
        ),
        "replace": (
            "        if (!panel || !pop) return;\n"
            "        pop.addEventListener('keydown', function (e) {"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__menu_shortcuts.mjs",
    },
    {
        "id": "shortcuts-eat-typing",
        "why": (
            "Without the text-entry guard, typing 'model' in the composer "
            "opens four sheets and deletes the conversation."
        ),
        "find": "            if (_isTextEntryTarget(e.target)) return;\n",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__menu_shortcuts.mjs",
    },
    {
        "id": "shortcuts-skip-confirm",
        "why": (
            "A destructive action one keypress away, with focus placed on the "
            "menu automatically, must not be reachable without confirming."
        ),
        "find": (
            "            if (spec.key) { accelerators[spec.key.toUpperCase()] "
            "= activate; }"
        ),
        "replace": (
            "            if (spec.key) { accelerators[spec.key.toUpperCase()] = "
            "function () { pop.setAttribute('data-open','false'); handler(); }; }"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__menu_shortcuts.mjs",
    },

    # ── Unified Share conversation ────────────────────────────────────────
    {'id': 'share-panels-eager-again',
     'why': 'Switching formats must replace the one descriptor panel rather than accumulating duplicate '
            'per-format DOM surfaces inside the unified sheet.',
     'find': '            while (formatHost.firstChild) formatHost.removeChild(formatHost.firstChild);',
     'replace': '            void formatHost.firstChild;',
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs'},
    {
        "id": "share-dispatch-skips-format-selection",
        "why": (
            "Opening the unified sheet without selecting the clicked format "
            "makes a JSON click show whichever format was previously active. "
            "One shell is only correct if dispatch still preserves user intent."
        ),
        "find": "            if (!convShareSheet._selectExportFormat(fmt)) { return; }",
        "replace": "            if (!convShareSheet) { return; }",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {'id': 'share-new-chat-does-not-reset-cached-panels',
     'why': 'New chat must clear active result state without discarding page-memory Global edit '
            'capabilities needed to revoke links created by the previous conversation.',
     'find': '            resultState = null;\n            _globalShareState = null;',
     'replace': '            managedArtifacts = [];\n'
                '            resultState = null;\n'
                '            _globalShareState = null;',
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs'},
    {'id': 'share-permanent-save-crosses-conversation',
     'why': 'A delayed privacy decision for a self-contained/local share must remain bound to the '
            'conversation that opened the dialog and must not publish after New chat.',
     'find': "            if (reviewed.action === 'cancel' || opConversationId !== boundConversationId || "
             'opConversationId !== _getConversationId()) return null;',
     'replace': "            if (reviewed.action === 'cancel') return null;",
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs'},
    {'id': 'share-global-save-crosses-conversation',
     'why': 'A delayed Global Share response belongs to the conversation that created it; removing the '
            'success identity guard can attach old server state to a new chat.',
     'find': '            function success(res) {\n'
             '                if (opConversationId !== boundConversationId || opConversationId !== '
             '_getConversationId()) return;',
     'replace': '            function success(res) {\n                void opConversationId;',
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation.mjs'},
    {
        "id": "share-export-mode-observers-not-notified",
        "why": (
            "There are many Download/Share controls (main header plus sheet "
            "toolbars). If the setter stops notifying observers, only the "
            "control clicked by the reader appears current and the others lie."
        ),
        "find": "        _notifyExportState();",
        "replace": "        void _exportLinkMode;",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {
        "id": "share-export-mode-singleton-id-returns",
        "why": (
            "Giving every cloned export-mode control the same id recreates the "
            "original stale-toolbar bug and invalid duplicate-id DOM. Controls "
            "must be observer views, never singleton-id authorities."
        ),
        "find": (
            "        var row = document.createElement('button');\n"
            "        row.type = 'button';\n"
            "        row.className = (options.rowClass || '') +"
        ),
        "replace": (
            "        var row = document.createElement('button');\n"
            "        row.type = 'button';\n"
            "        row.id = 'ai-assistant-export-link-toggle';\n"
            "        row.className = (options.rowClass || '') +"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {'id': 'share-conversation-id-follows-trimmed-head',
     'why': 'Transcript position is not conversation identity. The unified Share sheet must bind to the '
            'explicit conversation UUID so trimming cannot change ownership.',
     'find': '        var boundConversationId = _getConversationId();',
     'replace': "        var boundConversationId = _transcript.length ? String(_transcript[0].ts) : '';",
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation.mjs'},
    {
        "id": "share-unified-sheet-missing-from-registry",
        "why": (
            "The sheet registry is the single dependency map for open/close, "
            "Escape, toolbar, and focus wiring. Omitting Share from it means "
            "one of those cross-sheet behaviors silently stops covering Share."
        ),
        "find": (
            "            { key: 'conversation-share', sheet: convShareSheet,   "
            "toolbarId: 'conv-share' },"
        ),
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {'id': 'share-session-copy-overstates-privacy',
     'why': 'Self-contained links are embedded readable data, not encrypted or remotely revocable. UI copy '
            'must not promise stronger privacy/lifecycle semantics.',
     'find': "            'Reviewed static HTML · not encrypted · copied links cannot be revoked');",
     'replace': "            'Reviewed static HTML · encrypted · removable everywhere');",
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation.mjs'},

    # ── Client secret lifecycle boundary (B18 Run 1) ───────────────────
    {
        "id": "endpoint-token-persisted-again",
        "why": (
            "Endpoint bearer tokens are intentionally page-memory-only. "
            "Putting a token field back into the localStorage serializer makes "
            "the credential recoverable by any same-origin script and recreates "
            "the exact contradiction B18 closed."
        ),
        "find": (
            "                        datasetRepo: p.datasetRepo || '',\n"
            "                        ttlDays:     p.ttlDays     || 30,"
        ),
        "replace": (
            "                        datasetRepo: p.datasetRepo || '',\n"
            "                        shareToken:  p.shareToken  || '',\n"
            "                        ttlDays:     p.ttlDays     || 30,"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__endpoint_secret_lifecycle.mjs",
    },
    {
        "id": "endpoint-legacy-token-storage-not-scrubbed",
        "why": (
            "Ignoring legacy token fields in memory is insufficient: the raw "
            "v1/v2 localStorage blob still contains the credential until it is "
            "rewritten. Removing the migration rewrite must be caught."
        ),
        "find": "            if (needsRewrite) _persistCustom();",
        "replace": "            void needsRewrite;",
        "harness": "_static/ai_assistant/test_ai_assistant__endpoint_secret_lifecycle.mjs",
    },

    {'id': 'share-unified-sheet-host-use-before-create',
     'why': 'The unified format host must exist before rendering the active descriptor; source-only checks '
            'cannot catch a browser construction crash.',
     'find': "        var formatHost = document.createElement('div');\n"
             "        formatHost.className = 'ai-assistant-conv-share-format-host';",
     'replace': '        var formatHost;\n        void 0;',
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs'},

    # ── Export / Share active-content isolation (B18 Run 2) ─────────────
    {
        "id": "export-html-raw-json-breakout",
        "why": (
            "JSON.stringify output is not safe inside an HTML script raw-text "
            "element. Restoring it allows a conversation containing </script> "
            "to terminate the inert JSON block and create executable markup."
        ),
        "find": "            jsonPayload: _jsonForHtmlRawText(snap, 2),",
        "replace": "            jsonPayload: JSON.stringify(snap, null, 2),",
        "harness": "_static/ai_assistant/test_ai_assistant__active_content_isolation.mjs",
    },
    {
        "id": "export-source-url-unsanitized",
        "why": (
            "Raw location.href can contain URL credentials, access tokens in "
            "query/fragment data, or a local filesystem path. The canonical "
            "snapshot must redact it before every serializer sees it."
        ),
        "find": "        var pageUrl   = _sanitizePage(rawPage);",
        "replace": "        var pageUrl   = rawPage;",
        "harness": "_static/ai_assistant/test_ai_assistant__active_content_isolation.mjs",
    },
    {
        "id": "share-c1-html-executable-again",
        "why": (
            "Legacy c1 payloads are attacker-controlled bytes. Serving a c1 "
            "HTML payload as text/html recreates the same-origin arbitrary-HTML "
            "execution gadget closed by Run 2."
        ),
        "find": (
            "            var legacyMime = legacyFmt === 'json'\n"
            "                ? 'application/json;charset=utf-8'\n"
            "                : 'text/plain;charset=utf-8';"
        ),
        "replace": (
            "            var legacyMime = legacyFmt === 'json'\n"
            "                ? 'application/json;charset=utf-8'\n"
            "                : legacyFmt === 'html' ? 'text/html;charset=utf-8'\n"
            "                : 'text/plain;charset=utf-8';"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__active_content_isolation.mjs",
    },
    {
        "id": "share-c2-loses-structured-envelope",
        "why": (
            "c2 exists specifically so self-contained links carry structured "
            "data instead of rendered HTML. Removing the schema marker makes "
            "the transport contract indistinguishable from arbitrary content."
        ),
        "find": "            share_schema: 'c2',",
        "replace": "            share_schema: 'raw',",
        "harness": "_static/ai_assistant/test_ai_assistant__active_content_isolation.mjs",
    },

    {
        "id": "share-c2-skips-canonicalization",
        "why": (
            "Shape validation alone still lets attacker-controlled unknown and "
            "nested fields ride through the envelope. The decoder must rebuild "
            "a known-field snapshot and re-sanitize source metadata."
        ),
        "find": "            var normalized = _normalizeShareSnapshot(env.snapshot);",
        "replace": "            var normalized = env.snapshot;",
        "harness": "_static/ai_assistant/test_ai_assistant__active_content_isolation.mjs",
    },

    # ── Global Share server capability boundary (B18 Run 3) ─────────────
    {'id': 'global-share-client-mime-authority-returns',
     'why': 'Global Share must send canonical snapshot + allowlisted format only; restoring client MIME '
            'lets direct callers regain representation authority.',
     'find': '            var payload = recoveringGlobal ? _pendingGlobalCreate.payload : { snapshot: snapshot, format: meta.fmt, ttlDays: g.ttlDays };',
     'replace': '            var payload = recoveringGlobal ? _pendingGlobalCreate.payload : { snapshot: snapshot, format: meta.fmt, mimeType: meta.mime, '
                'ext: meta.ext, ttlDays: g.ttlDays };',
     'harness': '_cf_worker/test_index__global_share_capability.mjs'},
    {'id': 'global-share-patch-uses-endpoint-token',
     'why': 'Share update ownership is the per-share edit capability, never the endpoint create credential.',
     'find': "                _patchGlobalShare(base, _globalShareState.uuid, _globalShareState.editToken,",
     'replace': "                _patchGlobalShare(base, _globalShareState.uuid, g.token,",
     'harness': '_cf_worker/test_index__global_share_capability.mjs'},
    {'id': 'global-share-edit-token-persisted',
     'why': 'The Global edit capability is intentionally page-memory-only; persisting it enlarges '
            'same-origin credential exposure.',
     'find': "                conversationId: state.conversationId || '',\n"
             "                format: state.format || '',",
     'replace': "                conversationId: state.conversationId || '',\n"
                "                editToken: state.editToken || '',\n"
                "                format: state.format || '',",
     'harness': '_cf_worker/test_index__global_share_capability.mjs'},


    # ── Run 4: server-owned prompt authority ──────────────────────────────
    {
        "id": "chat-contract-negotiation-bypassed",
        "why": "If the advertised contract no longer controls the structured path, the bundled proxy receives legacy client-authored system authority or custom endpoints receive an incompatible body.",
        "find": "        var useStructuredProxy = (proxyContract === _CHAT_CONTRACT_V1 ||",
        "replace": "        var useStructuredProxy = false && (",
        "harness": "_cf_worker/test_index__chat_authority.mjs",
    },
    {
        "id": "chat-history-sent-under-legacy-contract",
        "why": "History must ride only on the contract that declares it. Sending it under v1 means the server never fences it as untrusted evidence, so a forged assistant turn reaches the model with role authority.",
        "find": "            if (proxyContract === _CHAT_CONTRACT_V2) {",
        "replace": "            if (true) {",
        "harness": "_cf_worker/test_index__chat_authority.mjs",
    },
    {
        "id": "content-revision-advanced-by-preview-eviction",
        "why": "Losing a preview is not an edit. If eviction advances the content revision, reader-facing numbers imply changes that never happened and a current answer is rejected as stale.",
        "find": "            contentRevision: _artifactContentRevision(old),\n            state: state,",
        "replace": "            contentRevision: _artifactContentRevision(old) + 1,\n            state: state,",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "file-preview-collapse-runs-per-chunk",
        "why": "Collapsing must happen once at finalization. Moving it into the per-chunk sync re-wraps a streaming file on every chunk, nesting disclosures and detaching the pre the ledger registered.",
        "find": "    function _syncExplicitCodeArtifacts(root, st) {",
        "replace": "    function _syncExplicitCodeArtifacts(root, st) {\n        _collapseArtifactPreBlocks(root);",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "tab-switcher-widens-the-sheet",
        "why": "Without the container-relative minimum a single column exceeds a narrow panel, so the switcher overflows at exactly the width wrapping exists to handle.",
        "find": "    grid-template-columns: repeat(auto-fit, minmax(min(6rem, 100%), 1fr));",
        "replace": "    grid-template-columns: repeat(auto-fit, minmax(6rem, 1fr));",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
        "target": "css",
    },
    {
        "id": "escape-ladder-documentation-drifts",
        "why": "A rung that swallows Escape without being documented makes the key look broken: the reader sees the panel stay open and does not notice that a filter they were not looking at was cleared instead.",
        "find": "then clears an active search filter, then closes the lightest open menu",
        "replace": "then closes the lightest open menu",
        "harness": "_static/ai_assistant/test_ai_assistant__escape_ladder.mjs",
    },
    {
        "id": "escape-swallowed-with-no-active-filter",
        "why": "Guarding on the key alone swallows Escape whenever focus is in a search box, so an unfiltered sheet becomes impossible to leave from its own filter field.",
        "find": "            if ((e.key === 'Escape' || e.keyCode === 27) && _query) {\n                e.stopPropagation();\n                _clearFilter();",
        "replace": "            if (e.key === 'Escape' || e.keyCode === 27) {\n                e.stopPropagation();\n                _clearFilter();",
        "harness": "_static/ai_assistant/test_ai_assistant__escape_ladder.mjs",
    },
    {
        "id": "first-message-orphans-the-toggle",
        "why": "Hiding the banner leaves the collapse toggle behind on its own: a control whose only purpose is to show and hide something no longer there, and expanding it yields an empty row.",
        "find": "            row.setAttribute('data-collapsed', 'true');",
        "replace": "            row.querySelector('#ai-assistant-panel-speak-banner').style.display = 'none';",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
    },
    {
        "id": "cleared-conversation-leaves-the-hint-collapsed",
        "why": "A cleared conversation is a fresh start and the hint should return as it was on first load. Leaving it collapsed means the reader never sees the shortcut again in that session.",
        "find": "            speakRowEl.setAttribute('data-collapsed', 'false');",
        "replace": "            void speakRowEl;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
    },
    {
        "id": "collapsed-pill-not-lifted-out-of-its-box",
        "why": "The positive-height row is intentionally offset by an equal negative block margin. Removing that offset makes the floating hint consume a new layout row and pushes the composer instead of sharing the transcript reservation.",
        "find": "    min-height: 2rem;\n    height: auto;\n    margin: -2rem 0.75rem 0;",
        "replace": "    min-height: 2rem;\n    height: auto;\n    margin: 0 0.75rem;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
        "target": "css",
    },
    {
        "id": "floating-hint-covers-the-last-line",
        "why": "A control floating over the transcript sits on the final line of an answer unless the transcript reserves room for it. The space saved is real only if what it costs is paid.",
        "find": "    padding-bottom: 2.75rem;",
        "replace": "    padding-bottom: 0;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
        "target": "css",
    },
    {
        "id": "expanded-toggle-has-no-ground",
        "why": "Collapsed and expanded states must retain the same owned surface. Removing the semantic ground makes the chevron depend on whatever transcript content happens to sit behind the floating row.",
        "find": "    background-color: var(--ai-speak-toggle-surface);\n    flex: 0 0 auto;",
        "replace": "    background-color: transparent;\n    flex: 0 0 auto;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
        "target": "css",
    },
    {
        "id": "collapsed-hint-parks-in-the-far-corner",
        "why": "Collapsed the row is a single small target whose only purpose is to be pressed. At the leading edge it sits in the far corner for a right-handed one-handed grip, which is the hardest place on a phone to reach.",
        "find": ".ai-assistant-panel-speak-row[data-collapsed=\"true\"] { justify-content: flex-end; }",
        "replace": ".ai-assistant-panel-speak-row[data-collapsed=\"true\"] { justify-content: flex-start; }",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
        "target": "css",
    },
    {
        "id": "speak-hint-collapse-not-remembered",
        "why": "Without applying the stored state at build time the hint reopens expanded after every answer, so a reader who collapsed it collapses it again and again.",
        "find": "            _applySpeakCollapsed(_ssGet(_SPEAK_HINT_COLLAPSED_KEY) === '1');",
        "replace": "            _applySpeakCollapsed(false);",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
    },
    {
        "id": "speak-toggle-label-states-only-its-state",
        "why": "A control announcing its current state leaves a screen-reader user guessing what activating it does. The label must name the action the next press performs.",
        "find": "                    collapsed ? 'Show the speak hint' : 'Collapse the speak hint');",
        "replace": "                    collapsed ? 'Speak hint hidden' : 'Speak hint shown');",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
    },
    {
        "id": "toolbar-dropdown-borrows-speak-token",
        "why": "The toolbar dropdown is outside the speak-toggle subtree. Borrowing that scoped token makes background-color invalid at computed-value time and the menu becomes transparent.",
        "find": "    background-color: var(--ai-assistant-dropdown-surface, #fff);",
        "replace": "    background-color: var(--ai-speak-toggle-surface);",
        "harness": "_static/ai_assistant/test_ai_assistant__toolbar_dropdown_surface.mjs",
        "target": "css",
    },
    {
        "id": "toolbar-dropdown-background-has-two-colors",
        "why": "background-color accepts one color. Appending a hard fallback after var() invalidates the whole declaration and the browser falls back to transparent.",
        "find": "    background-color: var(--ai-assistant-dropdown-surface, #fff);",
        "replace": "    background-color: var(--pst-color-surface, var(--color-background-primary, #fff))\n    #29313d;",
        "harness": "_static/ai_assistant/test_ai_assistant__toolbar_dropdown_surface.mjs",
        "target": "css",
    },
    {
        "id": "toolbar-dropdown-dark-theme-contract-narrows",
        "why": "Real PyData pages use data-theme/data-mode as well as Bootstrap theme state. Narrowing the selector can drop the explicit dark fallback on deployed docs.",
        "find": ":is([data-theme=\"dark\"], [data-mode=\"dark\"], [data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-dropdown {",
        "replace": "[data-bs-theme=\"dark\"] .ai-assistant-dropdown {",
        "harness": "_static/ai_assistant/test_ai_assistant__toolbar_dropdown_surface.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-mobile-ground-erased",
        "why": "A later background shorthand erases the resting surface. Desktop hover can hide the defect, but touch has no dependable hover and the tiny chevron can disappear into transcript content.",
        "find": "    border-radius: 999px;\n    -webkit-appearance: none;",
        "replace": "    border-radius: 999px;\n    background: transparent;\n    -webkit-appearance: none;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_mobile_visibility.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-mobile-resting-color-muted",
        "why": "Touch cannot rely on hover/focus to boost contrast. Falling back to the muted token can make the chevron indistinguishable from its surface on host themes.",
        "find": "        color: var(--ai-speak-toggle-ink);\n        min-width: 2rem;",
        "replace": "        color: var(--pst-color-text-muted, #71717a);\n        min-width: 2rem;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_mobile_visibility.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-svg-loses-current-color",
        "why": "The chevron is an SVG stroke icon. If its stroke is no longer tied to currentColor, mobile theme/touch contrast rules can change the button color without changing the visible glyph.",
        "find": "    color: var(--ai-speak-toggle-ink);\n    fill: none;\n    stroke: var(--ai-speak-toggle-ink);",
        "replace": "    color: var(--ai-speak-toggle-ink);\n    fill: none;\n    stroke: none;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_mobile_visibility.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-sticky-hover-inherits-foreground",
        "why": "Touch browsers can keep :hover after a tap. If the interaction state uses inherited color, that higher-specificity state can override the touch resting color and make the chevron disappear while the button remains clickable.",
        "find": "    /* Never inherit here: touch browsers can keep :hover after a tap. */\n    color: var(--ai-speak-toggle-ink);",
        "replace": "    /* Regressed: sticky hover falls back to container color. */\n    color: inherit;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_sticky_hover_visibility.mjs",
        "target": "css",
    },
    {
        "id": "speak-banner-icon-returns-to-theme-primary",
        "why": "A host theme may use a primary accent close to the speak-control surface. The mic icon must use the readable foreground authority rather than assuming the theme accent always contrasts.",
        "find": "    /* Visibility first: host primary colours can equal a custom surface. */\n    color: var(--pst-color-text-base, var(--color-foreground-primary, #1f2328));",
        "replace": "    /* Regressed: accent may match the host surface. */\n    color: var(--pst-color-primary, var(--color-brand-primary, #2980b9));",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_sticky_hover_visibility.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-interaction-drops-active-state",
        "why": "The pressed state participates in the mobile tap lifecycle. Dropping :active leaves a window where browser/native state can paint the control differently from its rest and sticky-hover states.",
        "find": ".ai-assistant-panel-speak-toggle:hover,\n.ai-assistant-panel-speak-toggle:focus-visible,\n.ai-assistant-panel-speak-toggle:active {",
        "replace": ".ai-assistant-panel-speak-toggle:hover,\n.ai-assistant-panel-speak-toggle:focus-visible {",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_sticky_hover_visibility.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-hardware-row-zero-height",
        "why": "A zero-height flex container with transformed overflowing children can remain hit-testable while a mobile/WebKit compositor drops the painted SVG layer.",
        "find": "    min-height: 2rem;\n    height: auto;\n    margin: -2rem 0.75rem 0;",
        "replace": "    min-height: 0;\n    height: 0;\n    margin: 0 0.75rem;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_hardware_paint_stability.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-hardware-child-translate-restored",
        "why": "Restoring the -100% child translation recreates the nested transform/compositing path that can paint differently from its still-active hit target on mobile hardware.",
        "find": ".ai-assistant-panel-speak-row > * { transform: none; }",
        "replace": ".ai-assistant-panel-speak-row > * { transform: translateY(-100%); }",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_hardware_paint_stability.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-hardware-ink-not-on-surface",
        "why": "The toggle is painted on the surface token, so its icon must use the matching on-surface authority rather than a generic or muted host text colour.",
        "find": "        --pst-color-on-surface,\n        var(--pst-color-text-base, var(--color-foreground-primary, #222832))",
        "replace": "        --pst-color-text-muted,\n        var(--color-foreground-secondary, #71717a)",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_hardware_paint_stability.mjs",
        "target": "css",
    },
    {
        "id": "speak-toggle-hardware-pydata-dark-contract-forgotten",
        "why": "The captured production page uses data-theme/data-mode=dark. Supporting only Bootstrap/.dark selectors leaves hardware-specific fallbacks disconnected from the actual host theme contract.",
        "find": ":is([data-theme=\"dark\"], [data-mode=\"dark\"], [data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-panel-speak-toggle:hover,\n:is([data-theme=\"dark\"], [data-mode=\"dark\"], [data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-panel-speak-toggle:focus-visible,\n:is([data-theme=\"dark\"], [data-mode=\"dark\"], [data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-panel-speak-toggle:active",
        "replace": ":is([data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-panel-speak-toggle:hover,\n:is([data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-panel-speak-toggle:focus-visible,\n:is([data-bs-theme=\"dark\"], .dark)\n    .ai-assistant-panel-speak-toggle:active",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_toggle_hardware_paint_stability.mjs",
        "target": "css",
    },
    {
        "id": "preview-file-menu-frozen-at-build-time",
        "why": "The menu is built once, when the dialog is created. Resolving its items then binds them to whatever file was open at that moment, so every later preview downloads the first file.",
        "find": "        var docMenu = _buildOverflowMenu('File actions', function () {\n            var it = _attachmentPreviewState.item;",
        "replace": "        var docMenuItem = _attachmentPreviewState.item;\n        var docMenu = _buildOverflowMenu('File actions', (function () {\n            var it = docMenuItem;",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
    },
    {
        "id": "chevron-rotation-not-driven-by-aria",
        "why": "Rotating from a class rather than aria-expanded lets the glyph and the announced state disagree: the arrow says open while the trigger still reports closed.",
        "find": ".ai-assistant-panel-attachment-preview-menu-btn[aria-expanded=\"true\"] svg {\n    transform: rotate(180deg);\n}",
        "replace": ".ai-assistant-panel-attachment-preview-menu-btn.is-open svg {\n    transform: rotate(180deg);\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
        "target": "css",
    },
    {
        "id": "composer-shares-a-row-with-its-own-small-print",
        "why": "With flex-basis auto the composer only had its own row because the note and credit were unshrinkable. Once they could shrink, a wide panel fitted all three on one line and the composer sat beside its own disclaimer.",
        "find": "    flex: 1 1 100%;\n    min-width: 0;              /* attachment strip must scroll, never widen footer */",
        "replace": "    flex: 1 1 auto;\n    min-width: 0;              /* attachment strip must scroll, never widen footer */",
        "harness": "_static/ai_assistant/test_ai_assistant__footer_layout.mjs",
        "target": "css",
    },
    {
        "id": "footer-spends-two-rows-on-small-print",
        "why": "Both elements claiming width:100% gives each a full line of a wrapping flex row, so the footer spends two rows on two short pieces of small print in a panel whose vertical space is the scarce dimension.",
        "find": "    flex: 1 1 10rem;\n    min-width: 0;\n    text-align: start;\n    /*",
        "replace": "    width: 100%;\n    min-width: 0;\n    text-align: center;\n    /*",
        "harness": "_static/ai_assistant/test_ai_assistant__footer_layout.mjs",
        "target": "css",
    },
    {
        "id": "menu-glyph-is-announced",
        "why": "The accessible name is the label text. An announced glyph makes a screen reader read decoration as part of the item, so Download patch becomes the label plus whatever the SVG happens to expose.",
        "find": "                iconWrap.setAttribute('aria-hidden', 'true');",
        "replace": "                void iconWrap;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "menu-icon-gutter-collapses",
        "why": "With the column sized to content, items with a glyph indent and items without do not. A half-indented list is harder to scan than one with no icons at all.",
        "find": "    grid-template-columns: 1rem minmax(0, 1fr);",
        "replace": "    grid-template-columns: auto minmax(0, 1fr);",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "workspace-tabs-inherit-the-format-grid",
        "why": "The even grid is right for five interchangeable formats and wrong for two or three named sections: they are stretched to equal fractions of the row and the active edge is drawn across the whole of each cell.",
        "find": "    display: flex;\n    grid-template-columns: none;\n    justify-content: flex-start;",
        "replace": "    justify-content: flex-start;",
        "harness": "_static/ai_assistant/test_ai_assistant__workspace_tabs.mjs",
        "target": "css",
    },
    {
        "id": "workspace-tab-typography-inherits-host-line-height",
        "why": "The export tabs sit inside a privacy body with line-height 1.55 while workspace tabs are direct sheet children. Inheriting line-height therefore makes one visual component compute differently by placement.",
        "find": "    line-height: 1.55;\n    cursor: pointer;",
        "replace": "    line-height: inherit;\n    cursor: pointer;",
        "harness": "_static/ai_assistant/test_ai_assistant__workspace_tab_visual_parity.mjs",
        "target": "css",
    },
    {
        "id": "preview-chevron-left-out-of-the-shared-treatment",
        "why": "The two chevrons are the same kind of control. Listing only one means the same gesture behaves differently depending on which surface the reader is on, and a third chevron inherits nothing.",
        "find": ".ai-assistant-panel-inline-picker-more,\n.ai-assistant-panel-attachment-preview-menu-btn {",
        "replace": ".ai-assistant-panel-inline-picker-more {",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "preview-chevron-too-small-for-a-thumb",
        "why": "A thumb does not aim differently because the control is in a dialog rather than a composer. Left at its pointer size it is the one chevron a reader cannot hit.",
        "find": "        min-width: 2.75rem;\n        min-height: 2.75rem;\n    }\n    .ai-assistant-panel-attachment-preview-menu-btn svg",
        "replace": "    }\n    .ai-assistant-panel-attachment-preview-menu-btn svg",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "chevron-drawn-from-its-own-tokens",
        "why": "The chevron and the picker are joined into one segmented control. Given a border of its own, or none, the pair reads as a bordered button with a bare glyph stuck to it -- most visibly at rest on a desktop, where nothing hovers.",
        "find": "    border: 1px solid var(\n        --pst-color-border,\n        var(--color-foreground-border, rgba(127, 127, 127, 0.3))\n    );\n    border-inline-start: 0;",
        "replace": "    border: 0;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "enlarged-chevron-crowds-the-mic",
        "why": "Two adjacent thumb-sized controls with a 0.1rem gap and no wrapping means the reader aims at one and hits the other. Wrapping is the right failure: two rows of reachable controls beat one row of overlapping ones.",
        "find": "        gap: 0.4rem;\n        flex-wrap: wrap;",
        "replace": "        gap: 0.1rem;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "expanded-hint-sits-on-an-opaque-row",
        "why": "The row needs a real paint box but must still cancel its flow cost. Replacing the negative offset with an ordinary margin restores an opaque layout row beside the expanded hint and steals transcript height.",
        "find": "    min-height: 2rem;\n    height: auto;\n    margin: -2rem 0.75rem 0;",
        "replace": "    min-height: 2rem;\n    height: auto;\n    margin: 0 0.75rem 0.5rem;",
        "harness": "_static/ai_assistant/test_ai_assistant__speak_hint_collapse.mjs",
        "target": "css",
    },
    {
        "id": "chevron-too-small-for-a-thumb",
        "why": "1.6rem is a cursor target, not a thumb one. Below the platform minimum the reader misses it, and on touch there is no hover to help them aim.",
        "find": "        min-width: 2.75rem;\n        min-height: 2.75rem;\n        width: auto;",
        "replace": "        min-width: 1.6rem;\n        min-height: 1.6rem;\n        width: auto;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "chevron-invisible-until-hovered",
        "why": "A transparent control depends on hover to announce itself, and touch has no hover. Without a resting ground it is a glyph floating beside the picker rather than a button.",
        "find": "        background: color-mix(in srgb, var(--pst-color-surface, #fff) 92%, #808080 8%);\n        color: inherit;",
        "replace": "        color: inherit;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "chevron-keeps-browser-button-chrome",
        "why": "Safari and Firefox apply their own button chrome, so without the reset the chevron renders at a different height from the picker it is joined to and the pair looks broken.",
        "find": "    -webkit-appearance: none;\n    appearance: none;\n    /* Removes",
        "replace": "    appearance: auto;\n    /* Removes",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "credit-forces-an-early-stack",
        "why": "With nowrap the credit is an unbreakable block, so the column pair fails as soon as the note cannot also fit beside it -- the credit decides the layout for both and the footer stacks while there is still room for two columns.",
        "find": "    white-space: normal;\n}\n.ai-assistant-panel-footer-credit-link { white-space: nowrap; }",
        "replace": "    white-space: nowrap;\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__footer_layout.mjs",
        "target": "css",
    },
    {
        "id": "footer-stacks-while-two-columns-still-fit",
        "why": "The basis decides when two columns stop fitting. Too large a basis makes the footer jump from one row to three at a width where it could have used two narrow ones.",
        "find": "    flex: 1 1 10rem;\n    min-width: 0;\n    text-align: start;",
        "replace": "    flex: 1 1 15rem;\n    min-width: 0;\n    text-align: start;",
        "harness": "_static/ai_assistant/test_ai_assistant__footer_layout.mjs",
        "target": "css",
    },
    {
        "id": "disclaimer-truncated-to-one-line",
        "why": "A notice that the assistant can be wrong is the one piece of text a reader most needs to have seen. Clipping it makes the footer look tidier and mean less, and the hover that would reveal the rest does not exist on touch.",
        "find": "    overflow: visible;\n    text-overflow: clip;\n    white-space: normal;",
        "replace": "    overflow: hidden;\n    text-overflow: ellipsis;\n    white-space: nowrap;",
        "harness": "_static/ai_assistant/test_ai_assistant__footer_layout.mjs",
        "target": "css",
    },
    {
        "id": "minimum-size-exceeds-a-small-viewport",
        "why": "With the floor applied after the viewport cap, a 320px minimum wins on a 280px screen and the window is sized wider than the display, pushing the controls it exists to protect off the edge.",
        "find": "        var w = Math.max(Math.min(_PREVIEW_MIN_W, v.w), Math.min(geom.width, v.w));",
        "replace": "        var w = Math.max(_PREVIEW_MIN_W, Math.min(geom.width, v.w));",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
    },
    {
        "id": "preview-window-draggable-on-touch",
        "why": "No hover reveals the resize corner, the corner is smaller than a fingertip, and a finger-dragged window is easy to strand. On touch the preview should fill the screen, not float.",
        "find": "@media (max-width: 640px), (pointer: coarse) {",
        "replace": "@media (max-width: 0px) {",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
        "target": "css",
    },
    {
        "id": "window-controls-split-around-the-title",
        "why": "Appending a control to the header directly puts it wherever its construction site happens to run, which is how minimise and maximise ended up before the title with Close separated from its peers.",
        "find": "        header.appendChild(controls);",
        "replace": "        header.appendChild(minBtn); header.appendChild(maxBtn); header.appendChild(close);",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
    },
    {
        "id": "preview-window-draggable-out-of-reach",
        "why": "Without the keep-visible clamp a window can be dragged past any edge, taking its header -- and therefore every control including close -- with it. The reader has then lost the window with no way to get it back.",
        "find": "            left: Math.min(Math.max(geom.left, _PREVIEW_KEEP_VISIBLE - w), v.w - _PREVIEW_KEEP_VISIBLE),",
        "replace": "            left: geom.left,",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
    },
    {
        "id": "preview-window-keeps-centring-transform",
        "why": "translate(-50%,-50%) offsets every explicit position by half the window's own size, so the first drag makes the window jump away from the pointer.",
        "find": "        dialog.style.transform = 'none';",
        "replace": "        void dialog;",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
    },
    {
        "id": "preview-header-buttons-become-drag-handles",
        "why": "Without excluding buttons, pressing minimise or close starts a drag instead of activating the control, so the window moves and the action never fires.",
        "find": "            if (ev.button !== 0 || (ev.target && ev.target.closest &&\n                    ev.target.closest('button'))) return;",
        "replace": "            if (ev.button !== 0) return;",
        "harness": "_static/ai_assistant/test_ai_assistant__preview_window.mjs",
    },
    {
        "id": "numbered-code-scrolls-itself",
        "why": "If the code keeps its own max-height and overflow, the gutter beside it has neither and renders every line of the file: the block stretches to the file's length and the numbers slide out of step with the code the moment either scrolls.",
        "find": "    max-height: none;\n    /*\n     * `hidden`, not `visible`.",
        "replace": "    max-height: min(70vh, 28rem);\n    overflow-y: auto;\n    /*\n     * `hidden`, not `visible`.",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "code-block-swallows-the-wheel",
        "why": "CSS promotes a visible axis to auto when the other is not visible, so `overflow-y: visible` beside `overflow-x: auto` makes the block a scroll container on both axes -- only for files with long lines. It then swallows the wheel while having nothing to scroll, and the sheet's overscroll containment stops the gesture reaching the page.",
        "find": "    overflow-y: hidden;\n    overflow-x: auto;\n    white-space: pre;",
        "replace": "    overflow-y: visible;\n    overflow-x: auto;\n    white-space: pre;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "preview-sheet-traps-the-wheel",
        "why": "The overlay body is the scroller there. A sheet that does not scroll but still contains overscroll traps the gesture in an element with nothing to scroll, so the reader cannot scroll the file they are reading.",
        "find": "    overscroll-behavior: auto;\n}",
        "replace": "    overscroll-behavior: contain;\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "numbered-code-wraps-and-desyncs-the-gutter",
        "why": "A logical line wrapping to three visual rows fills three line boxes in the code and one in the gutter, so every number after the first wrapped line is wrong.",
        "find": "    overflow-x: auto;\n    white-space: pre;\n    overflow-wrap: normal;\n    word-break: normal;",
        "replace": "    overflow-x: auto;\n    white-space: pre-wrap;\n    overflow-wrap: anywhere;\n    word-break: normal;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "snippet-blocks-left-unnumbered",
        "why": "Only collapsed files were numbered, so a reader could cite a line in a large file but not in the snippet beside it -- and snippets are what most answers are made of.",
        "find": "        _numberRemainingCodeBlocks(root);",
        "replace": "        void root;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "numbering-nests-a-sheet-inside-a-sheet",
        "why": "The collapse pass runs first and already sheets the files it collapses. Without the parent check the numbering pass wraps those sheets again, putting a gutter beside a gutter.",
        "find": "            if (wrap.parentNode && wrap.parentNode.classList &&\n                    wrap.parentNode.classList.contains('ai-md-file-sheet')) {",
        "replace": "            if (false) {",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "section-title-darkens-on-a-dark-bubble",
        "why": "Darker is the light-theme direction only. Applying it on a dark bubble moves the title toward the background instead of away from it, so the heading a reader scans by becomes the least readable text in the block.",
        "find": "    --ai-section-title-color: color-mix(in srgb, var(--pst-color-text-base, var(--color-foreground-primary, #e6e6e6)) 86%, #fff 14%);",
        "replace": "    --ai-section-title-color: color-mix(in srgb, var(--pst-color-text-base, var(--color-foreground-primary, #e6e6e6)) 86%, #000 14%);",
        "harness": "_static/ai_assistant/test_ai_assistant__custom_section_dom.mjs",
        "target": "css",
    },
    {
        "id": "section-title-indistinguishable-from-body",
        "why": "Without its own colour the summary inherits the bubble's text, leaving a heading that differs from the paragraph beneath it only by weight and two percent of size.",
        "find": "    color: var(--ai-section-title-color, inherit);",
        "replace": "    color: inherit;",
        "harness": "_static/ai_assistant/test_ai_assistant__custom_section_dom.mjs",
        "target": "css",
    },
    {
        "id": "overlay-preview-loses-its-line-numbers",
        "why": "The overlay is where a long file is actually read, so it is where a reader asking for a change at a particular line needs to read that number off. Numbering only the inline sheet leaves the main reading surface unnumbered.",
        "find": "            _buildLineNumberedSheet(pre, item.previewText,\n                'ai-md-file-sheet ai-assistant-panel-attachment-preview-sheet');",
        "replace": "            void pre;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "panel-toggle-invisible-in-forced-colours",
        "why": "Author colours are discarded in forced-colours modes, so an rgba track and a white thumb both collapse toward the system background and the switch shows no state. On or off is the one thing a switch has to communicate.",
        "find": "    .ai-assistant-mic-popup-toggle[aria-pressed=\"true\"] .ai-assistant-mic-toggle-track,\n    .ai-assistant-mic-popup-toggle[aria-checked=\"true\"] .ai-assistant-mic-toggle-track,\n    .ai-assistant-panel-mode-switch[aria-checked=\"true\"] .ai-assistant-panel-toggle-track {\n        background: Highlight;\n    }",
        "replace": "    .ai-assistant-panel-mode-switch[aria-checked=\"true\"] .ai-assistant-panel-toggle-track {\n        background: rgba(0, 0, 0, 0.22);\n    }",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "format-tabs-scroll-out-of-sight",
        "why": "Horizontal scrolling hides tabs behind an edge -- TOML sat off-screen with nothing indicating it existed. A format the reader cannot see is a format they will not choose.",
        "find": "    display: grid;\n    grid-template-columns: repeat(auto-fit, minmax(min(6rem, 100%), 1fr));",
        "replace": "    display: flex;\n    overflow-x: auto;",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
        "target": "css",
    },
    {
        "id": "artifact-row-refuses-narrow-stacking",
        "why": "If the narrow container keeps the artifact in a row, metadata and lifecycle buttons compete for the same line again and the description loses readable width.",
        "find": "        flex-direction:column;\n        align-items:stretch;",
        "replace": "        flex-direction:row;\n        align-items:stretch;",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_action_responsive_layout.mjs",
        "target": "css",
    },
    {
        "id": "section-title-reads-as-the-preview-caption",
        "why": "A title tuned to follow ordinary text sits too close under a bordered full-height preview, where it reads as that preview's caption rather than the next section's heading.",
        "find": ".ai-assistant-conv-share-format-preview + .ai-assistant-conv-share-section-title {\n    margin-top: 1rem;\n}",
        "replace": ".ai-assistant-conv-share-format-preview + .ai-assistant-conv-share-section-title {\n    margin-top: .2rem;\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
        "target": "css",
    },
    {
        "id": "narrow-panel-labels-collapse-by-viewport",
        "why": "The panel is resizable, maximizable and embeddable, so its width and the viewport's are different numbers. A media query collapses labels on a wide panel inside a narrow window and keeps them on a narrow panel inside a wide one.",
        "find": "@container ai-artifact-surface (max-width: 26rem) {",
        "replace": "@media (max-width: 26rem) {",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "per-file-download-collapse-threshold-regressed-to-22rem",
        "why": "A 22rem cutoff is below the rendered artifact width of wider phones and landscape mobile, so Download text can keep squeezing long filenames even though the compact icon affordance is already needed.",
        "find": "@container ai-artifact-surface (max-width: 26rem) {",
        "replace": "@container ai-artifact-surface (max-width: 22rem) {",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "per-file-icon-only-label-removed-not-clipped",
        "why": "display:none takes the per-file label out of the box, shrinking the touch target toward the glyph. Clipping keeps the semantic/text box while hiding it visually.",
        "find": "    .ai-md-artifact-download-label .ai-md-artifact-btn-label,\n    .ai-assistant-panel-changed-file-download .ai-md-artifact-btn-label {\n        position: absolute;\n        width: 1px;\n        height: 1px;\n        overflow: hidden;\n        clip-path: inset(50%);",
        "replace": "    .ai-md-artifact-download-label .ai-md-artifact-btn-label,\n    .ai-assistant-panel-changed-file-download .ai-md-artifact-btn-label {\n        position: absolute;\n        width: 1px;\n        height: 1px;\n        overflow: hidden;\n        display: none;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "bulk-icon-only-label-removed-not-clipped",
        "why": "Bulk footer labels use the same accessible clipping contract. Removing them with display:none makes the compact footer control depend only on its glyph box.",
        "find": "    .ai-assistant-panel-changed-files-download-all .ai-md-artifact-btn-label,\n    .ai-assistant-panel-changed-files-series .ai-md-artifact-btn-label {\n        position: absolute;\n        width: 1px;\n        height: 1px;\n        overflow: hidden;\n        clip-path: inset(50%);",
        "replace": "    .ai-assistant-panel-changed-files-download-all .ai-md-artifact-btn-label,\n    .ai-assistant-panel-changed-files-series .ai-md-artifact-btn-label {\n        position: absolute;\n        width: 1px;\n        height: 1px;\n        overflow: hidden;\n        display: none;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
        "target": "css",
    },
    {
        "id": "icon-button-glyph-is-announced",
        "why": "The accessible name comes from aria-label. An announced glyph makes a screen reader read decoration as part of the button, so Download all becomes the label plus whatever the SVG happens to expose.",
        "find": "        glyph.setAttribute('aria-hidden', 'true');",
        "replace": "        void glyph;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "icon-button-decorated-twice-duplicates-its-label",
        "why": "The decorator is called on buttons that may be relabelled. Without clearing first, a second call appends a second glyph and a second label rather than replacing them.",
        "find": "        btn.textContent = '';",
        "replace": "        void btn;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "privacy-section-looks-like-a-formatting-option",
        "why": "The section governing what leaves the device must be identifiable while scrolling. Stripped of its accent edge it is indistinguishable from the formatting collapses beside it, and a privacy control that reads as a throwaway row gets skipped.",
        "find": "    border-inline-start: 3px solid var(--ai-artifact-accent);",
        "replace": "    border-inline-start: 1px solid rgba(127,127,127,.22);",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
        "target": "css",
    },
    {
        "id": "collapse-sections-are-bare-rows",
        "why": "One collapse was a bordered card while its siblings were hairline rules, so a reader had to click a row to learn it was interactive at all.",
        "find": ".ai-assistant-conv-share-collapse {\n    border:1px solid rgba(127,127,127,.22);",
        "replace": ".ai-assistant-conv-share-collapse {\n    border-top:1px solid rgba(127,127,127,.18);",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
        "target": "css",
    },
    {
        "id": "share-destinations-orphan-the-fourth",
        "why": "Four destinations in a three-column track leave the fourth alone on its own row, stretched full width. That reads as a hierarchy the destinations do not have: Global link is a peer of the other three, not a summary of them.",
        "find": ".ai-assistant-conv-share-destinations { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); grid-auto-rows:1fr; gap:.55rem; }",
        "replace": ".ai-assistant-conv-share-destinations { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; }",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
        "target": "css",
    },
    {
        "id": "menu-placed-against-an-assumed-containing-block",
        "why": "`position: fixed` resolves against the viewport only when no ancestor has a transform -- and the panel has one on its open state. Writing viewport coordinates without converting them leaves the menu offset by the panel's own position.",
        "find": "        menu.style.left = Math.round(left - origin.left) + 'px';",
        "replace": "        menu.style.left = Math.round(left) + 'px';",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "long-menu-runs-off-the-screen",
        "why": "Placement flips a menu above its trigger when there is no room below, but a twelve-item list is taller than the space either way. Without a viewport bound the last items are cut off with nothing indicating they exist.",
        "find": "    max-height: min(50vh, 20rem);",
        "replace": "    max-height: min(500vh, 200rem);",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
        "target": "css",
    },
    {
        "id": "trigger-cannot-close-its-own-menu",
        "why": "The capture-phase outside-click handler sees a click before the trigger does. With an identity test, a click on the glyph inside the button closes the menu there, and the trigger then finds nothing open and reopens it -- a menu that can be opened and never closed from its own button.",
        "find": "                    if (!menu.contains(e.target) && !btn.contains(e.target)) _closeFileMenu();",
        "replace": "                    if (!menu.contains(e.target) && e.target !== btn) _closeFileMenu();",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "preview-menu-bounded-by-the-wrong-surface",
        "why": "The preview dialog is a floating window of its own, often wider and placed elsewhere. Bounding its menus by the panel clamps them to a box their trigger is not in, which is where the 1193px ceiling came from.",
        "find": "            ? (btn.closest('.ai-assistant-panel-attachment-preview') ||\n               btn.closest('.ai-assistant-panel'))",
        "replace": "            ? btn.closest('.ai-assistant-panel')",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "menus-share-only-a-caller-specific-name",
        "why": "One builder serves file rows, snippet rows, the preview and the model picker. With only the file-row name on them, a rule written for one surface silently restyles the other three.",
        "find": "            menu.className = 'ai-assistant-menu ai-assistant-panel-changed-file-menu';",
        "replace": "            menu.className = 'ai-assistant-panel-changed-file-menu';",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "model-list-keeps-its-own-scroller",
        "why": "Nested inside a menu that is itself bounded, a second scroll region takes the wheel while the outer menu cannot be scrolled to reveal it. On a short panel that leaves no model reachable at all.",
        "find": "    max-height: none;\n    overflow-y: visible;\n    overscroll-behavior: auto;\n    display: none;",
        "replace": "    max-height: min(50vh, 18rem);\n    overflow-y: auto;\n    overscroll-behavior: contain;\n    display: none;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "expanded-list-outgrows-a-stale-bound",
        "why": "The placement routine wrote a max-height for the collapsed menu. Expanding the list makes it taller than that bound, so the new rows sit below a clamp computed before they existed.",
        "find": "                if (ownerMenu) _positionBubbleMoreMenuWithinPanelBody(ownerMenu);",
        "replace": "                void ownerMenu;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
    },
    {
        "id": "hamburger-menu-runs-past-the-panel",
        "why": "With no height bound its lower entries run past the panel's bottom edge on a small panel -- nothing to scroll, and no indication the entries exist.",
        "find": "    max-height: min(calc(100% - 3.3rem - 0.75rem), 80vh);\n    overflow-y: auto;",
        "replace": "    overflow-y: visible;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
        "target": "css",
    },
    {
        "id": "composer-floor-squeezes-the-menu",
        "why": "Keeping a menu clear of the composer is a readability preference; showing all of its rows is not. On a short panel an unconditional floor left too little height and the lower entries were unreachable.",
        "find": "                    (fr.top - bounds.top) >= _MENU_MIN_USABLE_H) {",
        "replace": "                    true) {",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "bubble-menu-clamped-to-a-short-transcript",
        "why": "This menu carries the model list, the longest in the panel. Bounded by the transcript it was clamped to whatever height the body happened to have and its last entries fell off the bottom.",
        "find": "            boundarySelector: '.ai-assistant-panel',",
        "replace": "",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "menu-covers-the-composer",
        "why": "A menu drawn over the composer looks like it belongs to the composer, and hides the draft the reader is about to send. The bubble action menu has always been bounded this way; these were not.",
        "find": "                bounds.bottom = Math.min(bounds.bottom, fr.top);",
        "replace": "                void fr;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "footer-trigger-bounded-above-its-own-button",
        "why": "The model picker's trigger lives inside the footer. Applying the composer floor unconditionally puts the bound above its own button and leaves that menu nowhere to open.",
        "find": "            if (fr.height > 0 && t.bottom <= fr.top &&\n                    (fr.top - bounds.top) >= _MENU_MIN_USABLE_H) {",
        "replace": "            if (fr.height > 0) {",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "menu-width-measured-before-it-is-constrained",
        "why": "The rect read after this line drives every later calculation. Measuring an unconstrained max-content menu means placing a box the menu will never actually have, so the clamp is computed from a width that does not exist.",
        "find": "        menu.style.maxWidth = Math.max(160, (bounds.right - bounds.left) - margin * 2) + 'px';\n\n        var m = menu.getBoundingClientRect();",
        "replace": "        var m = menu.getBoundingClientRect();\n        menu.style.maxWidth = Math.max(160, (bounds.right - bounds.left) - margin * 2) + 'px';",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "menu-hint-decides-the-menu-width",
        "why": "With max-content the widest row wins. A non-wrapping secondary hint makes the menu as wide as a sentence of explanatory text, rather than as wide as the labels a reader is choosing between.",
        "find": ".ai-assistant-panel-changed-file-menu-hint {\n    white-space: normal;\n    overflow-wrap: anywhere;\n}",
        "replace": ".ai-assistant-panel-changed-file-menu-hint {\n    white-space: nowrap;\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
        "target": "css",
    },
    {
        "id": "menu-spills-outside-the-panel",
        "why": "A narrow panel inside a wide window has viewport to spare beside it. Clamping only to the screen lets a list belonging to a panel control be drawn over the documentation behind it, reading as part of neither.",
        "find": "                bounds.right = Math.min(bounds.right, pr.right);",
        "replace": "                bounds.right = bounds.right;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "hidden-panel-clamps-the-menu-to-a-point",
        "why": "A display:none or zero-height ancestor reports an empty rect. Intersecting with it collapses the bounds to nothing and every menu is clamped into a single point, which is worse than not clamping at all.",
        "find": "            if (pr.width > 0 && pr.height > 0) {",
        "replace": "            if (true) {",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "menu-lands-in-the-middle-of-the-screen",
        "why": "A menu must sit against the button that opened it. Centring it leaves the reader hunting for the connection between a control they pressed and a list that appeared somewhere else.",
        "find": "        var left = t.right - m.width;",
        "replace": "        var left = (bounds.left + bounds.right - m.width) / 2;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "quick-swap-list-frozen-at-build-time",
        "why": "The menu is built once with the composer. Reading candidates then binds the list to the models configured at that moment and to whichever model was active, so the tick marks the wrong row for the rest of the session.",
        "find": "            var quickModelBtn = _buildOverflowMenu('Try a different model', function () {\n                var live = _quickModelCandidates(_cfg());",
        "replace": "            var frozen = _quickModelCandidates(_cfg());\n            var quickModelBtn = _buildOverflowMenu('Try a different model', function () {\n                var live = frozen;",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
    },
    {
        "id": "picker-wrapper-swallows-aria-expanded",
        "why": "The sync code writes aria-expanded on inlinePicker. If that name refers to the wrapper rather than the button, the picker stops reporting whether its sheet is open and the wrapper reports it instead, where nothing reads it.",
        "find": "            pickerWrap.appendChild(inlinePicker);",
        "replace": "            pickerWrap.appendChild(_buildInlineModelPicker());",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
    },
    {
        "id": "quick-model-menu-silently-truncates",
        "why": "A cap keeps the first N and drops the rest in silence, so a twelve-model configuration is indistinguishable from a six-model one under a heading that reads Try a different model. Length belongs to scrolling, not to truncation.",
        "find": "                if (!activeQuickModel || candidate.id !== activeQuickModel.id) {",
        "replace": "                if (quickDisplayModels.length >= 6) return;\n                if (!activeQuickModel || candidate.id !== activeQuickModel.id) {",
        "harness": "_static/ai_assistant/test_ai_assistant__first_message_privacy_and_quick_model.mjs",
    },
    {
        "id": "file-menu-placed-before-it-is-measured",
        "why": "The placement routine reads the rendered box. Positioning before the menu is in the document sizes it from nothing and clamps everything to the corner, which looks like a broken menu rather than a mis-measured one.",
        "find": "            _positionMenuNearTrigger(menu, btn);\n            _fileMenuOpen = rec;",
        "replace": "            _fileMenuOpen = rec;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "file-menu-detaches-when-the-body-scrolls",
        "why": "The menu is anchored to a row inside a scrolling body. Without a reflow listener it stays where it opened while its trigger moves away -- and a long file list is exactly when the reader is scrolling.",
        "find": "            if (scroller) scroller.addEventListener('scroll', rec.onReflow, true);",
        "replace": "            void scroller;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "share-export-leaks-the-source-page",
        "why": "A share is published. A format that renders a field the share policy nulled leaks it to everyone with the link, and the five real exports were correct only by construction -- nothing asserted it.",
        "find": "        if (session.page_url) lines.push('Page: ' + session.page_url);",
        "replace": "        lines.push('Page: ' + (session.page_url || 'https://scikit-plots.github.io/dev/user_guide/impute/index.html'));",
        "harness": "_static/ai_assistant/test_ai_assistant__share_redaction_parity.mjs",
    },
    {
        "id": "text-export-drops-a-reviewed-session-field",
        "why": "The snapshot has already had the reader's review applied, so every format renders the same decisions. A format that drops a field the reader chose to include decides for them, and invisibly -- the same export in another format carries it.",
        "find": "        if (session.id) lines.push('Session: ' + session.id);",
        "replace": "        void session;",
        "harness": "_static/ai_assistant/test_ai_assistant__export_formats.mjs",
    },
    {
        "id": "local-save-redacted-like-a-published-share",
        "why": "Share redaction strips timestamps, model attribution, the session id and the source page. Applied to a local device file it protects nobody and removes the provenance a saved transcript is kept for -- a real export produced a file where every one of those fields was null.",
        "find": "        return (destination === 'download' || destination === 'local')\n            ? 'complete' : 'standard';",
        "replace": "        return 'standard';",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {
        "id": "queue-change-leaves-the-chips-on-screen",
        "why": "The removal path mutates _composerAttachments without redrawing -- its own callers did that. A caller that forgets leaves the registry emptied and every chip still on screen, so a bulk remove looks like it removed nothing.",
        "find": "        try { _renderComposerAttachments(); } catch (_) {}",
        "replace": "        void 0;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "bulk-clear-redraws-per-file",
        "why": "Clearing N files reaches the refresh N times. Without the depth guard each removal redraws every surface from an intermediate state, and the reader watches the queue empty one chip at a time.",
        "find": "        if (_continuationRefreshDepth > 0) return;",
        "replace": "        if (false) return;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "bulk-control-keeps-its-build-time-label",
        "why": "The control is built once. Without relabelling from the live count it keeps saying Continue editing all while the next click removes everything, which is the worst kind of wrong label -- one that describes the opposite of what happens.",
        "find": "        Array.prototype.forEach.call(buttons, _applyContinueAllLabel);",
        "replace": "        void buttons;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "bulk-control-only-attaches",
        "why": "Attaching several files was one click while detaching them meant dismantling the queue one menu at a time. The reader who wants none of it should not have the most work to do.",
        "find": "                if (_continuationCount()) _generatedArtifactClearContinuations();",
        "replace": "                if (false) _generatedArtifactClearContinuations();",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "chip-removal-leaves-the-continuation-registered",
        "why": "The chip and the registry are two views of one intent and the reader can act on either. Removing the chip without clearing the registry sends a file the reader explicitly removed, bound to a revision, as though they had asked for it.",
        "find": "        _syncContinuationForRemovedItem(item);",
        "replace": "        void item;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "menu-items-frozen-at-build-time",
        "why": "A Continue/Stop toggle built from a static array keeps its row-build label. After dropping a file the item still reads Stop continuing, so the next click stops an already-stopped continuation and the file looks impossible to re-add.",
        "find": "            resolveItems().forEach(function (item) {",
        "replace": "            (typeof items === 'function' ? items() : items).forEach(function (item) {",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "stop-continuing-leaves-the-bytes-staged",
        "why": "Deleting only the registry key means the dropped file still travels with the next message, and re-adding it stages a second copy. Stop has to undo everything Continue did.",
        "find": "        if (entry) _unstageContinuationAttachment(entry.path);",
        "replace": "        if (false) _unstageContinuationAttachment(entry.path);",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "pre-discovery-cap-drops-every-file-but-one",
        "why": "Guessing one file silently drops the rest from Continue-all, which reads as a broken button rather than a limit. Guessing the server default risks a rejection the reader can see and act on.",
        "find": "    var _WORKING_FILE_FALLBACK = { maxFiles: 4, maxFileChars: 48000, maxTotalChars: 96000 };",
        "replace": "    var _WORKING_FILE_FALLBACK = { maxFiles: 1, maxFileChars: 12000, maxTotalChars: 12000 };",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "continuation-sends-the-bytes-twice",
        "why": "A file carried as a bound working file must be dropped from the multipart body too. The form is built from requestResources, not bodyObj.resources, so filtering only the descriptor uploads the bytes with nothing describing them.",
        "find": "                    requestResources = requestResources.filter(function (row) {\n                        var p = row && (row.relative_path || row.relativePath || row.name);\n                        return !(p && carried[p]);\n                    });",
        "replace": "                    void carried;",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "continue-all-ignores-endpoint-limits",
        "why": "Batch queuing must apply the published per-request bounds. Without them the reader discovers the cap by having the request rejected, after the tokens are already spent composing it.",
        "find": "            if (_continuationCount() >= limits.maxFiles) { skipped++; continue; }",
        "replace": "            if (false) { skipped++; continue; }",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "continuation-overwrites-the-composer",
        "why": "Priming an instruction over text the reader already typed destroys their question to save them a sentence.",
        "find": "        if (!String(input.value || '').trim()) {",
        "replace": "        if (true) {",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "solo-patch-footer-loses-its-chrome",
        "why": "A one-file footer must read as the same kind of control it becomes when a second file arrives. A bare button beneath a bordered file row looks like a different feature, not the same one with less in it.",
        "find": "            series.classList.add('ai-assistant-panel-changed-files-solo');",
        "replace": "            void series;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "one-file-export-called-a-series",
        "why": "A series of one file is just a patch. Calling it a series makes a reader look for the other files it supposedly contains.",
        "find": "            many ? 'Download patch series' : 'Download patch');",
        "replace": "            'Download patch series');",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "artifact-accent-not-shared",
        "why": "Four controls express one idea -- take this file. Dropping one from the shared accent is how they drifted apart before: the snippet label had the theme colour while the presented-file and bulk downloads inherited body text.",
        "find": ".ai-assistant-panel-changed-files-download-all,\n.ai-assistant-panel-changed-files-head strong {\n    color: var(--ai-artifact-accent);",
        "replace": ".ai-assistant-panel-changed-files-head strong {\n    color: var(--ai-artifact-accent);",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_diff_stat.mjs",
        "target": "css",
    },
    {
        "id": "presented-file-forks-its-own-card-markup",
        "why": "The two artifact surfaces are the same control on the same kind of object. Parallel class trees styled to match had already diverged once -- a trailing Preview word, a badge in a different place, a different truncation point for the same filename.",
        "find": "            preview.className = 'ai-md-artifact-card ai-assistant-panel-changed-file-preview';",
        "replace": "            preview.className = 'ai-assistant-panel-changed-file-preview';",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "artifact-download-nested-inside-the-preview",
        "why": "A button nested in a button is invalid HTML and browsers resolve it by dropping one of the two click targets -- which one varies. Download must be a sibling with its own focus stop and accessible name.",
        "find": "        group.appendChild(primary);",
        "replace": "        primary.appendChild(secondary);\n        group.appendChild(primary);",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_contextual_naming.mjs",
    },
    {
        "id": "artifact-group-loses-its-accessible-name",
        "why": "Two segments drawn as one control must say so. Without role=group and a name, assistive technology announces two unrelated buttons and the relationship visible on screen is invisible to everyone else.",
        "find": "        group.setAttribute('role', 'group');",
        "replace": "        void 0;",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_contextual_naming.mjs",
    },
    {
        "id": "snippet-card-click-downloads-instead-of-previewing",
        "why": "Clicking the big card should let a reader check a snippet before committing to a download. Downloading on card click is the behaviour this change removed.",
        "find": "                card.setAttribute('aria-label', 'Preview ' + fname);",
        "replace": "                card.setAttribute('aria-label', 'Download ' + fname);",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_contextual_naming.mjs",
    },
    {
        "id": "snippet-card-bypasses-shared-capability-wrapper",
        "why": "The snippet trigger must go through _buildSnippetOverflow so it gets the same menu mechanics and workflow grammar as Presented files. Bypassing that wrapper lets the two surfaces drift again.",
        "find": "            var snippetMenu = _buildSnippetOverflow(\n                root, card, codeEl ? codeEl.textContent : '', lang, filename, typeLabel);",
        "replace": "            var snippetMenu = _buildOverflowMenu('More options for ' + filename, [], 'ai-md-artifact-overflow');",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_file_menu_parity.mjs",
    },
    {
        "id": "file-menu-leaks-document-listeners",
        "why": "The menu registers capture-phase document listeners while open. Leaving them attached after close makes every later click and keypress run a handler for a menu that no longer exists.",
        "find": "        document.removeEventListener('click', rec.onDocClick, true);",
        "replace": "        void rec.onDocClick;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "file-menu-escape-strands-focus",
        "why": "Closing a menu with Escape must return focus to its trigger. Without it, keyboard focus is stranded at the top of the document and the reader loses their place.",
        "find": "            if (typeof btn.focus === 'function') btn.focus();",
        "replace": "            void btn;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "timeline-file-row-drops-its-diff-counts",
        "why": "A reader scanning the timeline to see what a turn did should not have to scroll to a file card to learn whether \"Updated file\" meant a typo or a rewrite.",
        "find": "        var stat = _diffStatElement(entry);\n        if (stat) button.appendChild(stat);",
        "replace": "        var stat = null;\n        if (stat) button.appendChild(stat);",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_persistence.mjs",
    },
    {
        "id": "assistant-turn-records-no-activity",
        "why": "The persistence, the validation and the renderer are all inert unless a caller supplies the activity object. Shipped once already: every source-level assertion passed while the feature stored nothing.",
        "find": "        _recordMessage('assistant', accumulated || '(no response)', _streamModelInfo,\n            null, { activity: activity });",
        "replace": "        _recordMessage('assistant', accumulated || '(no response)', _streamModelInfo);",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_persistence.mjs",
    },
    {
        "id": "restored-activity-buries-its-step-list",
        "why": "After a reload the reader has lost every other cue about what happened. Requiring a click to see the step list makes the timeline as good as absent, which is the state this work exists to fix.",
        "find": "        panel.hidden = false;\n        var note = document.createElement('p');",
        "replace": "        panel.hidden = true;\n        var note = document.createElement('p');",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_persistence.mjs",
    },
    {
        "id": "restored-activity-not-revalidated",
        "why": "Session storage is same-origin but not trustworthy: any script on the page can write it. Trusting a persisted step's kind and state lets a tampered record render a timeline the panel never produced.",
        "find": "                var restoredActivity = (e.role === 'assistant')\n                    ? _activityRestoreSummary(e.activity) : null;",
        "replace": "                var restoredActivity = (e.role === 'assistant')\n                    ? (e.activity || null) : null;",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_persistence.mjs",
    },
    {
        "id": "restored-activity-offers-a-dead-stop-button",
        "why": "The turn is over. A Stop control that cannot act invites a click that does nothing, which is worse than no control at all.",
        "find": "        toggle.appendChild(icon); toggle.appendChild(summary); toggle.appendChild(caret);\n        head.appendChild(toggle);",
        "replace": "        toggle.appendChild(icon); toggle.appendChild(summary); toggle.appendChild(caret);\n        var stop = document.createElement('button');\n        stop.className = 'ai-assistant-panel-activity-stop';\n        head.appendChild(toggle); head.appendChild(stop);",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_persistence.mjs",
    },
    {
        "id": "line-numbers-written-into-the-code",
        "why": "Line numbers belong beside the code, never inside it. An announced gutter makes a screen reader read the file as one import sys two import os, which is worse than no numbers at all.",
        "find": "        gutter.setAttribute('aria-hidden', 'true');",
        "replace": "        gutter.setAttribute('aria-hidden', 'false');",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "raw-body-elision-disabled",
        "why": "Without elision a large file is held three times, and the data-raw copy is the one serialized into session storage, competing with the persistence budget for bytes nobody reads.",
        "find": "        return markdown.replace(/(`{3,})([^\\n`]*)\\n?([\\s\\S]*?)\\1/g,",
        "replace": "        return markdown; return markdown.replace(/(`{3,})([^\\n`]*)\\n?([\\s\\S]*?)\\1/g,",
        "harness": "_static/ai_assistant/test_ai_assistant__raw_body_dedup.mjs",
    },
    {
        "id": "save-as-accepts-a-typed-directory",
        "why": "Save-as writes to the reader's download folder. Honouring a typed path would be a claim the panel cannot make, and a traversal segment is a path the browser acts on.",
        "find": "        text = text.split(/[\\\\/]/).pop() || '';",
        "replace": "        text = text || '';",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_contextual_naming.mjs",
    },
    {
        "id": "presented-files-absent-from-activity-timeline",
        "why": "File events belong in the activity timeline. A presentation visible only in the answer body is invisible to a reader who consults the timeline to see what a turn did.",
        "find": "                id: 'presented-files', kind: 'file', state: 'done',",
        "replace": "                id: 'presented-files', kind: 'status', state: 'running',",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "presented-files-claims-files-changed",
        "why": "Nothing outside the browser changed. A heading that says otherwise is the same overstatement class as retry's as-is claim and the Remember switch.",
        "find": "            (combined.length === 1 ? ' file' : ' files');",
        "replace": "            (combined.length === 1 ? ' file' : ' files'); title.textContent = 'Changed files';",
        "harness": "_static/ai_assistant/test_ai_assistant__activity_and_latest_file_preview.mjs",
    },
    {
        "id": "persistence-failure-silently-swallowed",
        "why": "If _ssSet reports success unconditionally, the panel keeps showing Remember conversation ON while quota exhaustion or private browsing discards every save, and the reader learns only after a reload.",
        "find": "        try { sessionStorage.setItem(key, val); return true; }\n        catch (_) { return false; }",
        "replace": "        try { sessionStorage.setItem(key, val); } catch (_) {}\n        return true;",
        "harness": "_static/ai_assistant/test_ai_assistant__persistence_honesty.mjs",
    },
    {
        "id": "truncated-restore-reported-as-complete",
        "why": "A transcript the browser truncated looks exactly like a complete one. Without the marker comparison the reader is shown a partial conversation with nothing indicating anything is missing.",
        "find": "        var missing = _persistenceRestoredShortfall(restored.length);",
        "replace": "        var missing = 0;",
        "harness": "_static/ai_assistant/test_ai_assistant__persistence_honesty.mjs",
    },
    {
        "id": "retry-claims-exact-replay",
        "why": "The panel retains no context snapshot, so a control promising an as-is resend overstates what it can do once history and working files travel with a request.",
        "find": "            userRetryBtn.setAttribute('aria-label', 'Ask this question again with the current context');",
        "replace": "            userRetryBtn.setAttribute('aria-label', 'Retry - resend this question as-is');",
        "harness": "_static/ai_assistant/test_ai_assistant__retry_context_honesty.mjs",
    },
    {
        "id": "working-file-stale-response-committed",
        "why": "An answer generated from a superseded revision must not be committed as the next revision. Without the guard a slow reply built from r3 lands as r5 over the reader's own r4.",
        "find": "            if (binding && !_workingFileBindingIsCurrent(binding)) {",
        "replace": "            if (false) {",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "working-file-sent-without-digest",
        "why": "A working file with no digest cannot be checked for staleness later, so sending it produces the appearance of protection without the substance.",
        "find": "            if (!digest) { skipped++; continue; }",
        "replace": "            if (!digest) { digest = '0'.repeat(64); }",
        "harness": "_static/ai_assistant/test_ai_assistant__working_file_binding.mjs",
    },
    {
        "id": "chat-history-accepts-any-role",
        "why": "Only user and assistant turns may enter history. Admitting system/developer/tool turns hands a browser transcript the ability to assert server authority.",
        "find": "            if (!entry || (entry.role !== 'user' && entry.role !== 'assistant')) continue;",
        "replace": "            if (!entry) continue;",
        "harness": "_cf_worker/test_index__chat_authority.mjs",
    },
    {
        "id": "chat-history-bounds-guessed-when-unpublished",
        "why": "A v2 endpoint that publishes no history bounds must degrade to v1, not have limits invented for it; guessing high turns a recoverable local decision into a rejected request.",
        "find": "        if (contract === _CHAT_CONTRACT_V2 && !bounds) contract = _CHAT_CONTRACT_V1;",
        "replace": "        if (contract === _CHAT_CONTRACT_V2 && !bounds) bounds = _CHAT_HISTORY_FALLBACK;",
        "harness": "_cf_worker/test_index__chat_authority.mjs",
    },
    {
        "id": "chat-structured-user-message-replaced-by-system",
        "why": "The trusted proxy contract must carry typed user input, never a client-authored system message.",
        "find": "                user_message: question,",
        "replace": "                messages: [{ role: 'system', content: systemPrompt }],",
        "harness": "_cf_worker/test_index__chat_authority.mjs",
    },
    {
        "id": "chat-structured-context-skips-redaction",
        "why": "A structured proxy request that sends the cleaned-but-unredacted page would reintroduce page-secret exfiltration.",
        "find": "                    page_text: _redacted.text.slice(0, contextLimit),",
        "replace": "                    page_text: _cleaned.text.slice(0, contextLimit),",
        "harness": "_cf_worker/test_index__chat_authority.mjs",
    },

    # ── Run 6: feedback / contribution privacy lifecycle ────────────────
    {
        "id": "feedback-network-recollects-query",
        "why": "Ordinary rating telemetry must never silently recollect the user question; full Q&A belongs only to explicit contribution consent.",
        "find": "            ratingMode: detail.ratingMode || null,\n            ts: detail.ts || Date.now()",
        "replace": "            ratingMode: detail.ratingMode || null,\n            query: detail.query || '',\n            ts: detail.ts || Date.now()",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_contribution_privacy.mjs",
    },
    {
        "id": "feedback-telemetry-consent-fails-open",
        "why": "Network rating telemetry must require an explicit current structured consent record; missing enabled=true must never inherit authority.",
        "find": "            if (!saved || saved.version !== _FEEDBACK_TELEMETRY_CONSENT_VERSION ||\n                    typeof saved.enabled !== 'boolean') {",
        "replace": "            if (!saved || saved.version !== _FEEDBACK_TELEMETRY_CONSENT_VERSION ||\n                    saved.enabled === false) {",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_telemetry_consent.mjs",
    },
    {
        "id": "feedback-telemetry-helper-consent-gate-removed",
        "why": "Even an internal caller must not be able to send feedback telemetry when the user has not opted in.",
        "find": "        if (!_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt) { return false; }",
        "replace": "        if (false && (!_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt)) { return false; }",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_telemetry_consent.mjs",
    },
    {
        "id": "feedback-public-event-reexposes-content",
        "why": "The public feedback DOM event must never rebroadcast Q&A/note/model/page content to arbitrary page listeners.",
        "find": "        var out = _feedbackTelemetryPayload(detail);",
        "replace": "        var out = Object.assign({}, detail);",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_telemetry_consent.mjs",
    },
    {
        "id": "feedback-retract-ignores-opt-out",
        "why": "Turning telemetry off must stop all future feedback network traffic, including hidden retraction housekeeping requests.",
        "find": "        if (!url || !lineage || !_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt) {",
        "replace": "        if (!url || !lineage) {",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_telemetry_consent.mjs",
    },
    {'id': 'contribution-consent-version-disabled',
     'why': 'Explicit contribution must carry the active versioned consent so stale pages cannot submit '
            'under materially changed terms.',
     'find': "    var _CONTRIBUTION_CONSENT_VERSION = '2.0.0';",
     'replace': "    var _CONTRIBUTION_CONSENT_VERSION = '1.0.0';",
     'harness': '_static/ai_assistant/test_ai_assistant__dataset_contribution_ux.mjs'},
    {'id': 'contribution-session-linkage-restored',
     'why': 'Contribution does not need the stable browser conversation identifier; re-adding it increases '
            'linkability of personal records.',
     'find': "            page: _sanitizePage(((typeof _pageUrl === 'function') ? _pageUrl() : ((typeof location !== 'undefined') ? location.href : ''))),",
     'replace': "            sessionId: _sessionId,\n            page: _sanitizePage(((typeof _pageUrl === 'function') ? _pageUrl() : ((typeof location !== 'undefined') ? location.href : ''))),",
     'harness': '_static/ai_assistant/test_ai_assistant__dataset_contribution_ux.mjs'},

    # ── Run 7: local privacy preflight / sensitive-input protection ─────
    {
        "id": "privacy-preflight-inference-bypassed",
        "why": "A user-entered credential or personal datum must be reviewed before the browser sends it to an external inference endpoint.",
        "find": "            var privacyDecision = await _privacyPreflightReview(outboundCandidate, {",
        "replace": "            var privacyDecision = { action: 'continue', value: outboundCandidate }; void _privacyPreflightReview; ({",
        "harness": "_static/ai_assistant/test_ai_assistant__privacy_preflight.mjs",
    },
    {'id': 'privacy-preflight-share-bypassed',
     'why': 'Every Share destination must review the exact canonical outbound snapshot before '
            'serialization.',
     'find': '            var reviewed = await _privacyPreflightReview(snapshot, {',
     'replace': "            var reviewed = { action: 'continue', value: snapshot }; void "
                '_privacyPreflightReview; ({',
     'harness': '_static/ai_assistant/test_ai_assistant__privacy_preflight.mjs'},
    {'id': 'privacy-preflight-contribution-bypassed',
     'why': 'Contribution consent does not waive the final local sensitive-data preflight.',
     'find': "            var review = await _privacyPreflightReview(payload, {\n                title: 'Review dataset contribution',",
     'replace': "            var review = { action: 'continue', value: payload }; void _privacyPreflightReview; ({\n                title: 'Review dataset contribution',",
     'harness': '_static/ai_assistant/test_ai_assistant__privacy_preflight.mjs'},
    {
        "id": "privacy-preflight-finding-retains-source-text",
        "why": "The warning object itself must never become a secondary secret/PII store; findings are category/count only.",
        "find": "                count: count\n            });",
        "replace": "                count: count,\n                value: text\n            });",
        "harness": "_static/ai_assistant/test_ai_assistant__privacy_preflight.mjs",
    },
    {
        "id": "privacy-preflight-redaction-keeps-invisible-controls",
        "why": "When the reader explicitly chooses Redact, bidi/zero-width controls must not remain hidden in the outgoing copy.",
        "find": "        out = out.replace(_privacyFreshRegex(_INVISIBLE_CHARS_RE), '');",
        "replace": "        void _INVISIBLE_CHARS_RE;",
        "harness": "_static/ai_assistant/test_ai_assistant__privacy_preflight.mjs",
    },
    {'id': 'privacy-preflight-share-race-guard-removed',
     'why': 'A delayed Share privacy dialog must not publish a stale snapshot after the conversation '
            'identity changes.',
     'find': "            if (reviewed.action === 'cancel' || opConversationId !== boundConversationId || "
             'opConversationId !== _getConversationId()) return null;',
     'replace': "            if (reviewed.action === 'cancel') return null;",
     'harness': '_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs'},
    {
        "id": "privacy-preflight-no-dom-fails-open",
        "why": "If the warning UI cannot be constructed, flagged data must not silently leave the browser.",
        "find": "            return Promise.resolve({ action: 'cancel', value: value, scan: scan });",
        "replace": "            return Promise.resolve({ action: 'continue', value: value, scan: scan });",
        "harness": "_static/ai_assistant/test_ai_assistant__privacy_preflight.mjs",
    },

    # ── Run 8: YAML/TOML + artifact lifecycle ────────────────────────────
    {
        "id": "run8-yaml-raw-scalar-injection",
        "why": "YAML-looking user text must stay a quoted scalar; emitting it raw can turn tags, anchors, document markers, or mapping syntax into structure.",
        "find": "        return JSON.stringify(String(value));",
        "replace": "        return String(value);",
        "harness": "_static/ai_assistant/test_ai_assistant__serializers.mjs",
    },
    {
        "id": "run8-toml-raw-string-injection",
        "why": "TOML user strings must be escaped/quoted by one helper; raw strings can terminate values and inject tables or keys.",
        "find": "    function _tomlString(value) { return JSON.stringify(String(value)); }",
        "replace": "    function _tomlString(value) { return String(value); }",
        "harness": "_static/ai_assistant/test_ai_assistant__serializers.mjs",
    },
    {
        "id": "run8-direct-download-untracked",
        "why": "Direct toolbar downloads must enter the page-memory artifact registry so every assistant-managed result has truthful lifecycle management.",
        "find": "        _registerManagedConversationArtifact({\n            kind: 'download',",
        "replace": "        void ({\n            kind: 'download',",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {
        "id": "run8-local-remove-skips-blob-revoke",
        "why": "Removing a managed local preview must actually revoke the Blob URL rather than merely hiding its UI record.",
        "find": "                    try { URL.revokeObjectURL(artifact.url); } catch (_e) {}",
        "replace": "                    void artifact.url;",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs",
    },
    {
        "id": "run8-self-contained-removal-overclaims-revocation",
        "why": "A self-contained URL already copied elsewhere cannot be revoked; removal copy must never imply remote deletion.",
        "find": "                showNotification('Removed from this browser. Already copied self-contained links cannot be revoked.', false);",
        "replace": "                showNotification('Link deleted everywhere and revoked.', false);",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation.mjs",
    },
    {
        "id": "run8-global-revoke-bypasses-server-delete",
        "why": "A Global artifact with its edit capability must be revoked at the server; dropping only the local record leaves the public link live.",
        "find": (
            "                _deleteGlobalShare(\n"
            "                    base, revokeUuid, artifact.editToken,"
        ),
        "replace": (
            "                _dropArtifact(artifact.id); void _deleteGlobalShare; (\n"
            "                    base, revokeUuid, artifact.editToken,"
        ),
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },

    # ── Run 9: Global public-link lifecycle tracking ─────────────────────
    {
        "id": "run9-global-ledger-persists-edit-capability",
        "why": "The session-scoped Global artifact ledger may persist public read links for user tracking, but must never persist the private edit/revoke capability.",
        "find": "            _ssSet(_GLOBAL_LEDGER_KEY, JSON.stringify({ schemaVersion: 1, items: safeItems }));",
        "replace": "            _ssSet(_GLOBAL_LEDGER_KEY, JSON.stringify({ schemaVersion: 1, items: safeItems, editToken: 'persisted-secret' }));",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run9-global-ledger-unbounded",
        "why": "A public bearer-link history in sessionStorage must remain bounded so repeated Share creation cannot create unbounded browser storage or UI growth.",
        "find": "            var safeItems = _globalLedger.map(_normalizeGlobalLedgerItem).filter(Boolean).slice(0, _GLOBAL_LEDGER_MAX);",
        "replace": "            var safeItems = _globalLedger.map(_normalizeGlobalLedgerItem).filter(Boolean);",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run9-global-status-downloads-content",
        "why": "Lifecycle status checks must use the fixed /status endpoint with the public locator in the request body, never a capability-bearing path that infrastructure URL logs could capture.",
        "find": (
            "            _fetch(loc.base + '/status', {\n"
            "                method: 'POST', cache: 'no-store', redirect: 'error',"
        ),
        "replace": (
            "            _fetch(loc.base + '/' + encodeURIComponent(loc.id), {\n"
            "                method: 'GET', cache: 'no-store', redirect: 'error',"
        ),
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run9-global-revoke-drops-history",
        "why": "Successful server revocation should transition the managed artifact to a revoked lifecycle tombstone until the user explicitly forgets it, so provided-link history is traceable.",
        "find": "                        _markGlobalArtifactState(artifact, 'revoked');",
        "replace": "                        _dropArtifact(artifact.id);",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs",
    },
    {
        "id": "run9-new-chat-clears-global-ledger",
        "why": "New chat must clear current update state but preserve the public artifact ledger so links already handed to the user remain trackable during the browser session.",
        "find": (
            "            _globalShareState = null;\n"
            "            _saveGlobalSS(null);\n"
            "            _setPreset('standard');"
        ),
        "replace": (
            "            _globalShareState = null;\n"
            "            _saveGlobalSS(null); _ssDel(_GLOBAL_LEDGER_KEY);\n"
            "            _setPreset('standard');"
        ),
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs",
    },

    # ── Run 10: fail-closed Global lifecycle recovery ────────────────────
    {
        "id": "run10-global-recovery-trusts-legacy-object",
        "why": "Session storage is untrusted recovery input; returning a parsed legacy object wholesale can resurrect a persisted edit capability or conversation-derived fields.",
        "find": "                _saveGlobalSS(safe); // destructive scrub of forbidden legacy fields\n                return safe;",
        "replace": "                return state;",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run10-global-404-made-terminal",
        "why": "HTTP 404 is reason-unknown/unavailable, not proof of revocation or expiry; making it terminal destroys re-checkability and can silently discard a live page-memory revoke capability.",
        "find": "            if (state === 'revoked' || state === 'expired') {\n                artifact.editToken = '';",
        "replace": "            if (state === 'revoked' || state === 'expired' || state === 'unavailable') {\n                artifact.editToken = '';",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs",
    },
    {
        "id": "run10-global-patch-410-no-fallback",
        "why": "A server-confirmed expired current share must detach from update state and allow Create Global link to POST a fresh object instead of remaining stuck on a dead PATCH target.",
        "find": "                        if (err.status === 404 || err.status === 405 || err.status === 410) {",
        "replace": "                        if (err.status === 404 || err.status === 405) {",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run10-global-storage-fingerprint-restored",
        "why": "Reload recovery needs only public lifecycle metadata; persisting a conversation-derived content fingerprint adds unnecessary linkable material after mutation authority is intentionally discarded.",
        "find": "                expiresAt: state.expiresAt || null,\n                conversationId: state.conversationId || '',",
        "replace": "                expiresAt: state.expiresAt || null,\n                contentHash: state.contentHash || '',\n                conversationId: state.conversationId || '',",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run10-global-unavailable-forget-escape-removed",
        "why": "A reason-unknown 404 with a retained page-memory edit capability can make Revoke repeatedly return 404; the user still needs an explicit truthful local Forget escape hatch.",
        "find": "                if (artifact.kind === 'global' && artifact.state === 'unavailable' && artifact.editToken) {",
        "replace": "                if (false) {",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs",
    },

    # ── Contribution receipt lifecycle ─────────────────────────────────────
    {
        "id": "contribution-withdraw-button-pending-only",
        "why": (
            "After promotion the same receipt capability must still let the user "
            "withdraw training use; reverting the control to pending-only strands "
            "the lifecycle capability at the point durable copies may exist."
        ),
        "find": "                'Delete pending / withdraw training use',",
        "replace": "                'Delete pending data',",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_contribution_privacy.mjs",
    },
    {
        "id": "contribution-withdraw-overclaims-erasure",
        "why": (
            "Training withdrawal and current-view deletion do not prove removal "
            "from versioned repository history, backups, or provider infrastructure."
        ),
        "find": "                            text.textContent = 'Training withdrawal recorded. Current provider views were removed where possible; versioned provider history is not claimed physically erased.';",
        "replace": "                            text.textContent = 'Training withdrawal recorded. All copies were permanently erased.';",
        "harness": "_static/ai_assistant/test_ai_assistant__feedback_contribution_privacy.mjs",
    },

    # ── Run 17: first-class dataset contribution UX / conversation records ──
    {
        "id": "run17-conversation-error-row-reintroduced",
        "why": (
            "Runtime/error UI rows are not user/assistant conversation training content. "
            "Reintroducing them leaks operational failures into the contributed dialogue."
        ),
        "find": "            if (m.role === 'error') answerIndex++;",
        "replace": "            if (m.role === 'error') { messages.push({ role: 'assistant', content: m.text, ts: m.ts || null }); answerIndex++; }",
        "harness": "_static/ai_assistant/test_ai_assistant__dataset_contribution_ux.mjs",
    },
    {
        "id": "run17-whole-conversation-split-into-message-records",
        "why": (
            "Whole-conversation contribution is one ordered record; splitting messages into "
            "independent records destroys conversational structure and changes consent scope."
        ),
        "find": "            if (conversation) records.push(conversation);",
        "replace": "            if (conversation) records = conversation.messages || [];",
        "harness": "_static/ai_assistant/test_ai_assistant__dataset_contribution_ux.mjs",
    },
    {
        "id": "run17-share-reclaims-contribution-controller",
        "why": (
            "Share and dataset contribution are separate control planes. Reintroducing the "
            "contribution controller inside Share recreates the discoverability and consent-boundary bug."
        ),
        "find": "    function _buildConversationShareSheet(initialFmt) {",
        "replace": "    function _buildConversationShareSheet(initialFmt) {\n        void _postTrainingContribution;",
        "harness": "_static/ai_assistant/test_ai_assistant__dataset_contribution_ux.mjs",
    },

    # ── Run 13: fragment-backed fixed-path Share transport ───────────────
    {
        "id": "run13-global-update-capability-in-path",
        "why": "Current Share update traffic must keep the public read capability out of infrastructure request paths; only the fixed /update path may be used.",
        "find": "        _remotePost(base.replace(/\\/$/, '') + '/update', '', payload, {",
        "replace": "        _remotePost(base.replace(/\\/$/, '') + '/' + encodeURIComponent(shareId), '', payload, {",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run13-global-revoke-capability-in-path",
        "why": "Current Share revoke traffic must use the fixed /revoke path and carry the public locator in the request body, never in the URL path.",
        "find": "        _remotePost(base.replace(/\\/$/, '') + '/revoke', '', { shareId: shareId }, {",
        "replace": "        _remotePost(base.replace(/\\/$/, '') + '/' + encodeURIComponent(shareId), '', { shareId: shareId }, {",
        "harness": "_cf_worker/test_index__global_share_capability.mjs",
    },
    {
        "id": "run13-fragment-ledger-rejected",
        "why": "The bounded lifecycle ledger must retain newly generated #share=<id> URLs across chat resets/reload without restoring private edit authority.",
        "find": "            if (hashAt >= 0) {",
        "replace": "            if (false && hashAt >= 0) {",
        "harness": "_static/ai_assistant/test_ai_assistant__share_conversation_dom.mjs",
    },

    # ── Run 24: bounded remote-response / context ingestion ─────────────
    {
        "id": "run24-response-stream-unavailable-whole-body-fallback",
        "why": (
            "A response boundary that cannot stream cannot prove a pre-buffer byte ceiling. "
            "Falling back to Response.text() restores the memory-exhaustion class B43 closes."
        ),
        "find": "        throw new Error('REMOTE_RESPONSE_STREAM_UNAVAILABLE');",
        "replace": "        return response.text();",
        "harness": "_cf_worker/test_index__bounded_remote_response.mjs",
    },
    {
        "id": "run24-response-actual-byte-limit-disabled",
        "why": (
            "Content-Length is advisory and may be absent or false; actual decoded bytes must "
            "remain bounded while the stream is consumed."
        ),
        "find": "                    if (total > maxBytes) throw new Error('REMOTE_RESPONSE_TOO_LARGE');",
        "replace": "                    if (false) throw new Error('REMOTE_RESPONSE_TOO_LARGE');",
        "harness": "_cf_worker/test_index__bounded_remote_response.mjs",
    },
    {
        "id": "run24-canonical-markdown-whole-body-reintroduced",
        "why": (
            "Canonical Markdown is untrusted remote context. Reintroducing response.text() "
            "would buffer an arbitrary body before the 1 MiB context ceiling can act."
        ),
        "find": "        return _readResponseTextBounded(response, _CANONICAL_RESPONSE_MAX_BYTES);",
        "replace": "        return response.text();",
        "harness": "_cf_worker/test_index__bounded_remote_response.mjs",
    },
    {
        "id": "run24-dataset-discovery-whole-body-reintroduced",
        "why": (
            "Proxy/dataset discovery is a control response and must remain under the 512 KiB "
            "pre-buffer ceiling rather than parsing an unbounded JSON body."
        ),
        "find": "                    return resp.ok ? _readResponseJsonBounded(resp, _CONTROL_RESPONSE_MAX_BYTES)",
        "replace": "                    return resp.ok ? resp.json()",
        "harness": "_cf_worker/test_index__bounded_remote_response.mjs",
    },

    # ── Run 25: semantic-context live rendered visibility authority ─────
    {
        "id": "run25-live-dom-pruning-call-removed",
        "why": (
            "Detached clones are not visibility authorities. Removing the live-DOM pruning call "
            "reintroduces model-only class/layout content before serialization."
        ),
        "find": "        _stripModelOnlyLiveNodes(content, cloned);",
        "replace": "        void cloned;",
        "harness": "_static/isolation/test_ai_assistant_isolation__semantic_context_integrity.mjs",
    },
    {
        "id": "run25-content-visibility-hidden-accepted",
        "why": (
            "content-visibility:hidden is a deterministic rendered-hidden surface and must not "
            "become model-visible context."
        ),
        "find": "                    cs.visibility === 'collapse' || cs.contentVisibility === 'hidden' ||",
        "replace": "                    cs.visibility === 'collapse' || false ||",
        "harness": "_static/isolation/test_ai_assistant_isolation__semantic_context_integrity.mjs",
    },

    # ── Run 173 T87: managed Share artifact responsive action grouping ──
    {
        "id": "share-artifact-metadata-flex-basis-collapses",
        "why": "Resetting the metadata flex shorthand to flex:1 restores a 0% basis, so narrow rows sacrifice the description before moving actions below it.",
        "find": ".ai-assistant-conv-share-artifact-text { min-width:0; flex:1 1 12rem;",
        "replace": ".ai-assistant-conv-share-artifact-text { min-width:0; flex:1;",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_action_responsive_layout.mjs",
        "target": "css",
    },
    {
        "id": "share-artifact-action-escapes-group",
        "why": "A direct-row action becomes an independent flex item again, so the card can no longer move all controls below readable metadata as one responsive unit.",
        "find": "actions.appendChild(copyLink);",
        "replace": "row.appendChild(copyLink);",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_action_responsive_layout.mjs",
    },
    {
        "id": "share-artifact-container-query-disabled",
        "why": "Without inline-size containment the responsive rule cannot follow a docked or resized Share surface independently of the browser viewport.",
        "find": "    container-type:inline-size;\n    container-name:conv-share-artifacts;",
        "replace": "    container-name:conv-share-artifacts;",
        "harness": "_static/ai_assistant/test_ai_assistant__artifact_action_responsive_layout.mjs",
        "target": "css",
    },

    # ── Run 173 T88: mobile model action menu trigger anchoring ──────────
    {
        "id": "model-mobile-actions-reanchored-to-full-row",
        "why": "Putting the popup back on the full model row makes tall metadata cards open the menu far below the ellipsis trigger.",
        "find": "            actionHost.appendChild(actionsWrap);",
        "replace": "            row.appendChild(actionsWrap);",
        "harness": "_static/ai_assistant/test_ai_assistant__model_responsive_actions.mjs",
    },
    {
        "id": "model-mobile-action-host-position-context-removed",
        "why": "Without a positioned action host, the absolutely positioned mobile menu falls back to the model row as its containing block and drifts away from the trigger.",
        "find": ".ai-assistant-panel-model-action-host {\n    position: relative;",
        "replace": ".ai-assistant-panel-model-action-host {\n    position: static;",
        "harness": "_static/ai_assistant/test_ai_assistant__model_responsive_actions.mjs",
        "target": "css",
    },
    {
        "id": "model-mobile-actions-edge-flip-disabled",
        "why": "A trigger near the bottom of the scrollable model sheet must flip its menu above when there is not enough visible space below.",
        "find": "                    actionHost.setAttribute('data-actions-placement', 'top');",
        "replace": "                    actionHost.removeAttribute('data-actions-placement');",
        "harness": "_static/ai_assistant/test_ai_assistant__model_responsive_actions.mjs",
    },

    # ── Run 173 T92: answer-section disclosure discoverability ──────────
    {
        "id": "section-disclosure-action-hint-removed",
        "why": "Without the explicit Show/Hide hint, an open section falls back to looking like an ordinary heading on touch devices where hover cannot teach the affordance.",
        "find": "            summary.appendChild(actionHint);",
        "replace": "            void actionHint;",
        "harness": "_static/ai_assistant/test_ai_assistant__section_disclosure_discoverability.mjs",
    },
    {
        "id": "section-disclosure-state-copy-swapped",
        "why": "The visible action copy must describe the next action. Saying Show while the section is already open reverses the disclosure model for new users.",
        "find": "            hideHint.textContent = 'Hide section';",
        "replace": "            hideHint.textContent = 'Show section';",
        "harness": "_static/ai_assistant/test_ai_assistant__section_disclosure_discoverability.mjs",
    },
    {
        "id": "section-disclosure-persistent-surface-erased",
        "why": "Making the summary transparent again makes discoverability depend on hover; touch users see a heading, not a control.",
        "find": "    background: var(--ai-section-summary-bg);\n    border: 1px solid var(--ai-section-summary-border);",
        "replace": "    background: transparent;\n    border: 1px solid transparent;",
        "harness": "_static/ai_assistant/test_ai_assistant__section_disclosure_discoverability.mjs",
        "target": "css",
    },
    {
        "id": "section-disclosure-closed-chevron-points-down",
        "why": "A down chevron on a closed disclosure reverses the conventional right-closed/down-open model and weakens state recognition.",
        "find": "    transform: rotate(-90deg);\n    transition:\n        transform 0.16s ease,",
        "replace": "    transform: rotate(0deg);\n    transition:\n        transform 0.16s ease,",
        "harness": "_static/ai_assistant/test_ai_assistant__section_disclosure_discoverability.mjs",
        "target": "css",
    },

    # ── Run 173 T93: inline snippet scroll handoff ──────────────────────
    {
        "id": "inline-snippet-traps-vertical-scroll",
        "why": "An answer snippet is prose, not a nested document viewport. Giving it bounded vertical overflow and overscroll containment makes mouse-wheel and touch scrolling stop inside the code region instead of continuing the conversation.",
        "find": "    max-height: none;\n    overflow: visible;\n    overscroll-behavior: auto;",
        "replace": "    max-height: min(70vh, 28rem);\n    overflow-y: auto;\n    overflow-x: hidden;\n    overscroll-behavior: contain;",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_scroll_handoff.mjs",
        "target": "css",
    },
    {
        "id": "inline-snippet-touch-loses-vertical-pan",
        "why": "A horizontally scrollable code cell must still let a finger or pen pan vertically through the conversation. Removing pan-y turns the code block into a touch-scroll dead zone on phones and tablets.",
        "find": "    overscroll-behavior-y: auto;\n    touch-action: pan-x pan-y pinch-zoom;",
        "replace": "    overscroll-behavior-y: auto;\n    touch-action: pan-x pinch-zoom;",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_scroll_handoff.mjs",
        "target": "css",
    },
    {
        "id": "file-sheet-overflow-shorthand-erases-axis-contract",
        "why": "A later overflow shorthand silently overrides overflow-y:auto and recreates the contradictory scroll ownership that made wheel trapping browser-dependent and difficult to reproduce.",
        "find": "    /* A real file sheet owns bounded vertical scrolling. Keep the axes\n       explicit: an `overflow: hidden` shorthand here used to silently erase\n       overflow-y:auto while leaving overscroll containment behind. */\n    overflow-y: auto;\n    overflow-x: hidden;\n    overscroll-behavior: contain;",
        "replace": "    /* A real file sheet owns bounded vertical scrolling. Keep the axes\n       explicit: an `overflow: hidden` shorthand here used to silently erase\n       overflow-y:auto while leaving overscroll containment behind. */\n    overflow-y: auto;\n    overflow-x: hidden;\n    overscroll-behavior: contain;\n    overflow: hidden;",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_scroll_handoff.mjs",
        "target": "css",
    },
    {
        "id": "inline-snippet-restores-scroll-container-clipping",
        "why": "Using overflow:hidden for rounded corners makes the snippet a clipping scroll container again. The visual crop must stay independent of scroll ownership so vertical gestures can chain out.",
        "find": "    clip-path: inset(0 round 8px);",
        "replace": "    overflow: hidden;",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_scroll_handoff.mjs",
        "target": "css",
    },


    # ── Run 173 T94: Presented-file segmented-control parity ─────────────
    {
        "id": "presented-file-download-drops-base-segment-class",
        "why": "Presented-file Download must reuse the exact normal-artifact segment class; a parallel visual class lets padding, color and separator geometry drift again.",
        "find": "download.className = 'ai-md-artifact-download-label ai-assistant-panel-changed-file-download';",
        "replace": "download.className = 'ai-assistant-panel-changed-file-download';",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_segment_parity.mjs",
    },
    {
        "id": "presented-file-primary-restores-third-column",
        "why": "A third primary-row column recreates the old preview|download|overflow authority instead of one shared artifact group plus overflow.",
        "find": ".ai-assistant-panel-changed-file-primary {\n    display: grid;\n    grid-template-columns: minmax(0, 1fr) auto;\n    align-items: stretch;",
        "replace": ".ai-assistant-panel-changed-file-primary {\n    display: grid;\n    grid-template-columns: minmax(0, 1fr) auto auto;\n    align-items: stretch;",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_segment_parity.mjs",
        "target": "css",
    },
    {
        "id": "artifact-segment-card-restores-full-width-constraint",
        "why": "A width:100% card inside the segmented flex group competes with the separator and Download segment instead of yielding the remaining width cleanly.",
        "find": "    flex: 1 1 auto;\n    min-width: 0;\n    width: auto;",
        "replace": "    flex: 1 1 auto;\n    min-width: 0;\n    width: 100%;",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_segment_parity.mjs",
        "target": "css",
    },
    {
        "id": "artifact-segment-separator-moved-before-preview",
        "why": "The visual contract is preview content first, then the separator, then Download. Reordering the shared builder recreates the exact misplaced-divider defect.",
        "find": "        group.appendChild(primary);\n        var sep = document.createElement('span');",
        "replace": "        var sep = document.createElement('span');\n        group.appendChild(sep);\n        group.appendChild(primary);",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_segment_parity.mjs",
    },
    {
        "id": "presented-file-parallel-preview-geometry-reintroduced",
        "why": "A Presented-only geometry override recreates the duplicate CSS authority that allowed the same component to render differently from normal artifact cards.",
        "find": ".ai-assistant-panel-changed-file-primary > .ai-md-artifact-group {\n    min-width: 0;\n}",
        "replace": ".ai-assistant-panel-changed-file-primary > .ai-md-artifact-group {\n    min-width: 0;\n}\n.ai-md-artifact-group > .ai-assistant-panel-changed-file-preview { width: 100%; }",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_segment_parity.mjs",
        "target": "css",
    },


    # ── Run 173 T95: Presented-file responsive segment continuity ───────
    {
        "id": "presented-file-mobile-download-reclaims-full-width",
        "why": "At <=560px the Download control lives inside the shared segmented group. width:100% makes that trailing segment consume the group and collapses Preview toward zero, which visually moves the separator to the left edge on mobile.",
        "find": "@media (max-width: 560px) {\n    .ai-assistant-panel-activity { margin-left: .08rem; margin-right: .08rem; }\n}",
        "replace": "@media (max-width: 560px) {\n    .ai-assistant-panel-activity { margin-left: .08rem; margin-right: .08rem; }\n    .ai-assistant-panel-changed-file-download { width: 100%; }\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_responsive_layout.mjs",
        "target": "css",
    },
    {
        "id": "presented-file-responsive-heading-returns-to-viewport",
        "why": "Presented files live in a resizable panel, so a viewport query can fire while the component is wide or fail while the component is narrow. The component's own inline size is the responsive authority.",
        "find": "@container ai-artifact-surface (max-width: 35rem) {",
        "replace": "@media (max-width: 560px) {",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_responsive_layout.mjs",
        "target": "css",
    },
    {
        "id": "presented-file-list-restores-min-content-trap",
        "why": "Without min-width:0 on the Presented-files list, long filenames may impose their min-content width on the grid and force the segmented control beyond a small phone panel.",
        "find": ".ai-assistant-panel-changed-files-list {\n    display: grid;\n    gap: .35rem;\n    min-width: 0;\n}",
        "replace": ".ai-assistant-panel-changed-files-list {\n    display: grid;\n    gap: .35rem;\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_responsive_layout.mjs",
        "target": "css",
    },
    {
        "id": "presented-file-primary-restores-min-content-trap",
        "why": "The primary grid must be allowed to shrink below a long filename's intrinsic width; otherwise the overflow trigger stays visible by pushing the shared artifact group outside the panel.",
        "find": ".ai-assistant-panel-changed-file-primary {\n    display: grid;\n    grid-template-columns: minmax(0, 1fr) auto;\n    align-items: stretch;\n    gap: .35rem;\n    min-width: 0;\n    position: relative;\n}",
        "replace": ".ai-assistant-panel-changed-file-primary {\n    display: grid;\n    grid-template-columns: minmax(0, 1fr) auto;\n    align-items: stretch;\n    gap: .35rem;\n    position: relative;\n}",
        "harness": "_static/ai_assistant/test_ai_assistant__presented_file_responsive_layout.mjs",
        "target": "css",
    },


    # ── Run 173 T97: snippet / Presented-file menu workflow parity ──────
    {
        "id": "snippet-menu-drops-open-in-sheet",
        "why": "Both file menus should begin with the same low-risk inspection action. Removing Open in a sheet makes snippets and Presented files teach different workflows again.",
        "find": "                { label: 'Open in a sheet', hint: 'Full view with line numbers', icon: ICONS.terms,\n                  run: function () { _openAttachmentPreview(snippetPreviewItem(), card); } },",
        "replace": "                { label: 'Preview details', hint: 'Full view with line numbers', icon: ICONS.terms,\n                  run: function () { _openAttachmentPreview(snippetPreviewItem(), card); } },",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_file_menu_parity.mjs",
    },
    {
        "id": "snippet-menu-offers-patch-before-tracking",
        "why": "An anonymous snippet has no stable repository path or revision base. Offering Download patch before tracking promises a git operation the system cannot define honestly.",
        "find": "                { label: 'Track as file\\u2026', hint: 'Add revisions, diffs and patch export', icon: ICONS.gitMark,",
        "replace": "                { label: 'Download patch', hint: 'Apply with git am', icon: ICONS.gitMark,",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_file_menu_parity.mjs",
    },
    {
        "id": "promoted-snippet-menu-does-not-graduate",
        "why": "After a snippet gains tracked identity, keeping the pre-tracking menu forces users to manage one file through two inconsistent action vocabularies and hides patch export from the original trigger.",
        "find": "            if (entry) return _fileOverflowItems(entry.key);",
        "replace": "            if (entry) void entry;",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_file_menu_parity.mjs",
    },
    {
        "id": "snippet-continue-bypasses-tracking",
        "why": "Continue editing needs stable file identity so the next-turn attachment is revision-bound. Bypassing tracking turns it back into anonymous bytes with no diff or patch lineage.",
        "find": "                      var tracked = trackSnippet();\n                      if (tracked) _generatedArtifactContinueEditing(tracked.key);",
        "replace": "                      var tracked = promotedEntry();\n                      if (tracked) _generatedArtifactContinueEditing(tracked.key);",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_file_menu_parity.mjs",
    },
    {
        "id": "snippet-promotion-stops-returning-identity",
        "why": "The menu workflow needs the registered entry immediately so Continue editing can stage the exact promoted revision and the same trigger can graduate to tracked-file actions.",
        "find": "        _generatedArtifactRefreshRefs(entry.key);\n        return entry;",
        "replace": "        _generatedArtifactRefreshRefs(entry.key);\n        return;",
        "harness": "_static/ai_assistant/test_ai_assistant__snippet_file_menu_parity.mjs",
    },

]
