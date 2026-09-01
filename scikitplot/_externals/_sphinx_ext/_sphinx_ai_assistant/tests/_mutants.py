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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
    },
    {
        "id": "fence-ignores-limit",
        "why": "An unbounded context blows the model's window and the bill.",
        "find": "        var body = text.slice(0, max);",
        "replace": "        var body = text;",
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
    },
    {
        "id": "custom-prompt-unfenced",
        "why": (
            "Substituting raw text into {context} would make the safer path "
            "the one nobody takes."
        ),
        "find": "            ? cfg.panelSystemPrompt.replace('{context}', _fenced)",
        "replace": "            ? cfg.panelSystemPrompt.replace('{context}', _cleaned.text)",
        "harness": "test_untrusted_context.mjs",
    },
    # ── Egress redaction ──────────────────────────────────────────────────
    {
        "id": "redaction-removed",
        "why": "A key published in a docstring is sent to a third-party proxy.",
        "find": "        var _redacted = _redactSecrets(_cleaned.text);",
        "replace": "        var _redacted = { text: _cleaned.text, findings: [] };",
        "harness": "test_untrusted_context.mjs",
    },
    {
        "id": "redaction-silent",
        "why": (
            "A silent redaction protects the secret and leaves the reader "
            "never learning their key is published on the page."
        ),
        "find": "        _announceRedaction(_redacted.findings);\n",
        "replace": "",
        "harness": "test_untrusted_context.mjs",
    },
    {
        "id": "redaction-loses-kind",
        "why": (
            "The placeholder tells the model what sort of thing was removed so "
            "it can answer sensibly; a bare marker tells it nothing."
        ),
        "find": "                return '[redacted:' + spec.name + ']';",
        "replace": "                return '[redacted]';",
        "harness": "test_untrusted_context.mjs",
    },
    {
        "id": "secret-pattern-too-loose",
        "why": (
            "A pattern that fires on prose mangles the documentation it is "
            "protecting, and a redactor that does that gets switched off."
        ),
        "find": "{ name: 'openai_key',         re: /\\bsk-[A-Za-z0-9]{20,}\\b/g },",
        "replace": "{ name: 'openai_key',         re: /\\bsk-[A-Za-z0-9]{2,}\\b/g },",
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_untrusted_context.mjs",
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
        "harness": "test_panel_trigger.mjs",
    },
    {
        "id": "trigger-minimize-strands",
        "why": (
            "Honouring a hidden preference while minimized strands a live "
            "transcript behind two dropdown clicks."
        ),
        "find": "            pill:       minimized || (panel !== 'open' && preference),",
        "replace": "            pill:       (panel !== 'open' && preference),",
        "harness": "test_panel_trigger.mjs",
    },
    # ── Export surfaces ───────────────────────────────────────────────────
    {'id': 'export-duplicate-preview',
     'why': 'The live format-card list must contain each registry format exactly once; duplicating the live '
            'registry recreates duplicate data-fmt cards.',
     'find': '    var _EXPORT_CARD_FORMATS = _EXPORT_FORMATS.concat(_EXPORT_STUB_FORMATS);',
     'replace': '    var _EXPORT_CARD_FORMATS = _EXPORT_FORMATS.concat(_EXPORT_FORMATS);',
     'harness': 'test_export_formats.mjs'},
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
        "harness": "test_export_formats.mjs",
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
        "harness": "test_reasoning_support.mjs",
    },
    {
        "id": "discovery-reserved-names-allowed",
        "why": (
            "Without the denylist a compromised proxy can name 'messages' as "
            "its effort parameter and rewrite every request body."
        ),
        "find": "        if (_CAPS_RESERVED_PARAMS.indexOf(name) !== -1) return null;\n",
        "replace": "",
        "harness": "test_reasoning_support.mjs",
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
        "harness": "test_reasoning_support.mjs",
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
        "harness": "test_reasoning_support.mjs",
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
        "harness": "test_reasoning_support.mjs",
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
        "harness": "test_effort_levels.mjs",
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
        "harness": "test_effort_levels.mjs",
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
        "harness": "test_model_overrides.mjs",
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
        "harness": "test_model_overrides.mjs",
    },
    {
        "id": "override-mutates-the-source",
        "why": (
            "Mutating the build-time entry makes the diff unshowable and the "
            "original unrecoverable, so 'reset' cannot restore anything."
        ),
        "find": "                var merged = {};",
        "replace": "                var merged = m;",
        "harness": "test_model_overrides.mjs",
    },
    {
        "id": "active-model-skips-overrides",
        "why": (
            "The request would go to the endpoint the reader just corrected "
            "away from -- the bug the feature exists to fix, one layer down."
        ),
        "find": "        var models = _MODEL_STORE.applyOverrides(builtins).filter(function (m) {",
        "replace": "        var models = builtins.filter(function (m) {",
        "harness": "test_model_overrides.mjs",
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
        "harness": "test_model_overrides.mjs",
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
        "harness": "test_model_overrides.mjs",
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
        "harness": "test_model_overrides.mjs",
    },
    {
        "id": "edit-id-stays-editable",
        "why": (
            "The id keys the override and matches the radio row. Editing it "
            "silently creates a second entry instead of correcting the first."
        ),
        "find": "            idInp.disabled = true;",
        "replace": "            idInp.disabled = false;",
        "harness": "test_model_overrides.mjs",
    },
    {
        "id": "edit-click-selects-the-model",
        "why": (
            "The button lives inside a <label>; without preventDefault a click "
            "on 'edit' also switches the active model."
        ),
        "find": "            function _requestModelEdit(ev) {\n                if (ev) {\n                    ev.preventDefault();\n                    ev.stopPropagation();",
        "replace": "            function _requestModelEdit(ev) {\n                if (ev) {\n                    ev.stopPropagation();",
        "harness": "test_model_overrides.mjs",
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
        "harness": "test_custom_section_dom.mjs",
    },

    # ── Composition wiring ────────────────────────────────────────────────
    # These are caught by tests/test_end_to_end_context.py rather than a .mjs
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
        "harness": "test_menu_shortcuts.mjs",
    },
    {
        "id": "shortcuts-eat-typing",
        "why": (
            "Without the text-entry guard, typing 'model' in the composer "
            "opens four sheets and deletes the conversation."
        ),
        "find": "            if (_isTextEntryTarget(e.target)) return;\n",
        "replace": "",
        "harness": "test_menu_shortcuts.mjs",
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
        "harness": "test_menu_shortcuts.mjs",
    },

    # ── Unified Share conversation ────────────────────────────────────────
    {'id': 'share-panels-eager-again',
     'why': 'Switching formats must replace the one descriptor panel rather than accumulating duplicate '
            'per-format DOM surfaces inside the unified sheet.',
     'find': '            while (formatHost.firstChild) formatHost.removeChild(formatHost.firstChild);',
     'replace': '            void formatHost.firstChild;',
     'harness': 'test_share_conversation_dom.mjs'},
    {
        "id": "share-dispatch-skips-format-selection",
        "why": (
            "Opening the unified sheet without selecting the clicked format "
            "makes a JSON click show whichever format was previously active. "
            "One shell is only correct if dispatch still preserves user intent."
        ),
        "find": "            if (!convShareSheet._selectExportFormat(fmt)) { return; }",
        "replace": "            if (!convShareSheet) { return; }",
        "harness": "test_share_conversation.mjs",
    },
    {'id': 'share-new-chat-does-not-reset-cached-panels',
     'why': 'New chat must clear active result state without discarding page-memory Global edit '
            'capabilities needed to revoke links created by the previous conversation.',
     'find': '            resultState = null;\n            _globalShareState = null;',
     'replace': '            managedArtifacts = [];\n'
                '            resultState = null;\n'
                '            _globalShareState = null;',
     'harness': 'test_share_conversation_dom.mjs'},
    {'id': 'share-permanent-save-crosses-conversation',
     'why': 'A delayed privacy decision for a self-contained/local share must remain bound to the '
            'conversation that opened the dialog and must not publish after New chat.',
     'find': "            if (reviewed.action === 'cancel' || opConversationId !== boundConversationId || "
             'opConversationId !== _getConversationId()) return null;',
     'replace': "            if (reviewed.action === 'cancel') return null;",
     'harness': 'test_share_conversation_dom.mjs'},
    {'id': 'share-global-save-crosses-conversation',
     'why': 'A delayed Global Share response belongs to the conversation that created it; removing the '
            'success identity guard can attach old server state to a new chat.',
     'find': '            function success(res) {\n'
             '                if (opConversationId !== boundConversationId || opConversationId !== '
             '_getConversationId()) return;',
     'replace': '            function success(res) {\n                void opConversationId;',
     'harness': 'test_share_conversation.mjs'},
    {
        "id": "share-export-mode-observers-not-notified",
        "why": (
            "There are many Download/Share controls (main header plus sheet "
            "toolbars). If the setter stops notifying observers, only the "
            "control clicked by the reader appears current and the others lie."
        ),
        "find": "        _notifyExportState();",
        "replace": "        void _exportLinkMode;",
        "harness": "test_share_conversation.mjs",
    },
    {
        "id": "share-export-mode-singleton-id-returns",
        "why": (
            "Giving every cloned export-mode control the same id recreates the "
            "original stale-toolbar bug and invalid duplicate-id DOM. Controls "
            "must be observer views, never singleton-id authorities."
        ),
        "find": "        row.type = 'button';",
        "replace": (
            "        row.type = 'button';\n"
            "        row.id = 'ai-assistant-export-link-toggle';"
        ),
        "harness": "test_share_conversation.mjs",
    },
    {'id': 'share-conversation-id-follows-trimmed-head',
     'why': 'Transcript position is not conversation identity. The unified Share sheet must bind to the '
            'explicit conversation UUID so trimming cannot change ownership.',
     'find': '        var boundConversationId = _getConversationId();',
     'replace': "        var boundConversationId = _transcript.length ? String(_transcript[0].ts) : '';",
     'harness': 'test_share_conversation.mjs'},
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
        "harness": "test_share_conversation.mjs",
    },
    {'id': 'share-session-copy-overstates-privacy',
     'why': 'Self-contained links are embedded readable data, not encrypted or remotely revocable. UI copy '
            'must not promise stronger privacy/lifecycle semantics.',
     'find': "            'Reviewed static HTML · not encrypted · copied links cannot be revoked');",
     'replace': "            'Reviewed static HTML · encrypted · removable everywhere');",
     'harness': 'test_share_conversation.mjs'},

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
        "harness": "test_endpoint_secret_lifecycle.mjs",
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
        "harness": "test_endpoint_secret_lifecycle.mjs",
    },

    {'id': 'share-unified-sheet-host-use-before-create',
     'why': 'The unified format host must exist before rendering the active descriptor; source-only checks '
            'cannot catch a browser construction crash.',
     'find': "        var formatHost = document.createElement('div');\n"
             "        formatHost.className = 'ai-assistant-conv-share-format-host';",
     'replace': '        var formatHost;\n        void 0;',
     'harness': 'test_share_conversation_dom.mjs'},

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
        "harness": "test_active_content_isolation.mjs",
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
        "harness": "test_active_content_isolation.mjs",
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
        "harness": "test_active_content_isolation.mjs",
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
        "harness": "test_active_content_isolation.mjs",
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
        "harness": "test_active_content_isolation.mjs",
    },

    # ── Global Share server capability boundary (B18 Run 3) ─────────────
    {'id': 'global-share-client-mime-authority-returns',
     'why': 'Global Share must send canonical snapshot + allowlisted format only; restoring client MIME '
            'lets direct callers regain representation authority.',
     'find': '            var payload = recoveringGlobal ? _pendingGlobalCreate.payload : { snapshot: snapshot, format: meta.fmt, ttlDays: g.ttlDays };',
     'replace': '            var payload = recoveringGlobal ? _pendingGlobalCreate.payload : { snapshot: snapshot, format: meta.fmt, mimeType: meta.mime, '
                'ext: meta.ext, ttlDays: g.ttlDays };',
     'harness': 'test_global_share_capability.mjs'},
    {'id': 'global-share-patch-uses-endpoint-token',
     'why': 'Share update ownership is the per-share edit capability, never the endpoint create credential.',
     'find': "                _patchGlobalShare(base, _globalShareState.uuid, _globalShareState.editToken,",
     'replace': "                _patchGlobalShare(base, _globalShareState.uuid, g.token,",
     'harness': 'test_global_share_capability.mjs'},
    {'id': 'global-share-edit-token-persisted',
     'why': 'The Global edit capability is intentionally page-memory-only; persisting it enlarges '
            'same-origin credential exposure.',
     'find': "                conversationId: state.conversationId || '',\n"
             "                format: state.format || '',",
     'replace': "                conversationId: state.conversationId || '',\n"
                "                editToken: state.editToken || '',\n"
                "                format: state.format || '',",
     'harness': 'test_global_share_capability.mjs'},


    # ── Run 4: server-owned prompt authority ──────────────────────────────
    {
        "id": "chat-contract-negotiation-bypassed",
        "why": "If the advertised contract no longer controls the structured path, the bundled proxy receives legacy client-authored system authority or custom endpoints receive an incompatible body.",
        "find": "        var useStructuredProxy = (proxyContract === _CHAT_CONTRACT_V1);",
        "replace": "        var useStructuredProxy = false;",
        "harness": "test_chat_authority.mjs",
    },
    {
        "id": "chat-structured-user-message-replaced-by-system",
        "why": "The trusted proxy contract must carry typed user input, never a client-authored system message.",
        "find": "                user_message: question,",
        "replace": "                messages: [{ role: 'system', content: systemPrompt }],",
        "harness": "test_chat_authority.mjs",
    },
    {
        "id": "chat-structured-context-skips-redaction",
        "why": "A structured proxy request that sends the cleaned-but-unredacted page would reintroduce page-secret exfiltration.",
        "find": "                    page_text: _redacted.text.slice(0, contextLimit),",
        "replace": "                    page_text: _cleaned.text.slice(0, contextLimit),",
        "harness": "test_chat_authority.mjs",
    },

    # ── Run 6: feedback / contribution privacy lifecycle ────────────────
    {
        "id": "feedback-network-recollects-query",
        "why": "Ordinary rating telemetry must never silently recollect the user question; full Q&A belongs only to explicit contribution consent.",
        "find": "            ratingMode: detail.ratingMode || null,\n            ts: detail.ts || Date.now()",
        "replace": "            ratingMode: detail.ratingMode || null,\n            query: detail.query || '',\n            ts: detail.ts || Date.now()",
        "harness": "test_feedback_contribution_privacy.mjs",
    },
    {
        "id": "feedback-telemetry-consent-fails-open",
        "why": "Network rating telemetry must require an explicit current structured consent record; missing enabled=true must never inherit authority.",
        "find": "            if (!saved || saved.enabled !== true ||\n                    saved.version !== _FEEDBACK_TELEMETRY_CONSENT_VERSION) {",
        "replace": "            if (!saved || saved.enabled === false ||\n                    saved.version !== _FEEDBACK_TELEMETRY_CONSENT_VERSION) {",
        "harness": "test_feedback_telemetry_consent.mjs",
    },
    {
        "id": "feedback-telemetry-helper-consent-gate-removed",
        "why": "Even an internal caller must not be able to send feedback telemetry when the user has not opted in.",
        "find": "        if (!_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt) { return false; }",
        "replace": "        if (false && (!_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt)) { return false; }",
        "harness": "test_feedback_telemetry_consent.mjs",
    },
    {
        "id": "feedback-public-event-reexposes-content",
        "why": "The public feedback DOM event must never rebroadcast Q&A/note/model/page content to arbitrary page listeners.",
        "find": "        var out = _feedbackTelemetryPayload(detail);",
        "replace": "        var out = Object.assign({}, detail);",
        "harness": "test_feedback_telemetry_consent.mjs",
    },
    {
        "id": "feedback-retract-ignores-opt-out",
        "why": "Turning telemetry off must stop all future feedback network traffic, including hidden retraction housekeeping requests.",
        "find": "        if (!url || !prevSessionId || !_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt) {",
        "replace": "        if (!url || !prevSessionId) {",
        "harness": "test_feedback_telemetry_consent.mjs",
    },
    {'id': 'contribution-consent-version-disabled',
     'why': 'Explicit contribution must carry the active versioned consent so stale pages cannot submit '
            'under materially changed terms.',
     'find': "    var _CONTRIBUTION_CONSENT_VERSION = '2.0.0';",
     'replace': "    var _CONTRIBUTION_CONSENT_VERSION = '1.0.0';",
     'harness': 'test_dataset_contribution_ux.mjs'},
    {'id': 'contribution-session-linkage-restored',
     'why': 'Contribution does not need the stable browser conversation identifier; re-adding it increases '
            'linkability of personal records.',
     'find': "            page: _sanitizePage(((typeof _pageUrl === 'function') ? _pageUrl() : ((typeof location !== 'undefined') ? location.href : ''))),",
     'replace': "            sessionId: _sessionId,\n            page: _sanitizePage(((typeof _pageUrl === 'function') ? _pageUrl() : ((typeof location !== 'undefined') ? location.href : ''))),",
     'harness': 'test_dataset_contribution_ux.mjs'},

    # ── Run 7: local privacy preflight / sensitive-input protection ─────
    {
        "id": "privacy-preflight-inference-bypassed",
        "why": "A user-entered credential or personal datum must be reviewed before the browser sends it to an external inference endpoint.",
        "find": "            var privacyDecision = await _privacyPreflightReview(outboundCandidate, {",
        "replace": "            var privacyDecision = { action: 'continue', value: outboundCandidate }; void _privacyPreflightReview; ({",
        "harness": "test_privacy_preflight.mjs",
    },
    {'id': 'privacy-preflight-share-bypassed',
     'why': 'Every Share destination must review the exact canonical outbound snapshot before '
            'serialization.',
     'find': '            var reviewed = await _privacyPreflightReview(snapshot, {',
     'replace': "            var reviewed = { action: 'continue', value: snapshot }; void "
                '_privacyPreflightReview; ({',
     'harness': 'test_privacy_preflight.mjs'},
    {'id': 'privacy-preflight-contribution-bypassed',
     'why': 'Contribution consent does not waive the final local sensitive-data preflight.',
     'find': "            var review = await _privacyPreflightReview(payload, {\n                title: 'Review dataset contribution',",
     'replace': "            var review = { action: 'continue', value: payload }; void _privacyPreflightReview; ({\n                title: 'Review dataset contribution',",
     'harness': 'test_privacy_preflight.mjs'},
    {
        "id": "privacy-preflight-finding-retains-source-text",
        "why": "The warning object itself must never become a secondary secret/PII store; findings are category/count only.",
        "find": "                count: count\n            });",
        "replace": "                count: count,\n                value: text\n            });",
        "harness": "test_privacy_preflight.mjs",
    },
    {
        "id": "privacy-preflight-redaction-keeps-invisible-controls",
        "why": "When the reader explicitly chooses Redact, bidi/zero-width controls must not remain hidden in the outgoing copy.",
        "find": "        out = out.replace(_privacyFreshRegex(_INVISIBLE_CHARS_RE), '');",
        "replace": "        void _INVISIBLE_CHARS_RE;",
        "harness": "test_privacy_preflight.mjs",
    },
    {'id': 'privacy-preflight-share-race-guard-removed',
     'why': 'A delayed Share privacy dialog must not publish a stale snapshot after the conversation '
            'identity changes.',
     'find': "            if (reviewed.action === 'cancel' || opConversationId !== boundConversationId || "
             'opConversationId !== _getConversationId()) return null;',
     'replace': "            if (reviewed.action === 'cancel') return null;",
     'harness': 'test_share_conversation_dom.mjs'},
    {
        "id": "privacy-preflight-no-dom-fails-open",
        "why": "If the warning UI cannot be constructed, flagged data must not silently leave the browser.",
        "find": "            return Promise.resolve({ action: 'cancel', value: value, scan: scan });",
        "replace": "            return Promise.resolve({ action: 'continue', value: value, scan: scan });",
        "harness": "test_privacy_preflight.mjs",
    },

    # ── Run 8: YAML/TOML + artifact lifecycle ────────────────────────────
    {
        "id": "run8-yaml-raw-scalar-injection",
        "why": "YAML-looking user text must stay a quoted scalar; emitting it raw can turn tags, anchors, document markers, or mapping syntax into structure.",
        "find": "        return JSON.stringify(String(value));",
        "replace": "        return String(value);",
        "harness": "test_run8_serializers.mjs",
    },
    {
        "id": "run8-toml-raw-string-injection",
        "why": "TOML user strings must be escaped/quoted by one helper; raw strings can terminate values and inject tables or keys.",
        "find": "    function _tomlString(value) { return JSON.stringify(String(value)); }",
        "replace": "    function _tomlString(value) { return String(value); }",
        "harness": "test_run8_serializers.mjs",
    },
    {
        "id": "run8-direct-download-untracked",
        "why": "Direct toolbar downloads must enter the page-memory artifact registry so every assistant-managed result has truthful lifecycle management.",
        "find": "        _registerManagedConversationArtifact({\n            kind: 'download',",
        "replace": "        void ({\n            kind: 'download',",
        "harness": "test_share_conversation.mjs",
    },
    {
        "id": "run8-local-remove-skips-blob-revoke",
        "why": "Removing a managed local preview must actually revoke the Blob URL rather than merely hiding its UI record.",
        "find": "                    try { URL.revokeObjectURL(artifact.url); } catch (_e) {}",
        "replace": "                    void artifact.url;",
        "harness": "test_share_conversation_dom.mjs",
    },
    {
        "id": "run8-self-contained-removal-overclaims-revocation",
        "why": "A self-contained URL already copied elsewhere cannot be revoked; removal copy must never imply remote deletion.",
        "find": "                showNotification('Removed from this browser. Already copied self-contained links cannot be revoked.', false);",
        "replace": "                showNotification('Link deleted everywhere and revoked.', false);",
        "harness": "test_share_conversation.mjs",
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
        "harness": "test_global_share_capability.mjs",
    },

    # ── Run 9: Global public-link lifecycle tracking ─────────────────────
    {
        "id": "run9-global-ledger-persists-edit-capability",
        "why": "The session-scoped Global artifact ledger may persist public read links for user tracking, but must never persist the private edit/revoke capability.",
        "find": "            _ssSet(_GLOBAL_LEDGER_KEY, JSON.stringify({ schemaVersion: 1, items: safeItems }));",
        "replace": "            _ssSet(_GLOBAL_LEDGER_KEY, JSON.stringify({ schemaVersion: 1, items: safeItems, editToken: 'persisted-secret' }));",
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run9-global-ledger-unbounded",
        "why": "A public bearer-link history in sessionStorage must remain bounded so repeated Share creation cannot create unbounded browser storage or UI growth.",
        "find": "            var safeItems = _globalLedger.map(_normalizeGlobalLedgerItem).filter(Boolean).slice(0, _GLOBAL_LEDGER_MAX);",
        "replace": "            var safeItems = _globalLedger.map(_normalizeGlobalLedgerItem).filter(Boolean);",
        "harness": "test_global_share_capability.mjs",
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
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run9-global-revoke-drops-history",
        "why": "Successful server revocation should transition the managed artifact to a revoked lifecycle tombstone until the user explicitly forgets it, so provided-link history is traceable.",
        "find": "                        _markGlobalArtifactState(artifact, 'revoked');",
        "replace": "                        _dropArtifact(artifact.id);",
        "harness": "test_share_conversation_dom.mjs",
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
        "harness": "test_share_conversation_dom.mjs",
    },

    # ── Run 10: fail-closed Global lifecycle recovery ────────────────────
    {
        "id": "run10-global-recovery-trusts-legacy-object",
        "why": "Session storage is untrusted recovery input; returning a parsed legacy object wholesale can resurrect a persisted edit capability or conversation-derived fields.",
        "find": "                _saveGlobalSS(safe); // destructive scrub of forbidden legacy fields\n                return safe;",
        "replace": "                return state;",
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run10-global-404-made-terminal",
        "why": "HTTP 404 is reason-unknown/unavailable, not proof of revocation or expiry; making it terminal destroys re-checkability and can silently discard a live page-memory revoke capability.",
        "find": "            if (state === 'revoked' || state === 'expired') {\n                artifact.editToken = '';",
        "replace": "            if (state === 'revoked' || state === 'expired' || state === 'unavailable') {\n                artifact.editToken = '';",
        "harness": "test_share_conversation_dom.mjs",
    },
    {
        "id": "run10-global-patch-410-no-fallback",
        "why": "A server-confirmed expired current share must detach from update state and allow Create Global link to POST a fresh object instead of remaining stuck on a dead PATCH target.",
        "find": "                        if (err.status === 404 || err.status === 405 || err.status === 410) {",
        "replace": "                        if (err.status === 404 || err.status === 405) {",
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run10-global-storage-fingerprint-restored",
        "why": "Reload recovery needs only public lifecycle metadata; persisting a conversation-derived content fingerprint adds unnecessary linkable material after mutation authority is intentionally discarded.",
        "find": "                expiresAt: state.expiresAt || null,\n                conversationId: state.conversationId || '',",
        "replace": "                expiresAt: state.expiresAt || null,\n                contentHash: state.contentHash || '',\n                conversationId: state.conversationId || '',",
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run10-global-unavailable-forget-escape-removed",
        "why": "A reason-unknown 404 with a retained page-memory edit capability can make Revoke repeatedly return 404; the user still needs an explicit truthful local Forget escape hatch.",
        "find": "                if (artifact.kind === 'global' && artifact.state === 'unavailable' && artifact.editToken) {",
        "replace": "                if (false) {",
        "harness": "test_share_conversation_dom.mjs",
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
        "harness": "test_feedback_contribution_privacy.mjs",
    },
    {
        "id": "contribution-withdraw-overclaims-erasure",
        "why": (
            "Training withdrawal and current-view deletion do not prove removal "
            "from versioned repository history, backups, or provider infrastructure."
        ),
        "find": "                            text.textContent = 'Training withdrawal recorded. Current provider views were removed where possible; versioned provider history is not claimed physically erased.';",
        "replace": "                            text.textContent = 'Training withdrawal recorded. All copies were permanently erased.';",
        "harness": "test_feedback_contribution_privacy.mjs",
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
        "harness": "test_dataset_contribution_ux.mjs",
    },
    {
        "id": "run17-whole-conversation-split-into-message-records",
        "why": (
            "Whole-conversation contribution is one ordered record; splitting messages into "
            "independent records destroys conversational structure and changes consent scope."
        ),
        "find": "            if (conversation) records.push(conversation);",
        "replace": "            if (conversation) records = conversation.messages || [];",
        "harness": "test_dataset_contribution_ux.mjs",
    },
    {
        "id": "run17-share-reclaims-contribution-controller",
        "why": (
            "Share and dataset contribution are separate control planes. Reintroducing the "
            "contribution controller inside Share recreates the discoverability and consent-boundary bug."
        ),
        "find": "    function _buildConversationShareSheet(initialFmt) {",
        "replace": "    function _buildConversationShareSheet(initialFmt) {\n        void _postTrainingContribution;",
        "harness": "test_dataset_contribution_ux.mjs",
    },

    # ── Run 13: fragment-backed fixed-path Share transport ───────────────
    {
        "id": "run13-global-update-capability-in-path",
        "why": "Current Share update traffic must keep the public read capability out of infrastructure request paths; only the fixed /update path may be used.",
        "find": "        _remotePost(base.replace(/\\/$/, '') + '/update', '', payload, {",
        "replace": "        _remotePost(base.replace(/\\/$/, '') + '/' + encodeURIComponent(shareId), '', payload, {",
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run13-global-revoke-capability-in-path",
        "why": "Current Share revoke traffic must use the fixed /revoke path and carry the public locator in the request body, never in the URL path.",
        "find": "        _remotePost(base.replace(/\\/$/, '') + '/revoke', '', { shareId: shareId }, {",
        "replace": "        _remotePost(base.replace(/\\/$/, '') + '/' + encodeURIComponent(shareId), '', { shareId: shareId }, {",
        "harness": "test_global_share_capability.mjs",
    },
    {
        "id": "run13-fragment-ledger-rejected",
        "why": "The bounded lifecycle ledger must retain newly generated #share=<id> URLs across chat resets/reload without restoring private edit authority.",
        "find": "            if (hashAt >= 0) {",
        "replace": "            if (false && hashAt >= 0) {",
        "harness": "test_share_conversation_dom.mjs",
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
        "harness": "test_run24_bounded_remote_response.mjs",
    },
    {
        "id": "run24-response-actual-byte-limit-disabled",
        "why": (
            "Content-Length is advisory and may be absent or false; actual decoded bytes must "
            "remain bounded while the stream is consumed."
        ),
        "find": "                    if (total > maxBytes) throw new Error('REMOTE_RESPONSE_TOO_LARGE');",
        "replace": "                    if (false) throw new Error('REMOTE_RESPONSE_TOO_LARGE');",
        "harness": "test_run24_bounded_remote_response.mjs",
    },
    {
        "id": "run24-canonical-markdown-whole-body-reintroduced",
        "why": (
            "Canonical Markdown is untrusted remote context. Reintroducing response.text() "
            "would buffer an arbitrary body before the 1 MiB context ceiling can act."
        ),
        "find": "        return _readResponseTextBounded(response, _CANONICAL_RESPONSE_MAX_BYTES);",
        "replace": "        return response.text();",
        "harness": "test_run24_bounded_remote_response.mjs",
    },
    {
        "id": "run24-dataset-discovery-whole-body-reintroduced",
        "why": (
            "Proxy/dataset discovery is a control response and must remain under the 512 KiB "
            "pre-buffer ceiling rather than parsing an unbounded JSON body."
        ),
        "find": "                    return resp.ok ? _readResponseJsonBounded(resp, _CONTROL_RESPONSE_MAX_BYTES)",
        "replace": "                    return resp.ok ? resp.json()",
        "harness": "test_run24_bounded_remote_response.mjs",
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
        "harness": "test_run25_semantic_context_integrity.mjs",
    },
    {
        "id": "run25-content-visibility-hidden-accepted",
        "why": (
            "content-visibility:hidden is a deterministic rendered-hidden surface and must not "
            "become model-visible context."
        ),
        "find": "                    cs.visibility === 'collapse' || cs.contentVisibility === 'hidden' ||",
        "replace": "                    cs.visibility === 'collapse' || false ||",
        "harness": "test_run25_semantic_context_integrity.mjs",
    },

]
