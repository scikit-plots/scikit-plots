# B16 — Share conversation UX and state architecture

Status: **COMPLETE**
Type: **bounded browser UX/state architecture checkpoint**
Subsystem: **_sphinx_ai_assistant / browser Share conversation**

## Objective

Replace repeated JSON/HTML/TXT Share-conversation sheet construction and
hard-coded routing with one registry-driven, format-switchable Share surface,
while preserving direct offline download behavior and making asynchronous state,
conversation lifetime, accessibility, and cross-sheet focus behavior explicit.

## Source anchor

The review started from the user-supplied repository overlay:

```text
scikitplot__sphinx_ai_assistant_runtime_maintenance_repo_overlay(1).zip
sha256: 4f072448cf96730d51d54deca07ce61d56c4b166e4416b75a8330fd99661022d
```

The source anchor is the **input reviewed**, not an output-package hash. The final
package hash is intentionally recorded outside this self-contained archive so
repacking the archive cannot invalidate its own maintenance state.

## Scope

- `_static/ai-assistant.js` export registry, export-mode state, Share panel,
  cross-sheet registry, focus restoration, and conversation-lifetime state.
- `_static/ai-assistant.css` state selectors, unified format switcher, responsive
  and reduced-motion behavior.
- Node source-contract, execution-smoke, and mutation gates for the new boundary.
- Maintenance records for the browser UX contract and deferred follow-ons.

## Non-goals

- Do **not** claim that a browser conversation ID is authentication or
  authorization.
- Do **not** close `AIA-006` / `AIA-C09`; server read/edit capability separation
  remains an independent P0 security campaign.
- Do not redesign the global persistence API or invent new server semantics.
- Do not add speculative export formats merely to exercise the registry.

## Dependency map — before

```text
_EXPORT_FORMATS
   |
   +--> direct download serializers
   |
   +--> format cards/dropdown

export-mode toggle copies
   +--> duplicated DOM id / first-element synchronization

JSON button --> JSON Share sheet --> JSON closure/state
HTML button --> HTML Share sheet --> HTML closure/state
TXT button  --> TXT Share sheet  --> TXT closure/state

several sheet arrays --> open/close/Escape/toolbar/focus behavior
```

The format registry was already the right primitive. The repeated responsibility
lived in sheet construction, routing, state lifetime, and control synchronization.

## Dependency map — after

```text
_EXPORT_FORMATS (metadata + canonical serializers)
   |
   +--> direct offline download
   |
   +--> _openConversationShare(fmt)
             |
             v
      one Share conversation sheet
             |
             +--> registry-driven format tabs
             +--> lazy immutable JSON panel
             +--> lazy immutable HTML panel
             +--> lazy immutable TXT panel

shared export-mode state --> observer controls in every rendered surface

_sheetRegistry --> open/close/Escape/toolbar/focus behavior

conversation identity --> panel state + async completion guards
```

Format, delivery mode, and link lifetime are deliberately separate axes:

- **format** — JSON / HTML / Text;
- **delivery** — direct Download or open Share conversation;
- **link lifetime/scope** — session-only, portable, persistent, or server-backed
  according to the existing feature path.

## Execution record

```yaml
checkpoint: B16
status: COMPLETE
started_at: 2026-08-29
completed_at: 2026-08-29
source_anchor: scikitplot__sphinx_ai_assistant_runtime_maintenance_repo_overlay(1).zip @ 4f072448cf96730d51d54deca07ce61d56c4b166e4416b75a8330fd99661022d
upstream_anchor: null
production_code_modified: true
contracts_touched:
  - export format registry and canonical serialization metadata
  - direct-download versus Share-conversation delivery state
  - unified Share shell with lazy immutable format panels
  - conversation-scoped browser UI state
  - asynchronous format/conversation completion isolation
  - canonical sheet registry and focus restoration
  - repeated-control synchronization and keyboard accessibility
files_changed:
  - _static/ai-assistant.js
  - _static/ai-assistant.css
  - tests/test_export_formats.mjs
  - tests/test_hamburger_more.mjs
  - tests/test_share_conversation.mjs
  - tests/test_share_conversation_dom.mjs
  - tests/_mutants.py
  - maintenance checkpoint/registry/state/tracker/verification/history/lessons/todo records
findings_opened:
  - AIA-018
  - AIA-019
findings_closed:
  - AIA-018
risks:
  - portable self-contained URLs may become impractically long for large conversations
  - fake-DOM execution does not prove responsive layout or browser focus rendering
  - browser conversation identity scopes UI state only and provides no security authority
  - server Share authorization remains separately open as AIA-006 / AIA-C09
rollback: revert B16 runtime and its B16 tests together; do not restore duplicate-id toggle synchronization
```

## Implemented architecture

### One outer Share conversation sheet

`_buildConversationShareSheet(initialFmt)` owns the slide-over shell. It renders
one registry-driven tablist and lazily creates a format panel only when that
format is first selected. Switching JSON ⇄ HTML ⇄ Text therefore changes the
active view without constructing or reopening three independent sheets.

### Immutable per-format panels

`_buildFmtSharePanel(fmt)` resolves its registry metadata once. Delayed
IndexedDB/network callbacks remain bound to the format that initiated them; a
user changing the visible format cannot relabel or overwrite a different
format's result.

### Explicit conversation-scoped UI identity

Share UI state uses an explicit session-persisted conversation identity that
rotates on New chat / Clear conversation. It does **not** derive identity from
the first retained transcript message, because transcript trimming can change
that message during the same conversation. Async completions capture both their
format and conversation identity and refuse to attach stale results to a new
conversation.

This identity is an internal UI-state boundary only. It is not an authorization
token and must never be promoted into one.

### Repeated controls are observers, not DOM authority

Download/Share controls subscribe to one export-mode state. The old duplicate DOM
ID and `getElementById`/first-match synchronization path is removed. The actual
interactive primitive is one button per rendered control; nested interactive
`role=button` + `<button>` semantics are not used.

### Canonical sheet registry

One `_sheetRegistry` now feeds cross-sheet open/close, toolbar injection, Escape
handling, and focus-restoration logic. Opening Share from another sheet carries
forward the original visible opener instead of remembering a control that is
about to become hidden.

### Truthful link language

User-facing labels distinguish **Session-only link** and **Portable link**.
Portable-link copy explicitly states that anyone who receives the complete URL
can read the embedded conversation. Legacy internal keys remain unchanged for
compatibility; user-facing wording no longer claims browser-local URLs are
private authorization boundaries or calls a self-contained URL server-public.

## Verification gates

Final B16 gates on the reviewed working tree:

- [x] JavaScript syntax: `node --check _static/ai-assistant.js`.
- [x] Export registry contract: **60/60** assertions.
- [x] Share architecture source contract: **97/97** assertions.
- [x] Share execution smoke: **35/35** assertions.
- [x] Python Node-harness gate: **27 passed**.
- [x] Share-inclusive mutation gate: **103 passed**; no listed mutant survived.
- [x] Standalone/non-Sphinx suite: **409 passed, 3 skipped**.
- [x] Maintenance checker: GREEN in repository layout.
- [x] Full suite attempted: **875 passed, 3 skipped, 5 failed, 62 errors**; every
  failure/error is inside `test___init__.py` and blocked by missing `sphinx` /
  Sphinx fixture construction in this environment, before Share runtime code.
- [ ] Full Sphinx fixture suite in the canonical repository environment —
  **DEFERRED environment gate**, required before release certification.
- [ ] Real-browser responsive/focus/layout acceptance — **DEFERRED** to
  `AIA-019`; fake DOM proves construction and state, not pixels/layout.

## Suggestions / follow-on design candidates

These are intentionally not mixed into B16 production behavior unless evidence
requires them:

1. **Payload preflight before portable-link creation.** Show turn count and
   serialized byte size, and warn before creating a self-contained URL above a
   documented threshold. This is the highest-value follow-on because URL
   transport limits vary by browser and receiving application.
2. **Registry capabilities instead of new branches.** Future formats should add
   declarative fields such as `canDownload`, `canShare`, `shareKind`, and an
   optional size hint so adding CSV/YAML/PDF-like outputs does not recreate
   format-specific dispatch.
3. **Real browser E2E.** Exercise mobile/desktop widths, dark/light themes,
   tab order, focus restoration, Escape, reduced motion, and clipboard behavior
   in Playwright or the project's existing browser harness. Source/fake-DOM
   gates cannot detect off-screen or unmatched CSS.
4. **Optional remembered preferred format.** If local testing shows repeated
   switching, persist only the preferred format ID in localStorage. Keep the
   first live registry entry as the fallback so stale/removed IDs fail safely.
5. **Simplify link taxonomy only after user testing.** Portable and persistent
   local links both embed/retain content differently. A future UX could present
   one “Portable link” action with a separate “Remember in this browser” option,
   but B16 preserves current behavior to avoid semantic churn.
6. **Abort obsolete server saves.** Conversation/format identity guards already
   make late completion safe. An `AbortController` can additionally stop wasted
   network work when New chat/Clear occurs, where the transport supports it.
7. **Revoke outstanding blob URLs on teardown/pagehide.** Current replacement
   revocation is covered. If panel teardown becomes dynamic, centralize final
   blob cleanup as a lifecycle hook.
8. **Clipboard success should follow actual completion.** Avoid optimistic
   “Copied!” labels if a browser denies clipboard access; tie visual success to
   the asynchronous clipboard result when that helper is modernized.
9. **Optional “Download this format” inside Share.** This can be useful while the
   global delivery mode is Share, but should be driven by local UX testing rather
   than adding a second download affordance pre emptively.
10. **Keep service authorization separate.** `AIA-006` remains P0: a readable
    server Share locator must not automatically grant PATCH/DELETE authority.

`AIA-019` tracks the two strongest deferred acceptance improvements: payload
preflight and real-browser E2E. The remaining candidates stay here/to-do until
usage evidence justifies another bounded production checkpoint.

## Closure

B16 is complete for the browser architecture and its repository-reproducible
non-Sphinx gates. No service-level security finding is closed by this browser
refactor. The next release-level action is the existing B13 closure campaign,
with canonical Sphinx integration and local real-browser acceptance required
before release certification; `AIA-019` remains an explicit deferred follow-on.
