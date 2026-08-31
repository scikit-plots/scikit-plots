# B55 — Remember Conversation Site Default

Status: **GREEN**

## Scope

Make the existing **Remember conversation in this tab** switch configurable from Sphinx while preserving reader control and session-only persistence.

## Configuration

Two settings have distinct responsibilities:

- `ai_assistant_panel_persist = True` — master capability. `False` disables transcript persistence and disables the reader switch.
- `ai_assistant_panel_remember_conversation = True` — initial state for a tab that has no explicit reader choice yet. Default is `True` so conversations survive same-tab documentation page changes/reloads.

After the reader toggles the switch, the explicit `true` or `false` value is stored in `sessionStorage` and wins until that tab is closed. Storing explicit `false` is required: deleting the key would incorrectly reactivate a default-ON site setting on the next page.

## Privacy/lifecycle boundary

- transcript persistence remains `sessionStorage` only;
- no transcript is transmitted merely because remember mode is enabled;
- tab close clears browser session storage under normal browser semantics;
- `New chat` continues to rotate/clear conversation state;
- unavailable `sessionStorage` fails closed to persistence OFF;
- oversized/malformed restored state remains bounded and destructively cleared;
- same-origin scripts can read `sessionStorage`, so the UI continues to disclose that fact and lets the reader disable remembering.

## Verification

- Python extension syntax: GREEN.
- Browser JS syntax: GREEN.
- lifecycle/privacy harness: 60/60.
- registered Node harnesses: 50/50.
- config contract: registration + serialization + default True verified.
