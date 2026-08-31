# Fresh-chat handoff — Run 36 / B55

B55 adds `ai_assistant_panel_remember_conversation` as the Sphinx-controlled initial state of the existing **Remember conversation in this tab** switch.

Core invariants:

1. `ai_assistant_panel_persist` remains the master capability switch.
2. `ai_assistant_panel_remember_conversation` defaults to `True`; a new tab therefore keeps chat across same-tab page changes/reloads by default.
3. The reader's explicit ON/OFF choice is stored as a literal `true`/`false` in `sessionStorage` and wins for the lifetime of that tab.
4. Explicit OFF must never be represented by deleting the preference key, because missing means "use the site default".
5. Transcript bytes remain session-only and are not sent over the network by this feature.
6. Inaccessible storage fails closed; malformed/oversized transcript recovery remains bounded and cleared.
7. `ai_assistant_panel_persist = False` forces persistence off and disables the switch regardless of the initial-state setting.

Suggested conf.py:

```python
ai_assistant_panel_persist = True
ai_assistant_panel_remember_conversation = True
```
