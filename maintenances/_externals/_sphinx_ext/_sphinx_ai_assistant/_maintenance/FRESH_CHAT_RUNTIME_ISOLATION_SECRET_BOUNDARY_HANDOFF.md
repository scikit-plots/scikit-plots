# Fresh-chat handoff — Run 21 / B40 Runtime Isolation & Secret Boundary

Input anchor: Run 20 / B39 final overlay, SHA-256
`b64b4e56174f88f5de05789fd1457092949f0d036f84afc558bdd5afb044b748`.

## Current architecture

- Internal assistant lifecycle events use a private in-bundle bus.
- Public same-origin lifecycle events are Off by default and require the
  structured `ai-assistant-page-integration-consent` v2 permission.
- Public event detail is event-specific and bounded; do not re-expose internal
  model/profile/endpoint/token/conversation objects.
- Network feedback telemetry permission is independent from page integration.
- `Origin: null` Share read compatibility and mutation authority are separate;
  strict HF deployment forbids opaque-origin write authority.
- Browser-entered Share/Feedback bearer tokens are disabled unless the Sphinx
  site owner explicitly sets `ai_assistant_allow_runtime_tokens=True`.
- Viewer pages deny framing and sensitive browser permissions.
- `SEC-P1-38` remains external release/provider evidence. Separate-origin browser
  isolation remains a future architectural residual.

## Verification already green on the working tree

B40 Python 8/8, B40 browser 21/21, focused compatibility 57 passed, complete
runnable non-Sphinx 742 passed / 3 skipped. Continue with compile, JS/TOML,
maintenance, Sphinx-boundary, controlled diff and candidate → prefinal → final
exact-byte acceptance. Do not claim the final archive until the final extraction
reproduces those gates.
