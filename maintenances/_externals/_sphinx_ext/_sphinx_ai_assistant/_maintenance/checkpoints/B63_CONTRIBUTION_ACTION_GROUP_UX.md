# B63 — Contribution Action Group UX

Status: **GREEN**

## Scope

Refine only the Contribute-to-dataset sheet action surface so its controls reuse the
established Endpoint Configuration I/O primitives instead of presenting a flat wall of
share-style buttons.

## UI contract

The contribution sheet now uses `ai-assistant-panel-ep-io-row` and
`ai-assistant-panel-ep-io-btn` for secondary actions while keeping the full-width consent
submit button as the primary call to action.

Actions are grouped by authority and intent:

1. **Payload tools** — Inspect JSON, copy JSON locally, download JSON locally;
2. **Private recovery** — save the private management receipt or copy the private
   withdrawal code;
3. **Maintainer support** — copy the non-secret support reference or a ready-to-send
   removal request;
4. **Review lifecycle** — check provider status or delete/withdraw using the management
   capability;
5. **Recover withdrawal access** — load a private code or import a private receipt.

The destructive withdrawal action reuses the Endpoint Configuration danger-button
variant so it cannot be visually confused with local copy/export actions.

## Security / behavior boundaries

- Copy/download payload actions are local-only and explicitly say that nothing was
  submitted.
- No contribution endpoint, consent, receipt, provider-review, or withdrawal semantics
  changed.
- Private recovery and non-secret maintainer-support actions remain visually and
  semantically separated.
- Existing management-capability headers and provider lifecycle logic are unchanged.

## Accessibility / responsive behavior

- Inspect JSON exposes `aria-expanded` and `aria-controls` for the preview.
- Action groups expose descriptive group labels.
- Endpoint-style button rows wrap on desktop and become full-width stacked actions on
  narrow screens.
- Disabled payload tools are visibly non-interactive when no eligible payload exists.

## Verification

- dataset contribution UX source contract: **52/52**;
- dataset contribution DOM contract: **35/35**;
- registered Node/browser harnesses: **51/51**;
- contribution/provider/privacy/lifecycle focused Python plane: **54/54**;
- full runnable non-Sphinx suite: **849 passed, 3 skipped**;
- mutation catalogue retargeted to the new semantic withdrawal-button constructor and
  continues to kill the pending-only wording mutant;
- JavaScript syntax: GREEN.
