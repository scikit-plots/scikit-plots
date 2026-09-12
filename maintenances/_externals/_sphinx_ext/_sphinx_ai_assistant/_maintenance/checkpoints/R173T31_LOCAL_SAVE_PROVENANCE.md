# R173T31 — A local save is not a publication

Status: COMPLETE

Base: R173T30 one refresh for four surfaces, same package.

## Found by reading a real export

An uploaded `ai-conversation-local-save-json` was inspected rather than
described. Across its twelve records, **every** one of these was `null`:

```
ts  ts_iso  model_id  model_provider  model_name
feedback_rating_value  feedback_rating_label  feedback_message
session_id  page_url
```

as were `session.id`, `session.page_url`, `session.exported_at` and
`exported_at_iso`. The file records what was said and nothing about when, by
which model, or from which page.

## Cause

Not a redaction bug — a **default applied to the wrong exposure**.

Every destination in the share sheet, including `download`, runs through
`_reviewShareSnapshot`. The sheet opened with `contentPreset = 'standard'`,
whose whole purpose is to protect a *published* artifact by stripping
timestamps, model attribution, the session id and the source page.

Applied to a local device file that protects nobody: the reader already has the
conversation on screen. It removes precisely the provenance a saved transcript
is kept for — and the panel told them, in the same breath, that "the file is
controlled by your device".

## Fix

`_conversationPresetForDestination()` supplies the default: `complete` for
`download` and `local`, `standard` for anything that leaves the device.
Changing destination re-defaults the preset **only** when the reader has not
chosen one; a chosen preset (including `custom`) survives.

This changes a default, not the redaction. The review screen still lists what
will be included and the reader can still pick `standard` or `minimal`.
Nothing is added behind their back — which is the constraint that made this the
right fix rather than exempting local saves from review.

## Verification

- browser wrapper gate: **147/147**;
- architecture gates: **480/480**, one new mutant caught:
  `local-save-redacted-like-a-published-share`.

The DOM harness needed `_conversationPresetForDestination` supplied because the
sheet now consults it — the real function, not a stub, so the harness cannot
pass against a sheet that ignores it.

## Also observed, not fixed here

The export carries **no activity timeline**. R173T15/T17 persist a bounded step
summary per assistant turn and restore it on reload, but `_buildExportRecords`
does not emit it, so an exported transcript has no account of how each answer
was produced — the same gap on disk that the reload bug left on screen.

That is a schema addition (`schema_version` bump, share-policy decision about
whether activity is publishable), so it belongs in its own checkpoint rather
than folded into a default change.
