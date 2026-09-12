# R173T30 — Four surfaces, one refresh

Status: COMPLETE

Base: R173T29 bulk continuation toggle, same package.

## Why "Remove all" removed nothing visible

`_removeComposerResourceItem` mutates `_composerAttachments` but does not
redraw. Its two original callers each redrew afterwards — the chip's × and the
attachment manager's Remove — so the redraw was a **caller's responsibility**,
recorded nowhere.

R173T28 added a third caller, `_unstageContinuationAttachment`, which did not
know that. Clearing the queue therefore emptied the registry, detached the
items, and left every chip on screen: state changed, screen unchanged.

## The pattern, not the instance

Four surfaces answer *"what travels with my next message?"* — the composer
chips, the attachment manager, this section's tray, and the bulk control. Each
mutation refreshed whichever ones its author remembered. That is the same
defect three checkpoints in a row have now corrected:

- R173T27 — the ⋮ menu kept its build-time label;
- R173T29 — the bulk control kept its build-time label;
- R173T30 — the chips kept their pre-removal DOM.

`_refreshContinuationSurfaces()` now redraws all four, and every mutation calls
it through a single entry point. The redraw stops being anyone's
responsibility to remember.

## Coalescing

Clearing N files reaches the refresh N times, once per removal. A depth guard
holds the redraw until the sweep finishes, so the surfaces are drawn once from
the final state rather than N times from intermediate ones — and the reader
does not watch the queue empty one chip at a time. Released in a `finally`, so
a throwing redraw cannot wedge every later refresh.

## Verification

- browser wrapper gate: **147/147** (78 assertions in the owning harness);
- architecture gates: **478/478**, two new mutants, both caught:
  `queue-change-leaves-the-chips-on-screen` and `bulk-clear-redraws-per-file`.

## Note for the next change here

If a fifth surface starts describing the queue, it belongs inside
`_refreshContinuationSurfaces` and nowhere else. The three staleness bugs above
were each written by someone adding a surface and refreshing it from the call
site that happened to be in front of them.
