# R173T29 — The bulk control says what the next click does

Status: COMPLETE

Base: R173T28 one owner for continuation, same package.

## The asymmetry

Attaching several files was one click. Detaching them meant dismantling the
queue one ⋮ menu at a time, so the reader who wanted **none** of it had the most
work to do — the opposite of the effort the interface should ask for.

The footer control is now a toggle:

| queue | label |
|---|---|
| empty | `Continue editing all N files` |
| any files attached | `Remove all N attached files` |

The click follows the **live count**, not the label the button was built with,
and the label is rewritten from that same count. Building it once and trusting
it is how the ⋮ toggle became unusable in R173T27; this control is relabelled
from the one refresh point every queue operation already calls.

The accessible name and title change with it. A control whose visible text says
one thing and whose accessible name says another is worse than either alone.

## The tray became status only

Clearing now lives on the footer control, so the tray's own `Clear` was a
second button for one action — the duplication this section has spent several
checkpoints removing. With a single presented file there is no footer control
and the row's own menu offers `Stop continuing`, so no path lost its way to
detaching.

## Verification

- browser wrapper gate: **147/147** (72 assertions in the owning harness);
- architecture gates: **474/474**, two new mutants, both caught:
  `bulk-control-keeps-its-build-time-label` and `bulk-control-only-attaches`.

The label function is **driven**, not read: the harness constructs it against a
fake button and a stubbed count, and asserts both directions, the singular form,
and that clearing the queue restores the attach label. A source-level check
would have passed against a control that never relabels.
