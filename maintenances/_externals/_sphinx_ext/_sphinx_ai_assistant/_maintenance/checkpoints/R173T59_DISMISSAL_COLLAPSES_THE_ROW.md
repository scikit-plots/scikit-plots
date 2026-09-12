# R173T59 — Sending a message collapses the hint instead of gutting it

Status: COMPLETE

Base: R173T58 collapsed hint zero height, same package.

## The orphaned toggle

`_dismissSpeakBanner()` — the long-standing behaviour that puts the hint away
once the reader sends their first message — hid the **banner element** with
`display: none`.

Since R173T55 the banner sits inside a row alongside a collapse toggle. Hiding
the banner therefore left the toggle behind on its own:

```
| …the answer continues to the end of this line.        [ › ] |
```

A control whose only purpose is to show and hide something that was no longer
there. Expanding it produced an empty row, because the thing it expands had
been removed rather than collapsed — so the hint was gone permanently, which is
exactly what R173T55 set out to prevent, reintroduced from a code path that
predates it.

## One state, two ways to reach it

`_dismissSpeakBanner` now collapses the row — the same attribute the reader's
own toggle writes — and relabels the toggle to match. The automatic path and
the manual one reach the same state, so **either can undo the other**: a reader
whose first message collapsed the hint can expand it again with the control
that is right there.

Clearing the conversation expands it, as it always did: a cleared conversation
is a fresh start and the hint returns as on first load.

The element-level `display: none` survives as a fallback for a layout with no
row, so this never becomes a no-op if the markup changes again.

## The general fault

A feature added around an element left an older code path acting on the element
directly. Nothing was wrong with either piece in isolation — the fault was that
the newer one changed what "putting the hint away" means and the older one was
never told.

The harness now asserts that no path removes the row, so anything that puts the
hint away must do it in a way expanding can undo.

## Verification

- browser wrapper gate: **152/152** (40 assertions in the owning harness);
- architecture gates: **575/575**, two new mutants, both caught:
  `first-message-orphans-the-toggle` and
  `cleared-conversation-leaves-the-hint-collapsed`.
