# R173T53 — Escape does what it says

Status: COMPLETE

Base: R173T52 speak hint dismiss, same package.

## What the report described, and what it was

"Escape exits from nested sheets and menus, but on the main sheet it sometimes
does nothing — especially around a query surface."

Traced rather than guessed. The panel implements a **layered** Escape: each
transient surface binds its own handler and calls `stopPropagation`, and a
final dispatcher closes the assistant when nothing lighter claimed the key.
Nested sheets work because their handler is the one focus reaches first.

The model sheet's search input binds one of these rungs:

```js
if ((e.key === 'Escape' || e.keyCode === 27) && _query) {
    e.stopPropagation();
    _clearFilter();
}
```

Correct behaviour, and **undocumented**. The shortcuts sheet described the
ladder as: microphone → live response → lightest menu/popup/sheet → close the
assistant. Clearing a filter was not on that list.

So with text in a sheet's filter box, Escape cleared the filter and stopped.
The reader — looking at the sheet, not the filter field — saw the panel stay
open and concluded the key had done nothing. The behaviour was right; the
description was incomplete in exactly the way that made it look broken.

## The fix

The ladder now names the rung: *"…stops microphone capture or a live response,
then clears an active search filter, then closes the lightest open menu, popup
or sheet; otherwise it closes the AI Assistant."*

The handler is annotated with the same fact, so the next person to read either
one finds the other.

No behaviour changed. Clearing a filter before leaving a sheet is a reasonable
convention and readers may rely on it; the defect was that the panel documented
a ladder it did not implement.

## The guard that keeps it safe

`&& _query` is what makes the rung terminate rather than swallow. With no
active filter the handler does nothing and the event continues to the
dispatcher, so Escape from an unfiltered sheet's search box still closes the
sheet. Guarding on the key alone would make an unfiltered sheet impossible to
leave from its own filter field — which is the mutant
`escape-swallowed-with-no-active-filter`.

## Verification

- browser wrapper gate: **152/152** (one new harness, 12 assertions);
- architecture gates: **560/560**, two new mutants, both caught.

The gate asserts each named rung individually, so a future handler that
swallows Escape without being added to the description fails here rather than
being reported as "Escape sometimes doesn't work".

## If it persists

This explains the query-surface case specifically. If Escape still fails on the
main sheet with **no** filter active, the useful detail is which element had
focus — the ladder is dispatched from the focused element outward, so the
answer is always "which rung claimed it first".
