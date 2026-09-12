# R173T68 — The trigger closes what it opened

Status: COMPLETE

Base: R173T67 menu surface and name, same package.

## Two handlers racing over one click

The trigger toggles correctly on its own:

```js
var wasOpen = _fileMenuOpen && _fileMenuOpen.btn === btn;
_closeFileMenu();
if (wasOpen) return;
```

But the outside-click dismissal runs on `document` in the **capture** phase, so
it sees the click first, and it tested identity:

```js
if (!menu.contains(e.target) && e.target !== btn) _closeFileMenu();
```

Since R173T51 the trigger holds an `<svg>`, so a click lands on the **glyph**,
not the button. `e.target !== btn` was therefore true: the capture handler
closed the menu, and the trigger's own handler — running afterwards — found
nothing open and reopened it. The menu could be opened and never closed from
its own button.

`btn.contains(e.target)` is the fix: the button and everything drawn inside it
is what "clicked the trigger" means. The capture phase is left alone, because
the ordering was never the problem — the test was.

This is R173T51's cost, arriving two checkpoints later. Adding a glyph to a
trigger changed what `e.target` is for every handler reasoning about that
trigger, and only one of them was written to survive it.

## Verification

- browser wrapper gate: **152/152** (136 assertions in the owning harness);
- architecture gates: **603/603**, one new mutant caught:
  `trigger-cannot-close-its-own-menu`, which restores the identity test;
- the capture handler is **driven**: a click on the glyph must not close, a
  click outside must.

## Two harness corrections

A fixture name collided with one already in the file — the harness has grown
several geometry fixtures and `fakeBtn` was taken. Renamed to say what it is.

`no identity test remains` matched the rule's **own comment**, which quotes the
old test while explaining why it is gone. Comments are stripped before matching
now. That is the fourth instance of this trap and the first in JavaScript
rather than CSS; the rule filed in R173T39 covers both and now says so.
