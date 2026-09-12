# R173T47 — The preview becomes a window

Status: COMPLETE

Base: R173T46 numbered sheet scrolling, same package.

## The promise nothing kept

`data-drag-handle="true"` was on the preview header with no drag behaviour
anywhere in the file. The dialog is now genuinely a window: movable by its
header, resizable from its corner, minimisable and maximisable, centred by
default.

## The invariant everything else is arranged around

**A window must never be movable somewhere it cannot be moved back from.**

Every geometry write — drag, maximise, minimise, restore, viewport resize —
goes through one `_previewClamp`, which keeps at least 64px of the window on
screen on each axis and never lets the top edge go above zero. The header
carries every control including Close, so a header dragged off-screen is a
window the reader has lost with no way to recover it.

The gate drives that arithmetic directly rather than inspecting markup: dragged
to −5000 or +99999 on either axis, the clamped result still leaves a grabbable
strip. It also pins the size floor (a window cannot be resized smaller than its
own controls) and ceiling, and that an ordinary geometry passes through
untouched.

## Details that would have been bugs

- **The centring transform is dropped** when explicit coordinates are set.
  `translate(-50%,-50%)` offsets every position by half the window's own size,
  so the first drag would make the window jump away from the pointer.
- **Pointer capture** on the header, so a fast drag does not drop the window
  where it lost contact; `pointercancel` ends the drag rather than leaving it
  stuck.
- **Header buttons are not drag handles.** Without excluding them, pressing
  minimise would move the window and never fire the action.
- **`touch-action: none`**, or a touch drag scrolls the page instead.
- **Viewport resize re-clamps**, so a window left half off-screen by a shrunk
  window stays reachable.
- **Minimise reopens as normal.** Position is a preference worth keeping for
  the session; collapsed is a transient state, and reopening is a new intent to
  read the file — a preview that reopens as a title bar looks broken.

Resizing uses native CSS `resize: both`, which works because the dialog already
sets `overflow: hidden`, and brings correct pointer and keyboard behaviour with
no JavaScript. It is disabled while minimised, where it could only produce an
unusable strip.

## Verification

- browser wrapper gate: **149/149** (one new harness, 25 assertions);
- architecture gates: **536/536**, three new mutants, all caught:
  `preview-window-draggable-out-of-reach`,
  `preview-window-keeps-centring-transform`,
  `preview-header-buttons-become-drag-handles`.

## One assertion the mutant corrected, again

`header buttons stay buttons` searched the drag binder for
`ev.target.closest('button')`. The double-click handler contains the same
substring, so the assertion passed while the pointerdown guard had been
deleted. Both guards are now named, and the count is asserted.

This is the same shape as R173T34, T42 and T44: **a substring that occurs more
than once tests whichever occurrence came first, not the one the assertion is
about.**
