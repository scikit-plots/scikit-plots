# R173T27 — Three continuation bugs, three distinct causes

Status: COMPLETE

Base: R173T26 continuation queue, same package.

Both reported failures — "after removing a file I cannot re-add it" and
"Continue all only adds one file" — turned out to be three separate causes.

## 1. The menu was frozen at row-build time

`_buildOverflowMenu` took an array, so `_workingFileContinuations[key]` was read
once, when the row was created. After dropping a file the item still read
`Stop continuing`; clicking it stopped an already-stopped continuation, and the
file looked impossible to re-add.

The builder now accepts a **function** and resolves items on each open, so the
label always describes the state the reader is actually in. Any menu whose
contents depend on state must be built this way; a static array is only correct
for a fixed list.

## 2. Stop deleted the key but left the bytes staged

On an endpoint without working-file support, Continue stages a composer
attachment. Stop deleted only the registry key — so the "dropped" file still
travelled with the next message, and re-adding it staged a **second** copy.

Stop now unstages through `_removeComposerResourceItem`, the pipeline that owns
the item. Clear does the same for every queued file. Stop has to undo
everything Continue did, or it is not a stop.

## 3. The pre-discovery cap was one file

`_WORKING_FILE_FALLBACK` used before discovery said `maxFiles: 1`. `Continue
editing all N files` therefore queued exactly one and reported the rest as
"left out" — a correct message about a limit the panel had invented.

That was the wrong kind of caution. Guessing high risks a rejection the reader
can see and act on; guessing low silently drops files and reads as a broken
button. The fallback now mirrors the server's own defaults (4 files / 48k per
file / 96k total), and anything beyond them is rejected by the server with a
message naming the bound.

## And the tray had the same staleness as the menu

It reported whatever count it held when the section rendered.
`_refreshContinuationTray()` is now called by every operation that changes the
queue, and hides the tray when the queue empties.

## Verification

- browser wrapper gate: **147/147** (55 assertions in the owning harness);
- architecture gates: **468/468**, three new mutants, all caught:
  `menu-items-frozen-at-build-time`, `stop-continuing-leaves-the-bytes-staged`,
  `pre-discovery-cap-drops-every-file-but-one`.

## A loose assertion, caught by its own mutant

`stopping also unstages any attachment` checked only that
`_unstageContinuationAttachment(entry.path);` appeared in the function. The
mutant wrapped that call in `if (false)`, the substring survived, and the
assertion passed while the mutant did not die. It now includes the guard.

That is the second time this run a substring assertion has been defeated by a
mutation that disabled rather than deleted the code. **Assert the guard, not
just the call** — a line that cannot run still reads as present.
