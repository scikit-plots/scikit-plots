# R173T41 — Format tabs wrap instead of scrolling

Status: COMPLETE

Base: R173T40 share sheet layout fixes, same package.

## Scrolling was the wrong answer, including mine

R173T40 made the switcher scroll correctly. Correct, and still wrong: a
horizontal scroller hides tabs behind an edge. TOML sat off-screen with nothing
indicating it existed, and a format the reader cannot see is a format they will
not choose.

The switcher now wraps, showing all five at every width:

```
JSON  HTML  Text  YAML  TOML
JSON  HTML  Text  YAML
TOML
JSON  HTML  Text
YAML  TOML
JSON  HTML
Text  YAML
TOML
JSON
HTML
…
```

## Why auto-fit rather than a column count

`repeat(auto-fit, minmax(min(6rem, 100%), 1fr))` gives every tab the same
width, so the rows line up as a grid rather than as ragged flex lines. Tracks
with no item collapse, so a trailing row of two sits at the two columns it
occupies instead of one tab stretching across the whole row — the orphan
problem R173T36 fixed for the destination cards, avoided here **by
construction** rather than by choosing a fixed column count that a sixth format
would break.

`min(6rem, 100%)` in the minimum is what stops a single column from exceeding a
very narrow container. A bare `6rem` minimum overflows below 6rem of width,
which is precisely the case wrapping exists to handle — the mutant for this
line is the one that would otherwise have shipped.

The narrow-panel override still forced `overflow-x: auto; flex-wrap: nowrap`,
which would have reinstated scrolling at exactly the width that needs wrapping
most. Removed.

## Verification

- browser wrapper gate: **148/148** (147 assertions in the owning harness);
- architecture gates: **513/513**;
- one new CSS mutant, `format-tabs-scroll-out-of-sight`, restores the flex
  scroller; `tab-switcher-widens-the-sheet` was retargeted from the superseded
  `min-width` line to the container-relative minimum, so it still guards the
  overflow it was written for.

## Note

R173T40's fix and this one are not contradictory: `min-width: 0` was genuinely
required for the scroller to work, and finding that out is what made it clear
the scroller should not exist. The assertions were rewritten, not deleted — the
contract moved from "scrolls correctly" to "does not need to scroll".
