# R173T60 — A quick-swap chevron beside the model picker

Status: COMPLETE

Base: R173T59 dismissal collapses the row, same package.

## Two intents, two targets

```
| color · text · effort · ▾ | ⌄ |
```

The picker is unchanged and still opens the full model sheet, where a model is
chosen deliberately from a list with descriptions and radio buttons. The new
chevron is for the other case: swapping to a model the reader already knows,
without leaving the composer.

One control cannot serve both without guessing which was meant, and the panel
already has the shape for this — `ai-assistant-mic-expand-wrapper` puts a
primary action beside its own options chevron. A reader who has met that has
met this.

Joined by a hairline so the pair reads as one control with two targets, and
kept as **two buttons**: they do different things, and a nested button is
invalid markup besides.

## Built from the shared menu

`_buildOverflowMenu` again, so the quick list inherits Escape returning focus
to the trigger, outside-click dismissal, one menu open at a time, and
`role="menu"`. The current model is ticked, because a list that does not say
where you are makes you open the sheet to find out — which is the trip this
control exists to save.

**Candidates resolve per open.** The menu is built once with the composer, so
reading them at build time would bind the list to the models configured at that
moment and to whichever model was then active, leaving the tick on the wrong
row for the rest of the session. That is the R173T27 staleness bug, now guarded
in its fourth place.

`inlinePicker` still names the **button**, not the wrapper: the sync code
writes `aria-expanded` on it, and a wrapper would have swallowed that silently.
Its mutant reassigns the name and is caught.

## Two mutants that had to be rewritten

Both first *appended* to the code rather than removing it, so the asserted
substrings survived and both survived with them. Rewritten to delete what the
assertions match — the rule established in R173T43: a mutation that decorates
tests the assertion's spelling, not the behaviour.

An assertion also had to be rewritten around a `\\u2713` escape: the source
holds the literal escape sequence, a JS string literal in the harness becomes
the character, and the two encodings can never compare equal. Same trap as
R173T26, now matched on the ASCII either side.

## Verification

- browser wrapper gate: **152/152** (56 assertions in the owning harness);
- architecture gates: **579/579**, two new mutants, both caught:
  `quick-swap-list-frozen-at-build-time` and
  `picker-wrapper-swallows-aria-expanded`.
