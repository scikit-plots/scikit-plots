# R173T52 — The speak hint can be put away

Status: COMPLETE

Base: R173T51 preview file actions menu, same package.

## Why the row was expensive

`.ai-assistant-panel-speak-banner` was removed only after the first message or
the first mic use. A reader who wanted that row back therefore had to do
something they may not have wanted to do — send a message, or switch on a
microphone — to reclaim space in the transcript.

On a mini panel, a phone, or a short window, that row is a meaningful fraction
of the visible conversation.

## What the hint actually costs

The banner points at the mic, and **the mic is in the footer whether the hint
is shown or not**. So the row buys discoverability of a shortcut, not access to
a feature. That is what makes it safe to dismiss, and it is the reason this is
a dismissal rather than a collapse: hiding a control would be a different
decision.

- A dismiss control, **sibling to the banner, not nested inside it** — a button
  within a button is invalid and browsers drop one of the two click targets.
  R173T20 learned that once; here it was applied before it could recur.
- `stopPropagation`, or hiding the hint would also switch the microphone on —
  the opposite of what the reader asked for.
- The dismissal is recorded and **read at build time**, so the hint does not
  return with the next answer. Session-scoped: a hint dismissed today should
  not be silently permanent, and a new session is a new reader as far as this
  panel can tell.
- Focus moves to the mic the hint was pointing at, so dismissing does not
  strand keyboard focus on a removed element.

## Suppressed where height is scarce

Below 620px of viewport height the row is not rendered at all.

Keyed on viewport **height**, unlike every width rule in this file, which uses
a container query. The distinction is deliberate and worth stating: the panel
is narrow by design and its width is unrelated to the window's, but it is laid
out full-height, so viewport height is a fair proxy for the height it has. A
container query would need `container-type: size`, which requires an explicit
height the panel does not set.

## Verification

- browser wrapper gate: **151/151** (one new harness, 14 assertions);
- architecture gates: **556/556**, two new mutants, both caught:
  `speak-hint-returns-after-dismissal` and
  `speak-dismiss-also-starts-recognition`.

## Another same-substring assertion, corrected

`dismissing does not also start speech recognition` searched for
`ev.stopPropagation();`, which appears elsewhere in the file — so it matched
another occurrence and passed while this handler's call had been deleted. Now
scoped to the two adjacent lines of this handler.

That is the fifth instance this run. The pattern is stable enough to restate:
**a substring assertion is only as specific as its rarest token** — if the
token appears anywhere else in the file, the assertion is testing that other
place.
