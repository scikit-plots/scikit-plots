# R173T10 — Persistence honesty

Status: COMPLETE

Base: R173T9 retry honesty and content revision, same package.

## The last claim the panel could not support

`_ssSet` was `try { sessionStorage.setItem(key, val); } catch (_) { /* ignore */ }`.

Defensible for a cache. Indefensible for the transcript: quota exhaustion,
private browsing and disabled storage all fail there, and the panel went on
showing "Remember conversation" switched on while nothing survived a reload. A
switch describing a capability the browser is refusing is worse than no switch,
because the reader stops taking their own notes.

This is the third instance of one pattern this run — after "resend as-is" and
"Latest revision r5" — of the UI asserting more than the implementation could
deliver. It is now closed.

## What changed

`_ssSet` returns whether the write landed. `_saveTranscript` checks the result
and distinguishes the two failure modes: a storage rejection (`quota`) from a
transcript that could not be serialized at all.

`_persistenceReportUnavailable` warns **once per condition**, not per save. A
warning that repeats becomes noise and is dismissed along with the ones that
matter. It names the consequence — this conversation will not survive a reload
— and deliberately does not instruct the reader to change a setting, because
the panel cannot know which browser control, if any, would help. Recovery
resets the latch, so a later failure reports again.

## Truncated restores

The more dangerous failure is partial persistence: turns 1–40 of 62 survive, and
on reload that looks exactly like a complete conversation, with nothing about it
appearing wrong.

Every successful save now also writes `_TRANSCRIPT_COUNT_KEY` — an integrity
marker, not a duplicate of the data. On restore, a transcript shorter than its
own marker reports exactly how many turns the browser dropped. The marker is
pruned everywhere the transcript is, including on Remember→OFF; a marker left
behind would make the next restore compare a fresh conversation against a stale
expected length and report a shortfall that never happened.

## Verification

- browser wrapper gate: **144/144** (one new harness, 20 assertions);
- architecture gates: **411/411**, two new mutants, both caught:
  `persistence-failure-silently-swallowed` and
  `truncated-restore-reported-as-complete`;
- the harness drives `_ssSet` against a throwing `sessionStorage` and asserts
  the reported result, drives the notifier to prove it latches, and exercises
  shortfall detection against absent, complete, truncated, longer, corrupt and
  zero markers.

## Tests corrected during development

`test_ai_assistant__context_state_idempotency_matrix.mjs` counted seven deletes
per Remember→OFF and now sees eight. The extra delete is the marker being
pruned with the transcript it describes — an intentional change, so the count
was updated and an assertion added naming the marker specifically, rather than
loosening the count.

One of my own assertions rejected the warning text for containing the word
"disabled". The message *describes* storage as disabled, which is the actual
condition; the assertion had conflated describing a state with instructing the
reader to change it. Rewritten to match imperatives only.

## Remaining from the original design note

Two items, both deliberately not built:

- **§18 conversation branching / version navigation.** Requires restructuring
  `_transcript` from a list into a tree, which reaches persistence, restore,
  export, share, feedback and contribution. That is a large blast radius for a
  presentation improvement with no correctness or security pressure behind it.
- **§20 virtualised long transcripts.** Worth doing when a real transcript is
  actually slow; the ceiling is 200 turns and no measurement in this run showed
  a rendering problem. Optimising before measuring would be guesswork.

Both are recorded rather than closed. The correctness and honesty backlog from
that document is complete.
