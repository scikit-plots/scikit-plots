# R173T17 — Activity actually persists, and the timeline becomes inspectable

Status: COMPLETE

Base: R173T16 scale measured and file-card overflow, same package.

## R173T15 shipped a feature that stored nothing

The user reported the activity section still disappearing on reload. They were
right, and the cause is the plainest possible one:

```js
_recordMessage('assistant', accumulated || '(no response)', _streamModelInfo);
```

No `turnMeta` argument at all. `turnMeta.activity` was therefore always
`undefined`, `_activityPersistSummary` was never called, and every reload
produced exactly the behaviour that existed before R173T15.

Every assertion in that checkpoint passed. They asserted that
`entry.activity = activitySummary` appeared in the source, that
`_activityRestoreSummary(e.activity)` appeared in the source, that
`restoredActivity: m.activity || null` appeared in the source. Nothing ever ran
`_recordMessage` with an activity object, so nothing noticed that no caller
supplied one. Three source-level assertions in a row, each true, describing a
chain with no input.

This is the exact failure the same run wrote into SKILL.md as *"gates must
exercise the real consumer"*, committed in the checkpoint immediately after
writing it.

**Fixed** by passing the activity object at the call site, and — more
importantly — by a gate that constructs `_recordMessage` from source and calls
it: with an activity object (a summary is stored), without one (nothing is
stored), and on a user turn (never stored). Plus a mutant,
`assistant-turn-records-no-activity`, that reverts the call site and is
confirmed caught.

## The timeline is now worth inspecting

**Restored timelines open their step list.** After a reload the reader has lost
every other cue about what happened; making them click twice to see the list is
the wrong default. What stays collapsed is each step's *detail* — the shape is
recoverable at a glance, the specifics on request.

**Each step is its own disclosure.** Every detail expanded is a wall; every
detail behind one outer toggle is all-or-nothing. Per-step is the only shape
that lets a reader inspect exactly the step they doubt.

**The collapsed summary names the outcome.** `1 file · 5 steps`, not `5 steps`.
A summary that says nothing a reader wanted to know is a row they learn to skip.

**Command steps render as code.** The `command` kind already existed in the
persisted enumeration; its detail now gets monospace and preserved whitespace.
Reading a command as prose invites misreading it — nothing here executes
anything, and the shape should not suggest otherwise. This is the seam the
requested "Run a command" step slots into.

## Verification

- browser wrapper gate: **147/147**;
- architecture gates: **438/438**, three new mutants, all caught:
  `assistant-turn-records-no-activity`,
  `restored-activity-buries-its-step-list`, and the retargeted
  `restored-activity-offers-a-dead-stop-button`;
- the persistence harness now drives `_recordMessage` end to end rather than
  reading it: record → persist → restore, with the restorer accepting exactly
  what the recorder produced.

## Lesson

A source-level assertion proves a line exists. It cannot prove the line runs, or
that anything reaches it. For any feature whose value depends on a chain of
calls, at least one gate must execute the chain end to end with real inputs — a
harness that only greps has verified the author's intent, not the product's
behaviour.
