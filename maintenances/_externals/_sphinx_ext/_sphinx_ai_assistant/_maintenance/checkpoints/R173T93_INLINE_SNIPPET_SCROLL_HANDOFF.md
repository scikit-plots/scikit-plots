# R173T93 — Inline snippet scroll handoff

Status: COMPLETE

Base: R173T92 answer-section disclosure discoverability.

## Reported UX issue

When the pointer was over a numbered code snippet inside a normal assistant
answer, mouse-wheel scrolling could stop or feel captured by the code region.
The same nested-scroll ownership was especially awkward on phones/tablets,
where a vertical finger gesture over code should continue the conversation.

## Root cause

R173T69 fixed one half of this class of bug: the inner `<pre>` no longer
promotes itself into a vertical scroller when it owns horizontal overflow.
However inline snippets reuse `.ai-md-file-sheet`, whose generic contract was
still for document/file previews:

- bounded `max-height`;
- vertical scroll ownership;
- `overscroll-behavior: contain`.

That is correct for a real file sheet that intentionally owns a scrollbar, but
wrong for a code snippet embedded in prose. The generic rule also declared
`overflow-y:auto` and later `overflow:hidden` in the same block; the shorthand
silently overrode the axis-specific declaration while overscroll containment
remained. Browser differences made the failure intermittent.

## Repair

- Keep real file sheets explicit: `overflow-y:auto`, `overflow-x:hidden`, and
  overscroll containment. Remove the contradictory `overflow` shorthand.
- Make `.ai-md-snippet-sheet` a non-scrolling participant in conversation flow:
  `max-height:none`, `overflow:visible`, `overscroll-behavior:auto`.
- Preserve rounded visual clipping with `clip-path`, which does not assign
  scrolling ownership.
- Keep the inner `<pre>` as the horizontal scroller only.
- Explicitly allow `pan-x pan-y pinch-zoom` on snippet code so a deliberate
  horizontal drag pans a long line while vertical touch/pen movement continues
  the conversation naturally.
- Do not add a JavaScript wheel interceptor; native scroll chaining is the
  authority.

## Verification

- T93 snippet scroll-handoff contract: **20/20**;
- existing numbered-file/activity preview neighbor: **200/200**;
- canonical static Node/UI harnesses: **154/154**;
- mutation catalogue metadata + unique anchors: **254/254**;
- deliberate mutation execution: **251/251 mutants caught**;
- four new T93 mutation controls: **8/8** architecture assertions/executions.

## Prevention

Scrollable ownership follows interaction intent, not DOM reuse. A reusable
numbered-sheet component may serve both document previews and inline prose, but
only the surface that actually presents itself as a viewport may own vertical
scrolling or overscroll containment. Inline code may own horizontal scrolling;
vertical wheel/touch/pen movement must chain to the reading surface.
