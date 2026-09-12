# R173T14 — File-editing sheet, fewer controls, and data-raw deduplication

Status: COMPLETE

Base: R173T13 save-as and card actions, same package.

## Two display modes, on purpose

**Snippet mode is unchanged.** A fenced block with no path renders as it always
has, and `data-raw` keeps the full markdown. For a snippet answer that is
correct: the fenced code IS the answer, and a few dozen lines cost nothing.

**File-editing mode reads like an editor sheet.** A complete, path-bearing file
opens into a sheet with a line-number gutter beside the code.

The numbers live in their **own element**, never interleaved into the `<code>`.
Prefixing each line with its number is the usual shortcut and it poisons every
copy, every download and every patch taken from the block. Here `<code>` still
holds exactly the file's bytes; the gutter is `aria-hidden` (a screen reader
announcing "one import two sys three" would be worse than no numbers at all),
`user-select: none` so a drag-selection cannot pick it up, and
`pointer-events: none` so it is not a click target. The harness asserts the
gutter and the code share a line height and font size — if either drifts, the
numbers stop lining up, and that is not visible from the JavaScript alone.

## data-raw stops holding a second copy of every file

`data-raw` preserves answer markdown for copy, share and export. In a
file-editing answer a 600-line `index.rst` was held **three** times: the
rendered `<pre>`, `data-raw`, and the artifact ledger. The `data-raw` copy is
also the one serialized into session storage with the transcript, where it
competes with the persistence budget for bytes nobody reads.

File bodies are now elided to a marker naming their path and rehydrated on
demand through `_bubbleRawText`. The ledger already owns those bytes and already
resolves the latest revision, so reading them back at copy time is a lookup
rather than a second copy. Fidelity is asserted, not assumed: the round-trip
must reproduce the answer byte for byte, including a body with no trailing
newline. When the ledger entry is gone the accessor falls back to the rendered
block, and when neither is available it leaves the marker visible — a visible
marker beats silently exporting an empty file.

Small bodies and anonymous snippets are never elided: below the threshold the
marker costs more than the text it replaces, and an anonymous block has no
ledger entry to rehydrate from.

## Fewer controls

Five visible buttons per file made the block read as a control panel rather
than a result. `Preview`, `Download` and `Save as…` stay; `Patch` and
`Continue` fold behind one quiet disclosure that names them plainly, so a
reader who wants them can still reach them in one click and everyone else sees
three.

## A regression found by its own round-trip test

The first elision appended a newline after the marker. The captured body
already carried its own line terminator, so every rehydrated answer gained a
blank line before the closing fence — invisible in the UI, wrong in every
export and every patch derived from it. Caught because the assertion demanded
byte-for-byte equality rather than "looks the same".

## Verification

- browser wrapper gate: **145/145** (one new harness, 20 assertions);
- architecture gates: **424/424**, two new mutants, both caught:
  `line-numbers-written-into-the-code` and `raw-body-elision-disabled`;
- the new harness needed a **regex-aware extractor**: the naive one used by
  older harnesses treats quotes and backticks inside a regex literal as string
  delimiters, and both functions under test match fenced code, so both are full
  of them.

## Not in this checkpoint

Activity persistence across reload (`section.ai-assistant-panel-activity`
vanishing on refresh) is the next run. It needs a bounded activity summary
serialized with each assistant transcript entry and re-rendered on restore, and
it touches the persistence budget this checkpoint just relieved — worth doing
with that headroom measured rather than assumed.
