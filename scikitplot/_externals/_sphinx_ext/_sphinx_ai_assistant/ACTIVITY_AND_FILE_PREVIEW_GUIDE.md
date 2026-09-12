# Activity timeline and generated-file preview

Run 172 adds an **observable work surface** to the assistant panel. It is deliberately not a chain-of-thought viewer. Hidden model reasoning is neither requested nor rendered. The browser may show request lifecycle, tool/command status, verification status, generated-file changes, and short endpoint-supplied **public summaries**.

## Reader experience

A live turn gets a compact status row with a **Stop** control. Expanding it shows bounded observable steps such as request preparation, endpoint connection, public tool/search/command summaries, verification, and file-preview changes. Completion collapses to a short summary such as `Ran 2 commands, changed 1 file · 18s`. Stop is turn-owned rather than UI-owned: it invalidates the request token, aborts the captured fetch controller when available, cancels the active stream reader as a second teardown path, and immediately releases the composer. Late continuations must prove they still own the same turn before they may append, preview, retry, or record output.

## Public summaries, not hidden reasoning

Custom endpoints may provide concise reader-facing facts, but must not send private scratchpads, hidden chain-of-thought, internal prompts, credentials, or provider debug dumps. Event kinds named `thinking`, `reasoning`, or `chain_of_thought` are normalized to the public `summary` presentation. Activity label/detail strings pass through the panel secret-pattern redactor before DOM insertion.

## Streaming protocol

A compatible SSE endpoint may interleave normal assistant text with optional public events:

```text
event: assistant.activity
data: {"id":"verify-1","kind":"verify","state":"running","label":"Verifying generated files"}

event: assistant.activity
data: {"id":"verify-1","kind":"verify","state":"done","label":"Verified generated files","detail":"12 checks passed"}
```

`event: activity` is accepted as an alias. Supported public kinds are `plan`, `status`, `network`, `search`, `command`, `file`, `verify`, `warning`, and `summary`. States are `running`, `done`, `error`, and `cancelled`.

Generated text files may arrive as:

```text
event: assistant.artifact
data: {"path":"src/example.py","media_type":"text/x-python","language":"python","content":"print('hello')\n"}
```

`event: artifact` is accepted as an alias. `operation: "remove"`/`"delete"` may publish a removal tombstone without content. An artifact event creates a **preview/download candidate only**. It does not claim that a repository, remote service, or user file has been modified. Non-streaming JSON endpoints may expose equivalent top-level `activity: []` and `artifacts: []` arrays.

## Complete files in Markdown

A complete file can be annotated in the answer:

````markdown
```python file=src/example.py
from __future__ import annotations
print("hello")
```
````

`file=`, `filename=`, and `path=` are accepted. Paths are browser-preview identifiers, not filesystem authority. They must be relative, use `/`, contain no `.`/`..` segments, and cannot be drive-qualified or absolute. Unannotated code blocks remain ordinary snippet downloads.

## Latest-revision semantics

The generated-file ledger is keyed by canonical relative path. If `src/example.py` evolves from r1 to r2 to r3, **every link opens the latest revision**, including a preview button rendered back when r1 existed. Buttons store a path key and resolve the current ledger entry at click time. Exact duplicates do not increment revisions. A later revision that is removed, malformed, too large, or otherwise cannot be retained becomes an explicit **latest unavailable/removed state** with its own revision number. Historical buttons then report that latest state instead of falling back to stale earlier bytes. A later valid revision can make the path previewable again.

## Changed files at the end of the answer

Each answer with generated-file candidates gets a separate deduplicated **Changed files** section after the prose/code content. It shows the latest revision/source metadata, individual Preview/Download controls, and for multiple files a **Download all latest files** ZIP. The ZIP resolves each path again at click time and includes only latest revisions that are still previewable; unavailable/removed latest revisions are reported and skipped rather than silently substituting older content. Before ZIP creation, paths are also compared through the portable filesystem alias boundary (Unicode normalization, case folding, and trailing-dot/space collapse); a collision fails closed and asks the reader to download files individually instead of creating an ambiguous archive.

## Immediate preview

Activity file rows and Changed files rows reuse the existing attachment preview viewer. Retained text is shown immediately with filename, revision/source status, size, and line count; no additional network fetch is required.

## Bounds

| Boundary | Limit |
|---|---:|
| Activity steps per turn | 48 |
| Activity label | 180 characters |
| Activity detail | 2,000 characters |
| Generated files per turn | 24 |
| Generated file | 256 KiB |
| Aggregate generated-file bytes per turn | 1 MiB |
| Retained generated-file preview bytes per browser session | 8 MiB |

Malformed optional activity events do not break assistant streaming. Oversized or invalid **new paths** are rejected locally and surfaced as bounded warnings. When a known path receives a newer but unretainable revision, the ledger records an unavailable latest-state tombstone so stale content can never masquerade as current. The session-wide 8 MiB retention budget keeps the newest useful previews available by releasing the oldest retained preview bytes first. A local retention eviction does **not** invent a file revision: the same revision becomes `preview unavailable`, and all historical links resolve that state. Existing stream byte/line caps remain authoritative.


## Cancellation and broken-pipe behavior

Cancellation is independent of the visualization flag and of browser AbortController support. Every submitted turn owns a private in-memory request token; Stop, Clear, or a newer submit invalidates that token. The network path additionally captures its own AbortController instead of consulting a replaceable global controller, and SSE streaming registers the live `ReadableStreamDefaultReader` so Stop can cancel a pending `reader.read()` directly.

If a stream closes before any visible token, the panel may retry **once** only when it is stripping optional reasoning/effort fields and the same turn is still active. A cancelled turn can never launch that fallback. If a connection breaks after partial output is visible, the partial answer is preserved and marked interrupted; it is not automatically replayed because replay could duplicate visible text or external side effects. Provisional streaming bubbles are removed on cancellation/reader fallback so a broken transport cannot leave an empty spinner row.

## Persistence

The ledger is browser-session state and is cleared with the conversation. A file represented in assistant Markdown can be reconstructed when that transcript is rendered again; endpoint-only artifacts are not promised durable persistence.

## Configuration

```python
ai_assistant_panel_activity_timeline = True
ai_assistant_panel_activity_auto_collapse = True
ai_assistant_panel_generated_file_preview = True
```

Disabling the timeline restores the older opaque typing indicator, but a headless per-turn state still enforces cancellation ownership and generated-file budgets. In other words, disabling visualization never disables the safety/resource boundary. Disabling generated-file preview leaves ordinary code-snippet downloads intact.

## Trust model

The UI keeps three statements separate: **observed browser action**, **endpoint-reported public activity**, and **Actual repository/storage mutation**. Repository writes require their own verified artifact/edit workflow and receipt. This preserves useful step-by-step transparency without making unverifiable claims about hidden reasoning or external side effects.
