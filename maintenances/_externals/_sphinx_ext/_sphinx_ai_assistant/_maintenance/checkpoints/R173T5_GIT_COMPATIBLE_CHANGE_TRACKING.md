# R173T5 — Git-compatible change tracking

Status: COMPLETE

Base: R173T4 conversation-context-v2 checkpoint, same package.

Status: COMPLETE

## The server-side git question

The request was for server-side git so changes are easy to track and the same
file can be edited and delivered under the same name. Git's *data model* is
exactly right for that: stable identity, a parent per revision, real diffs,
revert. Running git *on the server* is not, and the difference matters enough
to record.

A git service in the proxy would require the proxy to store the reader's
document content. That single consequence cascades:

- the privacy boundary moves from "this message plus the context you selected"
  to "everything you have ever edited", invalidating the panel's own privacy
  copy and the preflight that quotes it;
- per-reader identity becomes mandatory, where today a docs-site panel has
  none, which means accounts, sessions, and a deletion story;
- a working tree inside a request handler brings hooks (arbitrary execution),
  symlinks, submodules, `.git` path traversal, and `core.*` config injection
  into scope — a large new attack surface guarding data the server did not
  previously hold;
- the HF Spaces filesystem is ephemeral, so "durable history" would require
  real storage, retention policy, and quota — all owned by the server, all
  regulated;
- and the existing invariant that the browser is presentation state while the
  server owns authority gets harder to reason about, not easier, because file
  content would then have two authorities.

The benefit that survives all of that is cross-device history. Everything else
the request actually asked for is achievable without the server holding a byte.

## What shipped instead

The panel emits git's **interchange format** rather than running git.

- `_diffUnified` produces real unified-diff hunks from the retained previous
  revision, reusing the same prefix/suffix trim as the `+N/−M` statistic so the
  two can never disagree about what changed.
- `_gitPatchText` renders a `git am`-compatible mailbox: magic `From` line,
  headers, body, `---` separator, `diff --git`, hunks, `-- ` terminator.
  `new file mode 100644` when no base was retained.
- `_gitPatchFilename` gives ordered, portable names (`0002-docs-index-rst.patch`).
- A `Patch` control sits beside `Download` on every file row — offered, never
  assumed.

The reader applies the patch in their own repository, under their own identity,
and gets real git tracking: history, blame, revert, branch, merge. The server
stores nothing.

Two deliberate omissions. The `index <blob>..<blob>` line is not emitted:
computing git blob SHA-1s in the browser needs an async digest for a value
`git apply` and `git am` do not require for text patches, and a fabricated one
would make an otherwise valid patch fail. The author is the panel, never the
reader — this code has no identity for the person using it and must not invent
one.

## Retention budget

Producing a patch needs the predecessor's bytes. They are retained only when
the predecessor is still available and fits the existing per-file ceiling, and
they are charged to the **same** session budget as content via
`_generatedArtifactEntryBytes`. Eviction releases base and content together. A
second, uncounted owner of retained bytes would have made the session ceiling a
number that no longer described what was held.

## Verification

- browser wrapper gate: **141/141** (one new harness);
- architecture gates: **390/390**;
- the patch harness shells out to **real `git am`** and asserts the applied
  file is byte-identical to the revision, for both the update and new-file
  cases. Structural assertions alone would have passed the first
  implementation, whose hunks overlapped and which git rejected.

## Regression caught during development

The first hunk assembler counted trailing context into one hunk and then walked
back over the same lines as the next hunk's leading context, so hunk 2 began on
a line hunk 1 already claimed. `git am` refused it. Rewritten to precompute line
numbers and group changes with the standard 2×context merge rule, which makes
overlap structurally impossible rather than merely absent.

## Next: what server-side git would still need

If cross-device history is later judged worth the cost, the bounded version is
a **push-only** endpoint: the browser sends a completed patch series and a
target the operator configured in advance; the server applies it in a scratch
clone with `core.hooksPath=/dev/null`, `--no-verify`, symlink and submodule
rejection, a path allowlist, and returns a verified receipt. The server still
never becomes the durable owner of reader content — it becomes a courier. That
is a much smaller contract than a git service, and it composes with the ZIP
authorization pipeline that already exists.
