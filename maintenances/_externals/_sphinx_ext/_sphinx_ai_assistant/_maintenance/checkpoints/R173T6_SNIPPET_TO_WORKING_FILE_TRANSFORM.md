# R173T6 — Snippet-to-working-file transform

Status: COMPLETE

Base: R173T5 git-compatible change tracking, same package.

## Evidence that drove this

The rendered `Impute` page supplied for this slice shows the problem exactly:
two code blocks, fifteen `snippet-` references, and **zero**
`data-artifact-path` attributes. The model never declared `file=`, so nothing
it wrote could be tracked, diffed, patched or continued — regardless of how
good the naming heuristic became.

T4 fixed half of that by declaring the `file=` capability in the server-owned
policy, so the model is now told the option exists. This checkpoint fixes the
other half: identity the reader can confer, and a way to keep working on a file
once it has one.

## Snippet -> tracked file

A `Save as file…` control now sits beside every unnamed code block's download
card. The reader supplies a repository-relative path, validated by the existing
`_generatedArtifactSafePath` authority — no second validator. The block is then
registered through the ordinary `_registerGeneratedArtifact` path, so from that
moment it *is* a generated file: revisions, `+N/−M` statistics, patch export,
latest-state resolution, eviction. Nothing downstream needed a second code path.

Promotion onto an existing path becomes that file's next revision. This is the
one place a name-based merge is correct: T4 established that a *derived* name
must never merge, because the reader did not choose it. Here they typed it.

## Continue editing across turns

A `Continue` control on every tracked-file row attaches the latest revision to
the next message and primes the composer.

It routes through `_stageComposerFiles`, the existing composer attachment
pipeline, rather than a private context channel. That is the whole point: the
bounds, classification, transport rules and privacy preflight that already
govern an uploaded file then govern a working file automatically. A private
channel would have put its bytes outside the preflight that tells the reader
what is about to be sent. `relativePath` carries the directory, so a follow-up
addresses `docs/index.rst` rather than a bare `index.rst`.

Unavailable or evicted revisions refuse continuation with the reason, rather
than silently attaching stale bytes.

## Multiple files in one turn

`Download patch series` emits every tracked file as one concatenated `git am`
mailbox — the same artifact git itself produces for a commit range. A
three-file turn applies as three ordered commits instead of three downloads the
reader has to sequence by hand.

## Verification

- browser wrapper gate: **141/141**;
- architecture gates: **390/390**;
- the patch harness now runs **real `git am` on a three-file series** and
  asserts four commits in the log, every file present, and the updated file
  byte-identical to its revision;
- 30 assertions in the git-patch harness, including that continuation reuses
  the composer pipeline and that a reader-supplied path passes through the
  existing safe-path authority.

## Still open

- T4b browser v2 sender: continuation currently carries the file as an
  attachment, which works today; once v2 history ships, the same file should be
  carried as a `working_files` entry with `baseRevision`/`baseSha256` so a late
  response cannot target a stale revision.
- `revision` remains an event counter (bumped on `unavailable`/`removed`), not a
  pure content version. This matters the moment a diff base is addressed by
  number rather than by retained bytes.
- Promotion is per-block; a turn producing four files still needs four
  promotions when the model declares no paths.
