---
title: "ZipReader reads a video before its subtitle file is extracted, so the video gives no text (CX-55)"
status: open
kind: "bug"
area: "scikitplot/corpus"
discovered_during: "Pre-merge review of TarReader, PR 985 (2026-11-26)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# ZipReader reads a video before its subtitle file is extracted, so the video gives no text (CX-55)

## Summary

`VideoReader` and `AudioReader` read a companion subtitle file found next
to the media file (`clip.mp4` with `clip.srt`). `ZipReader` extracts and
reads one member at a time in natural order, so `clip.mp4` is read before
`clip.srt` exists on disk: the video gives no text. The same files in a
folder, or in a tar (`TarReader` extracts every member before reading
any), give the subtitle text.

## Why it matters

The same files give different text by container: zip loses the transcript
of every video or audio member whose companion sorts after it, which with
natural order is nearly always (`.mp4` < `.srt`, `.mp3` < `.vtt`).

## Current evidence

PR 985 branch. A zip of `clip.mp4` (64 zero bytes) and `clip.srt` (one
cue): `clip.mp4` is empty, `1 read, 1 empty of 2 member(s)`. The same two
files in a `.tar`: `clip.mp4` gives the cue's text, `2 read`.

## Root cause / current understanding

`ZipReader._iter_members` interleaves extraction and reading. A member's
reader can only see siblings extracted before it.

## Proposed direction

Extract every selected member first (within the same budget, in the
archive's stored order), then read in natural order, as `TarReader` does;
`on_text_limit="truncate"` keeps the members extracted before the limit.
Disk use is unchanged (members already stay until the archive is done).

## Acceptance criteria

- A zip and a tar of the same media and companion files give the same
  chunks, and the same as the files read from a folder.
- The extraction budget, truncate behaviour and coverage counts are
  unchanged for archives without companions.
