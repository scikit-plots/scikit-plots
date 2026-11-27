---
title: "lean: an archive of Lean files cannot be indexed in one call"
status: open
kind: "other"
area: "scikitplot/lean"
discovered_during: "Design review of agent and MCP access to scikitplot.lean, gap probed on main ca0870b (2026-11-27)"
release_note: "required"
towncrier_section: "scikitplot.lean"
towncrier_type: "feature"
towncrier_fragment: ""
---

# lean: an archive of Lean files cannot be indexed in one call

## Summary

`LeanIndex.from_folder` reads a folder; a project shipped as a `.zip` or `.tar.*` must be extracted, or read member by member with `LeanReader`, learned commands first (`learn_commands`, PR 1025), then added with `add_chunks`. No call does this, and the result has no digests, so `refresh` treats every member as changed.

## Why it matters

Projects arrive as release archives, CI artifacts and dataset bundles. Extracting them costs disk and time, and opens the extraction risks the corpus archive readers already guard against (path traversal, size and expansion bombs).

## Current evidence

- `grep -rEn 'zipfile|tarfile|from_archive' scikitplot/lean` finds nothing.
- The corpus `ZipReader` forwards per-extension reader options (`reader_kwargs={'.lean': {...}}`) and enforces member count, size and expansion limits; it yields documents, not the per-member raw chunks with labels that `add_chunks` takes.
- `add_chunks` records no digest (`_digests` stays `None`), so a later `refresh` re-reads every such file.
- `grep -rEil 'zipfile|tarfile|from_archive' scikitplot/lean --include=*.py` (tests and vendored code excluded) finds nothing.
- Related code exists: `scikitplot/corpus/_readers/__init__.py`, `scikitplot/corpus/_readers/_custom.py`, `scikitplot/corpus/_readers/_tar.py`, `scikitplot/corpus/_readers/_zip.py`.

## Root cause / current understanding

Not implemented: the index reads folders; archives were left to the caller.

## Expected behavior

`LeanIndex.from_archive(path, *, include_hidden=False, learn_commands=True, workers=1, **options)` that reads every `.lean` member under the corpus archive limits, labels members by their path in the archive, learns commands first, and gives the index `from_folder` gives on the extracted folder, digests included (of the member bytes).

## Affected paths and ownership

- `scikitplot/lean/_index.py`
- `scikitplot/corpus/_readers/_zip.py`, `_tar.py` (limits reused, not changed)

## Constraints and non-goals

Optional extras imported lazily, no new base dependency; Python 3.8 floor for everything SDK-free (the MCP SDK v2 needs Python >= 3.10); read-only by default; deterministic output for the same index and arguments; bounded responses; no network access. Members are never written to disk; a member whose path leaves the archive root, a link, or a member over the limits is listed as skipped with its reason, as `from_folder` lists files.

## Edge cases to cover

- Nested archives (refused or skipped, not recursed without a limit)
- Members with absolute paths or `..` components
- Two members with the same normalized path
- Non-UTF-8 members (a source problem, as for files)
- Archive with a top-level folder (labels keep it, as `from_folder` on the parent would)

## Proposed direction

Share one member iterator with the corpus archive readers (their limits and path checks), feed member bytes to the same `_read_file` logic through an in-memory path, compute digests from member bytes, and keep `from_folder`'s learning pass. `add_folder` gains the same for a layer (`add_archive`).

## Verification / acceptance criteria

- `from_archive(zip)` and `from_folder(extracted)` save to the same bytes apart from paths, on pfr and batteries
- A hostile archive (traversal, bomb, nested) is refused or skipped with reasons, under the corpus limits

## Documentation impact

Lean user guide (saving and layers sections), API reference, lean usage skill.

## Release-note promotion

Required: `scikitplot.lean` / `feature`, one bullet, when the change lands.
