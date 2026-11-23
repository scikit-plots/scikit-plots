---
title: "CorpusBuilder glob sources drop every file under a hidden parent folder and keep unsupported types (CX-53)"
status: open
kind: "bug"
area: "scikitplot/corpus"
discovered_during: "Real-data review of folder ingestion (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# CorpusBuilder glob sources drop every file under a hidden parent folder and keep unsupported types (CX-53)

## Summary

`CorpusBuilder._expand_sources` treats the same files differently when they
are given as a directory and as a glob:

1. The glob branch excludes a match if any part of its **absolute** path
   starts with `.`; the directory branch checks parts **relative to** the
   directory given. A glob under `~/.cache/...`, `~/.local/...` or a
   project's `.venv`/`.data` folder therefore matches nothing, silently.
2. The directory branch drops files whose extension is unsupported; the
   glob branch keeps them, so they fail later, one error each.

## Why it matters

The same folder gives a corpus from `"/x/.data/docs"` and an empty one from
`"/x/.data/docs/*"`, with no warning (the "Glob pattern matched no files"
warning fires only when the pattern itself matched nothing).

## Current evidence

`main` at 13b457d:

```python
CorpusBuilder._expand_sources(["/tmp/probe/.hidden_parent/docs"])  # [.../a.md]
CorpusBuilder._expand_sources(["/tmp/probe/.hidden_parent/docs/*"])  # []
CorpusBuilder._expand_sources(["/tmp/probe/.hidden_parent/docs/*.md"])  # []
CorpusBuilder._expand_sources(["/tmp/probe2/docs/*"])  # [a.md, b.py]
CorpusBuilder._expand_sources(["/tmp/probe2/docs"])  # [a.md]
```

## Root cause / current understanding

Verified in `scikitplot/corpus/_corpus_builder.py`: the glob branch tests
`mp.parts`; the directory branch tests `fp.relative_to(p).parts` and the
`supported` set; the glob branch never tests `supported`.

## Expected behavior

A glob and the directory it names give the same files: hidden parts are
judged below the pattern's fixed prefix (the part before the first wildcard),
unsupported types are filtered the same way, and when every match is
filtered out one warning names the counts by reason.

## Affected paths and ownership

`scikitplot/corpus/_corpus_builder.py` (`_expand_sources`). Owner:
`maintenances/corpus`.

## Constraints and non-goals

Symlink skipping stays. A pattern that names a hidden part explicitly
(`docs/.notes/*.md`) reads it. One shared filter function for both
branches.

## Edge cases to cover

Relative and absolute patterns, `**` recursion, a pattern whose fixed
prefix is hidden, a hidden file matched explicitly, Windows drive paths,
`__pycache__` below the prefix, a glob matching only unsupported files.

## Proposed direction

Factor one `_keep(path, base)` used by both branches; compute `base` for a
glob as the longest wildcard-free prefix.

## Verification / acceptance criteria

The probe above gives the same list for both forms; tests fail on the
current code for both defects.

## Documentation impact

`CorpusBuilder.build` docstring (source expansion rules).

## Release-note promotion

Required: `scikitplot.corpus` / `fix`.
