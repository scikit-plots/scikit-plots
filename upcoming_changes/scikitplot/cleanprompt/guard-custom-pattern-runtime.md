---
title: "Bound the run time of custom pack patterns"
status: open
kind: "reliability"
area: "scikitplot/cleanprompt"
discovered_during: "internal review 2026-10-10 (CP-NEW-07), reproduced in round 25"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Bound the run time of custom pack patterns

## Summary

A custom pack may carry a pattern with nested repetition. It is validated
(examples executed, unknown keys refused) and accepted, and on a near-match it
takes exponential time.

## Why it matters

One document can stall `batch`, the MCP server or the web app. Packs are the
main extension point, and the more teams share them the more likely a pattern
from someone else's file runs here.

## Current evidence

Round 25, CPython 3.13:

```text
load_custom(<pack with pattern '^(a+)+$'>)  -> accepted
re.match('^(a+)+$', 'a'*18 + 'b')            0.012 s
                       'a'*20 + 'b'          0.047 s
                       'a'*22 + 'b'          0.156 s
```

`_packs.py` states that `_MAX_PATTERN` (2000 characters of source) is not a
defence against catastrophic backtracking.

## Root cause / current understanding

Python's `re` backtracks and has no timeout; the pack validator checks meaning
(examples), not cost.

## Expected behavior

A custom pattern either provably cannot backtrack catastrophically, or its
run time per document is bounded and exceeding the bound fails loudly naming
the pack and pattern (never the text).

## Affected paths and ownership

`scikitplot/cleanprompt/_packs.py`, `_custom.py`, `_detectors.py`
(`RegexDetector`), `tests/test__packs.py`, `tests/test__custom.py`.

## Constraints and non-goals

- Base tier only; no `regex` dependency as a requirement.
- Do not use `sre_parse` / `re._parser` (private, moved between releases, and
  deprecated with a warning the suite turns into an error).
- Built-in patterns keep their current scanner (`CP-015`/`CP-016` reasoning).

## Edge cases to cover

`(a+)+`, `(a*)*`, `(a|a)+`, `(a|aa)+`, nested groups with lazy quantifiers,
a pattern that is slow only on long inputs, a legitimate bounded pattern that
must stay accepted (`\bEMP-\d{6}\b`).

## Proposed direction

Two complementary steps, decided with the maintainer:

1. **Static lint at load** (`packs --check`, `load_custom`): a small,
   documented tokenizer over the public pattern *source* (not `re` internals)
   that flags an unbounded quantifier applied to a group which itself
   contains an unbounded quantifier or an alternation whose branches can
   match the same text. Refuse by default for custom packs, with an explicit
   `--allow-unbounded-patterns` acknowledgement.
2. **Optional runtime bound**: when the optional `regex` package is
   installed, compile custom patterns with its `timeout=`; report the
   capability in `doctor` like any other tier.

## Verification / acceptance criteria

Every edge case is refused (or bounded) with a message naming the pack and
the pattern's `kind`; built-in packs and every existing custom-pack test stay
green; a killable child-process timing test (as in `test__patterns.py`) proves
the bound.

## Documentation impact

Update "A pack file is trusted like code" in
`docs/source/user_guide/cleanprompt/files_and_packs.rst` and the packs
gallery example.

## Release-note promotion

Required: an `enhancement` fragment under `scikitplot.cleanprompt`.
