---
title: "Bound custom pattern run time with an optional regex timeout"
status: open
kind: "reliability"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 26 (step 2 of guard-custom-pattern-runtime)"
release_note: "required"
towncrier_section: "security"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Bound custom pattern run time with an optional regex timeout

## Summary

Round 26 reports risky custom patterns when they load (`_pattern_risk.py`),
but a report is not a bound: an accepted or unreported pattern can still take
exponential time on a near-match, and Python's `re` has no timeout.

## Why it matters

One document can stall `batch`, the MCP server or the web app. The static
check covers the known shapes; this note covers everything it cannot prove.

## Current evidence

- `probe_round26.py`: `^(a+)+$` on `'a' * 26 + 'b'` takes about a second and
  doubles per character; nothing interrupts it.
- `_pattern_risk.py` notes: "it cannot prove a pattern is fast".

## Root cause / current understanding

`re` backtracks with no time limit. The third-party `regex` package accepts
`timeout=` per match.

## Expected behavior

When the optional `regex` package is installed and enabled, custom pack
patterns run under a per-document time limit; exceeding it fails the file
loudly, naming the pack and kind (never the text). Without `regex`, behaviour
is unchanged and `doctor` says the bound is unavailable.

## Affected paths and ownership

`_detectors.py` (`RegexDetector` compile/match), `_packs.py`
(`PackPatternDetector`), `_capabilities.py` (a new optional tier),
`_cli.py` (`doctor`), docs `files_and_packs.rst`.

## Constraints and non-goals

- Base tier unchanged; `regex` stays optional (a dependency range, never
  `==`).
- Built-in patterns keep `re` (their behaviour is pinned by tests).
- `regex` syntax differs from `re` in places: a pattern must compile under
  both, or the bound is refused for it with a clear message.

## Edge cases to cover

A pattern valid in `re` but not in `regex`; a timeout hit on the last chunk
of a file; the MCP server and web app (one request fails, the server lives);
threads (the timeout is per call).

## Proposed direction

A `CLEANPROMPT_PATTERN_TIMEOUT` / `--pattern-timeout SECONDS` setting that
only applies when `regex` is importable, reported in `doctor`.

## Verification / acceptance criteria

A killable child-process test proves `^(a+)+$` on a long near-match stops
within the bound with the named error; the base-tier suite is unchanged.

## Documentation impact

`files_and_packs.rst` ("A pack file is trusted like code"), `doctor` docs.

## Release-note promotion

Required: an `enhancement` fragment under `security`.
