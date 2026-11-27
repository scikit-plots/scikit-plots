---
title: "lean: near-duplicate search below a similarity of 0.7 takes minutes"
status: open
kind: "other"
area: "scikitplot/lean"
discovered_during: "Design review of agent and MCP access to scikitplot.lean, gap probed on main ca0870b (2026-11-27)"
release_note: "required"
towncrier_section: "scikitplot.lean"
towncrier_type: "feature"
towncrier_fragment: ""
---

# lean: near-duplicate search below a similarity of 0.7 takes minutes

## Summary

`LeanIndex.near_duplicates` is exact (prefix filtering). At 0.8 and 0.9 it takes 6 to 11 s on mathlib4 with PFR (shapes kept), but the number of pairs, and the time, grow fast below 0.7: at 0.5 the search alone takes about 4 minutes for 515,934 pairs.

## Why it matters

Low thresholds are how one finds loose restatements (a lemma generalized, hypotheses reordered). Exact search there is too slow to use interactively, and an agent calling it through a tool would time out.

## Current evidence

- Measured on mathlib4 with PFR, 196,029 statements, two workers: 0.8 in 11 s (29,062 pairs), 0.9 in 6 s (3,491 pairs) with `keep_shapes=True`; at 0.5 the search alone takes about 4 minutes (515,934 pairs).
- The search is `similar_pairs` in `scikitplot/lean/_similar.py`: exact, integer arithmetic, no sampling.
- Related code exists: `scikitplot/lean/_similar.py`.
- `grep -rEil 'minhash|annoy' scikitplot/lean --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Prefix filtering prunes less as the threshold falls: prefixes grow, and so do candidate lists.

## Expected behavior

An opt-in approximate mode for low thresholds that reports what it may miss: `near_duplicates(..., method='exact'|'minhash'|'annoy')`, exact by default; the approximate modes verify every reported pair exactly (no false positives) and state their recall and parameters in the report.

## Affected paths and ownership

- `scikitplot/lean/_similar.py`
- `scikitplot/annoy` (route 2)

## Constraints and non-goals

Optional extras imported lazily, no new base dependency; Python 3.8 floor for everything SDK-free (the MCP SDK v2 needs Python >= 3.10); read-only by default; deterministic output for the same index and arguments; bounded responses; no network access. Exact stays the default and the reference: approximate results are a subset of exact ones, verified pair by pair, with seed and parameters in the report so a run can be repeated.

## Edge cases to cover

- Threshold 1.0 and very small statements (approximate mode must not be used where exact is fast)
- Recall measured against exact on a sample, reported
- Layer and against sides as in exact mode

## Proposed direction

Route 1: MinHash signatures (seeded, standard library hashing) with LSH banding chosen from the threshold, candidates verified exactly. Route 2: an Annoy index over the same signatures or over hashed pair vectors, through `scikitplot.annoy`, candidates verified exactly. Either way, measure recall against exact at 0.5 to 0.7 on mathlib4 before choosing a default for the mode.

## Verification / acceptance criteria

- At 0.5 on mathlib4 with PFR, the approximate mode runs in under 30 s and finds at least 95% of the exact pairs, every reported pair verified
- The same seed gives the same report

## Documentation impact

Lean user guide (near-duplicate section and its timing table), API reference.

## Release-note promotion

Required: `scikitplot.lean` / `feature`, one bullet, when the change lands.
