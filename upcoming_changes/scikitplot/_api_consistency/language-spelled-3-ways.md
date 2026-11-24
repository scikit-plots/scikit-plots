---
title: "Public API spells 'language' 3 ways (default_language, lang, language)"
status: open
kind: "api"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "api"
towncrier_fragment: ""
---

# Public API spells 'language' 3 ways (default_language, lang, language)

## Summary

Across public callables, the idea 'language' is named `default_language`, `lang`, `language`. `language`, `default_language` and `lang` name the same idea (a language code).

## Why it matters

A user who learned one submodule guesses wrong in the next; wrappers and configuration files have to map names; documentation and generated skills repeat the inconsistency.

## Current evidence

- `python -B tools/maint_tools/sweep_runtime.py --json out.json` (main cba9383), cluster report:
- `default_language` in 21 public callable(s): `corpus.ALTOReader`, `corpus.AudioReader`, `corpus.BuilderConfig`, `corpus.CorpusPipeline`, `corpus.CustomReader`, `corpus.DocumentReader` ...
- `lang` in 2 public callable(s): `corpus.coerce_language`, `corpus.resolve_stopwords`
- `language` in 26 public callable(s): `cleanprompt.CleanPlan`, `cleanprompt.build_detectors`, `cleanprompt.describe_engines`, `cleanprompt.encode`, `cleanprompt.engine_readiness`, `cleanprompt.resolve_engine` ...

## Root cause / current understanding

Submodules were written at different times against different upstream conventions; there is no naming register.

## Expected behavior

One canonical name per idea (proposed: `language`), documented in a naming register in the engineering rules and used by every new or changed public callable; where two scopes differ (a per-file and a per-archive limit) the name says the scope with one pattern.

## Affected paths and ownership

- the public callables listed above
- `plugins/_shared/skills/scikitplot-engineering-rules/` (naming register)
- `tools/maint_tools/sweep_runtime.py` (`CLUSTERS`, the check)

## Constraints and non-goals

Renaming public parameters is an API decision; the campaign rule is no backward compatibility, so a rename lands with its fragment and usage-skill update (Rule 172).

## Edge cases to cover

- Classes whose `__init__` takes the old name positionally
- Configuration dataclasses serialised to JSON with the old key
- Names that look alike but differ in meaning (keep them, document why)

## Proposed direction

Decide the canonical name, rename in one change per submodule, add the cluster to a `sweep_runtime` gate so a new divergent spelling fails review.

## Verification / acceptance criteria

- `sweep_runtime.py` reports no cluster for this idea, or only the documented exceptions

## Documentation impact

API reference, usage skills, the naming register.

## Release-note promotion

Required: `scikitplot` / `api`, one bullet, when the change lands.
