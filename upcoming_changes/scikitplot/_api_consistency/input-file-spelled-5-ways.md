---
title: "Public API spells 'input file' 5 ways (file_path, filename, input_path, path, source_path)"
status: open
kind: "api"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "api"
towncrier_fragment: ""
---

# Public API spells 'input file' 5 ways (file_path, filename, input_path, path, source_path)

## Summary

Across public callables, the idea 'input file' is named `file_path`, `filename`, `input_path`, `path`, `source_path`. Readers take `input_path`, most other APIs `path`, a few `filename`, `file_path` or `source_path`.

## Why it matters

A user who learned one submodule guesses wrong in the next; wrappers and configuration files have to map names; documentation and generated skills repeat the inconsistency.

## Current evidence

- `python -B tools/maint_tools/sweep_runtime.py --json out.json` (main cba9383), cluster report:
- `file_path` in 1 public callable(s): `utils.get_path`
- `filename` in 2 public callable(s): `corpus.provenance_from_filename`, `utils.get_path`
- `input_path` in 21 public callable(s): `corpus.ALTOReader`, `corpus.AudioReader`, `corpus.CorpusDocument`, `corpus.CustomReader`, `corpus.DocumentReader`, `corpus.DummyReader` ...
- `path` in 18 public callable(s): `agent.JsonlAuditSink`, `agent.read_audit`, `cleanprompt.load_surrogate_set`, `corpus.ANNIndexArtifact`, `corpus.JSONLStorage`, `corpus.is_archive` ...
- `source_path` in 2 public callable(s): `cython.build_extension_module`, `cython.build_extension_module_result`

## Root cause / current understanding

Submodules were written at different times against different upstream conventions; there is no naming register.

## Expected behavior

One canonical name per idea (proposed: `path`), documented in a naming register in the engineering rules and used by every new or changed public callable; where two scopes differ (a per-file and a per-archive limit) the name says the scope with one pattern.

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
