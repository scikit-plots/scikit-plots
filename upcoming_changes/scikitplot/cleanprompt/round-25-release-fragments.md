---
title: "Create the release-note fragments for cleanprompt round 25"
status: blocked
kind: "other"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 25 (truthful readiness, safe deployment files, detection view)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Create the release-note fragments for cleanprompt round 25

## Summary

Round 25 delivered user-visible fixes (`CP-093` … `CP-101`). They are
implemented, tested and probed. Towncrier fragments are named
`<PULL_REQUEST>.<TYPE>.rst`, and no pull-request number exists yet, so the
fragments are drafted here instead of invented under a false number.

## Why it matters

Without the fragments the release notes would not tell users that `doctor`
now reports entity engines truthfully, that `--debug` is refused off loopback,
or that obfuscated values are now detected.

## Current evidence

- `maintenances/cleanprompt/REVIEW.json`: `CP-093` … `CP-101`, each closed with
  a named regression test in `scikitplot/cleanprompt/tests/test_regressions.py`
  and a probe in `maintenances/cleanprompt/_maintenance/evidence/probe_negative.py`.
- `docs/source/whats_new/upcoming_changes/README.md`: the naming rule.

## Root cause / current understanding

Blocked on an external fact (the PR number), not on any implementation step.

## Expected behavior

One fragment per user-visible change, in the section that owns it.

## Affected paths and ownership

`docs/source/whats_new/upcoming_changes/scikitplot.cleanprompt/` and
`docs/source/whats_new/upcoming_changes/security/`.

## Constraints and non-goals

Do not create fragments with a placeholder PR number; the structure check and
the changelog both depend on the real one.

## Edge cases to cover

`security` takes the two hardening items; everything else is
`scikitplot.cleanprompt`.

## Proposed direction

When the PR number `N` is known, create these files with exactly this text
(adjust only the contributor role):

`security/N.fix.rst`

    ``scikitplot.cleanprompt``: values written with invisible format
    characters (zero-width spaces and joiners, bidirectional controls, soft
    hyphens), full-width or other compatibility forms, or Unicode spaces and
    dashes inside them are now detected and restored exactly as written;
    previously they were sent unredacted.

`security/N.enhancement.rst`

    ``cleanprompt flask --debug`` is now refused on any non-loopback bind,
    container mode included, and the generated container files publish on
    ``127.0.0.1`` only.

`scikitplot.cleanprompt/N.fix.rst`

    ``doctor``, ``inspect``, ``encode`` and the web app now agree on whether an
    entity engine can run: an engine installed without its spaCy model or NLTK
    data is reported as not ready with the exact download command, ``auto``
    no longer selects it, and an explicit ``--ner`` request fails before any
    text is read. NLTK data present only under pre-3.9 names is detected as
    unloadable. The web app now honours ``ner_engine``, ``language`` and
    ``model_size``, and ``docker --with-ner`` installs the model it runs with.

`scikitplot.cleanprompt/N.enhancement.rst`

    The CleanPrompt user guide is now several task-oriented pages, including
    entity-engine readiness, web and container deployment, and an explicit
    list of what is and is not protected.

## Verification / acceptance criteria

`python tools/maint_tools/generate_towncrier_sections.py check` (or the
repository's fragment check) passes with the four files present.

## Documentation impact

None beyond the fragments.

## Release-note promotion

This note *is* the promotion step. Mark it `promoted` once the files exist,
and remove it when the PR merges.
