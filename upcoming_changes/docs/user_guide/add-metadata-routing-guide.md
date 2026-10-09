---
title: "Add a real Scikit-Plots metadata-routing documentation target"
status: open
kind: "docs-contract"
area: "docs/source/user_guide"
discovered_during: "Towncrier template and section review"
release_note: "unknown"
towncrier_section: "metadata-routing"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Add a real Scikit-Plots metadata-routing documentation target

## Summary

Current Scikit-Plots implementation docstrings refer to a Sphinx target named
`metadata_routing`, but no matching target or Scikit-Plots metadata-routing user
guide exists in `docs/source/`.

## Why it matters

Users reading generated API documentation can be sent to a missing cross
reference precisely where behavior depends on scikit-learn metadata routing.
The old Towncrier template repeated the same nonexistent reference.

## Current evidence

- `scikitplot/experimental/pipeline/pipeline.py` contains multiple
  `:ref:`Metadata Routing User Guide <metadata_routing>`` references.
- repository-wide search under `docs/source/` finds no
  `.. _metadata_routing:` target.
- metadata-routing behavior is also present in `scikitplot/annoy/_mixins/_meta.py`.
- the Towncrier template's broken metadata-routing reference was removed during
  the section-ownership review rather than inventing a guide target.

## Root cause / current understanding

Metadata-routing behavior was integrated from scikit-learn-facing code, while
the corresponding Scikit-Plots user-guide page/anchor was not brought into the
documentation tree.

## Expected behavior

Public/generated Scikit-Plots docs should link to a current page that explains
what metadata routing means for the affected Scikit-Plots APIs, required
scikit-learn configuration/version constraints, and relevant examples.

## Affected paths and ownership

- `scikitplot/experimental/pipeline/pipeline.py`
- `scikitplot/annoy/_mixins/_meta.py`
- `docs/source/user_guide/`
- generated API docs that render those docstrings

## Constraints and non-goals

Do not copy scikit-learn's guide wholesale or imply Scikit-Plots supports
metadata routing everywhere. Document only verified Scikit-Plots integration
points and link to authoritative scikit-learn documentation for upstream
semantics.

## Edge cases to cover

- metadata routing disabled (default upstream behavior where applicable);
- metadata routing enabled;
- missing/incompatible scikit-learn dependency;
- behavior across pipeline and Annoy integrations;
- version-specific upstream capability.

## Proposed direction

Create a focused Scikit-Plots metadata-routing guide/anchor after verifying the
current supported integration points, then update implementation docstrings and
release-note cross references to that canonical target.

## Verification / acceptance criteria

- `metadata_routing` resolves in the Sphinx build.
- Every Scikit-Plots claim is backed by current implementation/tests and the
  supported scikit-learn range.
- Examples cover enabled/disabled behavior without depending on private helpers.
- No broken metadata-routing references remain.

## Documentation impact

Add or integrate a metadata-routing section in the user guide and cross-link the
relevant API pages.

## Release-note promotion

Decide after implementation whether the resulting guide accompanies a delivered
metadata-routing feature/fix and therefore needs a `metadata-routing` fragment.
