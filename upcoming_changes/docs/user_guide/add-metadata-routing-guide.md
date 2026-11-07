---
title: "Add a real Scikit-Plots metadata-routing documentation target"
status: promoted
kind: "docs-contract"
area: "docs/source/user_guide"
discovered_during: "Towncrier template and section review"
release_note: "required"
towncrier_section: "documentation"
towncrier_type: "other"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/documentation/909.other.rst"
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

## Resolution (PR 909)

`docs/source/user_guide/metadata_routing.rst` defines `metadata_routing` and is
linked from the user-guide index and the corpus evaluation guide. It documents
only verified integration points (checked with scikit-learn 1.9.1):

- estimators subclassing `BaseEstimator` inherit scikit-learn's request
  methods (`issubclass(BaseEstimator, _MetadataRequester)` is True);
- `scikitplot.experimental.pipeline.Pipeline` inherits
  `sklearn.pipeline.Pipeline.get_metadata_routing` (identity checked); its
  five off/on cases are tabulated and pinned by
  `scikitplot/experimental/pipeline/tests/test_pipeline_metadata_routing.py`
  (6 tests; replacing its routing makes the 3 routing-on tests fail);
- corpus scorers: routed `sample_weight` reaches the fold scores exactly
  ((1+3)/10 and (5+7)/26 in a check), already pinned by
  `corpus/_metrics/tests/test__scorer.py::TestScikitLearn::test_metadata_routing`;
  the vendored base's refusal by `test__sklearn_compat.py`.

The pipeline docstrings keep their five `<metadata_routing>` references (now
resolving: a Sphinx build of the page and the references resolves all five);
their `versionchanged:: 1.4` / `versionadded:: 1.4` / `versionadded:: 0.20`
directives, copied from scikit-learn, now name scikit-learn's version in
prose. The fragment is `documentation` (a documentation-only change), not
`metadata-routing` (behaviour).

Not documented: the unused `scikitplot.annoy._mixins._meta.MetadataRoutingMixin`
was removed in PR 909 (it was exported nowhere and silently built an
owner-less request or router when the signature differed).
