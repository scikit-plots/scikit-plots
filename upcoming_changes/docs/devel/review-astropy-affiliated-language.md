# Review inherited Astropy-affiliated wording in developer documentation

## Finding

The new `docs/source/affiliated/index.rst` now distinguishes Scikit-Plots-owned
partial distributions from Astropy-style independently managed affiliated
packages. Several existing files under `docs/source/devel/` still contain
Astropy-specific affiliated-package language and names (for example references
to the `astropy` root package and `sphinx-astropy`).

## Why this is separate

Those developer guides are outside the current affiliated-index task and need a
section-by-section provenance review before wording is replaced. A blind rename
could turn Astropy-specific policy into unsupported Scikit-Plots policy.

## Next-release review

Review at least:

- `docs/source/devel/guide_code.rst`
- `docs/source/devel/guide_test.rst`
- `docs/source/devel/guide_document.rst`

For each statement, classify it as Scikit-Plots policy, useful upstream guidance
that should be attributed, or stale inherited text that should be removed.
Synchronize the resulting terminology with the partial-distribution model in
`docs/source/affiliated/index.rst`.
