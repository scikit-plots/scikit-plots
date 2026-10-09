# Regenerate stale partial-distribution metadata

## Finding

`python -m libs._tools check` currently reports all nine generated
`libs/*/pyproject.toml` files as stale.

A controlled regeneration shows the current canonical root metadata adds the
Python 3.15 classifier and normalizes the relative ordering of
`Programming Language :: Python :: 3` and `Programming Language :: Python :: 3 :: Only`.
The generated lib metadata has not yet been refreshed to match it.

## Scope

This is a packaging-generation consistency issue, not part of the affiliated
user-guide task. Do not hand-edit the nine generated files.

## Next release

1. Review the root Python-version/classifier policy and confirm Python 3.15 is
   intentionally supported by every partial distribution whose
   `requires_python` range admits it.
2. Run `python -m libs._tools generate`.
3. Review all generated diffs.
4. Run `python -m libs._tools check` and the normal partial-distribution
   verification matrix before release.
