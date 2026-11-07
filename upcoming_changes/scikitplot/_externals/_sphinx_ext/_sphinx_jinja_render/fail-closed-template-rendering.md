---
title: "Fail closed when Sphinx RST template rendering fails"
status: promoted
kind: "reliability"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render"
discovered_during: "source-grounded Sphinx extension user-guide synchronization"
release_note: "required"
towncrier_section: "scikitplot._externals"
towncrier_type: "fix"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot._externals/907.fix.rst"
---

# Fail closed when Sphinx RST template rendering fails

## Summary

The Sphinx ``builder-inited`` hook calls ``render_rst_templates(src_dir,
context=context)`` with the helper's default ``strict=False``.  In that mode a
template exception is collected internally and the failed template is skipped,
but the errors are not returned or logged.  If a previously generated sibling
``.rst`` file already exists, Sphinx can continue building that stale file.

## Why it matters

A documentation build can appear successful while publishing RST from an older
template/context.  That weakens reproducibility and makes a clean build behave
differently from an incremental build.

## Current evidence

- ``scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_extension.py``:
  ``_on_builder_inited`` calls ``render_rst_templates`` without ``strict=True``.
- ``scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_rst_renderer.py``:
  ``render_rst_templates(..., strict=False)`` catches every per-template
  exception, appends it to a local ``errors`` list, then returns only successful
  output paths.
- ``_render_one`` overwrites the sibling generated ``.rst`` only after a
  successful render, so a prior output remains present when the current render
  fails.

## Root cause / current understanding

The low-level helper supports best-effort batch rendering, but the Sphinx build
hook inherits that non-strict default even though build-time generated RST is a
source dependency and stale output is unsafe.

## Expected behavior

A Sphinx documentation build should either regenerate every extension-owned RST
template output successfully or fail with a located/actionable error.  A failed
render must not silently reuse an old generated sibling.

## Affected paths and ownership

- ``scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_extension.py``
- ``scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_rst_renderer.py``
- tests under the same submodule
- user guide: ``docs/source/user_guide/_externals/_sphinx_ext/_sphinx_jinja_render/``

## Constraints and non-goals

Keep ``render_rst_templates`` useful for explicit best-effort programmatic
callers if that behavior is still wanted.  The Sphinx integration, however,
must have an explicit fail-closed policy.  Do not delete unrelated hand-written
RST that is not demonstrably generator-owned.

## Edge cases to cover

- undefined Jinja variable with an existing old ``.rst`` output;
- syntax/read/write failure for one of several templates;
- first build with no old output;
- clean versus incremental builds;
- recursive and non-recursive direct helper calls;
- reporting multiple failures without hiding the first actionable cause.

## Proposed direction

Make the Sphinx event hook explicitly strict, or change the helper to return a
structured result containing successes/failures and make the hook reject any
failure.  Consider generator ownership markers before pruning stale outputs.

## Verification / acceptance criteria

- a deliberately broken template makes the Sphinx build fail;
- an old sibling ``.rst`` cannot make that build pass;
- successful templates remain deterministic and idempotent;
- direct best-effort helper behavior is tested if retained.

## Documentation impact

Update the Jinja-render user guide reliability note after the fail-closed
contract is implemented.

## Release-note promotion

A user-visible documentation-build reliability fix warrants a Towncrier ``fix``
fragment under the configured Sphinx-extension owner.

## Resolution (2026-11-04)

`_on_builder_inited` calls `render_rst_templates(..., strict=True)`. Strict
mode now renders every template first and then raises one `RuntimeError`
listing each failure (`path: ExcType: message`, plus "its existing output X
was not regenerated" when an old sibling exists), chained from the first
error; best-effort mode logs each failure through
`logging.getLogger(__name__)` instead of dropping it. No output is deleted:
the build stops, so a stale sibling cannot pass, and pruning would need the
ownership markers this note mentions.

Evidence: a minimal Sphinx project with `{{ missing_value }}` in
`index.rst.template` and an old `index.rst` built with exit 0 and served the
old text on the previous tree; it now exits 2 with the message above. The
repository's own `docs/source/index.rst.template` renders under strict mode
with `index_template_kwargs`. Tests: `tests/test__rst_renderer.py` (11) and
`tests/test__extension.py` (2), new; the submodule had none for either
module.

Found while fixing it, in the same environment: `autoescape=True` HTML-escaped
every value into the generated RST (`docs/source/index.rst.template` with a
`development_link` holding `&` rendered `&amp;`). It is now
`select_autoescape(("html", "htm", "xml"), default=False)`: on for HTML/XML
template names, off for `*.rst.template`; two tests pin both sides, and
restoring `autoescape=True` fails one. Remove this note once the PR merges.
