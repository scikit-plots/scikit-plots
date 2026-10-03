---
name: sphinx-collection-maintainer
description: Maintain, debug, review, test, and evolve scikitplot._externals._sphinx_ext._sphinx_collection, the shared collection browser used by gallery-grid and the YouTube gallery. Use for collection controls, live search and match status, search variants (pill-overflow or classic), the controls-status-results UI contract, collection asset registration and incremental-build invalidation, or the local versus installed extension-authority bootstrap checked by its tests.
---

# Sphinx Collection Maintainer

Start every fresh chat by reading, in this order:

1. `maintenances/_externals/_sphinx_ext/_notes/README.md` — what the notes are and which checkout tests what
2. `maintenances/_externals/_sphinx_ext/_notes/SPHINX_COLLECTION.md` — the current contract of this subsystem
3. `maintenances/_externals/_sphinx_ext/_notes/SPHINX_EXTENSION_STACK.md` — the two checkouts, import rules, the maintenance gate
4. current source and tests under `scikitplot/_externals/_sphinx_ext/_sphinx_collection/`

This subsystem has no `MAINTAINING.md`, `STATE.json` or tracker of its own yet;
the notes above are its maintenance state. The code, schemas and tests are
authoritative over the notes. Do not require or trust previous chat history.

## Choose the owner before editing

```text
collection parsing, selection, browser controls -> _sphinx_collection
cards, layout, source format                    -> _sphinx_gallery_grid
YouTube grammar and player options              -> _sphinx_youtube_core
catalog, query, sync                            -> _sphinx_youtube_gallery
leaf iframe player                              -> _sphinxcontrib_youtube
```

## The two checkouts

This stack is one tree in two repositories: the library, at
`scikitplot/_externals/_sphinx_ext/`, and the documentation site, at
`docs/source/scikitplot/_externals/_sphinx_ext/`. The shared packages are kept
byte-identical. The site's `conf.py`, canonical content and publication
workflow exist only in the documentation repository, so tests that read them
skip in the library checkout with the reason, and run there.

- Find the site from what is on disk, never from `parents[n]`.
- After any lint or format pass, run the suites before committing.
- A change to a shared package is made once and delivered to both repositories.

## Working rules

The collection layer is the one owner of the search/status DOM contract;
leaf YouTube layers stay free of search UI. Keep sibling imports
package-relative. Assets are HTML-only, written atomically, and their revision
invalidates incremental pages.

The YouTube family has its own maintained subsystem and skill
(`_sphinx_youtube_gallery`); use that entry for catalog, sync and player work.
Tests of the site's `conf.py`, `Makefile` and `make.bat` go through
`_docs_source()` and skip where there is no site.

## Verify

```sh
python -B -m pytest scikitplot/_externals/_sphinx_ext/_sphinx_collection -q
python -B maintenances/_externals/_sphinx_ext/_maintenance_core/tools/check_all.py
```

Run each test folder on its own as well as in a full run: a folder that relies
on another folder's `conftest.py` passes in the full run and fails alone. A
tool that is missing is `UNAVAILABLE`, not green.

When a contract changes, update `maintenances/_externals/_sphinx_ext/_notes/SPHINX_COLLECTION.md` in place, and the same file at
the root of the documentation repository.
