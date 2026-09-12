# Gallery revision: private package names and safer catalog rendering

## Install

This is a complete copy of the supplied extension tree, with the changes below.
Extract the archive INSIDE your existing `_sphinx_ext` directory, merging files.
Do not delete unrelated extensions, including `_sphinx_ai_assistant`.
After installing, remove the obsolete `youtube_catalog/`, `collection/`, and
`_pydata_sphinx_theme/` directories from this extension tree. They were replaced
by purpose-named packages; do not remove unrelated private extensions.
Update configuration/imports, then rebuild with `python -m sphinx -E -a -b html SOURCE BUILD`.

| Previous package | New package |
| --- | --- |
| youtube_catalog | _sphinx_youtube_gallery |
| collection | _sphinx_collection |
| _pydata_sphinx_theme.gallery_directive | _sphinx_gallery_grid |
| _pydata_sphinx_theme.component_directive | _pydata_component_list |

All extension/shared package names now begin with underscore. Internal Python
filenames, public directives, CSS classes and public configuration options are
not mechanically renamed: existing document syntax stays readable and stable.

Main-package configuration:

```python
extensions = [
    "scikitplot._externals._sphinx_ext._sphinx_youtube_gallery",
]
```

Standalone configuration (put the parent of `_sphinx_ext` on sys.path):

```python
extensions = [
    "_sphinx_ext._sphinx_youtube_gallery",
]
```

Choose one namespace per Sphinx application. Both automatically load the local
YouTube extension and generic gallery; the gallery loads sphinx_design.
Explicit entries for these dependencies can appear in any order.
The shared `_sphinx_collection` is an internal library, not a conf.py extension.
`_sphinx_gallery_grid` is the theme-independent owner of the public
`gallery-grid` directive. `_pydata_component_list` separately owns the
PyData-specific `component-list` directive. The YouTube parser can be imported
without enabling Sphinx setup because shared package exports are lazy.

Keep using `.. youtube::`, `.. gallery-grid::`, `.. youtube-gallery::`, and the
`youtube_catalog_path` / `youtube_catalog_max_embeds` configuration options.
Update direct Python imports and sync invocations to the new package name.
The full scikit-plots package and external sync wrappers were not supplied;
search those callers for the two old package paths during integration.

## Why retain a shared collection package?

The generic gallery and YouTube gallery share filtering, sorting, grouping,
section rendering, and browser assets. Keeping that code separate lets the
generic gallery serve projects, contributors, datasets, or videos without
importing YouTube data acquisition. It also avoids divergent implementations.

## Changes and intentional behavior

- Package-relative imports and dependency loading work in either layout.
- Mixed namespaces and obsolete catalog extension entries get clear errors.
- Filters parse the field/operator before the value. Quoted values containing
  operators, punctuation, commas, or an empty string are handled correctly.
- Repeated grouping labels do not duplicate cards. Numeric zero and False
  remain real grouping values rather than disappearing into Ungrouped.
- Catalog URLs are validated against the video ID and emitted as canonical
  HTTPS YouTube watch links. Unsafe URLs and conflicting IDs are rejected.
  Tracking, playlist, and timestamp query parameters are not retained by
  catalog links. Standalone `youtube` directive timestamp handling is unchanged.
- Catalog titles and descriptions are plain text: whitespace is normalized and
  markup punctuation escaped before RST/MyST generation. A remote description
  must not become an include/raw directive or inject HTML. Generic gallery
  `content` remains explicitly authored RST/MyST, with nested directives allowed.
- `youtube-gallery` video cards now match a generic `gallery-grid` + `youtube` card: title + player only. Descriptions stay metadata; the legacy `:show-description:` flag is accepted as a no-op.
- `:view: channels` can derive a deduplicated offline channel index from a video catalog with stable channel identity, so one reviewed dataset can power both video and channel exploration. Channel-level sort/group/pagination now run after projection rather than accidentally sorting the contributing videos.
- Typed YouTube records accept intentional site metadata only under `fields:`. The adapter flattens that metadata into the shared gallery query record, enabling the same custom `group-by`/sort/live facet/search paths as `gallery-grid` without allowing metadata to overwrite card/player options.
- `_sphinx_collection` now lazy-exports shared symbols, restoring the documented ability to import catalog/YAML tooling without Sphinx/docutils installed.
- Video records separate human-facing ``channel`` from optional linkable ``handle`` identity, so one catalog can group by a friendly name and project offline to canonical channel cards without an API lookup.
- Catalog sync now preserves authored ``handle``/``tags``/``fields`` enrichment by video id, makes ``--check`` compare exact normalized output rather than id sets, writes atomically, treats no-key RSS as a partial non-pruning refresh, and preserves unmatched records on keyed refreshes unless destructive intent is explicit with ``--prune``.
- Empty catalog queries show a message as well as the existing build warning.
- Failed filesystem asset writes warn and remain retryable. Successful repeated
  registration remains a no-op; other setup errors are no longer hidden here.

## Verification

The earlier hardening baseline was exercised with Sphinx 9.1.0,
sphinx-design 0.7.0 and MyST 5.1.0. Those full Sphinx integration builds are
kept as historical evidence only; the current v10 execution environment does
not have Sphinx/docutils/sphinx-design/MyST installed, so the namespace rename
and capability refactor were **not** falsely reported as a rerun of that suite.

Current v10 verification performed in this replacement tree:

- all Python sources compile under Python 3.13.5;
- 67 dependency-free selection/model/query/reference doctests pass;
- importing the root package, YouTube model, and shared selection engine does
  not import Sphinx or docutils;
- static parity checks confirm one delegated ``gallery-grid`` collection root
  and no reintroduced ``sd-card-text``/``Watch on YouTube`` card prose;
- capability checks cover homogeneous video/channel catalogs, ``fields:``
  metadata safety, separate channel display/handle identity, offline
  ``:view: channels`` projection, channel-level sorting, and typo/reserved-field
  rejection;
- sync-pipeline checks cover authored-enrichment preservation, partial RSS
  history retention, keyed default preservation plus explicit ``--prune``,
  exact ``--check`` drift detection, deterministic overlapping-playlist
  handling, and idempotent atomic writes;
- real headless Chromium passes 4 generic-vs-YouTube parity roots, the existing
  30 latest-source behaviors, the 21 direct/post behaviors, and responsive
  containment at 320/768/1440 CSS px;
- typed browser export produces reusable ``videos:`` or ``channels:`` YAML,
  removes the redundant format selector, and both exports round-trip through
  ``normalize_gallery_catalog`` as the expected record type;
- JavaScript and JSON syntax checks pass.

Re-run the full integration suite downstream where Sphinx dependencies are
available:

```bash
python _maintenance/_gallery_revision/verify_extension_layouts.py
python _maintenance/_gallery_revision/verify_gallery_rendering.py
# Put the parent of _sphinx_ext on PYTHONPATH for direct scripts.
python _maintenance/_gallery_revision/test_gallery_regressions.py
```

Live YouTube network/playback and cross-theme/device visual testing remain
downstream HTTP-preview responsibilities.

## Review boundary

The follow-up hardening pass bounded regex matching, YAML structure and catalog
size, made height-free players responsive, capped streamed thumbnail downloads,
and verified nested searchable-gallery ownership with a 1,000-card DOM fixture.
Remote catalog acquisition remains an explicitly invoked maintenance operation,
outside the Sphinx build. Generic gallery authored markup has the privileges of
documentation source and is not an untrusted-data sandbox. Live provider playback
and cross-theme visual inspection still require a downstream HTTP preview.

### YouTube gallery parity

`youtube-gallery` now delegates its rendered cards, grouping/pagination, and
reader controls to a single `gallery-grid` collection root. The compatibility
class `youtube-gallery` is retained on that same root; the old extra outer
collection wrapper is gone. Video cards contain only title + player in embed
mode, and channel catalogs contain only title + stretched channel link. Catalog
descriptions remain searchable metadata and no longer emit `p.sd-card-text`.

### Internal namespace cleanup

The historical `_pydata_sphinx_theme` bucket is removed. `gallery-grid` is now
owned by `_sphinx_gallery_grid`, while the genuinely PyData-specific
`component-list` lives in `_pydata_component_list`. Public directive names do not
change. Update any direct private Python imports; ordinary pages using
`.. gallery-grid::` or `.. youtube-gallery::` need no syntax change.

## Wide-repository maintenance integration (2026-09-10)

The standalone-v10 history above remains provenance; the current wide repository
adds a separate maintenance/state/skill plane and does not reinterpret those old
JSON or SHA files as repository-wide evidence.

A runtime dependency cycle was also removed. ``_sphinxcontrib_youtube`` previously
imported YouTube reference parsing and player-option validators from
``_sphinx_youtube_gallery`` even though the gallery loads the leaf player. Those
shared provider primitives now live in dependency-free ``_sphinx_youtube_core``:

```text
_sphinx_youtube_gallery ----> _sphinx_youtube_core <---- _sphinxcontrib_youtube
          |                                               ^
          +----> _sphinx_gallery_grid / _sphinx_collection |
          +-----------------------------------------------+
```

The old ``_sphinx_youtube_gallery.reference`` and ``_video_options`` modules are
compatibility facades. New shared-provider work belongs in the core rather than
creating another gallery/player cross-dependency. ``_pydata_component_list`` is
tracked as family-related UI infrastructure, not as a runtime requirement of the
YouTube gallery.

Maintenance tests now discover a marked runtime ``_sphinx_ext`` tree instead of
assuming a fixed ``parents[N]`` archive shape. This supports both the main
``scikitplot/_externals/_sphinx_ext`` layout and standalone ``_sphinx_ext``
verification.
