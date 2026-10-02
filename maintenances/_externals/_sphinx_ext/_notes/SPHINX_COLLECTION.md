# Sphinx collection, gallery, and YouTube stack

## Scope

This guide covers the cooperating private extensions:

- `_sphinx_collection` — domain-agnostic selection, grouping, browser controls, status, assets, and shared presentation contracts.
- `_sphinx_gallery_grid` — public `gallery-grid` directive and card-grid rendering.
- `_sphinx_youtube_core` — dependency-light YouTube reference grammar and player option primitives; intentionally UI-free.
- `_sphinx_youtube_gallery` — typed YouTube catalog/query adapter that delegates rendering to `gallery-grid`.
- `_sphinxcontrib_youtube` — standalone leaf YouTube/Vimeo/PeerTube player directives.
- `_pydata_component_list` — PyData Sphinx Theme component-list inventory; intentionally separate from the generic gallery engine.

All live under:

`scikitplot/_externals/_sphinx_ext/`

## Ownership model

The stack is intentionally layered:

`provider/reference data -> adapter/query -> gallery-grid -> shared collection browser`

Do not duplicate collection controls inside YouTube-specific code.

### `_sphinx_collection`

Owns:

- bounded YAML helpers used by collection consumers;
- filtering, sorting, grouping, pagination/selection;
- shared section rendering;
- browser metadata serialization;
- search/disclosure/filter/sort/add/export controls;
- collection CSS/JS asset lifecycle;
- the live result-status contract.

### `_sphinx_gallery_grid`

Owns the generic `gallery-grid` directive and conversion of validated records into Sphinx Design grid/card markup. Domain adapters normalize their records and delegate here.

### `_sphinx_youtube_core`

Owns YouTube URL/reference parsing and leaf-player option validation only. It must not create collection controls, match counts, or collection persistence state.

### `_sphinx_youtube_gallery`

Owns YouTube catalog normalization/querying and `youtube-gallery`. It delegates final grid/browser structure to `_sphinx_gallery_grid` and validates the shared collection contract rather than reimplementing it.

### `_sphinxcontrib_youtube`

Owns standalone video/player directives. It also contains the EPUB handler compatibility bridge needed when HTML-only node handlers from other extensions coexist with EPUB-specific media handling.

## Search/status DOM contract

The current shared UI contract is:

`controls-status-results-v4`

For an enhanced searchable collection, result status is **not inside the controls shell**. The intended structure is:

```html
<div class="sk-collection-controls" role="group">
  <!-- search row, disclosure button, options panel -->
</div>
<p class="sk-collection-status"
   data-sk-collection-status-source="document"
   role="status"
   aria-live="polite"
   aria-atomic="true">...</p>
<!-- result cards -->
```

This matches the AI Learn pattern: controls/form first, a separate live status row second, results third.

`_sphinx_gallery_grid` emits the document-owned status node. `_sphinx_collection._browser.is_document_status_node()` is the shared structural recognizer. Consumers must not scrape `rawsource` or search for HTML marker text themselves.

The browser runtime may create a fallback status only for backward compatibility with older generated markup; it must normalize that status to a sibling outside `.sk-collection-controls`.

## Search variants

The shared presentation names are:

- `pill-overflow`;
- `classic`.

`_search_variant.py` owns option validation and conflict resolution so directives cannot drift in accepted values or precedence.

Changing search presentation must not change status ownership: both variants use the same `controls -> status -> results` structure.

## Asset and incremental-build lifecycle

Collection CSS/JS are extension-owned output assets. Their content revision and the collection UI contract participate in Sphinx HTML rebuild decisions.

The collection setup layer:

- computes the asset revision;
- writes/registers assets;
- marks affected HTML outdated when asset/UI contracts change;
- remembers the active revision in the Sphinx environment;
- verifies final emitted assets after the build.

This prevents a successful incremental build from silently serving an older collection runtime with new Python/directive code.

## Data and build safety

Collection inputs are bounded and path-confined. Do not reintroduce unrestricted file reads from directive arguments.

HTML rendering should remain deterministic and offline. Network acquisition/synchronization belongs in explicit tooling, not in ordinary HTML directive rendering.

When a data file is a real build dependency, note it to Sphinx so changes invalidate the page. Do not mark nonexistent optional files as dependencies in a way that makes every build perpetually outdated.

## YouTube reference rule

Use the typed parser in `_sphinx_youtube_core.reference` rather than regex-extracting an eleven-character substring from arbitrary URLs. A copied YouTube URL can simultaneously carry video, playlist, index, channel, tab, or wrapper context; consumers should receive a structured reference and decide what they support.

Unknown/new URL shapes should fail or degrade explicitly rather than silently resolving to a different valid video.

## Files to inspect first

- `_sphinx_collection/contract.py` — UI contract constants.
- `_sphinx_collection/_browser.py` — document browser/status metadata nodes.
- `_sphinx_collection/assets.py` — browser JS/CSS.
- `_sphinx_collection/setup.py` — asset revision and Sphinx lifecycle.
- `_sphinx_collection/select.py` — selection/filtering.
- `_sphinx_gallery_grid/directive.py` — grid directive and delegation surface.
- `_sphinx_youtube_core/reference.py` — URL/reference grammar.
- `_sphinx_youtube_core/video_options.py` — player option validation.
- `_sphinx_youtube_gallery/directive.py` — YouTube adapter/directive.
- `_sphinx_youtube_gallery/model.py`, `query.py`, `sync.py` — catalog model/query/acquisition tooling.
- `_sphinxcontrib_youtube/` — leaf media directives.

## Safe editing rules

1. Keep one owner for shared browser controls: `_sphinx_collection`.
2. Keep YouTube core UI-free.
3. Make `youtube-gallery` delegate rather than copy gallery rendering.
4. Preserve the document-owned status sibling outside `.sk-collection-controls`.
5. Use structural Docutils markers/predicates for cross-directive contracts, not brittle raw HTML scraping.
6. Update asset/UI revision contracts when incompatible generated output changes.
7. Test both `classic` and `pill-overflow` without changing their semantics accidentally.
8. Keep this file current-state only; do not add build/debug chronology.

## Library checkout

Six tests in `_sphinx_collection/tests/test_assets.py` read the site's
`conf.py`, `Makefile` or `make.bat` to check the local/installed authority
bootstrap. They go through `_docs_source()` and skip in this checkout, where
there is no site; they run in the documentation checkout.

`tests/` is not a package. Its `conftest.py` puts the stack's parent on
`sys.path` so the folder passes when run alone.
