# Shared collection browser controls

`_sphinx_collection` owns the progressive browser UI for searchable/interactive
Sphinx card collections. Provider adapters must contribute metadata and options;
they must not create a second search toolbar.

## UI ownership

- `_sphinx_collection`: search/disclosure shell, local filtering, facets, sorting,
  bounded display presets, progressive Load 12 more, chips/suggestions, optional
  saved view/additions, add/export/revert controls.
- `_sphinx_gallery_grid`: static card/grid rendering and the single
  `sk-collection` DOM root.
- `_sphinx_youtube_gallery`: typed YouTube catalog adapter that forwards the
  collection options to `gallery-grid`.
- `_sphinx_youtube_core`: provider grammar/reference primitives only; no UI.
- `_sphinxcontrib_youtube`: leaf player directives only; no collection UI.

## Compact control contract

The browser enhancement has one controller and two presentation variants.
`collection_search_variant = "pill-overflow"` is the site-wide default;
`"classic"` keeps the original rounded-rectangle field and chevron. A specific `gallery-grid` / `youtube-gallery` may override the presentation with
`:search-variant:` (or the underscore alias `:search_variant:`). Authors may also
use the concise activation forms `:searchable: classic` or
`:interactive: pill-overflow`; the old valueless forms remain valid and inherit the
global default. If multiple forms specify different variants, the build fails closed.
A standalone variant option never activates controls by itself: `:searchable:` or
`:interactive:` remains required.

Equivalent per-directive examples:

```rst
.. gallery-grid::
   :interactive: classic

.. gallery-grid::
   :interactive:
   :search_variant: classic

.. youtube-gallery:: ./_data/youtube.yaml
   :searchable: pill-overflow
```

For `ai-topic-explorer`, search is intrinsic to the explorer, so only the
presentation override is needed: `:search_variant: classic` (or the hyphenated
alias). Its fallback remains `ai_learn_explorer_search_variant`.

```text
pill-overflow (default)
[ pill search input               | search icon ] [ ⋮ ]
---------------------------------------------------------
long-form controls when expanded
result metadata

classic
[ search input                    | search icon ] [ chevron ]
-------------------------------------------------------------
long-form controls when expanded
result metadata
```

The result count/status is a document-owned live region immediately after the
controls shell, matching AI Learn's controls → status → results ordering. The
directive emits that status placeholder as a sibling before JavaScript runs; the
browser asset inserts the controls immediately before it and only updates the
existing status text. The count is therefore never a child of the collapsed
search shell. When the options panel is expanded, it remains inside the controls
shell and the live count follows the panel. The search icon is part of the input surface. The chevron controls one in-flow panel inside the
same bordered shell; it does not open a second toolbar or popup. Typing filters
immediately, while submit/Enter remains an equivalent explicit action. IME
composition is not filtered until composition ends.

The shared Docutils status node carries an internal ownership sentinel. Adapters
such as ``youtube-gallery`` must use the shared ``is_document_status_node``
predicate rather than inspecting ``rawsource``: for ``nodes.raw`` the raw source
and rendered payload are distinct fields, and older code created the payload
with an intentionally empty raw-source string.
The expanded panel follows the same information hierarchy as AI Learn rather
than presenting every control at one visual level. **View** owns the bounded
**Display up to** selector (`12/25/50/75/100/125/150`), facets, sort, and Reset. Domain-specific capabilities live under **Gallery tools** as compact
nested panels such as Add, Browser preferences, and Export; a tool expands
across the available width when opened and collapses back into the tool grid
when closed. **Restore original gallery** is visually separated from ordinary
view reset because it also clears local additions and saved preferences.

There is no redundant Close button or explanatory footer. Escape collapses the
outer panel and returns focus to the disclosure; the overflow/chevron control
remains the primary explicit open/close control. The panel does not auto-close on arbitrary
outside pointer activity.


## Bounded card display

Searchable and interactive `gallery-grid` roots use the same reader-side bounded
presentation contract as AI Learn: **12 cards are visible by default**, the
expanded **View** panel can select `12`, `25`, `50`, `75`, `100`, `125`, or `150`,
and a grid-end **Load 12 more** action expands only the current view by one fixed
step. Selecting a preset replaces the current visible window; loading more does
not invent a new preset value.

This is deliberately presentation state, distinct from the directives' existing
build-time `:limit:` and `:offset:` options. Build-time selection decides which
records belong to the rendered collection; the bounded browser view decides how
many of those rendered, locally searchable cards are currently painted. Search,
facets, and sort still operate over the whole rendered collection, while status
text reports the visible count against the current matching count. Group headings
are hidden when none of their cards fall inside the current visible window.

The selector and pager are progressive enhancement. With JavaScript disabled or
if enhancement fails, all statically rendered cards remain available. The shared
YAML loader still enforces its documented resource ceiling, so this UI is not a
claim that one static Sphinx document can safely contain 100K/1B/1T cards. Truly
large catalogs need source-side shards or a cursor/index service; the bounded
`display preset + load-more` interaction is intentionally compatible with that
future backend boundary.

## Progressive-enhancement invariant

The collection CSS/JS are generated into the HTML builder's ``_static`` directory
and registered globally. Their content digest has two independent rebuild paths:

1. ``sk_collection_asset_revision`` is registered as a native Sphinx config value
   with rebuild scope ``"html"``. A changed CSS/JS digest therefore participates in
   Sphinx's own configuration comparison and forces HTML documents to be rewritten
   even when their RST sources are unchanged.
2. The extension also stores its digest in the Sphinx environment and watches whether
   ``builder-inited`` physically replaced an old output asset. This remains a
   defense-in-depth path for stale or externally modified build directories.

At ``build-finished`` the final emitted ``sk-collection.css`` and
``sk-collection.js`` are compared byte-for-byte with the current extension source.
The generated HTML is also checked: every searchable collection root must carry
the V4 contract class and exactly the document-owned status marker emitted by the
directive. A late overwrite, stale doctree, or mixed-version HTML/static pair fails
the build instead of silently publishing a count inside the collapsed controls.
Registration on one live Sphinx application remains digest-aware rather than a
one-way boolean.

The browser enhancement also stamps the collection root with
``data-sk-collection-status-placement="sibling"`` and
``data-sk-collection-ui-contract="controls-status-results-v4"``. This is a
diagnostic contract: a post-load DOM/MHTML snapshot can prove that the current
asset ran. The status node itself carries
``data-sk-collection-status-source="document"`` when it came from the directive;
a runtime-created status exists only as a backward-compatible fallback for HTML
built by an older extension.

The static gallery is complete without JavaScript. Browser controls only hide,
reorder, or add local cards after page load. Provider/network lookup remains an
explicit optional capability and is not required for ordinary search/filtering.
