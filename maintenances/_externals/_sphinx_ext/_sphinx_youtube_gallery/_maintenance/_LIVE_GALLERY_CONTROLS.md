# Live gallery controls

The package-level user guide is `_sphinx_youtube_gallery/README.md`. It is the canonical copy-and-paste reference for installation, input forms, queries, live controls, presentation options, schema, and migration.

## Install and apply the page update

Merge this ZIP inside the existing `_sphinx_ext` directory. The generic
`gallery-grid` extension now lives at `_sphinx_gallery_grid`; the unrelated
PyData component inventory lives at `_pydata_component_list`. `youtube-gallery`
automatically loads `_sphinx_gallery_grid`. No new pip or JavaScript runtime
dependency is needed for the controls.

The updated version of the supplied page is:
`_maintenance/_live_controls/examples/index.rst`.
Copy its content into your existing YouTube resources page. It preserves all
8 channels and 13 video IDs/titles, combines the videos into one gallery grouped
by the five existing topics, and retains the `resource-youtube-index` anchor.
The channels have a separate toolbar so channel search does not hide videos.
A sample catalog page and the supplied youtube.yaml are included beside it.
The sample YAML's outdated sync-module comment is corrected.

Rebuild with `python -m sphinx -E -a -b html SOURCE BUILD`.
Preview over HTTP. No deployment has been performed.

## Options for both directives

| Option | Behavior |
| --- | --- |
| `:searchable:` | Simple live search (existing option). |
| `:interactive:` | Search, sort, Reset, counts, and any requested field filters. |
| `:filter-fields: category,tags` | One dropdown for each field with data; values discovered from rendered cards. |
| `:sort-fields: title,published,duration` | Sort choices for fields with values; defaults to title. |
| `:search-fields: link-alt,tags` | Additional metadata included in text search. |
| `:search-label: Search videos` | Accessible search name and placeholder. |

Generic galleries can use arbitrary data fields. YouTube catalogs keep strict
identity keys and place site-specific metadata under ``fields:`` (for example
``fields: {category: Agent Skills}``). Those keys are exposed to the same
``group-by``/filter/sort/search field paths as ``gallery-grid`` while remaining
non-rendered data. Only fields explicitly needed by browser controls are embedded
in HTML; the entire input record is never serialized.

Search is case/accent-insensitive and matches every word, in any order.
Search and dropdown selections combine; values within a list field are alternatives.
Facet counts reflect the other current filters and search. Counts measure rendered
cards, including repeated appearances under different groups, not unique video IDs.

Sorting preserves each category heading and sorts cards within its grid. The
control says "Sort within categories" when several grids are present. Numeric
values sort numerically, ISO dates chronologically, and missing values remain last.
Ties preserve source order. Reset restores the original build order and all filters.
Search and filtering do not mutate the URL, use analytics, or make provider
requests. With a unique `:collection-id:`, visitors can separately opt in to
remember additions and to remember validated filter/sort choices; free-form
search text is never stored. See the package README. Embedded media and remote
images still contact their providers normally. Filters only cover items emitted by the Sphinx build; they
cannot fetch records excluded by build-time filtering/limits.

## Inline YAML and file YAML

Both forms were already supported. This release adds `:grid-columns:` as an alias
for the existing `:columns:` on youtube-gallery. Conflicting values produce a
located error. Existing `:columns:` configurations continue to work.

```rst
.. youtube-gallery::
   :grid-columns: 1 1 2 2
   :interactive:
   :filter-fields: channel,tags
   :sort-fields: title,published,duration

   videos:
     - id: JXtISpdDPNY
       title: Principal Component Analysis in Python
       channel: Statistics Globe
       tags: [pca, dimensionality-reduction]
```

A bare list without `videos:` is also valid. A channel collection instead uses a homogeneous `channels:` list; do not mix the two in one directive. The equivalent file input is:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :grid-columns: 1 1 2 2
   :group-by: channel
   :sort: -published
   :interactive:
   :filter-fields: channel,tags
```

For a typed YouTube catalog with arbitrary categories, keep the player identity
strict and add the category as metadata:

```rst
.. youtube-gallery::
   :grid-columns: 1 1 2 2
   :group-by: category
   :interactive:
   :filter-fields: category

   videos:
     - id: UNzCG3lw6O0
       title: Building Great Agent Skills
       fields:
         category: Agent Skills
```

Use raw ``gallery-grid`` when the card body itself needs arbitrary authored RST/MyST.
`.. gallery-grid:: path/to/gallery.yaml` still uses the same generic options.
MyST uses the same options in its directive fences; enable colon_fence when using
nested catalog rendering as documented by the existing extension.

## Behavior and limits

- Controls appear only when JavaScript runs; the complete static gallery remains
  usable otherwise. Print CSS restores filtered cards and hides the controls.
- Each toolbar owns its gallery. Nested interactive galleries are independent.
- Loading the controls twice does not create duplicate toolbars.
- Responsive, theme-variable CSS and keyboard-native inputs/selects are included.
- Sorting moves actual DOM nodes, keeping visual and keyboard order aligned.
  Where supported, moveBefore preserves iframe state. On other browsers moving
  a card may reload its player. Filtering hides a player; it does not pause it.
- Live controls target card modes (embed/thumbnail for youtube-gallery). The
  list mode is deliberately static; pairing it with reader-control options is a
  located configuration error instead of a silently ignored request.
- Real headless Chromium checks exercise the local runtime and responsive
  containment at 320, 768, and 1440 CSS px. They verify DOM/layout behavior,
  not live provider playback or every downstream theme's focus painting.

## Verification included

- 21 existing namespace/order checks and seven earlier regression methods passed.
- Existing RST/MyST metadata and nested-video rendering checks passed.
- New input checks cover inline list, inline videos wrapper, file data,
  column aliases/conflicts, MyST options, and safe/limited metadata emission.
- 21 local JSDOM checks cover independent galleries, combined search/facets,
  heading collapse, no results, Reset, stable sorting, decimal/date/missing-value
  ordering, accent-insensitive search, nested controls, repeated initialization,
  no-JavaScript content, and a 1000-card fixture.

To reproduce, install Sphinx, sphinx-design, PyYAML, myst-parser, and jsdom for tests.
Build the examples folder with Sphinx, then run `test_controls.cjs BUILD_DIRECTORY`
with jsdom available to Node. Run `test_inputs.py` for input/build checks. The
preceding regression scripts remain in `_maintenance/_gallery_revision`.

## Nested presentation customization

`youtube-gallery` forwards validated presentation settings to its generated grid,
cards, and players. Options apply to every corresponding element in that directive;
use separate gallery directives for groups with different settings.

| Prefix | Examples | Target |
| --- | --- | --- |
| grid- | grid-columns, grid-gutter, grid-margin, grid-padding, grid-outline, grid-reverse, grid-class-container, grid-class-row | Grid layout |
| card- | card-shadow, card-padding, card-margin, card-class-card, card-class-body, card-class-title, card-width, card-text-align | Grid card |
| video- | video-width, video-height, video-aspect, video-align, video-title, video-privacy-mode, video-url-parameters | YouTube player |

All grid/card options present in the installed Sphinx Design option_spec are
available with their prefix. Validators are taken from that same installed version.
`grid-columns` supplies the grid argument. Existing `columns`, `class-card`, and
`class-container` aliases remain supported. Namespaced card-class-card overrides
class-card; grid-class-container is added to class-container. Conflicting columns
and grid-columns values are rejected.

The custom gallery's data-selection controls are owned by youtube-gallery rather
than forwarded a second time: this keeps filtering, grouping, pagination and counts
consistent. Presentation settings have no effect in the lightweight list mode.

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :group-by: channel
   :grid-columns: 1 1 2 2
   :grid-gutter: 3
   :card-shadow: sm
   :card-class-card: learning-video-card
   :video-width: 100%
   :video-aspect: 16:9
   :video-privacy-mode: true
   :video-url-parameters: rel=0
   :interactive:
   :filter-fields: channel,tags
```

Player titles default to each video's title. video-title overrides that accessible
name for the whole gallery. Privacy accepts an empty flag or true/on/yes/1; false
/off/no/0 uses the normal embed host. URL parameters accept a leading question mark
or plain key=value pairs and are normalized before forwarding. The iframe referrer
policy remains unchanged. Sizes and aspect ratios must be positive; malformed
options produce source-located errors. Normalization does not guarantee a provider
supports every query parameter.

Card link/image overrides are explicit author settings; they can replace generated
links/images. Avoid card-link on embedded video cards because a whole-card link can
intercept player interaction. HTTP(S), mailto and relative paths are accepted by the
forwarding layer; unsafe schemes and multiline source injection are rejected.
For responsive players prefer video-width: 100% with video-aspect, leaving height
unset. Percentage heights still depend on the containing layout's height.

When a reader selects live sorting, source grid-reverse is temporarily removed so
the chosen sort order matches the visual order. Reset restores it.

Additional RST/MyST build checks verify forwarded grid/card classes, responsive
player aspect/width, privacy true/false, query parameters, custom player titles,
and located failures for invalid options. Direct youtube invalid size/aspect
errors are also checked to ensure they do not crash the build.

Percentage player heights now retain their percentage value; the 30-pixel control allowance is added only to pixel heights.

## Video catalog as a channel explorer

``youtube-gallery`` can project a reviewed ``videos:`` catalog into channel cards
with ``:view: channels``. Stable ``channel_id``/explicit-handle identity is used
offline; no provider request is made during the build. The result still uses the
exact same ``gallery-grid`` controls and title-only channel-card DOM. Aggregated
``video_count`` and tags are available to live sort/filter fields without adding
visible card prose. Channel-level sorting happens after projection, so sorting by
``title``, ``video_count`` or latest ``published`` activity has the expected meaning.

## Reader additions, export and restoration

The toolbar uses a compact search form and an options disclosure. Active settings
appear as removable chips, while a metadata-driven **Try** row suggests matching
search titles, useful facet values, and an available high-value sort. Add/export
forms are collapsed by default. The options panel closes on Escape, its Close
button, or an outside pointer interaction. Reset keeps added cards; Revert restores
the built gallery and clears this gallery's saved additions and saved view.

YouTube additions keep channel and video rendering separate. Channel-gallery additions keep only the title visible, preserve the leading `@`, and use the same invisible stretched link as published channel cards so the whole card remains clickable. Channel galleries do not show the videos/shorts/streams/courses selector; video galleries keep that selector for channel-derived video sources. Direct video, Short, live, and watch URLs immediately use the same `video_wrapper` + iframe markup as published video cards; watch URLs with `list=` still use their exact video ID. Recognized `http://` and scheme-less YouTube URLs are upgraded to canonical HTTPS. Channel sections and playlists in a video gallery resolve first to one exact video ID, then use that same single-video renderer; unresolved sources create no placeholder card. YouTube `/post/<id>` URLs are accepted as resolver sources but never treated as video IDs: a custom site resolver must return the attached exact video ID, otherwise the form asks for the video/Short URL from the post. The options pane includes compact newcomer guidance, a live source-readiness hint, and non-submitting prefill stubs for Video, Short, Live, Watch + list, Channel latest, Playlist latest, and Post. A collapsed optional site-setup disclosure shows a same-origin resolver stub so provider credentials can stay server-side. Resolver and browser API waits are bounded, so an unavailable integration cannot leave the Add action disabled forever. Saved latest-source additions retain the original source URL and only a visitor-supplied custom title; resolver-generated titles are refreshed with the resolved video on reload. Obsolete saved mode values are ignored on restore.

Export is now content-aware rather than format-selectable: video galleries emit
``videos:`` YAML and channel galleries emit ``channels:`` YAML, each directly
reusable by ``youtube-gallery``. See `_sphinx_youtube_gallery/README.md` for
`:collection-id:`, the two browser opt-ins, resolver contract, failed-storage
recovery, data limits, projection behavior, and RST/MyST examples.

## YouTube-gallery v10 capability alignment

``youtube-gallery`` is now a typed adapter over the exact same
``_sphinx_gallery_grid`` engine rather than a parallel UI implementation. A
video catalog may also be projected to a deduplicated offline channel explorer
with ``:view: channels`` when stable channel identity is present. Site-specific
facet/group/search metadata belongs under each record's explicit ``fields:``
mapping and is forwarded into the shared collection vocabulary without becoming
visible card prose.

Browser export is content-aware and homogeneous: video galleries export a
``videos:`` catalog containing the exact currently rendered video IDs, while
channel galleries export ``channels:`` with canonical HTTPS channel roots. The
old export-format selector and generic link-card flattening are removed. Saved
dynamic additions still preserve their original source intent so a future reload
can resolve “latest” again; exported YAML intentionally captures a reproducible
current snapshot instead.

Video records may carry `handle:` separately from `channel:`; sync preserves authored `handle`/`tags`/`fields`, checks exact normalized output, and writes atomically. Partial no-key RSS refreshes never prune unseen history. Keyed refreshes also preserve unmatched records unless `--prune` is explicitly requested; `--prune` is rejected without `YOUTUBE_API_KEY`.
