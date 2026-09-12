# YouTube Gallery for Sphinx

`_sphinx_youtube_gallery` renders a reviewed homogeneous YouTube catalog as
responsive Sphinx Design cards. It is a typed adapter over the local
`gallery-grid` directive: video records supply the same nested `youtube`
content an author could write manually, while channel records supply the same
title + stretched-link card a generic gallery uses. After normalization and
YouTube-specific build filtering, presentation and reader controls are delegated
to one `gallery-grid` and therefore one `sk-collection` root—there is no second
visual or browser-control layer to drift out of sync. The generic engine lives
in ``_sphinx_gallery_grid``; shared selection/browser behavior lives in
``_sphinx_collection``.

The Sphinx build reads local YAML and produces complete HTML. Live search,
filters, and sorting run locally over the rendered cards. They make no fetch,
cookie, analytics, or URL-history calls. Visitor additions and validated
filter/sort choices can each be remembered locally only after separate explicit
opt-ins; free-form search text is never stored. If JavaScript is unavailable,
the complete static gallery remains readable.

## Enable the extension

Use the fully qualified name in the main scikit-plots package:

```python
extensions = [
    "scikitplot._externals._sphinx_ext._sphinx_youtube_gallery",
]
```

Use the short name when the parent of a standalone `_sphinx_ext` directory is
on `sys.path`:

```python
extensions = [
    "_sphinx_ext._sphinx_youtube_gallery",
]
```

Choose one namespace consistently in a Sphinx application. The extension loads
its local YouTube and gallery dependencies, and the gallery loads
`sphinx_design`. Explicitly listing those dependencies remains safe in any
order. Do not also enable upstream `sphinxcontrib.youtube`, which owns the same
`youtube`, `vimeo`, and `peertube` directive names.

## Small inline catalog

A `videos:` wrapper is optional. It is useful when a catalog may later carry
sibling metadata:

```rst
.. youtube-gallery::
   :grid-columns: 1 1 2 2

   videos:
     - id: JXtISpdDPNY
       title: Principal Component Analysis in Python
       description: A practical PCA tutorial.
       channel: Statistics Globe
       tags: [pca, dimensionality-reduction]
       fields:
         category: Machine Learning & Algorithms
```

A bare YAML list is equivalent:

```rst
.. youtube-gallery::
   :grid-columns: 1 1 2 2

   - id: JXtISpdDPNY
     title: Principal Component Analysis in Python
```

A channel collection uses `channels:` instead. Keep one directive homogeneous:
do not mix `videos:` and `channels:` in the same invocation.

```rst
.. youtube-gallery::
   :interactive:
   :collection-id: learning-channels
   :grid-columns: 1 1 2 2

   channels:
     - "@youtube"
     - handle: claude
       title: "@claude"
       description: Searchable metadata; not rendered in the card body.
       tags: [ai]
```

Each channel becomes the same simple clickable card as `gallery-grid`: only the
title is visible in `.sd-card-body`; Sphinx Design's stretched link makes the
whole card clickable. A supplied `/videos`, `/shorts`, or other channel-tab URL
is canonicalized back to the channel root for this catalog type.

MyST uses the same content and options:

````markdown
```{youtube-gallery}
:grid-columns: 1 1 2 2

videos:
  - id: JXtISpdDPNY
    title: Principal Component Analysis in Python
    channel: Statistics Globe
    tags: [pca, dimensionality-reduction]
```
````

## File-backed catalog

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :group-by: channel
   :sort: -published
   :grid-columns: 1 1 2 2
```

The file may contain a bare video list, a mapping with `videos:`, or a
mapping with `channels:`. `videos:` and `channels:` are mutually exclusive in
one directive so runtime additions never have to guess which card model is
intended.

```yaml
videos:
  - id: JXtISpdDPNY
    title: Principal Component Analysis in Python
    channel: Statistics Globe
    published: 2024-03-01T10:00:00Z
    duration: PT12M30S
    tags: [pca, dimensionality-reduction]
```

```yaml
channels:
  - "@youtube"
  - handle: claude
    title: "@claude"
    tags: [ai]
```

### Explore channels from the same video catalog

A separate channel file is optional. If reviewed video records already carry a
stable ``channel_id`` or video-level ``handle`` (with ``@handle``/channel URL
in ``channel`` retained as a compatibility fallback), project the selected videos
into one deduplicated channel card per channel:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :view: channels
   :interactive:
   :filter-fields: tags
   :sort-fields: title,video_count
   :collection-id: learning-channel-index
```

This projection is offline and deterministic. It does not query YouTube during
the Sphinx build. YouTube-specific predicates first choose the contributing
videos; channels are then deduplicated; only after that are channel-level sorting,
grouping and pagination applied. Consequently ``:sort: title``, ``:sort:
-video_count`` and ``:sort: -published`` describe channels rather than whichever
video happened to be encountered first. Tags are unioned, ``video_count`` records
the selected-video count, and ``published`` is the newest known catalog timestamp.
Plain display names alone are intentionally skipped because they cannot form a
trustworthy channel link. Prefer ``channel: Claude`` plus ``handle: claude`` when
you want a human display name and a linkable identity; a stable ``channel_id`` is
even stronger when it is available.

``:view: auto`` (the default) follows the catalog kind. ``:view: channels`` may
project a video catalog or render a native ``channels:`` catalog. ``:view:
videos`` is valid only for a video catalog. The projection changes content type,
not UI: the result still renders through the same ``gallery-grid`` root and
channel cards remain title-only clickable cards.

Catalog files must be UTF-8. Explicit paths resolve relative to the current
document and must remain inside the Sphinx source tree. Sphinx tracks the file
as a dependency, so an edit invalidates each page that consumes it.

You may also configure a default source:

```python
youtube_catalog_path = "_data/youtube.yaml"
youtube_catalog_max_embeds = 24
```

Then use `.. youtube-gallery::` without a path or inline body. Input precedence
is inline content, positional path, `:catalog:`, then `youtube_catalog_path`.
Only one source is used.

## Custom gallery metadata without losing schema safety

YouTube identity stays strict, but video/channel collections often need site-specific
metadata such as ``category``, ``audience``, ``language``, or ``series``. Put those
values under an explicit ``fields:`` mapping:

```yaml
videos:
  - id: UNzCG3lw6O0
    title: Building Great Agent Skills
    fields:
      category: Agent Skills & Design Patterns
      audience:
        level: intermediate
```

The adapter validates ``fields:`` as data, then flattens its top-level keys into the
same query/render metadata record used by ``gallery-grid``. That means ordinary
field paths work without another YouTube-specific option vocabulary:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :group-by: category
   :sort: category
   :interactive:
   :filter-fields: category,audience.level
   :sort-fields: title,category
   :search-fields: category,audience.level
```

Nested mappings use dotted paths. Identity and presentation names (for example
``title``, ``link``, ``content`` and ``shadow``) are reserved inside the top level of
``fields:`` so metadata cannot accidentally replace a player, channel link, or card
option. Unknown top-level catalog keys are still errors; this keeps typos such as
``title`` visible while giving intentional extension metadata one explicit home.

A native ``channels:`` catalog may use ``fields:`` in the same way. A derived
``:view: channels`` projection does not guess how arbitrary per-video custom fields
should aggregate; it exposes the well-defined derived fields ``video_count``,
``published`` and unioned ``tags``.

## Build-time catalog query

These options determine which records enter the built page:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :channel: Statistics Globe
   :tags: pca,python
   :since: 2024-01-01
   :until: 2027-01-01
   :match: principal component
   :sort: -published
   :group-by: playlist
   :limit: 24
   :offset: 0
   :show-count:
```

| Option | Meaning |
| --- | --- |
| `view` | `auto`, `videos`, or `channels`; `channels` can project a video catalog into a deduplicated offline channel index. |
| `channel`, `playlist` | Match display name or stable ID; accepted URLs are reduced to their relevant identity. |
| `tags` | Require every comma-separated tag. |
| `match` | Case-insensitive substring over title and description. |
| `match-regex` | Bounded case-insensitive expression subset over title and description; mutually exclusive with `match`. |
| `since`, `until` | Half-open publication range: `since <= published < until`. |
| `sort` | Built-ins (`title`, `published`, `duration`, `position`, `channel`, `playlist`) or any present `fields:` path; prefix `-` for descending. |
| `group-by` | Built-ins (`playlist`, `channel`, `year`) or any present `fields:` path; `none` disables grouping. |
| `limit`, `offset` | Apply pagination after filtering and sorting. |
| `show-count` | Use the same gallery-grid count message when pagination hides records. |
| `section-style` | `auto`, `section`, or `rubric`; `auto` uses real sections where the parser context permits them. |

An empty result produces a visible message and a build warning. It does not
leave an unexplained blank page.

## Rendering modes and scale

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :mode: auto
```

| Mode | Result |
| --- | --- |
| `auto` | Inline players through the configured embed budget; thumbnails above it. |
| `embed` | Always render players; exceeding the budget emits a warning. |
| `thumbnail` | Render linked thumbnail cards without players. |
| `list` | Render lightweight links without images or players. |

`show-duration` appends a known video duration to its title. Descriptions
remain catalog/search metadata and are not inserted into card bodies. Embedded
video cards therefore have the same visible structure as a `gallery-grid` card
containing `.. youtube::`: title + player, with no extra `sd-card-text`
paragraph. Channel cards are title + stretched link only.

For backward compatibility, the old `:show-description:` option is still
accepted as a no-op so existing pages do not fail immediately; new content
should omit it. The former nested `youtube-gallery > sk-collection` structure
is also flattened: `.youtube-gallery` now marks gallery-grid's own collection
root. Custom CSS should target that single root rather than depend on the old
extra wrapper.

## Live reader controls

Use `searchable` for search alone:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :searchable:
   :search-label: Search videos
```

Use `interactive` for search, Reset, result counts, optional facet dropdowns,
and optional sorting:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :interactive:
   :filter-fields: channel,tags
   :sort-fields: title,published,duration
   :search-fields: description,playlist
   :search-label: Search videos
```

Search is case- and accent-insensitive and requires every entered word, in any
order. Typed YouTube galleries index the human title by default; accessibility-only
link prose such as “Open … on YouTube” is deliberately excluded, while fields
listed in ``:search-fields:`` are added explicitly. Facet selections combine with
search and with each other. Facet counts
reflect the other active selections. Sorting is numeric for numbers,
chronological for ISO dates, and locale-aware text otherwise. Missing values
remain last. When a gallery is grouped, sorting stays inside each group.

Controls operate on rendered cards and visitor additions. They cannot reveal
records removed by build-time filters, limits, or offsets. `list` is deliberately
static: combining `:mode: list` with `:searchable:`, `:interactive:`, or their
reader-control options is a located configuration error instead of being silently
ignored. Reset restores the build order and any source `grid-reverse` layout.
Filtering hides a player without pausing it. Sorting may reload a player in
browsers that cannot preserve iframe state while moving DOM nodes.

Only metadata fields named by live filter, sort, or search options are embedded
for the browser controls. The full input record is never serialized into HTML.

## Grid, card, and player customization

Presentation options use prefixes to show which generated directive receives
them:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :grid-columns: 1 1 2 2
   :grid-gutter: 3
   :grid-margin: 2
   :grid-padding: 1
   :grid-outline:
   :grid-class-container: learning-gallery
   :grid-class-row: learning-row
   :card-shadow: sm
   :card-padding: 2
   :card-class-card: learning-video-card
   :card-class-body: learning-video-body
   :video-width: 100%
   :video-aspect: 16:9
   :video-align: center
   :video-privacy-mode: true
   :video-url-parameters: rel=0
```

- `grid-*` forwards validated Sphinx Design grid options.
- `card-*` forwards validated Sphinx Design card options.
- `video-*` forwards validated options to every generated `youtube` player.

`grid-columns` supplies the responsive grid argument. The existing `columns`
alias remains accepted, but different values for both produce a located error.
The existing `class-card` and `class-container` aliases also remain accepted.
The namespaced card class overrides the old card alias; the namespaced grid
container class is added to the old container alias.

Player settings are `video-width`, `video-height`, `video-aspect`,
`video-align`, `video-title`, `video-privacy-mode`, and
`video-url-parameters`. The title defaults to the current record title.
Privacy accepts an empty flag or `true/on/yes/1`; `false/off/no/0` selects the
normal embed host. Query parameters are normalized and limited to 128 fields.
Use `video-width: 100%` with an aspect ratio and no height for a responsive
player. Percentage heights depend on the containing layout.

Settings apply to all generated elements within that directive. Use separate
`youtube-gallery` directives when categories need different presentation.
Presentation settings have no effect in `list` mode. A whole-card `card-link`
can intercept an embedded player, so reserve link overrides for thumbnail cards.
Unsafe URL schemes and multiline option injection are rejected.

## Generic authored gallery

Use `gallery-grid` when each card needs arbitrary authored RST/MyST or presentation
markup that is not a typed YouTube record, including a manually nested `youtube`
directive:

```rst
.. gallery-grid::
   :grid-columns: 1 1 2 2
   :group-by: category
   :interactive:
   :filter-fields: category
   :grid-gutter: 3
   :card-shadow: sm

   - title: Building Great Agent Skills
     category: Agent Skills
     content: |
       .. youtube:: UNzCG3lw6O0
          :width: 100%
          :aspect: 16:9
```

`gallery-grid` accepts the same live options and `grid-*`/`card-*` presentation
options. It intentionally treats `content` as trusted documentation source.
`youtube-gallery` treats catalog metadata as data, supports intentional custom
metadata through `fields:`, and maps typed video/channel records into those same
`gallery-grid` card primitives. Use `youtube-gallery` when the content is YouTube;
use raw `gallery-grid` when authored card body markup itself is the feature.

## Catalog schema

A `youtube-gallery` is one homogeneous collection.

### Video catalog

Existing video catalogs are unchanged. Only `id` is required.

| Field | Type | Purpose |
| --- | --- | --- |
| `id` or `url` | string | A bare ID or supported single-video YouTube URL. |
| `title`, `description` | string | Visible title and searchable metadata. |
| `channel` | string | Human-facing channel display/grouping value. |
| `handle` | string | Optional linkable channel handle, with or without `@`; useful for offline `:view: channels` while keeping `channel` human-friendly. |
| `channel_id` | string | Stable `UC…` channel identity; strongest input for offline channel projection. |
| `playlist`, `playlist_id` | string | Display and stable playlist identity. |
| `position` | nonnegative integer | Editorial order within a playlist. |
| `published` | ISO date/datetime | Filtering, grouping, and sorting. |
| `duration` | seconds or ISO 8601 duration | Title decoration and sorting. |
| `tags` | string or list of strings | Topic filtering and live facets. |
| `fields` | mapping | Intentional site-specific gallery metadata; supports nested dotted paths and is never rendered as card prose. |

### Channel catalog

Use a `channels:` mapping. A channel entry may be a bare `@handle`, canonical
`UC…` channel ID, YouTube channel URL, or a mapping with:

| Field | Type | Purpose |
| --- | --- | --- |
| `id` or `url` | string | Channel reference. |
| `handle` | string | Handle, with or without the leading `@`. |
| `channel_id` | string | Stable `UC…` identifier. |
| `title` | string | Visible title; defaults to `@handle` when known. |
| `channel`, `description`, `tags` | metadata | Search/filter values; not rendered as card prose. |
| `fields` | mapping | Intentional channel-level gallery metadata; supports the same custom field paths as a video catalog. |

Unknown keys are rejected. Duplicate identities keep the first occurrence.
Channel-tab URLs are reduced to the channel root because this catalog renders
channel cards, not “latest video” requests. Supplying both `videos:` and
`channels:` is a located catalog error with guidance to split them into two
directives.

This separation is intentional: a video card owns a player; a channel card owns
a link. Shared search/filter/sort/addition behavior is provided by the same
collection/browser layer rather than by mixing the two card types.

## Standalone player

The local `youtube` directive accepts a bare ID or supported video URL:

```rst
.. youtube:: https://www.youtube.com/watch?v=JXtISpdDPNY&t=90s
   :width: 100%
   :aspect: 16:9
   :privacy_mode:
   :title: Principal Component Analysis tutorial
```

Channel, playlist, podcast, course, and other collection URLs do not identify a
single player. Use `youtube-gallery` and a synchronized catalog for those.
Always preview through an HTTP server; `file://` pages may not supply the HTTP
referrer YouTube now requires.

## Build and maintenance behavior

The HTML build performs no YouTube API call. Maintain or synchronize a catalog
explicitly, review the YAML diff, and build offline. For LaTeX-family builders,
the vendored player may download thumbnails with per-request timeouts and a
configurable aggregate limit:

```python
video_download_limit = 200
video_download_max_bytes = 8 * 1024 * 1024
```

Downloads are streamed with connect/read timeouts, checked for an image content
type, capped per response, and written atomically. A truncated response never
becomes a cached thumbnail.

Both gallery directives apply shared resource limits before and after YAML
parsing: 8 MiB of source, 100 aliases, 32 nesting levels, 100,000 parsed values,
1 MiB of scalar text, and 5,000 collection items. Oversized or recursive input
is a located Sphinx error. Split larger collections across pages.

`match-regex` uses a bounded expression subset to keep build time predictable:
literals, character classes, anchors, dot, `?`, and bounded `{m,n}` repeats up
to 100. Groups, alternation, backreferences, `*`, `+`, and open-ended repeats
are rejected. Use `:match:` for normal case-insensitive text search.

Players without an explicit height use a responsive CSS aspect-ratio wrapper.
This includes the plain `.. youtube:: VIDEO_ID` form, so default players retain
their aspect ratio inside narrow cards and on mobile screens.

The package declares parallel read/write safety. Catalog and query operations
are deterministic; unchanged synchronization avoids rewriting the catalog.

YouTube URL grammar and leaf-player option validation are owned by the
dependency-free sibling package ``_sphinx_youtube_core``. Both this gallery and
``_sphinxcontrib_youtube`` import that lower-level core, preventing the leaf
player from depending back upward on the gallery. The historical private imports
``_sphinx_youtube_gallery.reference`` and ``_sphinx_youtube_gallery._video_options``
remain compatibility facades over the canonical core.
The sync pipeline treats YouTube-owned metadata and author enrichment separately:
matching ``handle``, ``tags`` and ``fields`` survive provider refreshes by video
id, while title/channel/publication/playlist facts can refresh from YouTube.
``--check`` compares the exact normalized YAML that a sync would write, so
metadata drift is detected even when the set of video ids is unchanged. Catalog
writes use an atomic same-directory replacement. Credential-free RSS is a
partial recent-items feed, so a no-key refresh preserves older unmatched catalog
records instead of mistaking "not returned" for "deleted". Unmatched records are
also preserved on keyed refreshes by default: deletion intent is explicit via
``--prune``. ``--prune`` requires ``YOUTUBE_API_KEY`` because partial RSS cannot
prove absence. This keeps provider capability separate from destructive policy. A pasted
``watch?v=VIDEO&list=PLAYLIST`` source is treated consistently as the exact
video; use a playlist URL/ID or ``--playlist`` when the intended sync unit is
the whole playlist. Browser controls are scoped per gallery, tolerate multiple
and nested galleries, and guard repeated initialization.

## Migration

The package was renamed from `youtube_catalog` to `_sphinx_youtube_gallery`.
The shared engine was renamed from `collection` to `_sphinx_collection`.
Shared YouTube parsing/player-option primitives now live in ``_sphinx_youtube_core``;
direct callers may migrate to that canonical owner, while the old gallery-private
module paths remain compatibility facades.
After merging the replacement tree, remove the obsolete ``youtube_catalog/``,
``collection/``, and ``_pydata_sphinx_theme/`` directories, update direct
private imports to ``_sphinx_youtube_gallery``, ``_sphinx_collection``,
``_sphinx_gallery_grid`` or ``_pydata_component_list`` as appropriate, and
rebuild with a fresh Sphinx environment. Do not delete unrelated private
extensions that are not part of this replacement:

```bash
python -m sphinx -E -a -b html SOURCE_DIRECTORY BUILD_DIRECTORY
```

The public directives and configuration values retain their existing names:
`youtube`, `gallery-grid`, `youtube-gallery`, `youtube_catalog_path`, and
`youtube_catalog_max_embeds`. The legacy `:show-description:` option is accepted
as a no-op; descriptions remain searchable metadata instead of visible
`sd-card-text` card prose.

## Included verification

The distribution contains focused checks under `_maintenance` for both package
namespaces, extension ordering, RST/MyST parity, inline and file-backed input,
nested presentation forwarding, malformed options, bounded YAML and regex
inputs, streamed thumbnail failures, local browser controls, no-JavaScript
content, nested galleries, and a 1,000-card fixture.

These checks validate generated structure and local control behavior. They do
not constitute live YouTube playback testing or visual verification against
every downstream theme.

## Search, options, and reverting changes

The compact toolbar shows a rounded gallery search field, search button, and
an overflow (⋮) button. Typing filters locally; Enter and the search button
apply the same search. Active search, filters and sorting appear as removable
chips below the result count. Clear a chip to remove only that setting.

A small **Try** row offers context-aware shortcuts without hiding the canonical
controls. While typing it can suggest matching card titles. For interactive
galleries it also derives high-value facet suggestions from the current result
set and offers a useful sort such as **Newest first** or **Title A–Z** when that
sort exists. Suggestions are generated from the gallery metadata, so new filter
fields and sort fields participate without JavaScript changes.

The options disclosure holds filter/sort controls, Add video or channel,
Export additions, optional browser preferences, Reset view, Revert to initial
gallery, and Close options. Add and export forms expand only when needed. The Add
form includes compact beginner guidance plus prefill chips for Video, Short, Live,
Watch + list, Channel latest, Playlist latest, and Post source shapes. These chips only
fill the input and select the replaceable placeholder token; they never submit or add a
card automatically. A live readiness hint classifies what the visitor pasted before
submission: exact video sources are marked ready without an API key, while latest channel,
playlist, Courses, or Post lookups explain when optional site support is needed. A collapsed
**Site setup for latest/post sources (optional)** disclosure includes a same-origin resolver
stub so maintainers can keep provider credentials server-side. Escape
closes options and returns focus to the trigger. Clicking or tapping outside the
panel also closes it, but leaves focus with the destination the visitor chose.
Native controls use ordinary Tab navigation. The panel stays in document flow
to avoid covering players; its height is bounded, overscroll is contained and
its content scrolls when necessary. The layout follows available gallery width,
including narrow columns on desktop. Inputs retain 44px minimum height at the
standard root font size, theme colors, focus outlines and logical RTL spacing.

| Action | Visible cards | Saved additions / view |
| --- | --- | --- |
| Reset view | Clear search/filters/sort; keep added cards | Keep consent; saved filters/sort become clear |
| Forget saved additions / saved view | Keep the current visible gallery | Clear only that saved scope |
| Revert to initial gallery | Restore original cards/order, clear view and forms | Clear additions and saved view; turn both remembering choices off |

Revert does not reload the page or reset playback in original frames. If browser
storage cannot be cleared, the original visible gallery is still restored and
options stay open with a Forget saved additions retry button. It never clears
unrelated browser keys. The initial state is the gallery built from your source,
not the saved additions restored on page load.

## Adding cards and optional remembering

A gallery containing YouTube links or players offers runtime additions. Channel
galleries use a simple channel form with no videos/shorts/streams/courses selector.
Video galleries keep that section selector for channel-derived video sources. Supply
a video ID, a video/Short/live URL, a bare channel ID, an @handle, a channel URL, a
playlist URL, or a YouTube post URL, plus an optional title. Recognized `http://` and
scheme-less YouTube links are normalized to canonical `https://` URLs before use. Additions live in a separate Your additions
section. Channel cards keep only the title visible and retain the published gallery's
invisible stretched link so the whole card is clickable. Direct videos insert their
iframe immediately. Channel sections and playlists **resolve first** to one exact
video ID and only then use the same single-video renderer; an unresolved source never
creates a cosmetic card. The cards participate in search, counts and sorting.
Visitor-added records still have no authored tags or facet metadata, so metadata
filters can hide them.

References are canonicalized, duplicates rejected, and additions limited to 100
per gallery. Titles are literal text, up to 200 characters on one line. Direct
API-shaped or programmatic input is checked as well as browser form input.

By default additions last for this visit only. To offer optional remembering,
assign a stable ID unique among interactive galleries on the same page:

```rst
.. youtube-gallery:: ./_data/youtube.yaml
   :interactive:
   :collection-id: learning-videos
   :grid-columns: 1 1 2 2
   :filter-fields: channel,tags
```

The same `:collection-id:` option works on `gallery-grid` when its cards contain
YouTube references. The ID starts with a letter and allows up to 64 letters,
digits, underscores or hyphens. Missing or duplicated IDs leave remembering
unavailable and perform no storage access for those galleries.

Added cards intentionally preserve the published gallery card models instead of
sharing one mixed runtime card:

- **Channel additions** render with only the card title visible, while keeping the
  same invisible stretched link used by published channel cards so the whole card
  remains clickable. For a modern handle, the default title keeps the leading `@`.
- **Direct video additions** render immediately as the same `video_wrapper` +
  YouTube iframe structure used by published video cards. No extra source badge,
  note, external link, or play button is inserted into the card.
- **Channel-section and playlist additions in a video gallery** are not another
  card type. They resolve to one exact current video ID first, then render through
  that same direct-video path. While resolution is pending, feedback stays in the
  existing add form and no placeholder card is inserted.

Reader additions accept exact videos, whole playlists, YouTube posts, plain channel
names, stable channel IDs, and modern channel sections. Channel galleries keep channel
addition simple and show no section selector. Video galleries show the compact
**Latest channel section** selector, which normalizes bare channel inputs to `videos`,
`shorts`, `streams`, or `courses`. Direct video/Short/live/watch URLs ignore that
selector and render their exact video immediately. A watch URL that also carries a
`list=` parameter remains an exact-video source; the playlist context does not turn it
into a latest-playlist request. Examples:

```text
@claude
@claude/videos
https://www.youtube.com/@claude/shorts
https://www.youtube.com/@claude/streams
https://www.youtube.com/@claude/courses
https://www.youtube.com/playlist?list=PLHfy8mSAC18s
https://www.youtube.com/watch?v=wGA27zJEnaU
https://www.youtube.com/shorts/2TxRkGcPL-o
https://www.youtube.com/watch?v=cdPyuLr_rI0
https://www.youtube.com/watch?v=P8ggfb67ODs&list=PLbpi6ZahtOH6eTD4bB5qJ50QQRHz2leKM
https://www.youtube.com/watch?v=9rqQgOERpDo&list=PLKDZ1ig0uz-U
http://youtube.com/post/POST_ID?si=SHARE_TOKEN
```

Rendering is determined by the gallery and the resolved identity, not by a
visitor-selected card mode. Channel galleries keep channel sources as clickable
channel cards. Video galleries require one exact video ID before card creation.
Older saved additions that contain the previous `mode` property are still accepted;
the obsolete property is ignored when restoring them.

### Resolving a current channel-section or playlist video

The browser resolver has a strict boundary: it returns `{videoId, title?}` and the
normal video-card renderer owns everything after that. Direct video/Short/live/watch
URLs never need this resolver. The preferred integration for dynamic sources is a
site-provided resolver, which can keep provider credentials server-side behind a
same-origin endpoint, use a build-maintained service, or apply any policy appropriate
for the documentation host. The options pane includes a compact version of this stub:

```js
window.skCollectionResolveYouTubeLatest = async function (source) {
  // source: {kind, section, handle, channelId, playlistId, postId, name, url}
  const response = await fetch('/api/youtube/latest?' + new URLSearchParams({
    url: source.url,
    section: source.section || ''
  }));
  if (!response.ok) throw new Error('Latest-video lookup failed');
  return response.json();  // {videoId: '...........', title: 'Optional title'}
};
```

For a static site that intentionally exposes a browser API key, the built-in
resolver can instead use:

```js
window.skCollectionYouTubeDataApiKey = 'YOUR_REFERRER_RESTRICTED_BROWSER_KEY';
```

Treat that value as public client configuration, not a secret, and restrict it to
the documentation origins and required YouTube Data API. With it, modern `@handle`
inputs are resolved to their canonical `UC…` channel ID through `channels.list`
`forHandle`. For `videos`, `shorts`, or `streams`, the built-in resolver derives the
section compatibility playlist and then uses `playlistItems` publication timestamps
to select one exact newest video. Those section-playlist identifiers (`UULF…`,
`UUSH…`, `UULV…`) are YouTube compatibility behavior rather than a published Data
API contract, so a site resolver remains the most future-proof path. If a canonical
`UC…` channel ID is supplied without an API key, the browser can still use the
YouTube iframe player to inspect item zero of that compatibility playlist as a
best-effort zero-key fallback. Modern `@handle` inputs still require either the site
resolver or the Data API key because the iframe API does not resolve handles to
canonical channel IDs. The built-in resolver does not guess `/courses`: the public
Data API does not expose a stable Courses-tab classifier, so `/courses` requires the
site resolver.

For an explicit playlist URL, the API-key path walks a bounded number of
`playlistItems` pages and chooses the greatest `contentDetails.videoPublishedAt`.
It therefore means *newest published video in that playlist*, rather than silently
assuming the playlist's position zero is newest. A site resolver can implement a
different editorial definition if needed.

Browser resolution is bounded: Data API fetches time out rather than leaving the
Add action disabled indefinitely, and a custom ``skCollectionResolveYouTubeLatest``
promise has a 15-second safety timeout. Timeout/failure feedback stays in the add
form and no placeholder card is created. The zero-key iframe compatibility probe is
rendered off-screen (not ``display:none``) so the player can initialize without
adding visible UI.

A `/post/<post-id>` URL is accepted and canonicalized, but it is not itself a video
identity. YouTube posts may contain text, images, polls, playlists, or videos, and the
public Data API does not provide a stable post-to-attached-video lookup used here.
Therefore a post becomes a video card only when `skCollectionResolveYouTubeLatest`
returns an exact `videoId`; otherwise the visitor is asked to paste the attached
video/Short URL. This preserves the resolve-first rule and prevents a post ID from
being misused as an embed ID.

The source is preserved separately from the resolved card. Saving
`https://www.youtube.com/@claude/shorts`, for example, stores that source URL, not
the video ID that happened to be latest today. Persistence also stores only the
visitor's optional custom title; a title returned by the resolver is display data and
is refreshed with the video ID on the next restore. Restoring the page therefore
resolves the source again and renders a matching current ID/title pair. Direct-video
additions keep their exact source unchanged.

The visitor can separately select **Remember my additions in this browser** and
**Remember my filters & sorting**. Both are opt-in. The view preference stores
validated facet values and sorting only; free-form search text is never stored.
The keys are scoped to browser origin, page pathname and collection ID; query
strings and fragments are excluded.
Reload restores only validated saved records. Bad JSON, oversized data, invalid
references and duplicate records leave the published gallery available and
provide a clear-copy action. Records already in the published gallery are skipped.

A browser-storage event or a detected stale saved value pauses saving; current
cards remain for the visit. Turning remembering on again explicitly saves that
view. This is local browser state, not account sync or a transactional multi-tab
editor. Browser privacy settings or storage limits may prevent saving. Changing
a page path or its collection ID gives it a different saved scope; stable IDs
should be preserved when moving a gallery within a page.

## Export additions for maintainers

Open **Export additions** to inspect real, human-readable YAML before
downloading. There is no format selector because the collection already has one
unambiguous content type:

| Gallery | Export | Round trip |
| --- | --- | --- |
| Video gallery | ``videos:`` with the exact video IDs currently rendered | ``youtube-gallery``; no resolver is needed for the downloaded snapshot |
| Channel gallery | ``channels:`` with canonical HTTPS channel links | ``youtube-gallery`` as the same title-only clickable channel cards |

Resolved latest-channel, playlist, or post additions deliberately export their
**current exact video ID**. The browser therefore creates a reproducible catalog
snapshot rather than a hidden recipe that would need API access again. The
optional saved-additions feature is different: it keeps the original dynamic
source so a later browser session can resolve “latest” again.

The exporter writes YAML structure directly and uses JSON string quoting only for
individual scalar values, which is valid YAML and safely preserves punctuation,
Unicode, and authored titles. Only visitor-added records are exported; search
text, filters, original published records, browser permissions, and executable
HTML are excluded.

Review and commit the downloaded file, then reference it normally:

```rst
.. youtube-gallery:: youtube-video-additions.yaml
   :interactive:

.. youtube-gallery:: youtube-channel-additions.yaml
   :interactive:
```

Download failure leaves the preview selectable for copying. Export never writes
source files or publishes changes itself.

Plain display names are intentionally not guessed through provider search because
they are ambiguous; use an `@handle`, a canonical `UC…` channel ID, or a site
resolver that applies your own disambiguation policy. Shared publication remains an
author/maintainer workflow rather than browser state.

## Validation of reader changes

Rebuild documentation and refresh cached static assets after replacement.
Checks include the existing Sphinx/DOM suites plus focused coverage for local
strict channel/video runtime card separation, suggestion chips, outside-click
panel dismissal and separate saved-view state. Keyboard closing, composition
input, corrupt storage, deletion failures, stale tabs, unique disclosure IDs and
scoped storage remain covered. Browser visual layout and live provider playback
still require device/browser verification.
