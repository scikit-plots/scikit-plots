# `_sphinx_ai_learn`

`_sphinx_ai_learn` is a JSON-first Sphinx extension for the Scikit-Plots Learn
site. Canonical content lives below `docs/source/learn-ai` as validated JSON.
The extension deterministically materializes one sibling RST file for every
canonical JSON file during Sphinx `config-inited`; normal Sphinx rendering then
produces HTML or another builder output. Typed `design_grid` pages activate the
`sphinx_design` dependency through the extension itself, so correctness does not
depend on a project's extension-list ordering.

```text
*.json                     canonical repository authority
   ↓
_sphinx_ai_learn           deterministic compiler/materializer
   ↓
*.rst                      derived Sphinx source
   ↓
Sphinx                     renderer
   ↓
*.html                     build output
```

Generated Learn RST is never an authoring authority. Do not hand-edit it. The
Sphinx build performs no model calls, network fetches, Git operations, Hugging
Face writes, or canonical JSON mutations.

## Configuration and lifecycle

A normal documentation configuration needs the extension and its content root:

```python
extensions += ["_sphinx_ext._sphinx_ai_learn"]
ai_learn_content_root = "learn-ai"
ai_learn_site_id = "scikit-plots-learn"
ai_learn_runtime = "assistant"  # or "none"
ai_learn_explorer_search_variant = "pill-overflow"  # or "classic"
ai_learn_buttons_ratings = {
    "left_button_rating": "left",
    "right_button_rating": "right",
}
```

Materialization runs at `config-inited`, before Sphinx discovers/reads source
documents. That is required so RST newly derived from JSON participates in the
same build. The materializer validates the whole JSON tree, writes only changed
extension-owned RST, prunes only stale extension-owned RST, and records the
semantic content digest for Sphinx environment invalidation.

`ai_learn_buttons_ratings` controls only the reviewed-count placement inside the
two compact AI Learn feedback buttons. `left_button_rating` targets thumbs-down
and `right_button_rating` targets thumbs-up; each accepts `"left"` or `"right"`.
The balanced default is `0 | 👎` and `👍 | 0`. Partial dictionaries inherit the
missing side from that default. Unknown keys or unsupported values fail the
build instead of silently changing presentation. The setting does not alter the
`-1/+1` quick-feedback contract, detailed `-5..+5` ratings, canonical events, or
aggregate counts.

The extension also owns custom feedback-dependency state in the Sphinx build
environment. Its setup metadata therefore carries an explicit `env_version`, and
parallel reads use `env-purge-doc` plus `env-merge-info` with replace-on-reread
semantics: worker docnames are removed from the master dependency map before the
worker's current registrations are merged. This prevents stale record→document
dependencies from surviving cached or parallel rebuilds. Optional YouTube thumbnail
state follows the same version/purge/merge rule when `ai_learn_media=True`; its
environment schema is separately versioned so changes in thumbnail-cache semantics
invalidate old Sphinx pickles.

## Canonical contracts

Repository JSON uses five source contracts:

- `learn.page.v1`: structural/library/create/explorer pages.
- `learn.topic-prompt.v1`: reusable Topic Prompt definitions at
  `topic-prompts/<id>/index.json`.
- `learn.skill.v1`: reusable Skill definitions at `skills/<id>/index.json`.
- `learn.record.v2`: immutable Topic/Source/Problem/media record routes and
  ordered section references.
- `learn.section.v1`: one record-owned section/result artifact.

`learn.catalog.v3` is only the validated normalized in-memory graph assembled
from that JSON tree. There is no repository `catalog.json` input.

### Shared index explorer mastheads

Index/library `learn.page.v1` documents may declare an `explorer_header` object
with exactly two bounded plain-text fields, `kicker` and `title`. The materializer
projects that object to the shared `ai-index-explorer-header` directive. The
renderer owns the Bookmarks/Collections routes, so canonical JSON cannot inject
link targets and every supported index uses one responsive, accessible masthead
contract. Create-action labels remain canonical through `create_label`.

```json
{
  "create_label": "Create a Topic",
  "explorer_header": {
    "kicker": "Topic explorer",
    "title": "Trending Topics"
  }
}
```

`explorer_header` is accepted only on explorer/media-gallery/prompt-library/
skill-library page views; unsupported views and unknown nested fields fail
validation rather than being ignored.

### Optional Sphinx-Design page grids

`learn.page.v1` may declare a typed `design_grid` projection. It is the only
canonical contract that may describe **structural page composition**; record/section
bodies remain plain data and can never inject RST directives. A separate bounded
boolean layout policy, `hide_secondary_sidebar`, is shared by every JSON contract
that owns a materialized RST document. `design_grid` maps a bounded subset
of Sphinx-Design `grid` / `grid-item-card` options plus nested toctrees into derived
RST. Raw RST, arbitrary directive names, image/link directives, unsafe docnames and
unbounded class/options are rejected. The field is optional, so omitting it preserves
the existing page renderer.

```json
{
  "design_grid": {
    "columns": [1, 1, 1, 1],
    "items": [
      {
        "title": "topics",
        "card": {"padding": 2, "columns": [12, 12, 6, 6]},
        "toctree": {"maxdepth": 2, "children": ["topics/index"]}
      }
    ]
  }
}
```

Supported grid options are responsive `columns`, `gutter`, `margin`, `padding`,
`outline`, `reverse`, `class_container` and `class_row`. Cards support responsive
`columns`/`margin`/`padding`, child direction/alignment, outline, text alignment,
shadow, and bounded class hooks. Nested toctrees support safe child docnames,
`maxdepth`, `hidden`, `titlesonly` and a plain-text caption.
For the root page, a declared `design_grid` must cover `children` exactly once and
in the same order. Every nested toctree target must resolve to another canonical
JSON-owned document, so layout typos and omissions fail during materialization rather
than becoming broken or orphaned navigation at Sphinx build time.


### Secondary sidebar policy

Every canonical JSON artifact that owns a sibling RST document may declare:

```json
{
  "hide_secondary_sidebar": true
}
```

The field is optional at the validator boundary and defaults to `true`, so older or
programmatically constructed artifacts remain fail-safe. Repository-owned canonical
JSON writes the field explicitly so the layout decision is visible in review. `true`
materializes `:html_theme.sidebar_secondary.remove:`; `false` omits that page metadata
and lets the active Sphinx theme render its secondary sidebar normally.

The policy applies uniformly to `learn.page.v1`, `learn.record.v2`,
`learn.section.v1`/`v2`, `learn.topic-prompt.v1`, and `learn.skill.v1`. Feedback
sidecars do not materialize RST and therefore do not accept this field. Publication
projection preserves existing record/section values and gives newly created
renderable artifacts the explicit hidden default.

## Record routes and section composition

Record folders are deterministic:

```text
<kind-directory>/<YYYYMMDDTHHMMSSZ>-<16-hex-digest>/
```

The timestamp is immutable `created_at`; the digest is derived from stable
record kind + ID. A title edit therefore does not move the public route.

`learn.record.v2` has `add_toctree`, default `false`.

### Default: include composition

The record `index.rst` is the navigable page. It owns public headings/labels and
includes child bodies:

```rst
.. _learn-topic-example-summary:

Summary
-------

.. include:: summary.rst
   :start-after: .. ai-learn-fragment-start
```

Child RST is a directly renderable but non-navigation fragment document:

```rst
:orphan:
:no-search:

.. Generated by _sphinx_ai_learn from canonical JSON. DO NOT EDIT.

:html_theme.sidebar_secondary.remove:

.. ai-learn-fragment-start

.. ai-topic-section:: summary
   :topic-id: topic-example
```

The sentinel deliberately decouples include boundaries from Sphinx metadata.
The parent owns global labels so parsing the child independently cannot duplicate
an explicit anchor.

### Optional: toctree composition

With `"add_toctree": true`, the record `index.rst` emits `.. toctree::`
instead of includes. Children are normal navigable documents, omit `:orphan:`,
and own their headings/labels. Include and toctree modes use the same JSON and
renderer; this is only a composition/navigation policy.

## Topic Prompts and Skills

Reusable definitions are distinct from per-Topic generated results.

```text
learn-ai/topic-prompts/eli14/index.json
    reusable prompt definition

learn-ai/topics/<topic-route>/topic-prompts/eli14.json
    result of that prompt for one Topic

learn-ai/skills/skill-check-reference/index.json
    reusable Skill definition

learn-ai/topics/<topic-route>/skills/skill-check-reference.json
    result of that Skill for one Topic
```

Each Topic also has `topic-prompts/index.json` and `skills/index.json` group
artifacts. In default include mode they are orphan/no-search child sources just
like other Topic sections.

Prompt and Skill definitions share the same bounded interaction shape:
`id`, `title`, `author`, `description`, `instruction`, `empty_message`,
`default_enabled`, and explicit `order`. Skills additionally carry `domains`
and `related`. Their IDs are required to be disjoint from each other and from
fixed structural Topic section IDs.

Python owns renderer/generation policy only. Prompt/Skill wording, ordering,
enablement, and reusable instructions are JSON content rather than a duplicate
Python registry.

Reusable Skill definitions are projected into the normalized in-memory catalog
as `kind="skill"` so existing graph relations/navigation can address them, but
there are no timestamped `learn.record.v2` Skill records in the repository.

## Publication

Browser/private generation remains private until review. Publication is
revision-bound and JSON-only:

```text
private draft
   ↓
reviewed transaction against exact tree revision
   ↓
canonical JSON diff only
   ↓
GitHub pull request / review / merge
   ↓
ReadTheDocs Sphinx build
   ↓
JSON → RST → HTML
```

`_publication.py` uses the same canonical projection helpers as the materializer,
so publication cannot invent a second folder convention. `create-topic-prompt`
and `create-skill` are fan-out operations: a new reusable definition, every
Topic's updated `index.json`, and every Topic's empty result JSON are proposed
atomically.

The CLI accepts record, Topic Prompt, and Skill drafts. `--created-at` is needed
only for record drafts:

```bash
PYTHONPATH=docs/source python -m _sphinx_ext._sphinx_ai_learn._publication_cli \
  docs/source/learn-ai /path/to/private-draft.json /tmp/learn-review
```

For records add `--created-at 2026-09-24T00:00:00Z`. For reusable interactions,
`--author`, `--order`, and `--default-enabled` are optional review-owned fields.
The review bundle contains canonical `.json` changes plus a hashed manifest;
it never contains generated RST as publication authority. Browser publication
handoffs may add an optional public contributor display name; blank credit becomes
`Anonymous`, is bounded plain text, and is never sent to the generation model.
Published state and evidence-review state are rendered separately. Record-level
`authors` describe overall stewardship; accepted section generations carry their own
`contributors`, provenance, creation time, and reviewed feedback events. Legacy
`learn.section.v1` content is promoted lazily to the versioned generation ledger only
when a reviewed update/feedback requires it. Quick feedback is `-1/+1`; detailed
feedback is the eleven-point `-5..+5` scale with optional text/credit. Scores are
derived only from merged canonical events, and ambiguous retries reuse the same
feedback event nonce rather than double-counting. V71 browser feedback nonces are
192-bit CSPRNG values and contain no timestamp, account, device, browser, network,
or telemetry identity. New reviewed sidecar writes require that opaque nonce shape at
both the public proxy and repository workflow boundaries; V70 128-bit nonces remain
read/retry compatible. Reviewed feedback fan-out is bounded per generation, and abuse
limiting is separate, scope-pseudonymized server control-plane state rather than
feedback metadata.

See `PUBLICATION_LIFECYCLE.md` for the full lifecycle and security boundary.

## Materializer guarantees

`_materialize.py` is the sole JSON layout/RST compiler authority. It rejects:
unsafe paths, symlink canonical paths, duplicate JSON keys, non-finite values,
oversized files/trees, unknown contracts, orphan sections, duplicate IDs/orders,
Prompt/Skill/structural-ID collisions, invalid citation targets, cross-record
ownership, and handwritten RST collisions.

Model/user body text remains data consumed by trusted directives; it is not
interpreted as arbitrary model-authored RST structure. Writes use temporary
files plus `os.replace`. Unchanged outputs are not touched, preserving mtimes
and preventing source-watch rebuild loops.

Repository JSON uses deterministic pretty serialization; semantic revisions use
canonical compact serialization, so formatting-only edits do not alter the tree
revision.

Graph-integrity failures are reported as repository repair instructions, not generic
validation failures. Materialization aggregates every dangling `related` target and
invalid citation it can find before raising, and each message includes the owning
canonical JSON path, subject/section identity, and bad target. The compiler never
silently drops a broken relation or substitutes a placeholder subject.

## Shared index UI contract

Index presentation is derived from shared extension templates/CSS; generated RST
only selects directives and structural classes.

- Topic Prompt and Skill libraries intentionally share the same card, switch,
  action, edge, hover, focus, and responsive grid contract. Prompt/Skill semantics
  remain separate, but presentation must not fork. The nested Prompt/Skill detail
  shell owns the same `--learn-line`, `--learn-soft`, and `--learn-accent` tokens
  as the index libraries so identical `learn-switch` markup renders identically.
  Switch checkboxes own their accessible name through `aria-label`; do not add
  theme-owned `sr-only`/visually-hidden text inside the switch because hidden-helper
  availability is theme-dependent and any unhidden text changes the flex geometry.
  The visual switch footprint is fixed at 76px so title length never moves cards,
  topic rows, or detail headings.
- Every creation index uses `learn-index-actions` plus a page-specific hook such
  as `learn-topic-index-actions` or `learn-video-index-actions`. The shared class
  controls spacing only; the link deliberately keeps the native theme/Sphinx link
  treatment instead of introducing modality-specific filled buttons.
- Video and Whiteboard explorers are two-column media galleries at wide widths.
  Their `learn-card` surfaces use a complete border/radius/background instead of
  the generic list-row bottom divider so adjacent entries remain visually distinct.
- A Video record with a schema-validated `media.youtube_id` renders one semantic
  media card in `meta → title → youtube directive` order. The directive uses the
  privacy-mode player and lazy/responsive behavior supplied by the vendored
  YouTube extension.
- A Whiteboard with validated images uses the same media-card hierarchy in
  `meta → title → image` order. Only the first image is the index preview; the
  Whiteboard detail page remains the owner of the complete multi-image gallery.
  A media record without playable/preview media falls back to the generic text
  card rather than failing the explorer.
- Old `learn-*-index-gallery`/`sd-card` index selectors are not compatibility
  surfaces. Media indexes use the semantic `learn-media-card` structure only.
- Every catalog explorer shares one compact search-control contract, regardless
  of whether results render as table rows (Topics, Sources, Open Problems, Skills)
  or cards (Videos, Audio, Documents, Whiteboards): a visually hidden label, a
  search input with an integrated icon submit button, and one disclosure button
  for advanced filters. `ai_learn_explorer_search_variant` controls presentation
  only: `pill-overflow` (default) uses a pill search field plus circular vertical
  overflow button, while `classic` retains the rounded-rectangle field plus
  chevron disclosure. One `ai-topic-explorer` may override the global default with
  `:search-variant:` or `:search_variant:`; conflicting aliases fail the build.
  Both variants use the same controller, filter panel, URL state, keyboard
  semantics, and accessible labels; do not fork search behavior by
  variant or modality. Typing in the query field filters immediately and updates
  URL state; submit/Enter remains available as an equivalent accessible action.
  IME composition is allowed to finish before filtering. Advanced category,
  timeframe, sort, direction, display-size, and reset controls stay collapsed
  unless explicitly opened or restored from non-default URL state. Every catalog
  explorer is materialized into deterministic 12-record static shards; the first
  page is ``index.rst`` and overflow pages are derived ``page-N.rst`` files owned
  by the same canonical index JSON. The browser may progressively fetch these
  same-origin shards for the allowlisted ``12/25/50/75/100/125/150`` display
  presets, and every explicit **Load 12 more** action advances by exactly one
  shard. Automatic loading is bounded by the selected preset and never scans
  unbounded pages just to satisfy a rare client-side filter. The chosen preset is
  shareable URL state (``?limit=...``); incremental Load More state is deliberately
  ephemeral. No-JavaScript users retain real previous/next shard links. Result
  presentation is an adapter concern; query/filter/sort/disclosure/paging/URL
  behavior has one controller and one template shell. Do not reintroduce
  modality-specific search or pagination forms.
- UI motion added to cards must respect `prefers-reduced-motion`.

### Shared generation context chooser

Step 1 context selection is one shared component: `learn/generation-context.html`
plus `generation-context.js`. Every creation studio uses the same four visible
context types: **Topic**, **Source**, **URL**, and **Prompt**. Tabs switch the
inspector; the tab itself is not a selected context. The summary and tab badges
remain the authoritative visible audit of what is active.

The component has two explicit policies because visual parity must not disguise
different provider contracts:

- `composable` — Topic, Source, Open Problem, Skill, and Topic Prompt creation may
  combine zero or more Learn Topics, zero or more canonical Sources, an optional
  URL, and Prompt/notes. Topic/Source rows are checkboxes. Multi-selection is
  available but not globally required. Source creation is the exception: the
  user must supply a real source URL plus Source notes/excerpt; custom validation
  owns that requirement so a required field on a hidden tab never traps browser
  focus.
- `exclusive` — Video, Audio, Document, and Whiteboard generation keep exactly
  one active context type because their current provider requests encode one
  `mode`. Topic/Source rows are radio buttons. Activating another context type
  changes which branch is sent; selecting a Source clears a selected Topic and
  vice versa. Do not introduce context arrays into those provider requests until
  the provider contract explicitly supports them.

Primary grounding deliberately stays narrow: Learn Topics and canonical Sources
are selectable catalog context. Open Problems, Skills, Topic Prompts, Videos,
Audio, Documents, and Whiteboards are outputs/workflows/related artifacts, not
additional first-class context tabs. This prevents a visually convenient chooser
from creating unclear provenance semantics.

`window.AI_LEARN_CONTEXT_API` is the sole browser state API for this component.
Studio controllers consume `snapshot()`, `restore()`, `activeType()`,
`selectedIds()`/`singleId()`, `url()`, and `prompt()` rather than owning duplicate
context DOM logic. The component is responsive, uses proper tab/tabpanel ARIA
relationships, and reports selection counts/active context in a live status.

These are source/template contracts. Do not patch generated HTML to alter them.

## Development checks

Core tests do not require Sphinx:

```bash
PYTHONPATH=docs/source pytest -q \
  docs/source/_sphinx_ext/_sphinx_ai_learn/tests \
  --ignore=docs/source/_sphinx_ext/_sphinx_ai_learn/tests/_integration
```

A clean-room check should copy only `learn-ai/**/*.json`, call `materialize()`,
compare every generated RST byte with the checkout, then call `materialize()` a
second time and require zero changes.

The integration suite requires Sphinx plus the documentation theme/extensions
from the docs environment. It verifies real source discovery, warnings, routes,
and HTML rendering.

## Multi-select AI lens visibility

Audience, Purpose, Supporting skills, and Role lenses are one shared UI contract.
The canonical template is `learn/ai-lens-profile.html`; record creation, page-level
AI Overview, and in-place section generation must include that component rather
than maintaining separate checkbox grids.

Each group exposes a compact live selection summary (`N selected` plus up to two
visible labels and `+N more`). Audience and Purpose are marked required and show
an attention state when empty. A profile-level summary reports the total number
of selections across active lens groups. The visible preview stays compact on
small screens while the live region's accessible label contains the complete
selected-label list.

`lens-selection.js` is registered after the generation controllers so its first
pass reflects browser-restored draft state. User checkbox changes then update the
group and profile summaries live.

All creation studios now share the same numbered flow contract:

1. **Choose context**
2. **Shape the draft** for Topic/Source/Open Problem/Skill/Topic Prompt creation,
   or **Shape the explanation** for Video/Audio/Document/Whiteboard generation
3. **Choose AI lenses**
4. **Output settings** on media studios only

The four media studios reuse `learn/generation-lenses.html`, which wraps the same
`ai-lens-profile.html` component used elsewhere. Audience and Purpose remain
required groups; Supporting skills and Role lenses may be empty. Media draft
state persists the selected lens profile, and request construction deterministically
compiles that profile into the existing bounded instructions/prompt field. Do not
add ad-hoc `lenses` keys to provider request contracts merely for UI symmetry.
Video instructions remain capped at the server's 4,000-character contract limit;
free-form guidance is trimmed before the deterministic lens block when necessary.

The shaping preset buttons remain additive guidance rather than selections. They
must not be counted as lens selections.

## Unified generation studio architecture

Creation and nested AI generation now share one interaction vocabulary without
pretending that every provider has the same execution contract.

`STUDIO_DEFINITIONS` in `_pages.py` is the renderer-owned registry for all nine
creation studios, in this visible order: Topics, Sources, Open Problems,
Whiteboards, Videos, Audio, Documents, Skills, Topic Prompts. The registry owns
both the shared studio navigation model and generation-route lookup. Do not add a
second route/label list to templates or JavaScript.

All nine `new.html` studios include `generation-studio-header.html`, which owns
the AI Learn Studio hero and the shared `generation-studio-nav.html`. The nav is
a grid rather than a hard-coded row count: five columns on wide studio shells,
three at the medium container breakpoint, two on phone-width containers, and one
only on very narrow containers. Labels may wrap inside their cell; horizontal
scrolling is not the navigation contract.

Generation authority is one shared Assistant contract. `generation-authority.html`
and the global authority manager in `generation-ui.js` own model configuration,
quick model switching, effort/readout state, and synchronization. Video no longer
has a private model picker, and Audio/Document do not read `AI_ASSISTANT_MODEL_API`
directly just to reconstruct model provenance. Studio controllers consume
`assistantModelSnapshot()` instead. One model API subscription updates every
visible authority picker, including hidden nested generation panels; do not create
one subscription per section.

Page-level AI Overview and in-place section generation use the same embedded
Generation authority, lens helpers, and **text-generation workflow kernel**.
`text-generation-ui.js::createWorkflow()` owns the mechanics that must not drift:
profile validation, canonical request normalization, request-copy behavior,
abort/cancel handling, bounded runtime execution, serialized run/publication busy
states, request-time provenance snapshots, reviewed publication transport, receipt
links, and finally-state cleanup. Text-generation token budgets are strict integers
from 256 through 32000; malformed or oversized values fail before network I/O.
Runtime fetches omit credentials, disable cache, and reject redirects. The canonical
request returned for **Copy request** is the same object sent by `runRequest()`, and
generation adapters receive the exact request-start profile/context snapshot rather
than rereading live controls after validation. `overview-generation.js` and
`section-generation.js` remain domain adapters: they own their context/message,
token budget, draft contract, revision-scoped local persistence, and publication
target. Both local-draft paths use compare-and-save semantics: a request-start
storage snapshot may replace only the same browser value, and discard likewise
fails closed if another tab changed the draft. Storage/quota failures never replace
the previously reviewed in-memory draft.
Their common five-button lifecycle bar is rendered by
`text-generation-actions.html`; Overview-only result handoff actions remain in the
Overview template. Their visible private-generation lifecycle is rendered by
`generation-private-flow.html` with the shared vocabulary **Choose context →
Generate draft → Review → Handoff**.
`setFlowStage()` is the single lifecycle state helper. Workflow/skill/agent
metadata remains separate request metadata; it must not masquerade as a second
model-authority picker.

Action wording follows one operation vocabulary:

- page overview: **Generate AI Overview** / **Regenerate AI Overview**;
- in-place section generation: **Generate Now** / **Regenerate Now**;
- creation studios: **Generate Now**, **Save draft**, **Copy request**.

Provider/runtime ownership stays explicit. Video, Audio, Document, and Whiteboard
controllers remain thin modality adapters for endpoint/capability discovery,
output settings, provider request shape, artifact rendering, and lifecycle. Do
not merge those contracts into a universal request controller merely to reduce
line count. Shared UI state belongs in shared primitives; provider semantics stay
in provider-specific adapters.

`generation-ui.js` owns the common browser transport for those adapters. Runtime
requests are byte-bounded and timeout-bounded, omit browser credentials/referrers,
disable caching, reject redirects, require HTTPS except for loopback development,
and verify successful binary artifact MIME types before creating object URLs.
Private runtime-library persistence stores an allow-listed display-only receipt
schema; bearer generation/artifact capabilities and unknown authority fields are
never persisted by the shared library. If browser storage is unavailable, safe
receipt summaries remain available for the current tab only. Page disposal is
BFCache-aware: `pagehide.persisted` freezes keep listeners and object URLs alive,
while a true disposal releases subscriptions, timers, and generated URLs.

These changes are presentation/runtime orchestration only. Canonical Learn JSON
and materialized RST must remain byte-stable under this refactor.
