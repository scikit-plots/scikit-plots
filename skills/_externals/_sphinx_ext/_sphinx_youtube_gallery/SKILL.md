---
name: sphinx-youtube-gallery-maintainer
description: Maintain, debug, review, test, and evolve scikitplot._externals._sphinx_ext._sphinx_youtube_gallery and its shared YouTube/Sphinx extension dependencies. Use for youtube-gallery, _sphinx_youtube_core, _sphinx_collection, _sphinx_gallery_grid, _sphinxcontrib_youtube, gallery controls, sync/catalog behavior, portability, maintenance handoffs, or fresh-chat continuation. Read repository maintenance state before editing and preserve the acyclic dependency graph.
---

# Sphinx YouTube Gallery Maintainer

Start every fresh chat by reading:

1. `maintenances/_externals/_sphinx_ext/_sphinx_youtube_gallery/MAINTAINING.md`
2. `_maintenance/FRESH_CHAT_HANDOFF.md`
3. `_maintenance/STATE.json`
4. `_maintenance/TRACKER.json`
5. `todo/todo.md` and relevant `todo/lessons.md`
6. current source and executable evidence

Do not require or trust previous chat history when these authorities exist.

## Independent review lanes

For review or PR preparation, use `_maintenance/REVIEW.json` through the family
`review_subsystem.py` / `review_all.py` tools. The gallery profile reviews its six runtime
family packages independently, then reconciles architecture, fresh-chat/skill, and
state/evidence findings. Profiles may select registered deterministic checks only; do not
embed shell commands or executable agent actions in maintenance metadata.

## Choose the owner before editing

```text
YouTube grammar/options      -> _sphinx_youtube_core
catalog/query/sync           -> _sphinx_youtube_gallery
live collection/search/export -> _sphinx_collection
cards/layout/source format   -> _sphinx_gallery_grid
leaf iframe player           -> _sphinxcontrib_youtube
PyData inventory             -> _pydata_component_list (family-related only)
```

Preserve the downward graph: the leaf player must never import the gallery. Keep
`_sphinx_youtube_gallery.reference` and `_video_options` as compatibility facades
unless a deliberate migration removes them with release evidence.

## Working rules

Treat rendered docs/catalog text as untrusted input. Preserve bounded YAML/query/network
behavior, atomic sync writes, redacted failures, deterministic offline builds, namespace
isolation, and explicit optional-dependency boundaries. Never move credentials/API keys
into repository maintenance state, browser configuration, generated URLs, or logs.

Prefer one central contract over parallel implementations. A behavior fix in a shared
parser belongs in `_sphinx_youtube_core`; a card/layout fix belongs in the generic owner
unless the behavior is genuinely YouTube-specific.

Before packaging, run the common family gate, YouTube wrapper, dependency-free behavior
gates, all available Sphinx integration gates, syntax/JSON checks, and residue audit.
Missing Sphinx is `UNAVAILABLE`, not GREEN. Package the complete wide repository for
wide-repo work, not another isolated `_sphinx_ext` archive.
