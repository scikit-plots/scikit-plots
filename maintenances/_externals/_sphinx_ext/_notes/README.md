# Current developer context

This directory contains **current-state developer notes only**. It is intentionally limited to present architecture and active development contracts.

When a subsystem changes, update its corresponding file in place instead of adding another versioned Markdown report.

## Module guides

- [`AI_ASSISTANT.md`](AI_ASSISTANT.md) — Sphinx AI Assistant, browser runtime, proxy/model services, multimodal routes, resource handling, feedback surfaces, and security boundaries.
- [`AI_LEARN.md`](AI_LEARN.md) — JSON-first AI Learn materialization, page generation, publication, feedback/provenance, and authoring/runtime contracts.
- [`SPHINX_COLLECTION.md`](SPHINX_COLLECTION.md) — shared collection browser, `gallery-grid`, YouTube core/gallery adapters, player directives, and search/status ownership.
- [`SPHINX_FEEDBACK.md`](SPHINX_FEEDBACK.md) — privacy-minimal page feedback contracts, static UI, aggregation, service storage, and provider review.
- [`SPHINX_EXTENSION_STACK.md`](SPHINX_EXTENSION_STACK.md) — local-development versus installed-release extension authority and cross-module import rules.

## Source of truth

The code under `scikitplot/_externals/_sphinx_ext/` is authoritative. These notes summarize current contracts to make a fresh development session faster; they must not override code, schemas, tests, or security checks.

## Documentation rule

Keep root Markdown small and module-oriented:

1. Do not add versioned progress reports or chronological implementation journals here.
2. Update the relevant current-state module file instead.
3. Put detailed operator documentation beside the owning module when it is required by tests or operations.
4. Keep examples/content Markdown under `docs/source/` unchanged unless the content itself is being edited.
5. Prefer present-tense architecture, invariants, entry points, and verification commands over historical explanation.

## Where these notes live

The same six files are kept at the root of the documentation repository and
here, under `maintenances/_externals/_sphinx_ext/_notes/`. The module guides
name paths as they are in this repository, `scikitplot/_externals/_sphinx_ext/`;
in the documentation repository the same tree is at
`docs/source/scikitplot/_externals/_sphinx_ext/`. When a contract changes,
update the guide in both places in the same change.

Each guide ends with a **Library checkout** section. It holds what is true of
the tree as it sits in this repository and is not part of the documentation
repository's copy.

## Which checkout tests what

| What is tested | Library checkout | Documentation checkout |
|---|---|---|
| Extension code, templates, static assets | runs | runs |
| The site's `conf.py`, `Makefile`, `make.bat` | skips, with the reason | runs |
| Canonical `learn-ai/` content | skips, with the reason | runs |
| The publication workflow under `.github/` | skips, with the reason | runs |
| Anything needing `maintenances/` | runs | raises where it is first used |

A skip here is the truthful result, not a gap: there is no site in the library
checkout to test. The cost is that the site-dependent tests run only where
someone runs pytest in the documentation checkout; that repository's one
workflow publishes content and does not run them.

## Maintenance entry points

- `_maintenance_core/` — the family gate (`tools/check_all.py`) and review engine.
- `_sphinx_ai_assistant/`, `_sphinx_youtube_gallery/` — subsystem-owned state,
  trackers and handoffs. `_sphinx_ai_learn`, `_sphinx_feedback` and
  `_sphinx_collection` have no subsystem manifest yet; these notes and their
  skills are their maintenance entry.
- `skills/_externals/_sphinx_ext/` — one skill per subsystem, each routing here.
