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

The code under `docs/source/scikitplot/_externals/_sphinx_ext/` is authoritative. These notes summarize current contracts to make a fresh development session faster; they must not override code, schemas, tests, or security checks.

## Documentation rule

Keep root Markdown small and module-oriented:

1. Do not add versioned progress reports or chronological implementation journals here.
2. Update the relevant current-state module file instead.
3. Put detailed operator documentation beside the owning module when it is required by tests or operations.
4. Keep examples/content Markdown under `docs/source/` unchanged unless the content itself is being edited.
5. Prefer present-tense architecture, invariants, entry points, and verification commands over historical explanation.
