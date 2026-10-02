# AI Learn

## Scope

Primary package:

`docs/source/scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/`

Canonical content lives under `docs/source/learn-ai/`.

AI Learn is JSON-first. The authoritative content path is:

`canonical JSON -> deterministic materializer -> derived RST -> Sphinx -> HTML`

The build-time materializer is intentionally local and deterministic. It performs no model calls, provider requests, Git operations, telemetry, or publication.

## Core modules

- `_schema.py` — validates public/canonical content contracts and local media references.
- `_materialize.py` — loads canonical trees and deterministically emits RST.
- `_pages.py` — Sphinx directives and page/explorer/generation presentation.
- `_generation.py` — generation identifiers, active-generation projection, feedback statistics, and scoped invalidation helpers.
- `_publication.py` — validates publication operations and projects canonical JSON changes.
- `_publication_cli.py` / `_publication_request_cli.py` — reviewed publication command boundaries.
- `_registry.py` — shared interaction/target registries.
- `_sphinx.py` — Sphinx lifecycle integration.
- `_static/` — browser generation, feedback, context, explorer, and media UI.

## Canonical-content rule

JSON is the source artifact. RST is derived output. HTML is Sphinx output.

Do not hand-edit derived RST to represent a canonical content change. Update the corresponding JSON and run the materializer. The materializer must be idempotent: a second pass over unchanged JSON should produce no file changes.

## Content areas

The current system supports the Learn areas used by the site, including:

- topics;
- sources;
- open problems;
- videos;
- audios;
- documents;
- whiteboards;
- topic prompts;
- skills.

Index/detail/new-authoring pages are generated from validated definitions and records rather than independent hand-maintained HTML.

## Page composition

Include-based composition is the normal nested-content path. Toctree composition is available where an index intentionally owns child navigation.

Included generated RST must follow the Sphinx orphan/include contract so pages used only through `include` do not produce avoidable orphan warnings.

## Shared explorer contract

Index explorer pages share one masthead/search/results model. Page-specific JSON provides current copy such as kicker/title/create label, while reusable directives own layout and interaction.

Search presentation can be `pill-overflow` or `classic` where supported, but the page data should not duplicate browser-control implementation.

Catalog/media indexes use deterministic 12-record presentation shards. `index.rst` owns offset 0; the materializer derives owned `page-N.rst` overflow pages from the same canonical index JSON. The shared filter panel exposes display presets `12/25/50/75/100/125/150`, while the end-of-results control advances by exactly 12 records per explicit press. Keep selector, progressive loader, no-JS next/previous links, table/grid adapters, and URL `limit` validation in the shared explorer contract rather than modality-specific code. Prompt/Skill and browser-local library grids reuse the same 12-item visible-window UX; their underlying canonical/local-storage semantics remain separate.

Secondary sidebar behavior is a canonical page-level choice, not a template accident. Keep that policy centralized when adding a new page type.

## Generation studio

Media and text authoring pages share a common generation structure:

1. choose context;
2. shape the requested output/draft;
3. select output/lens options;
4. expose advanced options when needed;
5. show generation authority/runtime status;
6. provide generate/save/copy actions;
7. show the private runtime/user library when applicable.

The selected Assistant model represents requested provenance. The active modality runtime owns actual generation/rendering/publishing and must advertise the needed capability before `Generate` is enabled.

## Publication authority

Publication is a security boundary, separate from deterministic materialization.

Current invariants:

- publication accepts bounded validated operations, not arbitrary filesystem mutation;
- projected writes are confined to canonical JSON/sidecar paths;
- public contributor credit is validated separately from private provider credentials;
- generation provenance and reviewed feedback are explicit canonical data;
- browser-authored values are not trusted as filesystem paths, Git commands, or repository authority;
- provider/Git transport is outside the pure materializer;
- publication should be replay-safe and deterministic for the same accepted operation/state.

Publication planning validates the projected content tree before committing writes. Do not bypass that projection/validation layer for convenience.

## Generation history and feedback

Sections may contain accepted generation records and reviewed feedback. Active-generation projection and feedback aggregation are deterministic helpers, not UI-only calculations.

Feedback sidecars are canonical scoped data. Changes should invalidate only the documents that depend on the affected feedback/content when possible.

Quick-feedback count placement is presentation-only and is configured with
`ai_learn_buttons_ratings` (`left_button_rating` for thumbs-down,
`right_button_rating` for thumbs-up). Keep `_sphinx.py` validation, `_pages.py`
HTML-time config injection, `generation-feedback.html` DOM order, and
`topic.css` logical dividers synchronized. Do not persist this UI preference in
canonical generation/feedback JSON or publication requests.

Do not derive participant identity from network address, browser history, or a stable tracking identifier merely to support feedback counts.

## Sphinx lifecycle

At Sphinx startup the extension materializes canonical JSON before normal document reading. Derived assets/directives then participate in the normal Sphinx build.

If a change affects canonical schema, materialized RST shape, or static behavior, update the corresponding environment/build revision so incremental builds cannot silently retain an incompatible prior representation.

## Files to inspect first

For a new AI Learn change:

- `_schema.py` for canonical validation;
- `_materialize.py` for JSON-to-RST output;
- `_pages.py` for directive/template presentation;
- `_publication.py` for mutable/reviewed publication behavior;
- `_generation.py` for generation/feedback projections;
- `_static/*.js` for browser generation workflows;
- `_templates/learn/` for HTML templates;
- `tests/test__schema.py`, `test__materialize.py`, `test__pages.py`, and `test__publication.py` for contracts.

## Safe editing rules

1. Preserve JSON as canonical source.
2. Keep the materializer deterministic and network-free.
3. Separate preview/generation from publication authority.
4. Validate and confine every publication path.
5. Prefer shared generation/explorer primitives over modality-specific copies.
6. Preserve accessibility, light/dark theme behavior, and responsive layouts when changing UI.
7. Run idempotence and clean-tree checks after materializer changes.
8. Keep this file present-tense and focused on the active contract.
