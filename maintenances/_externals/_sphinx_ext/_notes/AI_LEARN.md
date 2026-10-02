# AI Learn

## Scope

Primary package:

`scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/`

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

## Library checkout

Canonical content (`docs/source/learn-ai/`), the site `conf.py` and the
publication workflow exist only in the documentation repository. Tests that
read them call `tests/_learn_site.py` (`content_root()`, `content_tree()`,
`docs_source()`, `site_repository()`), which returns the path or skips with
the reason. Tests of the schema, materializer logic on synthetic trees,
templates and static assets run in both checkouts.

Do not load canonical content at module import in a test. A module-level
`load_content_tree(...)` fails at collection where there is no content, and a
collection error stops the whole run.

Link targets taken from page data in `_static/ai-learn.js` and
`_static/topic.js` pass `safeHref` (http and https only); same-page targets are
built with `encodeURIComponent`.

A detail page that no index page owns is written with `:orphan:`. A record is
owned by the explorer or media gallery of its kind, a prompt by the prompt
library, a skill by the skill library; where that index page is not defined,
Sphinx would otherwise report the generated page as outside every toctree and
fail a `-W` build. Owned pages are unchanged.

`tests/_integration/test_sphinx_build.py` builds a throwaway Sphinx project
and needs `sphinx-design`. Of the themes it parametrizes over, `alabaster`
ships with Sphinx and always runs; a third-party theme (`pydata_sphinx_theme`,
`furo`) runs where it is installed and skips, naming the theme, where Sphinx
cannot load it. The project declares `pydata-sphinx-theme` and does not declare
`furo`, so a default CI install runs two themes and skips one. Adding a theme
to the parameter list does not require adding it to `pyproject.toml`. A fixture
that tests an explorer writes that explorer's page definition; records alone
produce detail pages and no explorer.
