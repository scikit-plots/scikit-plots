# `_sphinx_ai_assistant` Maintenance History

## 2026-08-20 — maintenance normalization package prepared

- Introduced Corpus/MCP-style live trackers, state, registry, checkpoints,
  schemas, and verification semantics for the multi-runtime assistant.
- Established sibling `_sphinx_llm` as the future owner of canonical Sphinx
  Markdown/`llms.txt`/`llms-full.txt`/directive representation.
- Kept current source behavior unchanged; runtime HTML conversion remains legacy
  current behavior until staged migration gates close.
- Recorded conservative security contract states instead of treating existing
  desired-policy prose as proof.

## 2026-08-22 — capability pivot: the assistant owns its representation

- `_sphinx_llm` frozen. It is not required by any assistant surface: the
  assistant already writes `page.md` and `llms.txt` itself on `build-finished`.
- Governing rule rewritten. It previously ended *"canonical representation is
  consumed from `_sphinx_llm`"*, which pointed the one dispute-resolving
  sentence at frozen code.
- B01 inverted (define -> sever), B08 inverted (consume -> produce), A12
  withdrawn.
- Reverse-dependency rule corrected: it scanned raw text, so a module docstring
  naming the assistant in an architecture diagram was reported as a dependency.
  It now detects imports. The gate was red for prose.
- Representation contract recorded explicitly: canonical is the build-time
  artifact, convenience is the browser conversion, and the distinction is not
  about quality — an external agent cannot fetch a `blob:` URL.

## 2026-08-22 — maintenance reconciled to the pivot

- `DEPENDENCY_MAP.md` rewritten: two live members, one live edge, and the false
  "Identical in all three `_maintenance/` folders" claim removed — the copies
  were never identical and each is now marked with its owner.
- `INTEGRATION_CONTRACT.md` re-titled and rewritten: the contract is no longer
  with `_sphinx_llm` but between this extension's own build and browser layers,
  plus the HTTP edge to the backend.
- `RULESET.md` 16-20 replaced: `_sphinx_llm` integration -> representation
  ownership.
- `REGISTRY.md`: `AIA-002` (P0) and `AIA-013` (P1) **WITHDRAWN**. Both existed to
  migrate canonical ownership away from the assistant; the pivot makes the
  assistant's own generation the intended design rather than a debt.
- `TRACKER_LOGICAL.md`: `AIA-C04` `RepresentationConsumer` (PLANNED) ->
  `RepresentationProducer` (ACTIVE).
- `MAINTAINING.md`: anchor and physical scope remeasured; ownership section
  inverted; representation contract stated.

## 2026-08-22 — capability increments 1-5 landed

Five bundles applied and verified against a pristine re-extraction. No new
runtime dependency; `__init__.py` still has zero module-scope non-stdlib imports.

- **Copy mode toggle.** `browser` (convenience, default) or `static` (canonical).
  A failed static fetch names the alternative rather than silently substituting
  the browser conversion.
- **`llms.txt`** now follows the llmstxt.org layout. Titles come from each page's
  own heading and descriptions from its first prose paragraph, so the index
  improves as the docs do.
- **Directive fidelity.** 15 rules in one table, declared in Python and
  serialised to the browser, so a rule cannot exist on one side only.
- **Root-cause fix.** `strong_em_symbol` was `"**"`; markdownify doubles it for
  `<strong>`, so every bold run in the documentation converted to `****text****`
  — 209 occurrences across 18 published pages, in every `page.md` and everything
  sent to an AI provider. Now 0.
- **Video links.** Embed URLs mapped to watch URLs using the templates
  `_sphinxcontrib_youtube` already uses for its own epub and latex output. Vimeo
  deliberately unchanged: its `_platform_url` *is* the player URL, so no watch
  form is invented.

Not proven from a source snapshot, and recorded as such: that COPY(static) and
ASK AI resolve to byte-identical content, and that no surface degrades with
`_sphinx_llm` absent. Both need a live build.
