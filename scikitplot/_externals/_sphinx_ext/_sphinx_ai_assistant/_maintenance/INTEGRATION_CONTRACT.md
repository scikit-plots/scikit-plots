# `_sphinx_ai_assistant` ↔ `_sphinx_llm` Integration Contract

## Dependency direction

```text
_sphinx_ai_assistant -> _sphinx_llm stable facade/static artifacts
_sphinx_llm          -X-> _sphinx_ai_assistant
```

## Consumer selection order

```text
1. canonical static Sphinx Markdown (fidelity=canonical)
2. static post-build HTML compatibility Markdown (fidelity=compatibility)
3. runtime DOM extraction (fidelity=runtime-fallback; last resort)
```

The assistant must make the selected fidelity observable for debugging and must
not silently call Tier 2/3 canonical.

## Preferred machine interface

The exact Python API is deferred to `_sphinx_llm` A11, but the consumer needs
capabilities equivalent to:

```text
resolve_current_page(document/url/build identity)
get_representation_metadata(...)
get_markdown_location(...)
get_llms_manifest(...)
get_capabilities/status(...)
```

The browser may use static `<link>` discovery and/or a published manifest rather
than importing Python directly.

## Prompt packaging rule

```text
SERVER SYSTEM POLICY
  immutable authority

REFERENCE CONTEXT
  canonical/compatibility Markdown
  explicitly untrusted documentation data

USER
  user question/input
```

No page Markdown, retrieved Corpus evidence, or browser-provided text is promoted
into the server's authoritative system role.

## Migration compatibility

The current assistant representation path remains until static coverage is
proved. Migration is staged:

```text
current DOM conversion primary
 -> static primary + DOM fallback
 -> static coverage measured
 -> DOM fallback disabled by default
 -> optional removal only with evidence
```
