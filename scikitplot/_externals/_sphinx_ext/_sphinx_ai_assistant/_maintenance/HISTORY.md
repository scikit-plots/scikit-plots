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
