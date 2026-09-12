# MCP family and ownership boundaries

`scikitplot.mcp` is an adapter subsystem, not a retrieval engine.

- **MCP owns:** the tool/resource wire shapes, official-SDK registration,
  transport configuration, capability truth, protocol-facing validation,
  read-only client bundles, and MCP-specific orchestration boundaries.
- **Corpus owns:** ingestion, chunking, retrieval semantics, canonical retrieval
  status/capability concepts, and index-neutral search behavior.
- **Annoy owns:** dense/vector backend behavior and native/Cython mechanics.
- **Project CLI owns:** the outer `scikitplot mcp` delegation contract.
- **External MCP SDK owns:** the MCP protocol implementation.

The bridge to Corpus/Annoy is deliberately late-bound. `_corpus_annoy.py` may
wire those systems at call time, but package import must not pull them in.
