# `compat` ownership

Compatibility code lives here when scikit-plots needs behavior around the pinned
NVIDIA baseline without modifying `sphinx_llm/**`.

Current A02 implementation:

- `primary_build_context.py` captures the effective primary Sphinx configuration,
  serializes it to a private short-lived snapshot, and integrity-checks the bytes
  with SHA-256 before child deserialization;
- `_child_config.py` is injected as the **first** Markdown-child user extension,
  restores primary values immediately for later extension setup, then reapplies
  them at `config-inited` priority 1 for Sphinx 5+ lifecycle parity;
- `markdown_generator.py` subclasses the pinned NVIDIA generator, retains its
  build/merge lifecycle, forwards only allow-listed early Sphinx core overrides
  on the process command line, and transfers all other pickleable effective
  configuration through the integrity-checked snapshot.

This is an upstream compatibility shim, not the owner of canonical semantic node
meaning. Directive/node meaning belongs in `adapters/`; curation belongs in
`curation/`; artifact lifecycle belongs in `core/`.

The preserved NVIDIA tree remains byte-identical. A02 cannot close until the per-environment parity runner is GREEN across all 10
required Python/Sphinx cells recorded in the compatibility baseline (or an explicitly
reviewed equivalent matrix). One GREEN environment is not sufficient.
