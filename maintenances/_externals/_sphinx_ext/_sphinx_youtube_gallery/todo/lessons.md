# YouTube gallery maintenance lessons

- A leaf extension must not import its orchestrator for shared parsing/validation; extract
  provider primitives downward into a dependency-free core instead.
- Family-related does not mean runtime-required. `_pydata_component_list` belongs in the
  Sphinx UI family but is not a dependency of `_sphinx_youtube_gallery`.
- Parent-count path discovery (`parents[2]`) is archive-shape coupling. Discover marked
  runtime roots explicitly so the same gate works in a main repo and standalone tree.
- Historical hashes are provenance, not a mutable cache. New repository scope requires new
  evidence rather than overwriting an older artifact's checksum file.
- Optional Sphinx absence is an unavailable verification layer. Do not convert it into a
  product failure or an inferred pass.
- A centralized gate should validate authority documents, not just their presence. A handoff
  that omits `TRACKER.json` or a skill that bypasses `STATE.json` can silently fork maintenance
  logic even when every runtime test passes.
- Optional test tooling has multiple layers: Sphinx/docutils and Node `jsdom` are independent.
  Record each unavailable layer explicitly and still run lower-level syntax/import evidence.
- Delivered archive hashes belong outside the archive's own mutable state; embedding the final
  hash into files inside that same ZIP creates a self-reference trap.

- Dependency graphs for plugin systems must include dynamically resolved runtime edges; AST
  imports alone would have missed the original gallery/player cycle.
- A dependency edge should explain *why* it exists. Capability IDs make ownership routing
  executable and help prevent consumers from re-implementing shared behavior.
- Source-reference evidence should come from executable syntax, not raw text grep; otherwise a
  comment can make a stale dependency declaration look valid.
- Centralized architecture logic does not require a centralized subsystem registry. Auto-discover
  local manifests so new subsystems can opt in without editing a global list.

- Review metadata is an input boundary. A profile that can embed shell/Python commands turns
  a maintenance convenience into an execution surface; registered deterministic checks keep
  agent-assisted review advisory and auditable.
- Independent review should run before reconciliation. A family-level green summary must not
  let one subsystem or reviewer suppress another subsystem's evidence.
- PR readiness and release readiness are different contracts. Missing optional Sphinx/jsdom
  evidence can block promotion without preventing useful code review.
- Reviewer false positives are reviewer defects. When valid handoff prose says “this file” or
  a mature checkpoint uses a unique filename-stem form, improve normalization instead of
  rewriting authoritative historical state to satisfy a brittle checker.
