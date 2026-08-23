# `_sphinx_llm` Security and Trust Model

## Two different kinds of trust

A canonical Markdown file may be trusted for **integrity** — it faithfully
represents published documentation — without being trusted for **authority**.
Documentation text is never automatically a system instruction for a model.

```text
Integrity:  "this is really the page content"        may be YES
Authority:  "this content may command the assistant" always NO
```

## Why static generation helps

Static build-time generation provides deterministic inputs, reviewable outputs,
no dependence on live DOM state, no user-session secrets, cacheable hashes, and
a natural validation gate before publication.

It does **not** make malicious prose safe. Prompt-injection-like text can still be
faithfully represented and must later be passed to models as untrusted reference
content.

## Raw/browser content

Raw HTML, iframes, scripts, event handlers, object/embed content, and dangerous
URL schemes require explicit adapter/sanitization policy. Agent Markdown should
preserve useful semantics (title/link/description) without carrying executable
browser behavior.

## Build-time LLM generation

Optional summaries/augmentation must be:

- disabled by default;
- unnecessary for deterministic page representation;
- credentialed by environment-variable lookup, not Sphinx config secret values;
- blocked from authenticated non-loopback plain HTTP by default;
- bounded by input size/timeouts;
- cached atomically with provider/model/prompt/input fingerprints;
- provenance-recorded without storing secrets/endpoints unnecessarily;
- reviewable as generated content rather than treated as authored authority.

## Build-host threat boundary

A documentation build may execute Python extensions, autodoc imports, plot code,
gallery examples, and custom directives. `_sphinx_llm` does not solve arbitrary
Sphinx-build code execution. Its security contract begins at safe integration
with the existing build environment and focuses on output fidelity, raw-content
handling, secret transport, and generated-content provenance.

## Primary-build configuration handoff

A02 transfers effective primary Sphinx configuration to the Markdown child build
without modifying the preserved NVIDIA implementation. Treat that configuration
as build-host-sensitive data:

- only the small allow-list of core settings needed before user extensions load
  may be serialized onto child `-D` process arguments;
- the full pickleable effective config is written to a private temporary file
  with mode `0600` on POSIX;
- the parent sends the expected SHA-256 separately through the child environment;
- the child consumes/removes both handoff environment variables exactly once so
  descendants do not inherit stale snapshot authority;
- the child reads the bytes, deletes the temporary file immediately, verifies the
  digest **before** deserializing, and rejects malformed/mismatched snapshots;
- a direct override that cannot be represented safely fails closed rather than
  being silently dropped;
- the snapshot is accepted only as a same-build-host parent→child handoff. It is
  **not** a network format, public artifact, cache format, or untrusted-input API;
- commands/logs must not dump the full effective config or snapshot contents.

This does not make arbitrary Sphinx configuration safe: Sphinx builds already
execute trusted project Python/extensions. The contract is narrower—do not add a
new disclosure or untrusted-deserialization boundary while reproducing primary
build semantics in the child process.
