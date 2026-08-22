# LLM Documentation Standards and Reference Baseline

Observed/reviewed: **2026-08-20**.

## Primary architecture reference

NVIDIA `sphinx-llm`:

- https://github.com/NVIDIA/sphinx-llm
- pinned maintenance baseline: `2a971d7da6a5d7df81f7bff3612ee1822a060c17`

At the pinned/public baseline it performs an additional Sphinx Markdown build,
merges Markdown with HTML output, handles `html`/`dirhtml` suffix modes, supports
exclusions, optional `llms-full.txt`, author/generated page descriptions, and
optional build-time LLM summaries with credential/cache safeguards.

## Curation reference

`sphinx-llms-txt`:

- https://github.com/jdillard/sphinx-llms-txt
- pinned maintenance reference: `9d0660ba71c3c5dfe3023ebc2d281ddcb3070241`

Useful downstream ideas include global/page/block exclusions, path handling,
source-code include/exclude globs, URI templates, deterministic collection, and
explicit full-output size policies.

## `llms.txt` proposal baseline

Current proposal/reference reviewed for this package:

- https://llmstxt.org/

The site identifies the current document as **The /llms.txt file, v2**. Our
maintenance policy treats `llms.txt` as the small standards-facing navigation
artifact and per-page Markdown as the primary fetchable detail representation.

## `llms-full.txt` policy

NVIDIA and other ecosystem tooling continue to support `llms-full.txt`, but our
architecture does not require it for correctness. It is an optional convenience/
compatibility artifact with explicit size policy. Canonical per-page Markdown +
a useful `llms.txt` must remain usable when full-file generation is disabled or
skipped.

## Standards-drift rule

Never hard-code a proposal assumption forever. When the `llms.txt` proposal,
Sphinx builder semantics, or NVIDIA behavior changes:

1. pin the new external revision/date;
2. diff behavior, not just prose;
3. update regression fixtures first;
4. record intentional divergence in `HISTORY.md` and `REGISTRY.md`;
5. keep backwards-compatible output only when its maintenance value is explicit.
