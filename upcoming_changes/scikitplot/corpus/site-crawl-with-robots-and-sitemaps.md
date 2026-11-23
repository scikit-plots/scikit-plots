---
title: "corpus: no polite multi-page crawl (robots.txt, sitemaps)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no polite multi-page crawl (robots.txt, sitemaps)

## Summary

`WebReader` reads one URL. Building a corpus from a documentation site or a public archive needs following links or a sitemap while obeying robots.txt and rate limits.

## Why it matters

Users write ad-hoc crawlers that ignore robots.txt or hammer servers; the SSRF and size policies of the corpus are bypassed.

## Current evidence

- `grep -rEil 'robotparser|sitemap' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A crawl source with a start URL or sitemap, same-site scope by default, robots.txt obeyed, a per-host rate limit, depth and page limits, deduplication by canonical URL, and the existing URL security checks on every page.

## Affected paths and ownership

- `scikitplot/corpus/_sources/`
- `scikitplot/corpus/_readers/_web.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Redirect loops
- Sitemaps of sitemaps
- Pages that differ only by query string
- Disallowed paths
- Huge pages

## Proposed direction

Breadth-first crawl over the existing fetcher; `urllib.robotparser` from the standard library.

## Verification / acceptance criteria

- A local test site with a robots.txt and a sitemap is crawled exactly within its rules

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
