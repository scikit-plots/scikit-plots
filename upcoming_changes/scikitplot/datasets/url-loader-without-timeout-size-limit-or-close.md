---
title: "datasets: URL loading has no timeout, no size limit, and never closes the response"
status: open
kind: "reliability"
area: "scikitplot/datasets"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.datasets"
towncrier_type: "fix"
towncrier_fragment: ""
---

# datasets: URL loading has no timeout, no size limit, and never closes the response

## Summary

`datasets/_data_loader.py` calls `urlopen(url=str(path))` and returns `io.BytesIO(response.read())`; `_load_dataset.py` calls `urlopen(DATASET_NAMES_URL)` the same way. There is no timeout, no limit on the bytes read, no integrity check, and the first response is never closed.

## Why it matters

A stalled server blocks the caller forever; a large or hostile response is read whole into memory; an unclosed response leaks a socket per call. The corpus downloaders already have `timeout` and `max_bytes` for the same job.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list urlopen-without-timeout` lists `scikitplot/datasets/_data_loader.py:137` and `scikitplot/datasets/_load_dataset.py:50`.
- `scikitplot/datasets/_data_loader.py` lines 134-137: `response = urlopen(url=str(path))` then `return io.BytesIO(response.read())`, no `with` block.

## Root cause / current understanding

Written before the corpus network policy existed; not migrated.

## Expected behavior

A timeout, a byte limit, an optional expected SHA-256, a context manager, and errors that name the URL; the cache (`get_data_home`) is used before the network.

## Affected paths and ownership

- `scikitplot/datasets/_data_loader.py`
- `scikitplot/datasets/_load_dataset.py`

## Constraints and non-goals

Reuse the corpus download helpers rather than a second implementation.

## Edge cases to cover

- Server that accepts and never answers
- Response larger than the limit
- Offline machine with a cached file
- Redirect to another host

## Proposed direction

Route both calls through the corpus downloader (timeout, max_bytes, retry) or a small shared helper.

## Verification / acceptance criteria

- A test with a local server that never answers returns within the timeout
- A response over the limit raises before reading it all

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.datasets` / `fix`, one bullet, when the change lands.
