# YTG-M001 — family maintenance bootstrap and provider-core extraction

## Goal

Give `_sphinx_youtube_gallery` the same fresh-chat maintenance discipline as the AI
assistant while centralizing only rules that genuinely apply across the Sphinx
extension family.

## Runtime change

The review found a real cycle: `_sphinx_youtube_gallery -> _sphinxcontrib_youtube`
while `_sphinxcontrib_youtube.utils -> _sphinx_youtube_gallery` for URL parsing and
option validation. Shared provider primitives are therefore extracted to
`_sphinx_youtube_core`; both consumers point downward to that core. The old gallery
modules remain compatibility facades.

## Rollback

This checkpoint includes a runtime architecture change. A rollback must restore the
prior gallery-owned `reference.py` / `_video_options.py`, reverse the imports in the
gallery and leaf player, remove `_sphinx_youtube_core`, and then rerun dependency and
behavior gates. Deleting maintenance files alone is not a complete rollback.

## Evidence boundary

Existing standalone-v10 validation JSON/SHA artifacts remain historical evidence;
current wide-repository results are recorded in `STATE.json` and this checkpoint.

## Current wide-repository evidence — 2026-09-10

The available local layers are green:

- common maintenance-core unit tests: **9/9 passed**;
- family gate: **GREEN** for `_sphinx_ai_assistant` and `_sphinx_youtube_gallery`;
- AI-specific composed maintenance gate: **GREEN (repository)**;
- YouTube-specific maintenance wrapper and core/facade identity: **GREEN**;
- canonical dependency-free doctests: **67/67 passed**;
- capability, sync pipeline, static parity, and strict latest-source controls: **GREEN**;
- dependency-free documentation: **14 public docstrings passed**;
- canonical core import smoke: **2 namespace layouts passed**;
- JavaScript syntax: **3/3 control scripts passed**.

The Sphinx integration layer is intentionally recorded as unavailable because this
workspace lacks `sphinx`, `docutils`, `sphinx_design`, and `myst_parser`. The
jsdom-backed browser controls are likewise unavailable because `jsdom` is absent.
Those are downstream verification requirements, not inferred passes.

The four historical standalone validation/SHA JSON files checked during this
checkpoint remain byte-for-byte unchanged from the user-wide source archive.
