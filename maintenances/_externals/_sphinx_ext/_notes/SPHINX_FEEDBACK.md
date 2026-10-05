# Sphinx feedback

## Scope

Primary package:

`docs/source/scikitplot/_externals/_sphinx_ext/_sphinx_feedback/`

The extension provides privacy-minimal page feedback for Sphinx/static HTML. Its dependency-free request/event helpers can be imported without Sphinx; Sphinx integration is loaded lazily.

## Core modules

- `_contracts.py` — request validation, canonical event bytes, request hashes, conflict/validation errors.
- `_config.py` — extension/service configuration helpers.
- `_aggregate.py` — aggregate/count projection.
- `_sphinx.py` — Sphinx directive, configuration, assets, and page-context integration.
- `_service/` — standalone service core plus SQLite/GitHub provider implementations.
- `_static/sphinx-feedback.js` / `.css` — reader UI.
- `_static/page-feedback-aggregate.json` — packaged aggregate snapshot fallback when configured.

## Privacy model

A feedback event represents a **page reaction, never a person**.

Core expectations:

- no network request occurs merely because a page is viewed;
- submission happens only after a reader acts;
- do not build stable user identity from IP address, cookies, browser history, or unrelated Assistant state;
- contributor credit is explicit optional content, not inferred identity;
- transport/storage credentials remain server-side;
- browser-visible configuration is treated as public.

## Reader UI

The Sphinx adapter supports quick and detailed feedback surfaces, optional comments/contributor fields, and optional counters. Placement can target sidebar/main content with fallback selectors.

Important configuration lives in `_sphinx.py`, including:

- `feedback_page_enabled`;
- `feedback_position` and fallback placement;
- quick/detailed/comment/contributor switches;
- `feedback_buttons_ratings` per-button quick-count placement;
- counter enable/source settings;
- `feedback_endpoint`;
- `feedback_site_id`;
- page revision and aggregate-file configuration;
- include/exclude and selector settings.


Quick-button reviewed counts are presentation-configurable without changing the
feedback event contract. `feedback_buttons_ratings` accepts the distinct keys
`left_button_rating` (thumbs-down) and `right_button_rating` (thumbs-up), each
with `"left"` or `"right"`. The site default is the balanced outer-edge layout
`[0 | 👎] [👍 | 0]`; selected labels stay attached to their thumb cluster.
Configuration is validated in `_config.py`, serialized by `_sphinx.py`, applied
by `sphinx-feedback.js`, and divided with logical-side CSS in
`sphinx-feedback.css`. Keep those four layers in sync when extending this UI.

Do not couple this generic page-feedback UI to AI Assistant answer ratings. They are separate product surfaces even when they share a backend deployment.

## Contract and idempotency

Requests are parsed and canonicalized before persistence. Hashes/ids are used to detect retries/conflicts so a repeated submission does not silently create an unrelated event.

Server code should distinguish:

- a valid retry of the same logical request;
- a conflicting mutation that reuses an identifier with different content;
- provider/storage failure;
- invalid/unbounded input.

## Aggregate snapshots

Counter data can come from embedded/packaged aggregate snapshots or configured runtime sources. A snapshot marked complete is authoritative for the included scope; incomplete data must not be presented as a complete zero.

Zero is a real value only when the selected authoritative snapshot/source says the count is zero. Missing/unavailable state should remain distinguishable from zero.

## Service/provider boundary

The service layer is deliberately small and does not require a specific web framework. Provider implementations own provider-specific persistence/review behavior while the contract/core remains provider-neutral.

GitHub/provider tokens are server authority. Never serialize them into Sphinx config, static assets, feedback events, or public aggregate files.

## Files to inspect first

- `__init__.py` — dependency-free public API and version.
- `_contracts.py` — canonical request/event semantics.
- `_aggregate.py` — counter/aggregate projection.
- `_sphinx.py` — Sphinx configuration and rendering.
- `_service/_core.py` — service behavior.
- `_service/_sqlite.py` / `_github.py` — storage/provider adapters.
- `_static/sphinx-feedback.js` — browser behavior.
- `tests/test_contracts.py`, `test_service.py`, `test_github_provider.py`, `test_static.py` — required invariants.

## Safe editing rules

1. Keep page feedback independent from Assistant rating/contribution flows.
2. Preserve explicit reader action before transport.
3. Avoid stable personal tracking identifiers.
4. Treat missing aggregate data differently from a verified zero.
5. Keep provider credentials out of static/browser data.
6. Preserve canonical hashing/retry semantics across transports.
7. Keep this file present-tense and focused on the active contract.
