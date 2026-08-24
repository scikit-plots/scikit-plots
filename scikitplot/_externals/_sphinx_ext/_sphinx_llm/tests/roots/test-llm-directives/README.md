# `tests/roots/test-llm-directives`

A03/A04/A05/A17 semantic compatibility fixture. It intentionally contains:

- native semantic content;
- dropdown and tab containers whose labels and complete bodies must survive;
- a video node represented from provider/title/URL metadata, never iframe HTML;
- an unregistered structural wrapper that preserves children but remains reported;
- an unregistered semantic leaf that warns in discovery mode and fails strict mode;
- executable raw HTML whose visible text may survive but script content must not;
- an explicit `llms-ignore` block that must be absent from machine output.

The default fixture uses `llms_txt_unknown_node_policy = "warn"` so A03 can
inventory discoveries. A strict A04/A13 run overrides it to `error` and is
expected to fail until every unregistered semantic leaf has an explicit adapter.
