# R173T2A1 — Conversation artifact provenance naming

Status: COMPLETE

Base: Run 173 T2A (`6096c0b44d21e2c46b4b1a9b70b97c5c001f81475e52cd138456729bcd519396`).

## Purpose

Apply the feedback artifact-identity rule consistently to conversation Save and Global Share across JSON, HTML, TXT, YAML, and TOML. Human-facing filenames identify lifecycle role and format without embedding a Share capability.

## Naming contract

- Local Save: `ai-conversation-local-save-<format>-<timestamp>.<ext>`
- Global Share: `ai-conversation-global-share-<format>.<ext>`
- Share UUID/read capability: never included in a filename
- Internal opaque storage keys remain internal and are not reused as human-facing artifact names.

The conversation artifact list surfaces the same provenance filename next to its format/lifecycle state.

## Global download lane

The fixed viewer now exposes `POST /v1/share/download`. The Share locator remains in a bounded JSON request body, not the URL path. The server re-canonicalizes and renders the stored snapshot, chooses MIME/extension, emits `Content-Disposition: attachment`, `nosniff`, `no-store`, and keeps the sandbox CSP for HTML. The fixed read response also exposes the server-owned filename/MIME so the viewer can label the artifact consistently.

`/v1/share/download` is classified as a public read operation for the narrow optional `Origin: null` read compatibility path; Share creation/update/revoke authority is unchanged.

The proxy remains version 7.4.0 because its versioning contract requires a bump for breaking changes; this endpoint is additive and backwards compatible.

## Verification

- Python Share contract: 34/34
- proxy app owner: 46/46
- registered Node architecture: 140/140
- layout architecture: 9/9
- collection: 2332 / 0 errors
- maintenance checker: GREEN (after metadata update)
