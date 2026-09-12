# R173T32 — Session-field parity across export formats

Status: COMPLETE

Base: R173T31 local save provenance, same package.

## Verified first: R173T31 landed

The new exports carry what the previous set did not — `session.id`,
`page_url`, `exported_at`, `exported_at_iso`, per-record `ts`/`ts_iso`, and
model attribution on every assistant record. The six remaining `model_id` nulls
are user records, which correctly have no model.

## Found by diffing five formats of one conversation

| field | json | html | txt | yaml | toml |
|---|---|---|---|---|---|
| `session.id` | yes | yes | **no** | yes | yes |

The text export rendered every session field the snapshot carried **except**
`id`. The snapshot has already had the reader's review applied, so every format
is rendering the same decisions; a format that drops a field the reader chose
to include is deciding for them, and invisibly — the same export in another
format carried it.

Every other session field in that header follows an if-present rule. This one
was simply missing from the list, and now follows it too.

## Two of my own findings were wrong, and checking is what caught them

The first pass reported the text export as also missing the resources manifest
and model attribution. Both were **probe errors**: `totalBytes` and
`model_id` are JSON field names, and a human-readable format has no reason to
use either. The text export carries a resources summary line with per-item
detail, and model name plus provider on every assistant header.

Grepping one format for another format's field names measures the schema, not
the content. Only the `session.id` gap survived being checked.

## Verification

- browser wrapper gate: **147/147** (80 assertions in the owning harness);
- architecture gates: **482/482**, one new mutant caught:
  `text-export-drops-a-reviewed-session-field`.

The parity assertions are written per field, so a future session field added to
the snapshot and forgotten in the text header fails here rather than being
noticed in a diff months later.

## Still open

The export carries **no activity timeline** in any format. Persisted since
R173T15 and restored on reload since R173T17, it is never emitted by
`_buildExportRecords`, so a saved transcript has no account of how each answer
was produced. That needs a `schema_version` bump and a share-policy decision —
is activity publishable, or local-only like the session id? — so it remains its
own checkpoint rather than being folded in here.
