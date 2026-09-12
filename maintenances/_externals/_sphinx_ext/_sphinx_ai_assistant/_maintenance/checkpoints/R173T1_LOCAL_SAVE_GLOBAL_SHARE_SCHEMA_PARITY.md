# R173T1 — Local Save + Global Share schema/format parity

Date: 2026-09-07

## Scope

First Run 173 improvement slice after the verified Run 172 closure:

- local conversation Save across JSON, HTML, TXT, YAML, TOML;
- Global Share across the same five formats;
- repair the `snapshot.schema_version must be '2.0'` break;
- preserve the current 2.1 resource/provenance manifest across cloud canonicalization;
- improve Save/Share UI wording and destination ownership without introducing a second serializer path.

Feedback-export work is intentionally deferred to the next checkpoint so schema transport and feedback-control-plane work do not become one migration.

## Root cause

The browser canonical conversation snapshot had advanced to schema `2.1`, including per-question resource metadata, while the Hugging Face/Python Share contract and Cloudflare Worker still enforced schema `2.0` and rebuilt the older shape.

Changing the server validator from `2.0` to `2.1` alone would have made Global links save but could still silently discard 2.1 resource provenance. The repair therefore treats 2.1 as the canonical current output and 2.0 only as an accepted migration input.

## Contract

```text
accepted input       2.0, 2.1
canonical output     2.1
unknown versions     reject
formats              json, html, txt, yaml, toml
```

The server rebuilds trusted `turns` and resource metadata from validated `records`; it does not trust client-supplied redundant projections.

## Privacy boundary

Local Save may retain reviewed local provenance. Portable/server-backed Share applies a stricter source-URL policy:

- HTTP(S) only;
- credentials removed;
- query removed;
- fragment removed;
- filesystem/custom schemes removed;
- resource `sourceUrl` is retained only for page resources and only when reviewed safe source URLs are enabled.

Python proxy and Worker independently re-canonicalize this boundary.

## UI/UX

The conversation export sheet now presents the workflow as **Save or share conversation** and promotes **Save file** to a first-class destination. The selected format is shared across Save/preview/self-contained/Global destinations rather than creating a hidden alternate serializer path.

TXT now includes schema, optional page title/source/export time, resource summary, model/provider timing metadata, rating, and optional feedback so human-readable export semantics do not lag structured formats.

The Global HTML viewer renders server-canonical resources, model/provider/time, rating, and feedback with DOM construction + `textContent`; no `innerHTML` rendering authority was introduced.

## Verification

Working-tree gates completed after the final TXT contract reconciliation:

- canonical Python Share owner: **34/34** with `ResourceWarning` promoted to error;
- explicit five-format Global create -> canonicalize -> read matrix: JSON/HTML/TXT/YAML/TOML **GREEN**;
- complete registered Node architecture plane: **140/140**;
- proxy Share/CORS/protocol neighbors: **16/16** warning-strict;
- test-layout architecture: **9/9**;
- full collection: **2330 / 0 errors**;
- maintenance drift checker: **GREEN**.

Collection increases from the Run 172 baseline because R173T1 adds real regression cases inside existing canonical owners; it does not add duplicate collected owner files.

## Next checkpoint

R173T2 may begin feedback export/review work from this checkpoint:

1. local each-feedback export via panel;
2. cloud merged-feedback export/review representation.

Keep local feedback artifacts and cloud merged review records as explicit control planes; do not infer network consent from local export actions.
