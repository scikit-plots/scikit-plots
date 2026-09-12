# R173T2B — Cloud merged feedback view

Status: COMPLETE

Base: Run 173 T2A1 (`4a44d6a1ff06f1b79f7e1dfc738b38afdc75faec83adecca76b3a3feee08611c`).

## Purpose

Provide one deterministic, human/developer-friendly merged JSONL view of canonical cloud maintainer-feedback records without replacing the individual provider files that own review, revision, withdrawal, deduplication, and concurrency semantics.

## Authority model

```text
individual canonical provider feedback files
        |  lifecycle/write authority
        v
existing dedup + feedback-lineage resolver
        |
        +-- exclude telemetry
        +-- exclude contributions
        +-- exclude unreviewed/quarantined rows
        +-- resolve superseded/retracted/withdrawn lineage
        v
deterministic current feedback-review rows
        v
ai-feedback-review-cloud-merged-jsonl-<UTC timestamp>.jsonl
        + .manifest.json integrity/authority sidecar
```

The merged artifact is explicitly derived and non-authoritative. It is never used as an input authority when found inside a local snapshot.

## Operator interface

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --feedback-review-cloud-merged
```

`--output` may override the human-facing filename. `--include-unreviewed` is rejected in merged-review mode.

The manifest binds the generated filename, byte count, row count, and SHA-256 and identifies the individual canonical provider feedback records as authority. Provider credentials are not serialized.

## Verification

- deduplicate/cloud-merge + feedback documentation focused: 23/23
- warning-strict feedback/dataset/lifecycle slice: 106/106
- registered Node architecture: 140/140
- layout architecture: 9/9
- collection: 2340 / 0 errors
- maintenance checker: GREEN after metadata freeze
