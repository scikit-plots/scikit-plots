# R173T3 — Contribution provenance, privacy, and cloud-merged hardening

Status: COMPLETE

Base: Run 173 T2B (`94e2b544549ca9b5598160e267d4df6dfe58d812b5f080ad6a7e9c78b38a0b24`).

## Purpose

Apply the same explicit artifact-role, local-projection/cloud-authority, privacy-minimization, and deterministic merged-view rules to AI dataset contribution for all three contribution scopes: single pair, rated answers, and whole conversation.

## Artifact identity

Human-facing local artifacts encode contribution scope plus lifecycle role:

```text
ai-contribution-single-pair-request-json-<timestamp>.json
ai-contribution-single-pair-cloud-projection-jsonl-<timestamp>.jsonl
ai-contribution-rated-answers-request-json-<timestamp>.json
ai-contribution-rated-answers-cloud-projection-jsonl-<timestamp>.jsonl
ai-contribution-whole-conversation-request-json-<timestamp>.json
ai-contribution-whole-conversation-cloud-projection-jsonl-<timestamp>.jsonl
```

The projection remains a local pre-save prediction, not proof of a successful cloud write. Provider objects such as `ct_<opaque>.jsonl` remain lifecycle authority and keep opaque storage identities.

## Privacy and canonicalization

Contribution model evidence is attribution-only: `id`, `provider`, and `model`. Endpoint URLs, labels, info URLs, descriptions, default/custom UI state, and malformed nested identity values are excluded. The canonical 8-key model shape keeps non-attribution fields explicitly `null`. The server repeats the minimization and sanitizes page provenance to portable HTTP(S) origin + path before digesting, normalizing, or deciding whether an update is idempotent.

Historical contribution rows are re-minimized when read/merged so retired transport metadata cannot reappear through derived exports. Whole-conversation assistant-message model evidence follows the same boundary.

## Derived cloud merged view

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --contribution-cloud-merged
```

The output is `ai-contribution-cloud-merged-jsonl-<UTC timestamp>.jsonl` plus a manifest containing SHA-256, byte/record counts, Q&A/conversation counts, source-identity fields, lifecycle role, and the explicit statement that individual canonical provider contribution records remain authoritative.

Derived merged files never become write authority. Default role-bearing merged names are excluded from local ingestion. Custom-renamed merged files are excluded only when a bounded sidecar manifest binds the exact filename and exact SHA-256; stale/tampered sidecars cannot hide unrelated JSONL inputs.

Merged JSONL and manifest writes use temporary files, flush + `fsync`, and atomic `os.replace`; failed replacement leaves prior valid artifacts intact and cleans temporary files.

## Resource-ownership review

A Python 3.13 `ResourceWarning` exposed a test-owned SQLite inspection connection that used `with sqlite3.connect(...)` without closing the connection. The fixture now owns and closes that connection explicitly. Production contribution-ledger ownership was not implicated.

A separate warning can be induced only when pytest-asyncio app tests and the `asyncio.run()` ledger owner share one Python 3.13 pytest process. Each owner is warning-strict clean in a fresh process; forced-GC isolation attributes that signal to test-framework event-loop teardown/order rather than contribution runtime ownership. Keep those owners as separate warning-strict gates instead of suppressing warnings or weakening product cleanup.

## Prefinal verification

- dataset/dedup/privacy/docs focused warning-strict: **77/77**;
- contribution ledger owner warning-strict, fresh process: **45/45**;
- proxy app owner warning-strict, fresh process: **46/46**;
- registered Node architecture: **140/140**;
- test-layout architecture: **9/9**;
- full collection: **2359 / 0 errors**;
- changed Python source syntax via in-memory `compile()`: **7/7**;
- changed JavaScript/MJS syntax via `node --check`: **4/4**;
- maintenance drift before metadata freeze: **GREEN**.

Candidate packaged-byte replay is GREEN: 77/77 focused warning-strict, 45/45 ledger warning-strict, 46/46 proxy app warning-strict, 140/140 Node architecture, 9/9 layout, 2359 collection, maintenance GREEN. Final delivery is deterministically rebuilt after this metadata freeze and receives a final fresh focused replay; its SHA-256 is kept external to avoid self-reference.
