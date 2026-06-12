# Dataset Collection Guidance

**Applies to:** `scikit-plots/ai-assistant-contributions` (HuggingFace Dataset)
**Version:** 1.0
**Maintained by:** scikit-plots maintainers

---

## 1. Overview

The AI-assistant widget collects fine-tuning data through **two independent paths**
that can produce records for the same (conversation, answer) pair:

| Path | Endpoint | Folder | Trigger |
|------|----------|--------|---------|
| **Individual feedback** | `POST /v1/feedback` | `feedback/` | User clicks a rating button immediately after each answer |
| **Whole-conversation contribution** | `POST /v1/contribute` | `contributions/` | User clicks the **Contribute** button in the share sheet (GDPR-gated) |

When both paths are active (`FEEDBACK_PERSIST_ENABLED=true` **and** the user
later clicks Contribute), the **same Q&A pair is stored twice**.  Training on
duplicated examples artificially amplifies those examples — the dominant
contributor problem — so duplicates **must be resolved before any training run**.

---

## 2. Deduplication Key Contract

Every JSONL record stored in the dataset carries two mandatory metadata fields:

| Field | Type | Value | Set by |
|-------|------|-------|--------|
| `_source` | `"feedback"` \| `"contribution"` | Provenance tag | Server |
| `_dedup_key` | `string` | `"{conversationId}:{answerIndex}"` | Server |

### Key format

```
_dedup_key = f"{conversationId}:{answerIndex}"
```

- **`conversationId`** — stable per-page-load UUID (`_sessionId` in the JS
  widget, set once at module load and never re-generated).  Sent as
  `detail.conversationId` in the feedback POST and as `payload.sessionId` in the
  contribution POST.  Both endpoints use the *same* UUID so equality across
  folders is sufficient to detect cross-source duplicates.

- **`answerIndex`** — zero-based integer position of the answer within the
  conversation transcript.

> **Invariant:** Within a single conversation, `(conversationId, answerIndex)`
> is unique.  Across conversations, `conversationId` alone is unique (UUID).
> Therefore `_dedup_key` is globally unique for a given answer position within
> a given conversation session.

---

## 3. Priority Rule

When two records share the same `_dedup_key`, **always keep the
`contribution` record and discard the `feedback` record**.

Rationale:

- The contribution path is gated behind explicit GDPR consent.  It represents
  a deliberate, end-of-session review of the user's ratings and is the
  higher-quality signal.
- The feedback path fires immediately on button click (keepalive, fire-and-
  forget) — it may capture a transient or misclicked rating that the user
  later reconsiders before clicking Contribute.
- Contribution records carry the verified `consentVersion` flag; feedback
  records do not require consent to store.

```
Priority:  contribution  >  feedback
```

### Retraction tombstones

When a user edits a previously submitted quick-feedback rating, the browser
sends a **retraction tombstone** record before the new rating.  Tombstones
are stored in the `feedback/` folder alongside normal rating records and
carry these fields:

| Field | Value |
|-------|-------|
| `action` | `"retract"` |
| `prevSessionId` | `sessionId` of the original rating record |
| `_dedup_key` | identical to the original record's `_dedup_key` |
| `_ts` | server-write timestamp — always later than the original |
| `ratingValue` | *absent* |

Tombstones participate in the LWW loop inside `deduplicate()` because their
`_ts` is later than the original record's, ensuring the original rating is
not emitted.  They are then **unconditionally removed from the clean output**:
a tombstone is never a valid training example.

**Normal terminal state** (edit completed, all three records share the same
`_dedup_key`):

```
_ts 100  ratingValue=+1     ← original rating, kept during LWW, superseded
_ts 200  action="retract"   ← tombstone, LWW suppresses +1; filtered from output
_ts 201  ratingValue=-1     ← new rating, LWW winner, written to clean output ✓
```

**Degenerate case** (tombstone wins — follow-up rating never reached the
server, e.g. network failure after the retraction was sent):

```
_ts 100  ratingValue=+1     ← original rating, superseded
_ts 200  action="retract"   ← LWW winner, but FILTERED from clean output
```

Net result: the original +1 was explicitly retracted, so **no record is
emitted** for this key.  Training data is never contaminated by either the
abandoned +1 or the tombstone.

---

## 4. Dataset Folder Structure

```
scikit-plots/ai-assistant-contributions/
├── contributions/
│   └── {unix_ms}.jsonl      # 1 file per /v1/contribute POST
│                             # Each line = 1 Q&A record
│                             # Fields: answerIndex, query, answer,
│                             #         ratingValue, ratingLabel, message, ts,
│                             #         _sessionId, _page, _model,
│                             #         _consentVersion, _ts,
│                             #         _source="contribution",
│                             #         _dedup_key
└── feedback/
    └── {unix_ms}.jsonl      # 1 file per /v1/feedback POST (when
                              #   FEEDBACK_PERSIST_ENABLED=true)
                              # Each file = 1 record
                              # Fields: all feedback payload fields +
                              #         _ts, _source="feedback",
                              #         _dedup_key
```

---

## 5. Canonical Deduplication Script

Run this script **before every training job** to produce a clean, deduplicated
NDJSON file.  It reads all JSONL files from both folders, deduplicates by
`_dedup_key` applying the priority rule, and writes one record per unique key.

```python
"""
deduplicate_dataset.py
======================
Canonical deduplication script for scikit-plots/ai-assistant-contributions.

Usage
-----
    python deduplicate_dataset.py \\
        --repo-id scikit-plots/ai-assistant-contributions \\
        --output  clean_dataset.jsonl

Requirements
------------
    huggingface_hub>=0.23,<2
    (optional) hf_transfer for faster downloads

Notes
-----
* Priority rule: "contribution" beats "feedback" for the same _dedup_key.
* Records without _dedup_key are retained as-is (legacy, pre-v1.0 records).
* Retraction tombstones (action="retract") are always excluded from the
  clean output even if they win the LWW race — they carry no ratingValue
  and must never enter a training job.
* Script is idempotent: re-running produces the same output for the same
  dataset state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Priority order: lower index = higher priority.  Records with a _source not
# listed here are treated as lowest priority (integer max).
_SOURCE_PRIORITY: dict[str, int] = {
    "contribution": 0,
    "feedback":     1,
}
_DEFAULT_PRIORITY = 99


def _priority(record: dict) -> int:
    return _SOURCE_PRIORITY.get(record.get("_source", ""), _DEFAULT_PRIORITY)


def load_all_records(local_dir: Path) -> list[dict]:
    """Read every ``*.jsonl`` file under *local_dir* into a flat list.

    Parameters
    ----------
    local_dir : pathlib.Path
        Root of the locally downloaded dataset snapshot.

    Returns
    -------
    list[dict]
        All JSON-decoded records, preserving ``_source`` and ``_dedup_key``.
    """
    records: list[dict] = []
    for jsonl_path in sorted(local_dir.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] Skipping malformed JSON in {jsonl_path}:{lineno}: {exc}",
                        file=sys.stderr,
                    )
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate *records* by ``_dedup_key`` applying the priority rule.

    Parameters
    ----------
    records : list[dict]
        All raw records from both ``contributions/`` and ``feedback/`` folders.

    Returns
    -------
    list[dict]
        One record per unique ``_dedup_key``.  Records that have no
        ``_dedup_key`` (legacy, pre-v1.0) are retained unchanged.

    Notes
    -----
    Priority rule: for the same ``_dedup_key``, the record with the lowest
    ``_SOURCE_PRIORITY`` value is kept.  Ties are broken by server-write
    timestamp (``_ts``), keeping the most recent.  This is deterministic:
    given the same input, the output is always the same.

    Retraction tombstones (``action="retract"``) are always excluded from the
    final output even when they are the last-written record for a given key.
    A tombstone carries no ``ratingValue`` and must never reach a training
    job.  The LWW loop above still uses tombstones to suppress an earlier
    rating (correct behaviour: the tombstone's ``_ts`` is later than the
    original record's), but the post-loop filter ensures tombstones cannot
    leak into ``clean_dataset.jsonl``.
    """
    keyed:   dict[str, dict] = {}   # _dedup_key → winning record
    no_key:  list[dict]      = []   # legacy records without _dedup_key

    for rec in records:
        dk = rec.get("_dedup_key")
        if dk is None:
            no_key.append(rec)
            continue

        existing = keyed.get(dk)
        if existing is None:
            keyed[dk] = rec
            continue

        # Compare priorities; lower = better.
        new_pri = _priority(rec)
        old_pri = _priority(existing)
        if new_pri < old_pri:
            keyed[dk] = rec
        elif new_pri == old_pri:
            # Same source: keep the most recently written record (_ts).
            if rec.get("_ts", 0) > existing.get("_ts", 0):
                keyed[dk] = rec

    # Post-loop: discard retraction tombstones from the winning set.
    #
    # Scenario A (normal edit): user clicks +1 (saved), edits (tombstone
    # saved at _ts+100ms), then clicks -1 (saved at _ts+101ms).
    # LWW selects the -1 record. No tombstone in output. ✓
    #
    # Scenario B (orphaned tombstone): +1 saved, tombstone saved, but the
    # follow-up -1 never reaches the server.  LWW selects the tombstone.
    # Without this filter, clean_dataset.jsonl would contain a record with
    # action="retract" and ratingValue absent — corrupting any training job.
    # The filter silently drops the tombstone.  The net effect is correct:
    # the +1 was explicitly retracted, so no rating is emitted. ✓
    #
    # Tombstones are intentionally kept in the keyed dict *during* the loop
    # because their later _ts must still suppress the earlier +1.  Removal
    # happens only at this output stage.
    clean_keyed = [r for r in keyed.values() if r.get("action") != "retract"]
    return clean_keyed + no_key


def write_output(records: list[dict], output_path: Path) -> None:
    """Write *records* to *output_path* as newline-delimited JSON.

    Parameters
    ----------
    records : list[dict]
        Deduplicated records.
    output_path : pathlib.Path
        Destination file.  Parent directories are created if absent.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo ID, e.g. scikit-plots/ai-assistant-contributions",
    )
    parser.add_argument(
        "--output",
        default="clean_dataset.jsonl",
        help="Output path for the deduplicated NDJSON file (default: clean_dataset.jsonl)",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Use a pre-downloaded local snapshot instead of downloading.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace read token (optional; uses cached token if absent).",
    )
    args = parser.parse_args(argv)

    local_dir: Path
    if args.local_dir:
        local_dir = Path(args.local_dir)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                "ERROR: huggingface_hub is not installed.  "
                "Run: pip install 'huggingface_hub>=0.23,<2'",
                file=sys.stderr,
            )
            return 1
        print(f"Downloading {args.repo_id} …")
        local_dir = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                token=args.token,
            )
        )

    print(f"Reading records from {local_dir} …")
    all_records = load_all_records(local_dir)
    retract_raw = sum(1 for r in all_records if r.get("action") == "retract")
    print(f"  {len(all_records)} total records read")
    if retract_raw:
        print(
            f"  {retract_raw} retraction tombstone(s) in raw data "
            f"(always excluded from clean output)"
        )

    clean = deduplicate(all_records)
    # retract_raw are excluded by deduplicate() before returning; subtract
    # them so the "duplicates removed" count reflects actual cross-source
    # duplicates rather than tombstones.
    duplicates_removed = len(all_records) - retract_raw - len(clean)
    print(f"  {duplicates_removed} duplicate(s) removed (priority rule applied)")
    print(f"  {len(clean)} unique records retained")

    output_path = Path(args.output)
    write_output(clean, output_path)
    print(f"Clean dataset written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Example run

```bash
pip install "huggingface_hub>=0.23,<2"

python deduplicate_dataset.py \
    --repo-id scikit-plots/ai-assistant-contributions \
    --output  clean_dataset.jsonl \
    --token   hf_xxxxxxxxxxxxxxxx
```

---

## 6. Field Reference (Post-Dedup Record)

Every record in `clean_dataset.jsonl` contains at minimum:

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | The question asked by the user |
| `answer` | `string` | The AI-generated response that was rated |
| `ratingValue` | `int \| null` | Signed integer rating (positive = good) |
| `ratingLabel` | `string` | Human-readable label (e.g. `"Helpful"`) |
| `message` | `string` | Optional free-text comment from the user |
| `answerIndex` | `int` | Zero-based position within the conversation |
| `ts` | `int` | Client-side timestamp (ms since epoch) |
| `_source` | `string` | `"contribution"` or `"feedback"` |
| `_dedup_key` | `string` | `"{conversationId}:{answerIndex}"` |
| `_ts` | `int` | Server-write timestamp (ms since epoch) |

Contribution records additionally carry:

| Field | Type | Description |
|-------|------|-------------|
| `_sessionId` | `string` | Conversation session UUID |
| `_page` | `string` | Originating documentation page URL |
| `_model` | `object \| null` | Model ID and provider |
| `_consentVersion` | `string` | Consent version string (e.g. `"v1.0"`) |

> **Note — retraction tombstones (`action="retract"`)** are always excluded
> from `clean_dataset.jsonl` by `deduplicate_dataset.py`.  They are only
> present in the raw `feedback/` folder.  The `action` field therefore never
> appears in the clean output described by this table.

---

## 7. Enabling Feedback Persistence

By default, `POST /v1/feedback` is **log-only** (no dataset writes).  To enable
dual-source collection:

1. Set environment variable `FEEDBACK_PERSIST_ENABLED=true` in your HF Space
   (Settings → Repository secrets).
2. Ensure `TRAINING_DATASET_REPO` and `HF_WRITE_TOKEN` (or `HF_TOKEN`) are also
   set.
3. Without both of the above, `FEEDBACK_PERSIST_ENABLED` has no effect — the
   endpoint continues to log only.

> **Important:** enabling this flag means duplicates **will** appear in the
> dataset whenever a user both rates answers AND clicks Contribute.  Always run
> `deduplicate_dataset.py` before training.

---

## 8. Summary of Changes (v1.0)

| Component | Change |
|-----------|--------|
| `app.py` | Added `FEEDBACK_PERSIST_ENABLED` constant |
| `app.py /v1/contribute` | Every stored JSONL record now carries `_source="contribution"` and `_dedup_key` |
| `app.py /v1/feedback` | Optional HF persistence via `FEEDBACK_PERSIST_ENABLED`; stored records carry `_source="feedback"` and `_dedup_key` |
| `app.py /v1/feedback` | Retraction tombstones (`action="retract"`) are detected, validated (`prevSessionId` required), rate-limit-exempt, logged as `feedback.retract`, and committed with `"Retract 1 feedback record"` message |
| `app.py rate limiter` | `_MAX_RL_ENTRIES` sweep now active in the feedback rate limiter to bound memory under unique-IP floods |
| `ai-assistant.js detail` | Added `conversationId: _sessionId` to feedback POST payload (critical — server needs this for correct `_dedup_key`) |
| `ai-assistant.js _feedbackStore` | Added `conversationId` field for self-documenting store entries |
| `ai-assistant.js tRecords` | Added `_source: 'contribution'` to each record for self-describing POST payload |
| `deduplicate_dataset.py` | Post-loop filter removes tombstone winners from clean output; `main()` reports tombstone count separately so duplicate counts are accurate |
| `DATASET_COLLECTION_GUIDANCE.md` | This document; added §3 Retraction tombstones, §6 tombstone exclusion note, §8 this table |

## 9. Viewing the dataset from the AI panel

The Extended Settings sheet (Endpoint Configuration → **Dataset Endpoint**)
surfaces the dataset directly in the browser, with no secret ever leaving the
server.

How the panel resolves the dataset repo (two sources, first match wins):

1. **Explicit** — `ai_assistant_panel_dataset_repo = "org/repo"` in `conf.py`.
   Highest trust; no network call. Use for offline docs or to pin a specific
   repo.
2. **Auto-discovery** — when a training URL is configured, the panel issues
   `GET {proxyBase}/` and reads `training.dataset_repo` from the JSON. The
   proxy already publishes this field, so no `conf.py` change is required for
   standard deployments.

From the repo id the panel builds three clickable links:

| Card | URL |
|------|-----|
| Dataset root | `https://huggingface.co/datasets/<repo>` |
| Feedback records | `https://huggingface.co/datasets/<repo>/tree/main/feedback` |
| Contributions | `https://huggingface.co/datasets/<repo>/tree/main/contributions` |

**Token posture (read-only).** The same `GET /` response also reports
`tokens.hf_token_type`, `tokens.hf_write_token_type`, and
`tokens.least_privilege_mode`. The panel shows these as a short
read/write/fine-grained summary so operators can confirm least-privilege
configuration **without** reading Space logs. Token *values* are never sent to
the browser. When neither source is reachable, the section degrades to a
"Not configured" hint and the Space repository secret
(`TRAINING_DATASET_REPO`) continues to drive persistence server-side — both
the panel and repo-secret approaches are fully supported.

## 10. Summary of Changes (vNEXT)

| Component | Change |
|-----------|--------|
| `_static/ai-assistant.js` | Added **Dataset Endpoint** subsection (E) to `_buildEndpointSheet`: proxy-discovery + explicit-config dataset links and read-only HF token-posture row |
| `_static/ai-assistant.js` | Project Links sheet now renders HuggingFace **Space / Dataset / Active Endpoint** cards (config-driven or auto-derived) |
| `_static/ai-assistant.js` | New **self-contained URL-hash share** tier (`#ai-share-c1.<fmt>.<base64url>`): a real navigable, server-free, any-device/any-browser permanent link; routed in `_checkShareHash` and used by the permanent-link button (data: URI kept as fallback) |
| `_static/ai-assistant.css` | Added `.ai-assistant-panel-ep-ext-dataset-*` rules (light + dark) for the Dataset Endpoint section |
| `__init__.py` | Injects `panelDatasetRepo` + `panelHf*` keys into `window.AI_ASSISTANT_CONFIG` and registers the matching `ai_assistant_panel_*` config values |
| `_example_conf.py` | Documented examples for every new key |
| `app.py` | **No change** — `GET /` already exposes `training.*` and `tokens.*` |
