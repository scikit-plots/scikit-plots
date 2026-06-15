# Dataset Collection Guidance

**Applies to:** `scikit-plots/ai-assistant-contributions` (HuggingFace Dataset)
**Version:** 2.0
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

Every JSONL record stored in the dataset carries two mandatory server-assigned
provenance fields:

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

### v2 direct foreign key (preferred join key)

Since schema version 2, contribution records carry an additional
**direct foreign key** alongside `_dedup_key`:

| Field | Type | Description |
|-------|------|-------------|
| `feedbackId` | `string \| null` | `feedbackId` (= `sessionId`) of the matching `feedback/` record, when the user individually rated this answer before clicking Contribute. `null` when they contributed without ever rating the specific answer. |

This is a **1-to-1 join key** between `contributions/` rows and `feedback/` rows —
use it for precise cross-source joins instead of the coarser `_dedup_key`.

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

```
Priority:  contribution  >  feedback
```

### Retraction tombstones

When a user edits a previously submitted rating, the browser sends a
**retraction tombstone** record before the new rating.  Tombstones are
stored in the `feedback/` folder alongside normal rating records.

Since schema version 2 the canonical field name for the superseded record's
identifier is `prevFeedbackId` (previously `prevSessionId`).  Both field names
are handled by `normalize_record` in `_dataset_schema.py` for backward
compatibility.

| Field | Value |
|-------|-------|
| `action` | `"retract"` |
| `prevFeedbackId` | `feedbackId` of the original rating record (v2) |
| `_dedup_key` | identical to the original record's `_dedup_key` |
| `_ts` | server-write timestamp — always later than the original |
| `ratingValue` | `null` (tombstones carry no rating) |
| `editCount` | `null` (not applicable to a tombstone) |

Tombstones participate in the LWW loop inside `deduplicate()` because their
`_ts` is later than the original record's, ensuring the original rating is
not emitted.  They are then **unconditionally removed from the clean output**:
a tombstone is never a valid training example.

**Normal terminal state** (edit completed, all three records share the same
`_dedup_key`):

```
_ts 100  action="rate"    ratingValue=+1  prevFeedbackId=null     editCount=0
_ts 200  action="retract" ratingValue=null prevFeedbackId=<id-A>  editCount=null
_ts 201  action="rate"    ratingValue=-1  prevFeedbackId=<id-A>   editCount=1
```

LWW selects `_ts 201` (most recent `rate` action). No tombstone in output. ✓

**Degenerate case** (tombstone wins — follow-up rating never reached the
server, e.g. network failure after the retraction was sent):

```
_ts 100  action="rate"    ratingValue=+1
_ts 200  action="retract" ratingValue=null  <- LWW winner
```

Net result: the original +1 was explicitly retracted, so **no record is
emitted** for this key.  Training data is never contaminated.

### Supersession chain resolution (v2)

Beyond LWW, `feedbackId` + `prevFeedbackId` form a walkable edit-chain
per `(conversationId, answerIndex)`:

```
rating A: feedbackId="id-A"  prevFeedbackId=null    editCount=0  <- first rating
rating B: feedbackId="id-B"  prevFeedbackId="id-A"  editCount=1  <- edited
rating C: feedbackId="id-C"  prevFeedbackId="id-B"  editCount=2  <- edited again
```

`editCount` gives the chain length without walking it.  The record
with the highest `editCount` and `action="rate"` is the current active rating
for that `(conversationId, answerIndex)`.  `deduplicate_dataset.py` resolves
this automatically via LWW on `_ts`.

---

## 4. Dataset Folder Structure

```
scikit-plots/ai-assistant-contributions/
|-- contributions/
|   `-- {unix_ms}.jsonl      # 1 file per /v1/contribute POST
|                             # Each line = 1 Q&A record (canonical schema v2)
|                             # Key fields: conversationId, feedbackId (FK->feedback/),
|                             #             answerIndex, query, answer,
|                             #             ratingValue, ratingSlug, ratingTitle, ratingMode,
|                             #             prevFeedbackId, editCount, message, ts,
|                             #             model (8-key), page, consentVersion (null),
|                             #             _source="contribution", _dedup_key, _ts
`-- feedback/
    `-- {unix_ms}.jsonl      # 1 file per /v1/feedback POST (when
                              #   FEEDBACK_PERSIST_ENABLED=true)
                              # Each file = 1 record (1 rating or 1 retract tombstone)
                              # Key fields: conversationId, feedbackId,
                              #             action, answerIndex,
                              #             ratingValue, ratingSlug, ratingTitle, ratingMode,
                              #             prevFeedbackId, editCount, message, query, answer,
                              #             model (8-key), page, consentVersion (null),
                              #             _source="feedback", _dedup_key, _ts
```

---

## 5. Canonical Deduplication Script

Run this script **before every training job** to produce a clean, deduplicated
NDJSON file.  It reads all JSONL files from both folders, deduplicates by
`_dedup_key` applying the priority rule, and writes one record per unique key.

```python
r"""
deduplicate_dataset.py
======================
Canonical deduplication script for scikit-plots/ai-assistant-contributions.

Supports schema versions 1 (legacy) and 2 (current).  Records are normalised
to the canonical v2 schema by ``_dataset_schema.normalize_record`` before
deduplication so callers can always expect the full field set.

Usage
-----
    python deduplicate_dataset.py \
        --repo-id scikit-plots/ai-assistant-contributions \
        --output  clean_dataset.jsonl

    # Use a local pre-downloaded snapshot (faster on re-runs):
    python deduplicate_dataset.py \
        --repo-id scikit-plots/ai-assistant-contributions \
        --local-dir /tmp/ai-contributions-snapshot \
        --output clean_dataset.jsonl

Requirements
------------
    huggingface_hub>=0.23,<2
    (optional) hf_transfer for faster downloads
    (optional) _dataset_schema.py (from _hf_spaces_proxy/) for normalization

Notes
-----
* Priority rule: "contribution" beats "feedback" for the same _dedup_key.
* Retraction tombstones (action="retract") are always excluded from the
  clean output even if they win the LWW race.
* Script is idempotent: re-running produces the same output for the same
  dataset state.
* Output records are written with ``sort_keys=True``, so every record's
  keys (including nested objects) appear in a fixed alphabetical order in
  clean_dataset.jsonl.
* Progress and statistics are emitted via the module ``logging`` logger.
  INFO-level records route to stdout; WARNING and ERROR records route to
  stderr — preserving the previous ``print`` /
  ``print(..., file=sys.stderr)`` split so that callers capturing stdout
  see only the NDJSON data.
* When _dataset_schema is importable, records are normalised from v1 to v2
  schema automatically (legacy _sessionId/_page/_model fields mapped to
  conversationId/page/model; editCount/feedbackId/prevFeedbackId back-filled).
  When _dataset_schema is not importable (standalone usage), records are used
  as-is with a warning.
"""  # noqa: D205, D400

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional: normalize records from v1 to v2 schema when _dataset_schema is
# available alongside this script (standard _hf_spaces_proxy/ deployment).
# Falls back to identity function with a warning for standalone usage.
try:
    from _dataset_schema import normalize_record as _normalize_record

    _SCHEMA_AVAILABLE = True
except ImportError:

    def _normalize_record(raw: dict) -> dict:
        return raw

    _SCHEMA_AVAILABLE = False


# Priority order: lower index = higher priority.
_SOURCE_PRIORITY: dict[str, int] = {
    "contribution": 0,
    "feedback": 1,
}
_DEFAULT_PRIORITY = 99


def _priority(record: dict) -> int:
    return _SOURCE_PRIORITY.get(record.get("_source", ""), _DEFAULT_PRIORITY)


def load_all_records(local_dir: Path) -> list[dict]:
    """Read every *.jsonl file under local_dir into a flat list.

    Parameters
    ----------
    local_dir : pathlib.Path
        Root of the locally downloaded dataset snapshot.

    Returns
    -------
    list[dict]
        All JSON-decoded records, normalised to canonical v2 schema when
        ``_dataset_schema`` is importable.  Malformed lines are skipped with
        a WARNING-level log record.
    """
    records: list[dict] = []
    for jsonl_path in sorted(local_dir.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()  # noqa: PLW2901
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON in %s:%d: %s",
                        jsonl_path,
                        lineno,
                        exc,
                    )
                    continue
                if not isinstance(raw, dict):
                    logger.warning(
                        "%s:%d: expected JSON object, got %s -- skipped",
                        jsonl_path,
                        lineno,
                        type(raw).__name__,
                    )
                    continue
                records.append(_normalize_record(raw))
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate records by _dedup_key applying the priority rule.

    Parameters
    ----------
    records : list[dict]
        All raw records from both contributions/ and feedback/ folders,
        already normalised to v2 schema by load_all_records.

    Returns
    -------
    list[dict]
        One record per unique _dedup_key.  Records that have no
        _dedup_key (legacy, pre-v1.0 records) are retained unchanged.
        Retraction tombstones are excluded from the output.

    Notes
    -----
    Priority rule
        For the same _dedup_key, the record with the lowest
        _SOURCE_PRIORITY value is kept.  Ties (same source) are broken by
        server-write timestamp (_ts), keeping the most recent.  Deterministic:
        given the same input, the output is always the same.

    Retraction tombstones
        action="retract" records are still used during the LWW loop
        because their later _ts must suppress an earlier rate record
        (correct behaviour).  They are removed in the post-loop filter so they
        cannot leak into clean_dataset.jsonl.

        Degenerate case -- orphaned tombstone wins: silently discarded.
        Net effect: the original rating was explicitly retracted, so no record
        is emitted for that key -- correct for training data quality.

    feedbackId cross-source linkage (v2)
        When both a feedback/ record and a contributions/ record exist
        for the same _dedup_key, the contribution record's feedbackId
        field points directly to the feedback record's feedbackId (1-to-1
        FK).  The winning contribution record therefore carries the complete
        provenance chain without any additional join.
    """
    keyed: dict[str, dict] = {}  # _dedup_key -> winning record
    no_key: list[dict] = []  # legacy records without _dedup_key

    for rec in records:
        dk = rec.get("_dedup_key")
        if dk is None:
            no_key.append(rec)
            continue

        existing = keyed.get(dk)
        if existing is None:
            keyed[dk] = rec
            continue

        # Compare source priorities; lower = better (contribution > feedback).
        new_pri = _priority(rec)
        old_pri = _priority(existing)
        if new_pri < old_pri:
            keyed[dk] = rec
        elif new_pri == old_pri:  # noqa: SIM102
            # Same source: keep the most recently written record (_ts).
            if rec.get("_ts", 0) > existing.get("_ts", 0):
                keyed[dk] = rec

    # Post-loop: discard retraction tombstones from the winning set.
    #
    # Scenario A (normal edit): user rates +1 (_ts=100), edits (tombstone at
    # _ts=200), then rates -1 (_ts=201).  LWW selects -1.  No tombstone. OK
    #
    # Scenario B (orphaned tombstone): +1 at _ts=100, tombstone at _ts=200,
    # but follow-up -1 never reached the server.  LWW selects the tombstone.
    # Without this filter, action="retract" with ratingValue=null would corrupt
    # training.  Filter silently drops it. OK
    clean_keyed = [r for r in keyed.values() if r.get("action") != "retract"]
    return clean_keyed + no_key


def write_output(records: list[dict], output_path: Path) -> None:
    """Write records to output_path as newline-delimited JSON.

    Parameters
    ----------
    records : list[dict]
        Deduplicated records in canonical v2 schema.
    output_path : pathlib.Path
        Destination file.  Parent directories are created if absent.

    Notes
    -----
    Each record is serialised with ``sort_keys=True``, so object keys
    (at every nesting level) are written in a fixed alphabetical order.
    This keeps the output byte-for-byte reproducible across runs and
    makes line-level diffs between dataset snapshots meaningful.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _report_stats(records: list[dict]) -> dict[str, Any]:
    """Return summary statistics for a list of records.

    Parameters
    ----------
    records : list[dict]
        Records to summarise (raw or deduplicated).

    Returns
    -------
    dict
        Counters by source, action, schema version, and FK population.
    """
    by_source: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_schema: dict[Any, int] = {}
    with_feedback_id = 0
    with_prev_feedback = 0
    tombstones = 0

    for r in records:
        src = r.get("_source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

        act = r.get("action", "rate")
        by_action[act] = by_action.get(act, 0) + 1

        sv = r.get("schemaVersion", "?")
        by_schema[sv] = by_schema.get(sv, 0) + 1

        if r.get("feedbackId"):
            with_feedback_id += 1
        if r.get("prevFeedbackId"):
            with_prev_feedback += 1
        if act == "retract":
            tombstones += 1

    return {
        "total": len(records),
        "by_source": by_source,
        "by_action": by_action,
        "by_schema": by_schema,
        "with_feedback_id": with_feedback_id,
        "with_prev_feedback_id": with_prev_feedback,
        "tombstones": tombstones,
    }


class _MaxLevelFilter(logging.Filter):
    """Admit only log records whose level is at or below *max_level*.

    Parameters
    ----------
    max_level : int
        Maximum ``logging`` level number (inclusive) to pass through.
        Records with a higher level number are suppressed.  Pass
        ``logging.INFO`` to block WARNING and above.

    Notes
    -----
    Attached to the stdout handler inside ``_configure_logging`` so that
    WARNING / ERROR records are handled exclusively by the stderr handler
    and are not duplicated on stdout.
    """

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Return ``True`` if *record.levelno* is at or below *max_level*.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to evaluate.

        Returns
        -------
        bool
            ``True`` to emit the record; ``False`` to suppress it.
        """
        return record.levelno <= self.max_level


def _configure_logging() -> None:
    """Attach stdout and stderr handlers to the root logger for CLI use.

    Routes INFO-level records to stdout with a plain ``%(message)s``
    format, and WARNING / ERROR / CRITICAL records to stderr with a
    ``[%(levelname)s] %(message)s`` format.

    This preserves the stdout / stderr split that the original ``print``
    / ``print(..., file=sys.stderr)`` calls provided:

    * Callers that capture stdout (e.g. downstream JSONL pipelines) see
      only the NDJSON data, never progress lines.
    * Diagnostic warnings and errors still appear on stderr.

    The function overwrites ``logging.root.handlers`` directly, so it is
    idempotent: repeated calls replace handlers rather than stacking
    duplicates.

    Notes
    -----
    This is a CLI-only helper.  Library callers that import the domain
    functions (``load_all_records``, ``deduplicate``, …) should configure
    their own logging handlers; this function is only invoked from
    ``main()``.
    """
    plain_fmt = logging.Formatter("%(message)s")
    level_fmt = logging.Formatter("[%(levelname)s] %(message)s")

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(plain_fmt)
    out_handler.setLevel(logging.DEBUG)
    out_handler.addFilter(_MaxLevelFilter(logging.INFO))

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(level_fmt)
    err_handler.setLevel(logging.WARNING)

    root = logging.getLogger()
    root.handlers = [out_handler, err_handler]
    root.setLevel(logging.DEBUG)


def main(argv: list[str] | None = None) -> int:
    """Run Main."""
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
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print dataset statistics without writing an output file.",
    )
    args = parser.parse_args(argv)
    _configure_logging()

    if not _SCHEMA_AVAILABLE:
        logger.warning(
            "_dataset_schema.py not found on sys.path.  Records will not be "
            "normalised from v1 to v2 schema.  Copy _dataset_schema.py from "
            "_hf_spaces_proxy/ to the same directory as this script for full "
            "schema normalisation.",
        )

    local_dir: Path
    if args.local_dir:
        local_dir = Path(args.local_dir)
    else:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError:
            logger.error(
                "huggingface_hub is not installed.  "
                "Run: pip install 'huggingface_hub>=0.23,<2'",
            )
            return 1
        logger.info("Downloading %s ...", args.repo_id)
        local_dir = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                token=args.token,
            )
        )

    logger.info("Reading records from %s ...", local_dir)
    all_records = load_all_records(local_dir)

    raw_stats = _report_stats(all_records)
    logger.info("  %d total records read", raw_stats["total"])
    for src, cnt in sorted(raw_stats["by_source"].items()):
        logger.info("    %s: %d", src, cnt)
    for act, cnt in sorted(raw_stats["by_action"].items()):
        logger.info("    action=%r: %d", act, cnt)
    for sv, cnt in raw_stats["by_schema"].items():
        logger.info("    schemaVersion=%s: %d", sv, cnt)
    logger.info("  feedbackId populated:      %d", raw_stats["with_feedback_id"])
    logger.info("  prevFeedbackId populated:  %d", raw_stats["with_prev_feedback_id"])
    if raw_stats["tombstones"]:
        logger.info(
            "  %d retraction tombstone(s) in raw data "
            "(always excluded from clean output)",
            raw_stats["tombstones"],
        )

    if args.stats_only:
        return 0

    clean = deduplicate(all_records)
    duplicates_removed = raw_stats["total"] - raw_stats["tombstones"] - len(clean)
    logger.info("  %d duplicate(s) removed (priority rule applied)", duplicates_removed)
    logger.info("  %d unique records retained", len(clean))

    output_path = Path(args.output)
    write_output(clean, output_path)
    logger.info("Clean dataset written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Example run

```bash
pip install "huggingface_hub>=0.23,<2"

# Full pipeline (download + normalise + dedup):
python deduplicate_dataset.py \
    --repo-id scikit-plots/ai-assistant-contributions \
    --output  clean_dataset.jsonl \
    --token   hf_xxxxxxxxxxxxxxxx

# Stats-only (no output file written):
python deduplicate_dataset.py \
    --repo-id scikit-plots/ai-assistant-contributions \
    --stats-only \
    --token   hf_xxxxxxxxxxxxxxxx

# Faster re-runs using a pre-downloaded snapshot:
python deduplicate_dataset.py \
    --repo-id  scikit-plots/ai-assistant-contributions \
    --local-dir /tmp/ai-contributions-snapshot \
    --output   clean_dataset.jsonl
```

---

## 6. Field Reference (Post-Dedup Record, Canonical v2 Schema)

Every record in `clean_dataset.jsonl` uses the canonical key order from
`_dataset_schema.CANONICAL_COLUMNS`.  Old v1 records are normalised to
this schema by `normalize_record` in `_dataset_schema.py`.

### Schema metadata

| Field | Type | Description |
|-------|------|-------------|
| `schemaVersion` | `int` | `2` for all records normalised by this pipeline |

### Provenance (server-assigned)

| Field | Type | Description |
|-------|------|-------------|
| `_source` | `string` | `"contribution"` or `"feedback"` |
| `_ts` | `int` | Server receive timestamp (ms since epoch) |
| `_dedup_key` | `string` | `"{conversationId}:{answerIndex}"` — cross-source dedup key |

### Session identity

| Field | Type | Description |
|-------|------|-------------|
| `conversationId` | `string` | Stable per-page-load UUID (formerly `_sessionId` in contribution records) |
| `feedbackId` | `string \| null` | Per-rating event UUID. **Contribution records**: FK pointing at the matching `feedback/` record (`null` when the user contributed without rating individually). **Feedback records**: own idempotency UUID (formerly `sessionId`). |

### Record descriptor

| Field | Type | Description |
|-------|------|-------------|
| `answerIndex` | `int` | Zero-based position of the answer in the conversation |
| `action` | `"rate" \| "retract"` | `"retract"` tombstones are always excluded from clean output |
| `prevFeedbackId` | `string \| null` | `feedbackId` of the record this one supersedes. Set on `action="rate"` edits and `action="retract"` tombstones. `null` for a first-time rating. |
| `editCount` | `int \| null` | `0` for first rating, `+1` per edit. `null` for tombstones. |
| `status` | `"active" \| "retracted"` | Managed by the dedup pipeline |

### Rating

| Field | Type | Description |
|-------|------|-------------|
| `ratingValue` | `int \| null` | Signed integer: `-1`/`+1` for quick; `-5`...`+5` for panel |
| `ratingSlug` | `string \| null` | Snake_case canonical identifier (e.g. `"helpful"`, `"mostly_positive"`) |
| `ratingTitle` | `string \| null` | Human display string (e.g. `"Helpful"`, `"Mostly yes"`) |
| `ratingMode` | `"quick" \| "panel" \| null` | Which rating widget produced this record |
| `message` | `string` | Free-text comment (empty string when absent) |

### Conversation content

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | User question |
| `answer` | `string` | Model response that was rated |

### Model attribution (8-key object)

| Field | Type | Description |
|-------|------|-------------|
| `model` | `dict \| null` | Full model attribution; `null` when no model was configured |
| `model.id` | `string \| null` | Canonical model identifier |
| `model.provider` | `string \| null` | Inference provider (e.g. `"huggingface"`, `"anthropic"`) |
| `model.model` | `string \| null` | HF model path or model string |
| `model.label` | `string \| null` | Human display name |
| `model.endpoint` | `string \| null` | Inference endpoint URL |
| `model.info_url` | `string \| null` | Model info/documentation link |
| `model.description` | `string \| null` | Short description |
| `model.default` | `bool \| null` | `True` when this was the default model |

### Context

| Field | Type | Description |
|-------|------|-------------|
| `page` | `string` | Originating documentation page URL |
| `consentVersion` | `null` | Reserved; always `null` while `CONSENT_VERSION_ENABLED = False` |
| `ts` | `int` | Client-side event timestamp (ms since epoch) |

> **Note** — retraction tombstones (`action="retract"`) are always excluded
> from `clean_dataset.jsonl`.  The `action` field therefore never appears in
> the clean output described by this table.

---

## 7. Enabling Feedback Persistence

By default, `POST /v1/feedback` is **log-only** (no dataset writes).  To enable
dual-source collection:

1. Set environment variable `FEEDBACK_PERSIST_ENABLED=true` in your HF Space
   (Settings -> Repository secrets).
2. Ensure `TRAINING_DATASET_REPO` and `HF_WRITE_TOKEN` (or `HF_TOKEN`) are also
   set.
3. Without both of the above, `FEEDBACK_PERSIST_ENABLED` has no effect.

> **Important:** enabling this flag means duplicates **will** appear in the
> dataset whenever a user both rates answers AND clicks Contribute.  Always run
> `deduplicate_dataset.py` before training.

---

## 8. Consent Version (Reserved)

`consentVersion` is collected for future use but **not currently enforced**.

| Setting | Location | Current value |
|---------|----------|---------------|
| `CONSENT_VERSION_ENABLED` | `_dataset_schema.py` | `False` |
| `CONSENT_VERSION_ENFORCEMENT_ENABLED` | `app.py` | `False` |
| `RESERVED_CONSENT_VERSION` | `_dataset_schema.py` | `"1.0.0"` |
| JS constant | `ai-assistant.js` | commented out (`// var CONSENT_VERSION = '1.0.0'`) |

While both flags are `False`, all stored records have `consentVersion: null`
regardless of what the client sends, including historical records that stored
`"v1.0"`.  To activate enforcement, flip both flags to `True`, uncomment the
JS constant, and set `TRAINING_CONSENT_VERSION` in `app.py`.

---

## 9. Summary of Changes

### v1.0 (initial)

| Component | Change |
|-----------|--------|
| `app.py` | Added `FEEDBACK_PERSIST_ENABLED` constant |
| `app.py /v1/contribute` | Stored JSONL records carry `_source="contribution"` and `_dedup_key` |
| `app.py /v1/feedback` | Optional HF persistence; stored records carry `_source="feedback"` and `_dedup_key` |
| `app.py /v1/feedback` | Retraction tombstones detected, validated, rate-limit-exempt, committed with distinct message |
| `ai-assistant.js` | Added `conversationId: _sessionId` to feedback POST payload |
| `ai-assistant.js _feedbackStore` | Added `conversationId` field |
| `ai-assistant.js tRecords` | Added `_source: 'contribution'` to each record |
| `deduplicate_dataset.py` | Post-loop filter removes tombstone winners from clean output |

### v2.0 (current — additive, backward compatible)

| Component | Change |
|-----------|--------|
| `_dataset_schema.py` | `SCHEMA_VERSION` bumped 1->2; new `editCount` column; `feedbackId` FK on contribution records; `prevFeedbackId` on `action="rate"` edits; `consentVersion` always null; `_safe_id`/`_safe_int` guards; 8-key model shape unified |
| `_dataset_schema.py` | `normalize_feedback_record` / `normalize_contribution_record` always write `SCHEMA_VERSION` (2) — fixes `int(1 or 2) = 1` Python short-circuit bug that stored v1 headers on v2-content records |
| `app.py` | `TRAINING_CONSENT_VERSION` hardcoded check replaced by `CONSENT_VERSION_ENFORCEMENT_ENABLED = False` flag — eliminates 422 errors for `consentVersion: null` payloads |
| `app.py` | `supported_versions` expanded `{1}->{1, 2}` to accept updated JS clients |
| `ai-assistant.js` | `schemaVersion: 1->2` in all four payload sites (quick feedback, panel `_rebuildFeedbackFormIn`, panel `_buildFeedbackBlock`, contribution envelope) |
| `ai-assistant.js` | `feedbackId`, `prevFeedbackId`, `editCount` forwarded in `tRecords` from `_feedbackStore` |
| `ai-assistant.js` | Shared `_buildModelInfo(cfg)` helper produces canonical 8-key model for all sites |
| `deduplicate_dataset.py` | Integrates `_dataset_schema.normalize_record` for v1->v2 normalisation; adds `--stats-only` flag; reports schema version breakdown and FK population counts |
| `DATASET_COLLECTION_GUIDANCE.md` | Updated to v2; revised field reference; updated folder structure; v2 dedup script |

---

## 10. Viewing the Dataset from the AI Panel

The Extended Settings sheet (Endpoint Configuration -> **Dataset Endpoint**)
surfaces the dataset directly in the browser, with no secret ever leaving the
server.

How the panel resolves the dataset repo (two sources, first match wins):

1. **Explicit** — `ai_assistant_panel_dataset_repo = "org/repo"` in `conf.py`.
   Highest trust; no network call. Use for offline docs or to pin a specific repo.
2. **Auto-discovery** — when a training URL is configured, the panel issues
   `GET {proxyBase}/` and reads `training.dataset_repo` from the JSON.

From the repo id the panel builds three clickable links:

| Card | URL |
|------|-----|
| Dataset root | `https://huggingface.co/datasets/<repo>` |
| Feedback records | `https://huggingface.co/datasets/<repo>/tree/main/feedback` |
| Contributions | `https://huggingface.co/datasets/<repo>/tree/main/contributions` |
