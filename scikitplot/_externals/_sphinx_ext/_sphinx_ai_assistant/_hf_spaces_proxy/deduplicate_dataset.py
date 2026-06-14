# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/deduplicate_dataset.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

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
