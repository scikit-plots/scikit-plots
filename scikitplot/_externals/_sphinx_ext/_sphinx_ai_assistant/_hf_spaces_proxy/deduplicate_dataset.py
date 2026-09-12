# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/deduplicate_dataset.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
deduplicate_dataset.py
======================
Canonical dataset reader/deduplicator for AI-assistant feedback and contribution
records.

The script supports both deployment generations:

* **Legacy Hugging Face mode** — ``--repo-id`` continues to mean a Hugging Face
  Dataset repo unless ``--provider`` is explicitly changed.
* **Provider-neutral storage mode** — ``--from-storage-config`` or
  ``--targets-file`` consumes the same ``RECORD_STORAGE_TARGETS`` schema used by
  ``app.py``.  The primary target is selected by default; ``--target-id`` can
  select a mirror; ``--all-targets`` can safely union primary + mirrors.
* **Local snapshot mode** — ``--local-dir`` works with snapshots/clones from any
  supported provider and no longer requires a dummy ``--repo-id``.

Direct remote downloads are supported for Hugging Face, GitHub, GitLab, and
Bitbucket Cloud.  Provider credentials are read from environment variables in
storage-config mode.  ``--token-env`` is preferred for direct mode; legacy
``--token`` remains supported for backward compatibility.

Examples
--------
Legacy Hugging Face command (unchanged)::

    python deduplicate_dataset.py \\
        --repo-id scikit-plots/ai-assistant-contributions \\
        --output clean_dataset.jsonl

Local clone/snapshot::

    python deduplicate_dataset.py \\
        --local-dir /tmp/ai-assistant-records \\
        --output clean_dataset.jsonl

GitHub-only primary::

    export GITHUB_DATASET_READ_TOKEN=github_pat_...
    python deduplicate_dataset.py \\
        --provider github \\
        --repo-id scikit-plots/ai-assistant-records \\
        --token-env GITHUB_DATASET_READ_TOKEN \\
        --output clean_dataset.jsonl

Use the exact app.py storage topology; primary is selected automatically::

    export RECORD_STORAGE_TARGETS='[...]'
    export AI_RECORD_STORAGE_TOKEN_HF_PRIMARY='hf_...'
    export AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR='github_pat_...'
    python deduplicate_dataset.py --from-storage-config --output clean_dataset.jsonl

Read a specific mirror instead of the primary::

    python deduplicate_dataset.py \\
        --from-storage-config \\
        --target-id github-mirror \\
        --output clean_dataset.jsonl

Generate the deterministic derived cloud feedback-review view while keeping individual
provider records authoritative::

    python deduplicate_dataset.py \
        --from-storage-config \
        --feedback-review-cloud-merged

This writes ``ai-feedback-review-cloud-merged-jsonl-<UTC timestamp>.jsonl`` plus an
integrity/authority ``.manifest.json`` sidecar.

Generate the deterministic derived cloud contribution view while keeping individual
provider contribution records authoritative::

    python deduplicate_dataset.py \
        --from-storage-config \
        --contribution-cloud-merged

This writes ``ai-contribution-cloud-merged-jsonl-<UTC timestamp>.jsonl`` plus an
integrity/authority ``.manifest.json`` sidecar with Q&A/conversation counts and
source-identity fields. Derived merged artifacts are never re-ingested as source
authority.

Audit/recovery union across all configured targets::

    python deduplicate_dataset.py \\
        --from-storage-config \\
        --all-targets \\
        --stats-only

When ``--all-targets`` is used, byte-identical mirrored files are included only
once.  New-style files with the same canonical ``fb_<record-id>.jsonl`` or
``ct_<record-id>.jsonl`` identity but different bytes are treated as a hard
conflict instead of silently selecting one provider's copy.

Deduplication contract
----------------------
* Historical schema rows are normalized to the current canonical schema when
  ``_utils/_dataset_schema.py`` is importable.
* Storage lifecycle is resolved first by ``_dedup_key``; ``contribution`` wins
  over ``feedback`` for the same key and same-source ties use server ``_ts``.
* Semantic rating lineage is resolved second by ``feedbackChainId`` /
  ``prevFeedbackIds`` / ``editCount`` so the terminal rating wins even when
  provider-review updates and contribution snapshots use different storage keys.
* Forked or cyclic/inconsistent terminal lineages fail closed instead of using
  timestamp order to guess which rating the reader intended.
* Retraction/withdrawal tombstones are considered during storage LWW but never
  emitted for training.
* The output is idempotent for the same source snapshots.

Security notes
--------------
* Prefer environment variables over ``--token`` because command-line arguments
  may be visible to other local processes or shell history.
* Provider error bodies and credential values are never logged.
* Downloaded tar archives are size-bounded and extracted with traversal,
  symlink, hardlink, device, and absolute-path guards.
* ``RECORD_STORAGE_TARGETS`` contains token *environment-variable names*, never
  token values.
"""  # noqa: D205, D400

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Optional: import _RedactingFilter from _shared_logic when available so that
# credential-looking strings embedded in dependency exceptions are scrubbed.
try:
    from ._utils._shared_logic import _RedactingFilter as _REDACTING_FILTER_CLS
except (ImportError, ValueError):
    try:
        from _utils._shared_logic import _RedactingFilter as _REDACTING_FILTER_CLS
    except ImportError:
        _REDACTING_FILTER_CLS = None  # type: ignore[assignment,misc]

# Optional: normalize records from v1 to v2 schema when _dataset_schema is
# available alongside this script (standard _hf_spaces_proxy/ deployment).
try:
    from ._utils._dataset_schema import (
        normalize_model_attribution as _normalize_model_attribution,
    )
    from ._utils._dataset_schema import (
        normalize_record as _normalize_record,
    )
    from ._utils._share_contract import (
        sanitize_share_page_url as _sanitize_dataset_page,
    )

    _SCHEMA_AVAILABLE = True
except (ImportError, ValueError):
    try:
        from _utils._dataset_schema import (
            normalize_model_attribution as _normalize_model_attribution,
        )
        from _utils._dataset_schema import (
            normalize_record as _normalize_record,
        )
        from _utils._share_contract import (
            sanitize_share_page_url as _sanitize_dataset_page,
        )

        _SCHEMA_AVAILABLE = True
    except ImportError:

        def _normalize_record(raw: dict) -> dict:
            return raw

        def _normalize_model_attribution(raw: Any) -> dict[str, Any] | None:
            if not isinstance(raw, dict):
                return None
            provider = raw.get("provider")
            model = raw.get("model")
            if not isinstance(provider, str) or not provider.strip():
                return None
            if not isinstance(model, str) or not model.strip():
                return None
            raw_id = raw.get("id")
            return {
                "id": raw_id[:256] if isinstance(raw_id, str) and raw_id else None,
                "provider": provider.strip()[:128],
                "model": model.strip()[:512],
                "label": None,
                "endpoint": None,
                "info_url": None,
                "description": None,
                "default": None,
            }

        def _sanitize_dataset_page(_value: Any) -> str:
            return ""

        _SCHEMA_AVAILABLE = False


_SOURCE_PRIORITY: dict[str, int] = {
    "contribution": 0,
    "feedback": 1,
}
_DEFAULT_PRIORITY = 99
_SUPPORTED_PROVIDERS = {"huggingface", "github", "gitlab", "bitbucket"}
_CANONICAL_RECORD_FILE_RE = re.compile(r"^(?:fb|ct)_([0-9a-f]{24})\.jsonl$")
_MAX_SOURCE_COUNT = 8


class DatasetSourceError(RuntimeError):
    """Raised when a dataset source cannot be read safely."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DatasetMirrorConflict(  # ruff: ignore[error-suffix-on-exception-name]
    RuntimeError,
):
    """Raised when mirrors disagree for the same canonical record-file ID."""


@dataclass(slots=True)
class DatasetSource:
    """Provider-neutral read source used by the deduplication CLI."""

    id: str
    provider: str
    repo: str
    branch: str = "main"
    token: str = ""
    feedback_path: str = "feedback"
    contributions_path: str = "contributions"
    api_base: str = ""


@dataclass(slots=True)
class SourceLoadStats:
    """Counters emitted when loading one or more storage targets."""

    files_seen: int = 0
    files_loaded: int = 0
    mirrored_files_suppressed: int = 0
    exact_records_suppressed: int = 0


@dataclass(slots=True)
class FeedbackLineageStats:
    """Audit counters from semantic terminal-rating resolution."""

    chains_seen: int = 0
    chains_collapsed: int = 0
    superseded_records_removed: int = 0
    forked_chains_excluded: int = 0
    malformed_records_excluded: int = 0
    unresolved_legacy_records: int = 0


_LAST_LINEAGE_STATS = FeedbackLineageStats()


def _priority(record: dict) -> int:
    return _SOURCE_PRIORITY.get(record.get("_source", ""), _DEFAULT_PRIORITY)


def _parse_jsonl_bytes(data: bytes, *, display_path: str) -> list[dict]:
    """Decode one JSONL byte payload into normalized record dictionaries."""
    records: list[dict] = []
    text = data.decode("utf-8", errors="strict")
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON in %s:%d", display_path, lineno)
            continue
        if not isinstance(raw, dict):
            logger.warning(
                "%s:%d: expected JSON object, got %s -- skipped",
                display_path,
                lineno,
                type(raw).__name__,
            )
            continue
        records.append(_normalize_record(raw))
    return records


def _has_bound_derived_manifest(  # ruff: ignore[too-many-return-statements]
    path: Path,
) -> bool:
    """Return whether a sidecar cryptographically identifies *path* as derived.

    This recognizes renamed/custom-output merged artifacts without trusting an
    unbound sidecar.  A manifest can suppress ingestion only when it names the
    exact file and its SHA-256 matches the current bytes.
    """
    manifest_path = path.with_name(path.name + ".manifest.json")
    try:
        if not manifest_path.is_file() or manifest_path.stat().st_size > 64 * 1024:
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        if (
            manifest.get("derived") is not True
            or manifest.get("authoritative") is not False
        ):
            return False
        if (
            manifest.get("lifecycleRole") != "cloud-merged"
            or manifest.get("representation") != "jsonl"
        ):
            return False
        if manifest.get("artifactFamily") not in {
            "ai-feedback-review",
            "ai-contribution",
        }:
            return False
        if manifest.get("filename") != path.name:
            return False
        digest = manifest.get("contentSha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == digest
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _is_derived_cloud_merged_path(path: Path) -> bool:
    """Return whether *path* is a non-authoritative derived merged export."""
    name = path.name
    patterns = (
        r"ai-feedback-review-cloud-merged-jsonl-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}\.jsonl",
        r"ai-contribution-cloud-merged-jsonl-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}\.jsonl",
    )
    return any(
        re.fullmatch(pattern, name) for pattern in patterns
    ) or _has_bound_derived_manifest(path)


def load_all_records(local_dir: Path) -> list[dict]:
    """Read authoritative ``*.jsonl`` files under *local_dir* into a flat list.

    Legacy local snapshots are still scanned recursively, but human-facing
    ``cloud-merged-jsonl`` exports are derived views rather than source records
    and are therefore skipped if a previous export sits inside the snapshot.
    New provider-aware CLI paths use only configured feedback/contribution folders.
    """
    records: list[dict] = []
    for jsonl_path in sorted(local_dir.rglob("*.jsonl")):
        if _is_derived_cloud_merged_path(jsonl_path):
            logger.info("Skipping derived merged dataset artifact: %s", jsonl_path.name)
            continue
        try:
            data = jsonl_path.read_bytes()
        except OSError:
            logger.warning("Unable to read dataset file; skipped: %s", jsonl_path)
            continue
        try:
            records.extend(_parse_jsonl_bytes(data, display_path=str(jsonl_path)))
        except UnicodeDecodeError:
            logger.warning("Dataset file is not valid UTF-8; skipped: %s", jsonl_path)
    return records


def _iter_source_files(root: Path, source: DatasetSource) -> Iterable[tuple[str, Path]]:
    """Yield configured record files as ``(logical_path, local_path)`` pairs."""
    seen: set[Path] = set()
    for folder in (source.feedback_path, source.contributions_path):
        base = root.joinpath(*folder.split("/"))
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.jsonl")):
            if _is_derived_cloud_merged_path(path):
                logger.info("Skipping derived merged dataset artifact: %s", path.name)
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path.relative_to(root).as_posix(), path


def _record_fingerprint(record: dict) -> str:
    """Return a stable exact-record fingerprint after schema normalization."""
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_sources_records(
    source_roots: list[tuple[DatasetSource, Path]],
    *,
    merge_mirrors: bool,
) -> tuple[list[dict], SourceLoadStats]:
    """Load configured sources with mirror-aware duplicate/conflict handling.

    In single-source mode this is equivalent to reading the configured record
    folders. In multi-source mode:

    * byte-identical files are loaded once;
    * same canonical record-file ID with different bytes raises a hard conflict;
    * exact normalized records repeated in legacy differently-named files are
      suppressed once across sources.
    """
    records: list[dict] = []
    stats = SourceLoadStats()
    seen_file_hashes: set[str] = set()
    canonical_ids: dict[str, str] = {}
    seen_record_hashes: set[str] = set()

    for source, root in source_roots:
        for logical_path, path in _iter_source_files(root, source):
            stats.files_seen += 1
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise DatasetSourceError("SOURCE_FILE_READ") from exc
            digest = hashlib.sha256(data).hexdigest()
            canonical_match = _CANONICAL_RECORD_FILE_RE.fullmatch(path.name)
            if merge_mirrors and canonical_match:
                canonical_id = canonical_match.group(1)
                prior = canonical_ids.get(canonical_id)
                if prior is not None and prior != digest:
                    raise DatasetMirrorConflict(
                        f"canonical record file {canonical_id} differs across storage targets"
                    )
                canonical_ids[canonical_id] = digest

            if merge_mirrors and digest in seen_file_hashes:
                stats.mirrored_files_suppressed += 1
                continue
            seen_file_hashes.add(digest)
            stats.files_loaded += 1

            try:
                parsed = _parse_jsonl_bytes(
                    data, display_path=f"{source.id}:{logical_path}"
                )
            except UnicodeDecodeError as exc:
                raise DatasetSourceError("SOURCE_UTF8") from exc

            if not merge_mirrors:
                records.extend(parsed)
                continue

            # Legacy mirrors may contain timestamp-named files whose byte layout
            # differs while records are semantically identical. Suppress exact
            # normalized record duplicates across sources as a second guard.
            for rec in parsed:
                fp = _record_fingerprint(rec)
                if fp in seen_record_hashes:
                    stats.exact_records_suppressed += 1
                    continue
                seen_record_hashes.add(fp)
                records.append(rec)

    return records, stats


def _semantic_feedback_candidate(record: dict[str, Any]) -> bool:
    """Return whether *record* participates in top-level rating lineage."""
    return (
        record.get("recordType") == "qa"
        and record.get("action") in {"rate", "review"}
        and isinstance(record.get("feedbackId"), str)
        and bool(record.get("feedbackId"))
        and record.get("ratingValue") is not None
    )


def _lineage_edit_count(record: dict[str, Any]) -> int | None:
    value = record.get("editCount")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _explicit_lineage_root(  # ruff: ignore[too-many-return-statements]
    record: dict[str, Any],
) -> tuple[str | None, bool]:
    """Return ``(root, valid)`` for self-contained schema-v5 lineage evidence.

    An empty ``prevFeedbackIds`` vector is valid for a first rating.  Historical
    rows that have only ``prevFeedbackId`` are reported as unresolved here and
    may be connected by the graph fallback in ``_resolve_terminal_feedback_lineages``.
    """
    fid = record.get("feedbackId")
    chain = record.get("feedbackChainId")
    prev = record.get("prevFeedbackId")
    raw_prev_ids = record.get("prevFeedbackIds")
    edit_count = _lineage_edit_count(record)
    if edit_count is None:
        return None, False
    if raw_prev_ids is None:
        return None, True
    if not isinstance(raw_prev_ids, list):
        return None, False
    prev_ids = [item for item in raw_prev_ids if isinstance(item, str) and item]
    if len(prev_ids) != len(raw_prev_ids) or len(set(prev_ids)) != len(prev_ids):
        return None, False
    if fid in prev_ids:
        return None, False
    if prev_ids:
        if not isinstance(chain, str) or not chain or chain != prev_ids[0]:
            return None, False
        if prev != prev_ids[-1] or edit_count != len(prev_ids):
            return None, False
        return chain, True
    if prev is None and edit_count == 0:
        if chain is None:
            return str(fid), True
        return (str(chain), bool(chain == fid))
    # ``[]`` plus a parent/editCount is a historical partial-normalization shape.
    return None, True


def _choose_same_terminal(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose one representation of the same terminal feedback event."""
    best = records[0]
    for rec in records[1:]:
        new_pri = _priority(rec)
        old_pri = _priority(best)
        if new_pri < old_pri or (
            new_pri == old_pri and rec.get("_ts", 0) > best.get("_ts", 0)
        ):
            best = rec
    return best


def _resolve_terminal_feedback_lineages(  # ruff: ignore[too-many-branches]
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], FeedbackLineageStats]:
    """Collapse superseded Q&A rating records by semantic feedback lineage.

    Storage keys intentionally describe receipts/provider lifecycle, not user
    rating identity.  This second pass therefore resolves the terminal rating
    using the self-contained v5 lineage first and a bounded legacy parent graph
    second.  Same-revision forks are excluded rather than guessed by timestamps.
    """
    stats = FeedbackLineageStats()
    candidates = [r for r in records if _semantic_feedback_candidate(r)]
    passthrough = [r for r in records if not _semantic_feedback_candidate(r)]
    if not candidates:
        return records, stats

    by_id: dict[str, list[dict[str, Any]]] = {}
    for rec in candidates:
        by_id.setdefault(str(rec.get("feedbackId")), []).append(rec)

    root_cache: dict[str, str | None] = {}
    invalid_ids: set[str] = set()
    invalid_roots: set[str] = set()

    # The same semantic feedback event can legitimately appear in both feedback
    # review and contribution storage, but its ancestry must be identical.  A
    # reused feedbackId with conflicting parents/root/revision is ambiguous
    # identity evidence and must fail closed before source-priority resolution.
    for fid, same_id_records in by_id.items():
        signatures: set[tuple[Any, ...]] = set()
        claimed_roots: set[str] = set()
        for rec in same_id_records:
            prev_ids = rec.get("prevFeedbackIds")
            prev_tuple = (
                tuple(prev_ids) if isinstance(prev_ids, list) else ("<not-a-list>",)
            )
            signatures.add(
                (
                    rec.get("feedbackChainId"),
                    rec.get("prevFeedbackId"),
                    prev_tuple,
                    _lineage_edit_count(rec),
                )
            )
            claimed = rec.get("feedbackChainId")
            if isinstance(claimed, str) and claimed:
                claimed_roots.add(claimed)
        if len(signatures) > 1:
            invalid_ids.add(fid)
            invalid_roots.update(claimed_roots)

    def root_for(  # ruff: ignore[too-many-branches, too-many-return-statements]
        rec: dict[str, Any],
        visiting: set[str] | None = None,
    ) -> str | None:
        fid = str(rec.get("feedbackId") or "")
        if not fid:
            return None
        if fid in invalid_ids:
            root_cache[fid] = None
            return None
        if fid in root_cache:
            return root_cache[fid]
        explicit, valid = _explicit_lineage_root(rec)
        if not valid:
            invalid_ids.add(fid)
            claimed_root = rec.get("feedbackChainId")
            if isinstance(claimed_root, str) and claimed_root:
                invalid_roots.add(claimed_root)
            root_cache[fid] = None
            return None
        if explicit:
            root_cache[fid] = explicit
            return explicit

        prev = rec.get("prevFeedbackId")
        edit_count = _lineage_edit_count(rec)
        if not isinstance(prev, str) or not prev:
            if edit_count == 0:
                root_cache[fid] = fid
                return fid
            root_cache[fid] = None
            return None
        if visiting is None:
            visiting = set()
        if fid in visiting or prev == fid:
            invalid_ids.add(fid)
            claimed_root = rec.get("feedbackChainId")
            if isinstance(claimed_root, str) and claimed_root:
                invalid_roots.add(claimed_root)
            root_cache[fid] = None
            return None
        visiting = set(visiting)
        visiting.add(fid)
        parents = by_id.get(prev, [])
        if not parents:
            # A single edit can still reveal its root directly from the parent ID.
            if edit_count == 1:
                root_cache[fid] = prev
                return prev
            root_cache[fid] = None
            return None
        parent_roots = {root_for(parent, visiting) for parent in parents}
        parent_roots.discard(None)
        if len(parent_roots) != 1:
            invalid_ids.add(fid)
            root_cache[fid] = None
            return None
        root = next(iter(parent_roots))
        root_cache[fid] = root
        return root

    groups: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[dict[str, Any]] = []
    for rec in candidates:
        root = root_for(rec)
        if root is None:
            if str(rec.get("feedbackId") or "") in invalid_ids:
                stats.malformed_records_excluded += 1
            else:
                unresolved.append(rec)
                stats.unresolved_legacy_records += 1
            continue
        groups.setdefault(root, []).append(rec)

    resolved: list[dict[str, Any]] = []
    for _root, group in groups.items():
        stats.chains_seen += 1
        if _root in invalid_roots:
            stats.malformed_records_excluded += len(group)
            continue
        counts = [(_lineage_edit_count(rec), rec) for rec in group]
        if any(count is None for count, _rec in counts):
            stats.malformed_records_excluded += len(group)
            continue

        # A fork at *any* observed revision depth poisons the chain, even if one
        # branch later grows to a numerically larger editCount. Otherwise a
        # concurrent f2a/f2b split followed by f3-from-f2b would silently make
        # f3 look authoritative merely because it is deeper.
        ids_by_revision: dict[int, set[str]] = {}
        for count, rec in counts:
            ids_by_revision.setdefault(int(count), set()).add(
                str(rec.get("feedbackId") or "")
            )
        if any(len(ids) != 1 for ids in ids_by_revision.values()):
            stats.forked_chains_excluded += 1
            stats.malformed_records_excluded += len(group)
            continue

        # Validate every observed edge against every other known revision. Missing
        # historical rows are allowed because v5 carries self-contained ancestry,
        # but contradictory rows that *are* present are never ignored.
        known_by_revision = {
            depth: next(iter(ids)) for depth, ids in ids_by_revision.items()
        }
        inconsistent = False
        for count, rec in counts:
            depth = int(count)
            if (  # ruff: ignore[collapsible-if]
                depth > 0 and (depth - 1) in known_by_revision
            ):
                if rec.get("prevFeedbackId") != known_by_revision[depth - 1]:
                    inconsistent = True
                    break
            prev_ids = rec.get("prevFeedbackIds")
            if isinstance(prev_ids, list) and prev_ids:
                for known_depth, known_id in known_by_revision.items():
                    if known_depth >= depth:
                        continue
                    if (
                        known_depth >= len(prev_ids)
                        or prev_ids[known_depth] != known_id
                    ):
                        inconsistent = True
                        break
            if inconsistent:
                break
        if inconsistent:
            stats.forked_chains_excluded += 1
            stats.malformed_records_excluded += len(group)
            continue

        max_count = max(int(count) for count, _rec in counts)
        terminal = [rec for count, rec in counts if count == max_count]
        # Per-depth uniqueness above guarantees one semantic terminal ID; there
        # may still be multiple storage representations of that same event.
        winner = _choose_same_terminal(terminal)
        resolved.append(winner)
        removed = len(group) - 1
        if removed:
            stats.chains_collapsed += 1
            stats.superseded_records_removed += removed

    # Unresolved historical rows are retained rather than guessed. They do not
    # gain semantic deduplication, preserving backward-compatible output while
    # current v5 rows fail closed on explicit malformed/forked lineage.
    return passthrough + unresolved + resolved, stats


def deduplicate(records: list[dict], *, include_unreviewed: bool = False) -> list[dict]:
    """Build the training set from reviewed records, then deduplicate.

    By default reviewed ``contribution`` and explicitly training-consented
    ``feedback`` rows with ``trainingStatus=eligible`` are admitted, plus
    privacy-minimal ``action=withdraw``
    tombstones are admitted. Withdrawal tombstones participate in last-write-
    wins so a later participant withdrawal suppresses the matching eligible
    row, then the tombstone itself is excluded from training output. Feedback
    telemetry, quarantined intake, and historical unreviewed rows fail closed.
    ``include_unreviewed`` exists only for explicit audit/recovery workflows.


    Storage lifecycle is resolved first: ``contribution`` beats ``feedback`` for
    the same storage key and ties within one source use latest server ``_ts``.
    Retraction/withdrawal tombstones participate in that LWW pass and are then
    excluded.  A second semantic pass resolves the terminal Q&A rating by
    feedback lineage, where ancestry/editCount outrank source priority; source
    priority is used only for duplicate representations of the same terminal
    feedbackId.  Malformed cycles, conflicting same-ID ancestry, and same-revision
    forks fail closed. Records without ``_dedup_key`` remain for legacy
    compatibility unless explicit malformed lineage makes them unsafe.
    """
    keyed: dict[str, dict] = {}
    no_key: list[dict] = []

    filtered: list[dict] = []
    for rec in records:
        source = rec.get("_source")
        if source not in {"contribution", "feedback"}:
            continue
        status = rec.get("trainingStatus")
        action = rec.get("action")
        if (
            (action == "withdraw" and status == "withdrawn")
            or status == "eligible"
            or (
                include_unreviewed
                and source == "contribution"
                and status in {"quarantined", "legacy_unreviewed", None}
            )
        ):
            filtered.append(rec)

    for rec in filtered:
        dk = rec.get("_dedup_key")
        if dk is None:
            no_key.append(rec)
            continue

        existing = keyed.get(dk)
        if existing is None:
            keyed[dk] = rec
            continue

        new_pri = _priority(rec)
        old_pri = _priority(existing)
        if new_pri < old_pri or (
            new_pri == old_pri and rec.get("_ts", 0) > existing.get("_ts", 0)
        ):
            keyed[dk] = rec

    clean_keyed = [
        r for r in keyed.values() if r.get("action") not in {"retract", "withdraw"}
    ]
    semantic, lineage_stats = _resolve_terminal_feedback_lineages(clean_keyed + no_key)
    global _LAST_LINEAGE_STATS  # noqa: PLW0603
    _LAST_LINEAGE_STATS = lineage_stats
    return semantic


def _jsonl_bytes(records: list[dict]) -> bytes:
    """Return deterministic UTF-8 NDJSON bytes for *records*."""
    return (
        "".join(
            json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
            for rec in records
        )
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace one derived artifact without exposing partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:  # ruff: ignore[suppressible-exception]
            tmp.unlink()
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def write_output(records: list[dict], output_path: Path) -> None:
    """Write records to *output_path* as deterministic newline-delimited JSON."""
    _atomic_write_bytes(output_path, _jsonl_bytes(records))


def _feedback_review_merged_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Stable ordering key for the derived cloud feedback review view."""
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        int(record.get("_ts") or record.get("ts") or 0),
        str(record.get("feedbackChainId") or ""),
        int(record.get("editCount") or 0),
        int(record.get("answerIndex") if record.get("answerIndex") is not None else -1),
        str(record.get("feedbackId") or ""),
        hashlib.sha256(canonical).hexdigest(),
    )


def feedback_review_cloud_merged(records: list[dict]) -> list[dict]:
    """Return the deterministic derived view of eligible reviewed feedback.

    Individual provider feedback files remain the lifecycle/write authority.
    This helper deliberately emits only the current canonical reviewed-feedback
    rows after the normal deduplication + lineage resolver has run.
    """
    clean = deduplicate(records, include_unreviewed=False)
    feedback = [
        row
        for row in clean
        if row.get("_source") == "feedback"
        and row.get("feedbackReview") is True
        and row.get("recordType") == "qa"
        and row.get("trainingStatus") == "eligible"
        and row.get("action") not in {"retract", "withdraw"}
    ]
    return sorted(feedback, key=_feedback_review_merged_sort_key)


def _feedback_review_cloud_merged_filename(now: float | None = None) -> str:
    """Human-facing filename for one generated cloud merged feedback export."""
    stamp = datetime.fromtimestamp(now or time.time(), tz=timezone.utc).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )
    return f"ai-feedback-review-cloud-merged-jsonl-{stamp}.jsonl"


def write_feedback_review_cloud_merged(
    records: list[dict],
    output_path: Path,
    *,
    source_description: str,
) -> dict[str, Any]:
    """Write merged feedback JSONL plus a non-authoritative integrity manifest."""
    merged = feedback_review_cloud_merged(records)
    data = _jsonl_bytes(merged)
    _atomic_write_bytes(output_path, data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "artifactFamily": "ai-feedback-review",
        "lifecycleRole": "cloud-merged",
        "representation": "jsonl",
        "derived": True,
        "authoritative": False,
        "authority": "individual canonical provider feedback records",
        "source": source_description[:512],
        "filename": output_path.name,
        "recordCount": len(merged),
        "contentBytes": len(data),
        "contentSha256": digest,
        "lineageFields": [
            "feedbackId",
            "feedbackChainId",
            "prevFeedbackId",
            "prevFeedbackIds",
            "editCount",
            "_dedup_key",
        ],
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "records": merged,
        "manifest": manifest,
        "manifestPath": manifest_path,
    }


def _privacy_minimize_contribution_row_for_export(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Return a derived-view copy with current contribution privacy boundaries.

    Historical provider rows may predate model-attribution and page sanitization.
    A merged export must not revive those retired transport/source details.
    """
    row = json.loads(json.dumps(record))
    row["page"] = _sanitize_dataset_page(row.get("page") or "")
    row["model"] = _normalize_model_attribution(row.get("model"))
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            message["model"] = _normalize_model_attribution(message.get("model"))
    return row


def _contribution_merged_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Stable ordering key for the derived cloud contribution view."""
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        int(record.get("_ts") or record.get("ts") or 0),
        str(record.get("recordType") or ""),
        int(record.get("answerIndex") if record.get("answerIndex") is not None else -1),
        str(record.get("feedbackChainId") or ""),
        int(record.get("editCount") or 0),
        str(record.get("_dedup_key") or ""),
        hashlib.sha256(canonical).hexdigest(),
    )


def contribution_cloud_merged(records: list[dict]) -> list[dict]:
    """Return the deterministic derived view of eligible contributions.

    Individual provider contribution records remain the lifecycle/write
    authority.  This view includes only current eligible Q&A/conversation rows
    after normal deduplication, withdrawal suppression and feedback-lineage
    resolution have completed.
    """
    clean = deduplicate(records, include_unreviewed=False)
    contributions = [
        _privacy_minimize_contribution_row_for_export(row)
        for row in clean
        if row.get("_source") == "contribution"
        and row.get("recordType") in {"qa", "conversation"}
        and row.get("trainingStatus") == "eligible"
        and row.get("action") not in {"retract", "withdraw"}
    ]
    return sorted(contributions, key=_contribution_merged_sort_key)


def _contribution_cloud_merged_filename(now: float | None = None) -> str:
    """Human-facing filename for one generated cloud merged contribution export."""
    stamp = datetime.fromtimestamp(now or time.time(), tz=timezone.utc).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )
    return f"ai-contribution-cloud-merged-jsonl-{stamp}.jsonl"


def write_contribution_cloud_merged(
    records: list[dict],
    output_path: Path,
    *,
    source_description: str,
) -> dict[str, Any]:
    """Write merged contribution JSONL plus a non-authoritative manifest."""
    merged = contribution_cloud_merged(records)
    data = _jsonl_bytes(merged)
    _atomic_write_bytes(output_path, data)
    digest = hashlib.sha256(data).hexdigest()
    qa_count = sum(1 for row in merged if row.get("recordType") == "qa")
    conversation_count = sum(
        1 for row in merged if row.get("recordType") == "conversation"
    )
    manifest = {
        "schemaVersion": 1,
        "artifactFamily": "ai-contribution",
        "lifecycleRole": "cloud-merged",
        "representation": "jsonl",
        "derived": True,
        "authoritative": False,
        "authority": "individual canonical provider contribution records",
        "source": source_description[:512],
        "filename": output_path.name,
        "recordCount": len(merged),
        "qaRecordCount": qa_count,
        "conversationRecordCount": conversation_count,
        "contentBytes": len(data),
        "contentSha256": digest,
        "sourceIdentityFields": [
            "_dedup_key",
            "recordType",
            "conversationId",
            "answerIndex",
            "feedbackId",
            "feedbackChainId",
            "prevFeedbackId",
            "prevFeedbackIds",
            "editCount",
        ],
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "records": merged,
        "manifest": manifest,
        "manifestPath": manifest_path,
    }


def _report_stats(records: list[dict]) -> dict[str, Any]:
    """Return summary statistics for raw or deduplicated records."""
    by_source: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_schema: dict[Any, int] = {}
    with_feedback_id = 0
    with_prev_feedback = 0
    with_feedback_chain = 0
    with_feedback_history = 0
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
        if r.get("feedbackChainId"):
            with_feedback_chain += 1
        if isinstance(r.get("prevFeedbackIds"), list) and r.get("prevFeedbackIds"):
            with_feedback_history += 1
        if act == "retract":
            tombstones += 1

    return {
        "total": len(records),
        "by_source": by_source,
        "by_action": by_action,
        "by_schema": by_schema,
        "with_feedback_id": with_feedback_id,
        "with_prev_feedback_id": with_prev_feedback,
        "with_feedback_chain_id": with_feedback_chain,
        "with_feedback_history": with_feedback_history,
        "tombstones": tombstones,
    }


class _MaxLevelFilter(logging.Filter):
    """Admit only records whose logging level is at or below *max_level*."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return record.levelno <= self.max_level


def _configure_logging() -> None:
    """Attach deterministic stdout/stderr handlers for CLI use."""
    plain_fmt = logging.Formatter("%(message)s")
    level_fmt = logging.Formatter("[%(levelname)s] %(message)s")

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(plain_fmt)
    out_handler.setLevel(logging.DEBUG)
    out_handler.addFilter(_MaxLevelFilter(logging.INFO))

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(level_fmt)
    err_handler.setLevel(logging.WARNING)

    if _REDACTING_FILTER_CLS is not None:
        redactor = _REDACTING_FILTER_CLS()
        out_handler.addFilter(redactor)
        err_handler.addFilter(redactor)

    root = logging.getLogger()
    root.handlers = [out_handler, err_handler]
    root.setLevel(logging.DEBUG)


def _effective_hf_legacy_token() -> tuple[str, str]:
    """Resolve the same HF persistence-token precedence used by app.py."""
    dataset = os.environ.get("HF_DATASET_TOKEN", "").strip()
    legacy = os.environ.get("HF_WRITE_TOKEN", "").strip()
    inference = os.environ.get("HF_TOKEN", "").strip()
    if dataset:
        return dataset, os.environ.get("HF_DATASET_TOKEN_TYPE", "unknown")
    if legacy:
        return legacy, os.environ.get("HF_WRITE_TOKEN_TYPE", "unknown")
    return inference, os.environ.get("HF_TOKEN_TYPE", "unknown")


def _storage_sources(
    raw_json: str, *, target_id: str | None, all_targets: bool
) -> list[DatasetSource]:
    """Parse app.py's storage config through the shared _storage implementation."""
    try:
        try:
            from ._utils._storage import load_storage_targets  # noqa: PLC0415
        except (ImportError, ValueError):
            from _utils._storage import load_storage_targets  # noqa: PLC0415
    except ImportError as exc:
        raise DatasetSourceError("STORAGE_MODULE_MISSING") from exc

    legacy_token, legacy_type = _effective_hf_legacy_token()
    try:
        targets = load_storage_targets(
            raw_json,
            legacy_repo=os.environ.get("TRAINING_DATASET_REPO", "").strip(),
            legacy_token=legacy_token,
            legacy_token_type=legacy_type,
        )
    except Exception as exc:  # keep config internals/private values out of logs
        raise DatasetSourceError("STORAGE_CONFIG_INVALID") from exc

    if not targets:
        raise DatasetSourceError("STORAGE_NOT_CONFIGURED")

    selected = targets
    if target_id:
        selected = [t for t in targets if t.id == target_id]
        if not selected:
            raise DatasetSourceError("TARGET_ID_NOT_FOUND")
    elif not all_targets:
        selected = [t for t in targets if t.role == "primary"]

    if not selected or len(selected) > _MAX_SOURCE_COUNT:
        raise DatasetSourceError("SOURCE_COUNT")

    return [
        DatasetSource(
            id=t.id,
            provider=t.provider,
            repo=t.repo,
            branch=t.branch,
            token=t.token,
            feedback_path=t.feedback_path,
            contributions_path=t.contributions_path,
            api_base=t.api_base,
        )
        for t in selected
    ]


def _read_targets_file(path: Path) -> str:
    """Read a storage-target JSON file with a conservative size guard."""
    try:
        if path.stat().st_size > 256 * 1024:
            raise DatasetSourceError("TARGETS_FILE_TOO_LARGE")
        return path.read_text(encoding="utf-8")
    except DatasetSourceError:
        raise
    except OSError as exc:
        raise DatasetSourceError("TARGETS_FILE_READ") from exc


def _direct_source(args: argparse.Namespace) -> DatasetSource:
    provider = str(args.provider or "huggingface").lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise DatasetSourceError("PROVIDER")
    token = ""
    if args.token_env:
        token = os.environ.get(args.token_env, "").strip()
    elif args.token:
        token = args.token
    elif provider == "huggingface":
        token = os.environ.get("HF_TOKEN", "").strip()
    elif provider == "github":
        token = os.environ.get("GITHUB_TOKEN", "").strip()
    elif provider == "gitlab":
        token = os.environ.get("GITLAB_TOKEN", "").strip()
    elif provider == "bitbucket":
        token = os.environ.get("BITBUCKET_TOKEN", "").strip()
    return DatasetSource(
        id=f"{provider}-direct",
        provider=provider,
        repo=args.repo_id,
        branch=args.branch,
        token=token,
        feedback_path=args.feedback_path,
        contributions_path=args.contributions_path,
        api_base=args.api_base or "",
    )


def _stream_archive(
    *,
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None,
    destination: Path,
    max_bytes: int,
) -> None:
    """Download a provider archive without logging URL, headers, or body."""
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:
        raise DatasetSourceError("HTTPX_MISSING") from exc

    total = 0
    try:
        with httpx.Client(  # ruff: ignore[multiple-with-statements]
            follow_redirects=True,
            timeout=60.0,
        ) as client:
            with client.stream("GET", url, headers=headers, params=params) as response:
                if response.status_code != 200:  # ruff: ignore[magic-value-comparison]
                    raise DatasetSourceError(f"REMOTE_HTTP_{response.status_code}")
                content_length = response.headers.get("content-length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > max_bytes
                ):
                    raise DatasetSourceError("ARCHIVE_TOO_LARGE")
                with destination.open("wb") as fh:
                    for chunk in response.iter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise DatasetSourceError("ARCHIVE_TOO_LARGE")
                        fh.write(chunk)
    except DatasetSourceError:
        raise
    except Exception as exc:
        raise DatasetSourceError("REMOTE_DOWNLOAD") from exc


def _safe_extract_tar(
    archive_path: Path, destination: Path, *, max_extract_bytes: int
) -> Path:
    """Extract a provider tar archive with strict path/type/size guards."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    total = 0
    members_count = 0
    try:
        with tarfile.open(archive_path, mode="r:*") as tf:
            safe_members: list[tarfile.TarInfo] = []
            for member in tf.getmembers():
                members_count += 1
                if members_count > 100_000:  # ruff: ignore[magic-value-comparison]
                    raise DatasetSourceError("ARCHIVE_MEMBER_COUNT")
                # No links/devices/fifos: dataset snapshots need regular files + dirs only.
                if (
                    member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise DatasetSourceError("ARCHIVE_UNSAFE_MEMBER")
                name = member.name.replace("\\", "/")
                if not name or name.startswith("/"):
                    raise DatasetSourceError("ARCHIVE_PATH")
                parts = Path(name).parts
                if any(p in {"", ".", ".."} for p in parts):
                    raise DatasetSourceError("ARCHIVE_PATH")
                target = destination.joinpath(*parts).resolve()
                if (
                    target != destination_resolved
                    and destination_resolved not in target.parents
                ):
                    raise DatasetSourceError("ARCHIVE_PATH")
                if member.isfile():
                    total += max(0, int(member.size))
                    if total > max_extract_bytes:
                        raise DatasetSourceError("ARCHIVE_EXTRACT_TOO_LARGE")
                safe_members.append(member)
            tf.extractall(  # noqa: S202 - prevalidated above
                destination,
                members=safe_members,
            )
    except DatasetSourceError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise DatasetSourceError("ARCHIVE_INVALID") from exc

    children = [
        p
        for p in destination.iterdir()
        if p.name not in {".DS_Store"}  # ruff: ignore[single-item-membership-test]
    ]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return destination


def _download_hf(source: DatasetSource) -> Path:
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise DatasetSourceError("HUGGINGFACE_HUB_MISSING") from exc
    try:
        return Path(
            snapshot_download(
                repo_id=source.repo,
                repo_type="dataset",
                revision=source.branch,
                token=source.token or None,
            )
        )
    except Exception as exc:
        raise DatasetSourceError("HF_DOWNLOAD") from exc


def _download_http_source(
    source: DatasetSource,
    *,
    temp_root: Path,
    max_archive_bytes: int,
    max_extract_bytes: int,
) -> Path:
    archive = temp_root / f"{source.id}.tar.gz"
    extract_to = temp_root / f"{source.id}-extract"
    headers: dict[str, str] = {"Accept": "application/octet-stream"}
    params: dict[str, str] | None = None

    if source.provider == "github":
        owner, repo = source.repo.split("/", 1)
        url = (
            f"https://api.github.com/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
            f"/tarball/{quote(source.branch, safe='')}"
        )
        headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if source.token:
            headers["Authorization"] = f"Bearer {source.token}"
    elif source.provider == "gitlab":
        base = (source.api_base or "https://gitlab.com/api/v4").rstrip("/")
        url = f"{base}/projects/{quote(source.repo, safe='')}/repository/archive.tar.gz"
        params = {"sha": source.branch, "include_lfs_blobs": "false"}
        if source.token:
            headers["PRIVATE-TOKEN"] = source.token
    elif source.provider == "bitbucket":
        workspace, repo = source.repo.split("/", 1)
        # Bitbucket Cloud documents branch archives at /get/<branch>.gz.
        url = (
            f"https://bitbucket.org/{quote(workspace, safe='')}/{quote(repo, safe='')}"
            f"/get/{quote(source.branch, safe='')}.gz"
        )
        if source.token:
            # OAuth/access-token bearer auth matches the write adapter. Users of
            # Atlassian API-token basic auth can use a local clone/snapshot.
            headers["Authorization"] = f"Bearer {source.token}"
    else:
        raise DatasetSourceError("PROVIDER")

    _stream_archive(
        url=url,
        headers=headers,
        params=params,
        destination=archive,
        max_bytes=max_archive_bytes,
    )
    return _safe_extract_tar(archive, extract_to, max_extract_bytes=max_extract_bytes)


def _download_source(
    source: DatasetSource,
    *,
    temp_root: Path,
    max_archive_bytes: int,
    max_extract_bytes: int,
) -> Path:
    if source.provider == "huggingface":
        return _download_hf(source)
    return _download_http_source(
        source,
        temp_root=temp_root,
        max_archive_bytes=max_archive_bytes,
        max_extract_bytes=max_extract_bytes,
    )


def _log_stats(records: list[dict]) -> dict[str, Any]:
    stats = _report_stats(records)
    logger.info("  %d total records read", stats["total"])
    for src, cnt in sorted(stats["by_source"].items()):
        logger.info("    %s: %d", src, cnt)
    for act, cnt in sorted(stats["by_action"].items()):
        logger.info("    action=%r: %d", act, cnt)
    for sv, cnt in sorted(stats["by_schema"].items(), key=lambda x: str(x[0])):
        logger.info("    schemaVersion=%s: %d", sv, cnt)
    logger.info("  feedbackId populated:      %d", stats["with_feedback_id"])
    logger.info("  prevFeedbackId populated:  %d", stats["with_prev_feedback_id"])
    logger.info("  feedbackChainId populated: %d", stats["with_feedback_chain_id"])
    logger.info("  prevFeedbackIds populated: %d", stats["with_feedback_history"])
    if stats["tombstones"]:
        logger.info(
            "  %d retraction tombstone(s) in raw data (excluded from clean output)",
            stats["tombstones"],
        )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "Repository ID owner/repo. Legacy behavior: Hugging Face Dataset unless "
            "--provider is supplied. Optional when --local-dir or storage-config mode is used."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_SUPPORTED_PROVIDERS),
        default="huggingface",
        help="Provider for --repo-id direct mode (default: huggingface).",
    )
    parser.add_argument(
        "--branch", default="main", help="Repository branch/revision (default: main)."
    )
    parser.add_argument(
        "--feedback-path",
        default="feedback",
        help="Feedback folder (default: feedback).",
    )
    parser.add_argument(
        "--contributions-path",
        default="contributions",
        help="Contributions folder (default: contributions).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Custom GitLab API base for direct mode (advanced/self-managed GitLab).",
    )
    parser.add_argument(
        "--output",
        default="clean_dataset.jsonl",
        help="Output path for deduplicated NDJSON (default: clean_dataset.jsonl).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help=(
            "Use a local pre-downloaded snapshot/clone. Legacy --repo-id + --local-dir "
            "continues to work; --repo-id is no longer required for local mode."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Legacy direct-mode token. Prefer --token-env so credentials do not enter "
            "shell history/process arguments."
        ),
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help="Direct-mode environment variable containing the provider read token.",
    )
    parser.add_argument(
        "--from-storage-config",
        action="store_true",
        help=(
            "Read RECORD_STORAGE_TARGETS (or legacy TRAINING_DATASET_REPO/HF_* fallback) "
            "using the same parser as app.py. Selects the primary by default."
        ),
    )
    parser.add_argument(
        "--targets-file",
        default=None,
        help=(
            "Read RECORD_STORAGE_TARGETS JSON from a file. The file must contain token_env "
            "names only; token values stay in environment variables."
        ),
    )
    parser.add_argument(
        "--target-id",
        default=None,
        help="In storage-config mode, read one specific target ID (including a mirror).",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help=(
            "Read primary + all mirrors, suppress byte-identical mirror files, and fail on "
            "same canonical record ID with different content. Intended for audit/recovery."
        ),
    )
    parser.add_argument(
        "--max-archive-mb",
        type=int,
        default=512,
        help="Maximum compressed HTTP archive size in MiB (default: 512).",
    )
    parser.add_argument(
        "--max-extract-mb",
        type=int,
        default=2048,
        help="Maximum extracted HTTP archive size in MiB (default: 2048).",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help=(
            "AUDIT/RECOVERY ONLY: include quarantined or legacy-unreviewed contribution rows. "
            "Feedback telemetry remains excluded. Default training output accepts only trainingStatus=eligible."
        ),
    )
    parser.add_argument(
        "--feedback-review-cloud-merged",
        action="store_true",
        help=(
            "Export only the deterministic current view of eligible maintainer feedback "
            "reviews. Individual cloud feedback files remain authoritative; a .manifest.json "
            "sidecar binds the derived JSONL bytes and record count."
        ),
    )
    parser.add_argument(
        "--contribution-cloud-merged",
        action="store_true",
        help=(
            "Export only the deterministic current view of eligible dataset contributions. "
            "Individual cloud contribution files remain authoritative; a .manifest.json "
            "sidecar binds the derived JSONL bytes, record counts, and source identity fields."
        ),
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print dataset statistics without writing an output file.",
    )
    return parser


def main(  # ruff: ignore[too-many-branches, too-many-return-statements]
    argv: list[str] | None = None,
) -> int:
    """Run the dataset reader/deduplicator CLI."""
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parser.parse_args(raw_argv)
    output_explicit = any(
        token == "--output"  # ruff: ignore[hardcoded-password-string]
        or token.startswith("--output=")
        for token in raw_argv
    )
    _configure_logging()

    if not _SCHEMA_AVAILABLE:
        logger.warning(
            "_utils/_dataset_schema.py not found; records will not be normalized to the canonical schema."
        )

    if args.max_archive_mb < 1 or args.max_archive_mb > (
        16_384  # ruff: ignore[magic-value-comparison]
    ):
        parser.error("--max-archive-mb must be between 1 and 16384")
    if args.max_extract_mb < 1 or args.max_extract_mb > (
        65_536  # ruff: ignore[magic-value-comparison]
    ):
        parser.error("--max-extract-mb must be between 1 and 65536")
    if args.target_id and args.all_targets:
        parser.error("--target-id and --all-targets are mutually exclusive")
    if args.local_dir and (args.from_storage_config or args.targets_file):
        parser.error("--local-dir cannot be combined with storage-config mode")
    if args.repo_id and (args.from_storage_config or args.targets_file):
        parser.error("--repo-id cannot be combined with storage-config mode")
    if args.from_storage_config and args.targets_file:
        parser.error("use either --from-storage-config or --targets-file, not both")

    if args.feedback_review_cloud_merged and args.contribution_cloud_merged:
        parser.error(
            "--feedback-review-cloud-merged and --contribution-cloud-merged are mutually exclusive"
        )

    source_description = ""
    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser()
        if not local_dir.is_dir():
            logger.error(
                "Local dataset directory does not exist or is not a directory."
            )
            return 1
        logger.info("Reading records from local snapshot ...")
        source_description = f"local snapshot:{local_dir.name}"
        all_records = load_all_records(local_dir)
    else:
        sources: list[DatasetSource]
        merge_mirrors = False
        try:
            if args.from_storage_config or args.targets_file:
                raw_json = (
                    _read_targets_file(Path(args.targets_file).expanduser())
                    if args.targets_file
                    else os.environ.get("RECORD_STORAGE_TARGETS", "")
                )
                sources = _storage_sources(
                    raw_json,
                    target_id=args.target_id,
                    all_targets=args.all_targets,
                )
                merge_mirrors = args.all_targets
            elif args.repo_id:
                sources = [_direct_source(args)]
            else:
                parser.error(
                    "choose a source: --repo-id, --local-dir, --from-storage-config, or --targets-file"
                )
                return 2
        except DatasetSourceError as exc:
            logger.error("Dataset source configuration failed: code=%s", exc.code)
            return 1

        source_description = ",".join(
            f"{source.provider}:{source.id}" for source in sources
        )[:512]
        logger.info("Resolved %d dataset source(s).", len(sources))
        for source in sources:
            logger.info("  source=%s provider=%s", source.id, source.provider)

        max_archive_bytes = args.max_archive_mb * 1024 * 1024
        max_extract_bytes = args.max_extract_mb * 1024 * 1024
        try:
            with tempfile.TemporaryDirectory(prefix="ai-dataset-dedup-") as temp_dir:
                temp_root = Path(temp_dir)
                source_roots: list[tuple[DatasetSource, Path]] = []
                for source in sources:
                    logger.info(
                        "Downloading source=%s provider=%s ...",
                        source.id,
                        source.provider,
                    )
                    root = _download_source(
                        source,
                        temp_root=temp_root,
                        max_archive_bytes=max_archive_bytes,
                        max_extract_bytes=max_extract_bytes,
                    )
                    source_roots.append((source, root))
                all_records, source_stats = load_sources_records(
                    source_roots,
                    merge_mirrors=merge_mirrors,
                )
        except DatasetMirrorConflict:
            logger.error(
                "Storage mirrors disagree for the same canonical record ID; refusing to train from an ambiguous union."
            )
            return 1
        except DatasetSourceError as exc:
            logger.error("Dataset download/read failed: code=%s", exc.code)
            return 1

        logger.info("  record files seen:          %d", source_stats.files_seen)
        logger.info("  record files loaded:        %d", source_stats.files_loaded)
        if merge_mirrors:
            logger.info(
                "  mirrored files suppressed:   %d",
                source_stats.mirrored_files_suppressed,
            )
            logger.info(
                "  exact records suppressed:    %d",
                source_stats.exact_records_suppressed,
            )

    raw_stats = _log_stats(all_records)
    if args.stats_only:
        return 0

    if args.feedback_review_cloud_merged:
        if args.include_unreviewed:
            parser.error(
                "--feedback-review-cloud-merged cannot be combined with --include-unreviewed"
            )
        output_path = Path(args.output)
        if not output_explicit:
            output_path = Path(_feedback_review_cloud_merged_filename())
        result = write_feedback_review_cloud_merged(
            all_records, output_path, source_description=source_description
        )
        logger.info(
            "Merged feedback review view written to %s (%d records, sha256=%s)",
            output_path,
            len(result["records"]),
            result["manifest"]["contentSha256"],
        )
        logger.info("Integrity manifest written to %s", result["manifestPath"])
        return 0

    if args.contribution_cloud_merged:
        if args.include_unreviewed:
            parser.error(
                "--contribution-cloud-merged cannot be combined with --include-unreviewed"
            )
        output_path = Path(args.output)
        if not output_explicit:
            output_path = Path(_contribution_cloud_merged_filename())
        result = write_contribution_cloud_merged(
            all_records, output_path, source_description=source_description
        )
        logger.info(
            "Merged contribution view written to %s (%d records, sha256=%s)",
            output_path,
            len(result["records"]),
            result["manifest"]["contentSha256"],
        )
        logger.info("Integrity manifest written to %s", result["manifestPath"])
        return 0

    clean = deduplicate(all_records, include_unreviewed=args.include_unreviewed)
    if args.include_unreviewed:
        logger.warning(
            "AUDIT/RECOVERY MODE: unreviewed contribution rows may be included."
        )
    excluded = sum(
        1
        for r in all_records
        if r.get("_source") not in {"contribution", "feedback"}
        or r.get("trainingStatus") != "eligible"
    )
    logger.info("  %d record(s) excluded by training eligibility policy", excluded)
    duplicates_removed = max(
        0, raw_stats["total"] - raw_stats["tombstones"] - excluded - len(clean)
    )
    logger.info(
        "  %d duplicate/superseded record(s) removed", max(0, duplicates_removed)
    )
    lineage = _LAST_LINEAGE_STATS
    if lineage.chains_seen or lineage.unresolved_legacy_records:
        logger.info(
            "  lineage: %d chain(s), %d collapsed, %d superseded removed",
            lineage.chains_seen,
            lineage.chains_collapsed,
            lineage.superseded_records_removed,
        )
    if lineage.forked_chains_excluded or lineage.malformed_records_excluded:
        logger.warning(
            "Lineage safety excluded %d forked chain(s) / %d malformed record(s).",
            lineage.forked_chains_excluded,
            lineage.malformed_records_excluded,
        )
    if lineage.unresolved_legacy_records:
        logger.info(
            "  lineage: %d unresolved legacy record(s) retained without guessing",
            lineage.unresolved_legacy_records,
        )
    logger.info("  %d unique records retained", len(clean))

    output_path = Path(args.output)
    write_output(clean, output_path)
    logger.info("Clean dataset written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
