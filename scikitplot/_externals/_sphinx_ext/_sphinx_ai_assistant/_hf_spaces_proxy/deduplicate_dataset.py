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
* Schema versions 1 (legacy) and 2 (current) are normalised to canonical v2 when
  ``_utils/_dataset_schema.py`` is importable.
* ``contribution`` wins over ``feedback`` for the same ``_dedup_key``.
* Ties within the same source use last-write-wins on server ``_ts``.
* Retraction tombstones are considered during LWW but never emitted for
  training.
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
from dataclasses import dataclass
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
    from ._utils._dataset_schema import normalize_record as _normalize_record

    _SCHEMA_AVAILABLE = True
except (ImportError, ValueError):
    try:
        from _utils._dataset_schema import normalize_record as _normalize_record

        _SCHEMA_AVAILABLE = True
    except ImportError:

        def _normalize_record(raw: dict) -> dict:
            return raw

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


def load_all_records(local_dir: Path) -> list[dict]:
    """Read every ``*.jsonl`` file under *local_dir* into a flat list.

    This public helper intentionally preserves the legacy behavior: every JSONL
    file below the supplied directory is read recursively.  New provider-aware
    CLI paths use the configured feedback/contribution folders instead.
    """
    records: list[dict] = []
    for jsonl_path in sorted(local_dir.rglob("*.jsonl")):
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


def deduplicate(records: list[dict], *, include_unreviewed: bool = False) -> list[dict]:
    """Build the training set from reviewed records, then deduplicate.

    By default only ``_source=contribution`` rows with
    ``trainingStatus=eligible`` plus privacy-minimal ``action=withdraw``
    tombstones are admitted. Withdrawal tombstones participate in last-write-
    wins so a later participant withdrawal suppresses the matching eligible
    row, then the tombstone itself is excluded from training output. Feedback
    telemetry, quarantined intake, and historical unreviewed rows fail closed.
    ``include_unreviewed`` exists only for explicit audit/recovery workflows.


    ``contribution`` beats ``feedback`` for the same key. Ties within the same
    source are resolved by latest server ``_ts``. Retraction/withdrawal tombstones
    participate in LWW and are then unconditionally excluded from output.
    Records without ``_dedup_key`` are retained for legacy compatibility.
    """
    keyed: dict[str, dict] = {}
    no_key: list[dict] = []

    filtered: list[dict] = []
    for rec in records:
        if rec.get("_source") != "contribution":
            continue
        status = rec.get("trainingStatus")
        action = rec.get("action")
        if (
            (action == "withdraw" and status == "withdrawn")
            or status == "eligible"
            or (
                include_unreviewed
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
    return clean_keyed + no_key


def write_output(records: list[dict], output_path: Path) -> None:
    """Write records to *output_path* as deterministic newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _report_stats(records: list[dict]) -> dict[str, Any]:
    """Return summary statistics for raw or deduplicated records."""
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
    args = parser.parse_args(argv)
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

    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser()
        if not local_dir.is_dir():
            logger.error(
                "Local dataset directory does not exist or is not a directory."
            )
            return 1
        logger.info("Reading records from local snapshot ...")
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

    clean = deduplicate(all_records, include_unreviewed=args.include_unreviewed)
    if args.include_unreviewed:
        logger.warning(
            "AUDIT/RECOVERY MODE: unreviewed contribution rows may be included."
        )
    excluded = sum(
        1
        for r in all_records
        if r.get("_source") != "contribution" or r.get("trainingStatus") != "eligible"
    )
    logger.info("  %d record(s) excluded by training eligibility policy", excluded)
    duplicates_removed = max(
        0, raw_stats["total"] - raw_stats["tombstones"] - excluded - len(clean)
    )
    logger.info(
        "  %d duplicate(s) removed (priority rule applied)", max(0, duplicates_removed)
    )
    logger.info("  %d unique records retained", len(clean))

    output_path = Path(args.output)
    write_output(clean, output_path)
    logger.info("Clean dataset written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
