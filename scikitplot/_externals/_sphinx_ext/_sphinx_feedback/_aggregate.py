"""Offline aggregate helpers shared by build tooling and tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from ._contracts import (
    AGGREGATE_CONTRACT,
    MAX_AGGREGATE_BYTES,
    canonical_event_bytes,
    normalize_page_revision,
    normalize_site_id,
    parse_feedback_event,
)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd)
    except OSError:
        # The file itself is already fsync'd and atomically replaced. Directory
        # fsync is an additional durability guarantee where the platform permits.
        pass
    finally:
        if fd is not None:
            os.close(fd)


def write_aggregate(  # ruff: ignore[too-many-branches]
    path: str | Path,
    rows: Iterable[dict],
    *,
    site_id: str,
    page_revision: str | None = None,
    complete_snapshot: bool = False,
) -> None:
    """
    Write one deterministic, site-scoped aggregate JSON file atomically.

    ``page_revision`` is optional for page-lifetime aggregates. When a non-empty
    revision is supplied, output includes only events from that
    exact revision so a rebuilt page cannot inherit stale feedback from an older
    document revision. Duplicate copies of the exact same event are ignored so a
    caller can safely aggregate a de-duplicated union of primary/mirror exports.
    Reusing one feedback id for different durable content remains a hard conflict.

    ``complete_snapshot=True`` is an explicit coverage statement: ``rows`` contains
    every reviewed event for this site/revision scope. Sphinx may therefore treat a
    page missing from ``pages`` as a reviewed zero-count page. Leave it false for
    partial exports, mirrors, samples, or any input whose completeness is unknown.
    """
    if not isinstance(complete_snapshot, bool):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            "complete_snapshot must be a boolean",
        )
    try:
        site_id = normalize_site_id(site_id)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    revision_filter: str | None = None
    if page_revision not in (None, ""):
        try:
            revision_filter = normalize_page_revision(page_revision)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not revision_filter:
            raise ValueError(
                "revision-scoped aggregate requires a non-empty page_revision",
            )

    pages: dict[str, dict[str, int]] = {}
    seen: dict[str, bytes] = {}
    for raw_event in rows:
        try:
            event = parse_feedback_event(raw_event)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if event["site_id"] != site_id:
            raise ValueError("aggregate rows must all belong to the requested site_id")
        event_id = event["feedback"]["id"]
        canonical = canonical_event_bytes(event)
        previous = seen.get(event_id)
        if previous is not None:
            if previous != canonical:
                raise ValueError(
                    "duplicate feedback id has conflicting durable content",
                )
            continue
        seen[event_id] = canonical
        if (
            revision_filter is not None
            and event.get("page_revision", "") != revision_filter
        ):
            continue
        page_id = event["page_id"]
        rating = int(event["feedback"]["rating"])
        state = pages.setdefault(
            page_id,
            {
                "count": 0,
                "score": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
        )
        state["count"] += 1
        state["score"] += rating
        if rating > 0:
            state["positive_count"] += 1
        elif rating < 0:
            state["negative_count"] += 1
        else:
            state["neutral_count"] += 1
    payload = {
        "contract": AGGREGATE_CONTRACT,
        "site_id": site_id,
        "pages": {key: pages[key] for key in sorted(pages)},
    }
    if complete_snapshot:
        payload["complete"] = True
    if revision_filter is not None:
        payload["page_revision"] = revision_filter
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_AGGREGATE_BYTES:
        raise ValueError("feedback aggregate exceeds 2 MiB")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("feedback aggregate target must not be a symlink")
    if target.exists() and not target.is_file():
        raise ValueError("feedback aggregate target must be a regular file")
    if target.is_file():
        try:
            if target.read_bytes() == encoded:
                return
        except OSError as exc:
            raise ValueError("unable to read existing feedback aggregate") from exc
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        temp_name = ""
        _fsync_directory(target.parent)
    finally:
        if temp_name:
            try:  # ruff: ignore[suppressible-exception]
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
