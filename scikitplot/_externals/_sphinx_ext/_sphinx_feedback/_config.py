"""Dependency-free public configuration helpers for ``_sphinx_feedback``."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ._contracts import (
    AGGREGATE_CONTRACT,
    MAX_AGGREGATE_BYTES,
    normalize_page_id,
    normalize_page_revision,
    normalize_site_id,
)

DEFAULT_SIDEBAR_SELECTORS = (
    'aside[role="complementary"]',
    ".bd-sidebar-secondary",
    ".bd-toc",
    ".toc-sidebar",
    ".toc-drawer",
    ".sphinxsidebar",
)
DEFAULT_MAIN_SELECTORS = (
    'article[role="main"]',
    "article.bd-article",
    "div.rst-content",
    '[role="main"]',
    "main",
    "div.document",
    "div.body",
    "article",
)
DEFAULT_EXCLUDE = ("search", "genindex", "py-modindex", "404")
POSITIONS = frozenset({"auto", "sidebar", "main-bottom", "floating", "none"})
FALLBACKS = frozenset({"main-bottom", "none"})
COUNTER_SOURCES = frozenset({"embedded", "none"})


class FeedbackConfigError(ValueError):
    """Raised for unsafe or internally inconsistent public config."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FeedbackConfigError(
                f"feedback aggregate contains duplicate field: {key}",
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise FeedbackConfigError(
        f"feedback aggregate contains a non-finite JSON number: {value}"
    )


def _plain_string(value: Any, *, name: str, maximum: int, empty: bool = True) -> str:
    if not isinstance(value, str):
        raise FeedbackConfigError(f"{name} must be a string")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise FeedbackConfigError(f"{name} contains invalid Unicode") from exc
    value = value.strip()
    if len(value) > maximum or any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value
    ):
        raise FeedbackConfigError(f"{name} is too long or contains control characters")
    if not empty and not value:
        raise FeedbackConfigError(f"{name} must not be empty")
    return value


def validate_endpoint(value: Any) -> str:
    """Accept HTTPS or localhost HTTP endpoints with no credentials/fragments."""
    endpoint = _plain_string(value, name="feedback_endpoint", maximum=2048)
    if not endpoint:
        return ""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise FeedbackConfigError(
            "feedback_endpoint is not a valid URL",
        ) from exc
    host = (parsed.hostname or "").lower()
    local_http = parsed.scheme == "http" and host in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not local_http:
        raise FeedbackConfigError(
            "feedback_endpoint must use HTTPS (localhost HTTP is allowed)",
        )
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise FeedbackConfigError(
            "feedback_endpoint must not contain credentials, a query, or a fragment",
        )
    if not host:
        raise FeedbackConfigError(
            "feedback_endpoint must include a host",
        )
    if parsed.scheme == "https" and port not in (None, 443):
        raise FeedbackConfigError(
            "feedback_endpoint must use the standard HTTPS port",
        )
    if (
        local_http  # lint
        and port is not None  # lint
        and not 1 <= port <= 65535  # ruff: ignore[magic-value-comparison]
    ):
        raise FeedbackConfigError("feedback_endpoint contains an invalid local port")
    return endpoint.rstrip("/")


def resolve_endpoint(config: Any) -> str:
    """
    Return the explicit generic-feedback authority endpoint.

    Page feedback is intentionally independent from AI Assistant endpoint
    profiles. Sites that use different chat/share and feedback services must not
    inherit one authority from the other implicitly.
    """
    return validate_endpoint(getattr(config, "feedback_endpoint", ""))


def validate_patterns(
    value: Any,
    *,
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    _len = len(value) > 128  # ruff: ignore[magic-value-comparison]
    if not isinstance(value, (list, tuple)) or _len:
        raise FeedbackConfigError(
            f"{name} must be a list/tuple with at most 128 patterns",
        )
    result: list[str] = []
    for item in value:
        text = _plain_string(item, name=name, maximum=256, empty=False)
        result.append(text)
    return tuple(result)


def page_enabled(
    pagename: str,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> bool:
    if any(fnmatch.fnmatchcase(pagename, pattern) for pattern in exclude):
        return False
    return any(fnmatch.fnmatchcase(pagename, pattern) for pattern in include)


def _validate_selectors(
    value: Any,
    *,
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    _len = len(value) > 32  # ruff: ignore[magic-value-comparison]
    if not isinstance(value, (list, tuple)) or not value or _len:
        raise FeedbackConfigError(f"{name} must contain 1..32 CSS selectors")
    selectors: list[str] = []
    for item in value:
        text = _plain_string(item, name=name, maximum=256, empty=False)
        if "</" in text.lower():
            raise FeedbackConfigError(f"{name} contains invalid selector text")
        selectors.append(text)
    return tuple(selectors)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise FeedbackConfigError(f"{name} must be a boolean")
    return value


def _page_revision(value: Any) -> str:
    text = _plain_string(value or "", name="feedback_page_revision", maximum=128)
    return text  # ruff: ignore[unnecessary-assign]


def validate_config(config: Any) -> dict[str, Any]:
    """Validate Sphinx-facing settings and return normalized shared config."""
    page_enabled_flag = _strict_bool(
        getattr(config, "feedback_page_enabled", False), name="feedback_page_enabled"
    )
    page_main = _strict_bool(
        getattr(config, "feedback_page_main", True), name="feedback_page_main"
    )
    quick_enabled = _strict_bool(
        getattr(config, "feedback_quick_enabled", True), name="feedback_quick_enabled"
    )
    detailed_enabled = _strict_bool(
        getattr(config, "feedback_detailed_enabled", True),
        name="feedback_detailed_enabled",
    )
    comment_enabled = _strict_bool(
        getattr(config, "feedback_comment_enabled", True),
        name="feedback_comment_enabled",
    )
    contributor_enabled = _strict_bool(
        getattr(config, "feedback_contributor_enabled", True),
        name="feedback_contributor_enabled",
    )
    counter_enabled = _strict_bool(
        getattr(config, "feedback_counter_enabled", True),
        name="feedback_counter_enabled",
    )
    position = (
        str(
            getattr(config, "feedback_position", "sidebar") or "",
        )
        .strip()
        .lower()
    )
    if position not in POSITIONS:
        raise FeedbackConfigError(
            f"feedback_position must be one of {sorted(POSITIONS)}",
        )
    fallback = (
        str(getattr(config, "feedback_position_fallback", "main-bottom") or "")
        .strip()
        .lower()
    )
    if fallback not in FALLBACKS:
        raise FeedbackConfigError(
            f"feedback_position_fallback must be one of {sorted(FALLBACKS)}",
        )
    counter_source = (
        str(getattr(config, "feedback_counter_source", "embedded") or "")
        .strip()
        .lower()
    )
    if counter_source not in COUNTER_SOURCES:
        raise FeedbackConfigError(
            "feedback_counter_source must be 'embedded' or 'none'; live page-view fetches are intentionally unsupported",
        )
    site_id = _plain_string(
        getattr(config, "feedback_site_id", ""),
        name="feedback_site_id",
        maximum=96,
        empty=False,
    )

    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?", site_id):
        raise FeedbackConfigError("feedback_site_id must be a stable ASCII identifier")
    endpoint = resolve_endpoint(config)
    page_revision = _page_revision(getattr(config, "feedback_page_revision", ""))
    if page_enabled_flag and not (quick_enabled or detailed_enabled):
        raise FeedbackConfigError(
            "enabled page feedback requires quick and/or detailed interaction"
        )
    if page_enabled_flag and (quick_enabled or detailed_enabled) and not endpoint:
        raise FeedbackConfigError(
            "enabled interactive page feedback requires an explicit feedback_endpoint"
        )
    include = validate_patterns(
        getattr(config, "feedback_include", ["**"]),
        name="feedback_include",
        default=("**",),
    )
    exclude = validate_patterns(
        getattr(config, "feedback_exclude", list(DEFAULT_EXCLUDE)),
        name="feedback_exclude",
        default=DEFAULT_EXCLUDE,
    )
    sidebar = _validate_selectors(
        getattr(config, "feedback_sidebar_selectors", None),
        name="feedback_sidebar_selectors",
        default=DEFAULT_SIDEBAR_SELECTORS,
    )
    main = _validate_selectors(
        getattr(config, "feedback_main_selectors", None),
        name="feedback_main_selectors",
        default=DEFAULT_MAIN_SELECTORS,
    )
    return {
        "page_enabled": page_enabled_flag,
        "page_main": page_main,
        "quick_enabled": quick_enabled,
        "detailed_enabled": detailed_enabled,
        "comment_enabled": comment_enabled,
        "contributor_enabled": contributor_enabled,
        "counter_enabled": counter_enabled,
        "page_revision": page_revision,
        "position": position,
        "fallback": fallback,
        "counter_source": counter_source,
        "site_id": site_id,
        "endpoint": endpoint,
        "include": include,
        "exclude": exclude,
        "sidebar_selectors": sidebar,
        "main_selectors": main,
    }


def load_aggregate(  # ruff: ignore[too-many-branches]
    asset_root: str | Path,
    configured: Any,
    *,
    expected_site_id: str,
    expected_page_revision: str = "",
    return_metadata: bool = False,
):
    """
    Load the current bounded V3 aggregate from a packaged asset root.

    ``configured`` is a root-relative logical asset path such as
    ``/page-feedback-aggregate.json``. It is resolved only beneath ``asset_root``
    (the caller-provided asset root), never against the documentation source
    tree or host filesystem. Missing/blank means unknown.

    ``page.feedback-aggregate.v3`` is the only supported aggregate contract.
    Every page row carries total score/count plus explicit positive, negative,
    and neutral counts, so the UI never infers a distribution from an ambiguous
    score. ``complete=true`` is valid only for a full reviewed snapshot; then a
    page absent from ``pages`` is an authoritative zero rather than unknown.
    """
    try:
        expected_site_id = normalize_site_id(expected_site_id)
    except ValueError as exc:
        raise FeedbackConfigError(
            "expected feedback aggregate site_id is invalid",
        ) from exc
    try:
        expected_page_revision = normalize_page_revision(expected_page_revision)
    except ValueError as exc:
        raise FeedbackConfigError(
            "expected feedback aggregate page_revision is invalid"
        ) from exc
    raw = _plain_string(configured or "", name="feedback_aggregate_file", maximum=512)
    if not raw:
        empty = {}
        metadata = {"contract": "", "complete": False}
        return (empty, metadata) if return_metadata else empty
    if (
        not raw.startswith("/")
        or raw.startswith("//")
        or "?" in raw
        or "#" in raw
        or "\\" in raw
    ):
        raise FeedbackConfigError(
            "feedback_aggregate_file must be a root-relative packaged asset path",
        )
    parts = raw[1:].split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise FeedbackConfigError(
            "feedback_aggregate_file contains an unsafe asset path",
        )
    root = Path(asset_root).resolve()
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FeedbackConfigError(
            "feedback_aggregate_file must stay inside the packaged feedback asset root",
        ) from exc
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FeedbackConfigError(
            f"Unable to read feedback aggregate: {exc}",
        ) from exc
    if len(data) > MAX_AGGREGATE_BYTES:
        raise FeedbackConfigError(
            "feedback aggregate exceeds 2 MiB",
        )
    try:
        payload = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except FeedbackConfigError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise FeedbackConfigError(
            "feedback aggregate is not valid UTF-8 JSON",
        ) from exc
    if not isinstance(payload, dict) or payload.get("contract") != AGGREGATE_CONTRACT:
        raise FeedbackConfigError(
            f"feedback aggregate contract must be {AGGREGATE_CONTRACT}"
        )
    unknown = sorted(
        set(payload) - {"contract", "site_id", "pages", "page_revision", "complete"}
    )
    if unknown:
        raise FeedbackConfigError(
            "feedback aggregate contains unsupported field(s): " + ", ".join(unknown)
        )

    if "page_revision" in payload:
        raw_revision = payload.get("page_revision")
        try:
            aggregate_revision = normalize_page_revision(raw_revision)
        except ValueError as exc:
            raise FeedbackConfigError(
                "feedback aggregate page_revision is invalid",
            ) from exc
        if not aggregate_revision or aggregate_revision != raw_revision:
            raise FeedbackConfigError(
                "feedback aggregate page_revision must be non-empty and canonical",
            )
        if not expected_page_revision:
            raise FeedbackConfigError(
                "revision-scoped feedback aggregate requires feedback_page_revision",
            )
        if aggregate_revision != expected_page_revision:
            raise FeedbackConfigError(
                "feedback aggregate page_revision does not match feedback_page_revision",
            )
    elif expected_page_revision:
        raise FeedbackConfigError(
            "revisioned page feedback requires an aggregate pinned to feedback_page_revision",
        )

    complete = payload.get("complete", False)
    if not isinstance(complete, bool):
        raise FeedbackConfigError(
            "feedback aggregate complete must be a boolean",
        )
    if payload.get("site_id") != expected_site_id:
        raise FeedbackConfigError(
            "feedback aggregate site_id does not match feedback_site_id",
        )
    pages = payload.get("pages")
    _len = len(pages) > 200_000  # ruff: ignore[magic-value-comparison]
    if not isinstance(pages, dict) or _len:
        raise FeedbackConfigError(
            "feedback aggregate pages must be a bounded object",
        )

    distribution_fields = {"positive_count", "negative_count", "neutral_count"}
    allowed_page = {"count", "score"} | distribution_fields
    result: dict[str, dict[str, int]] = {}
    for page_id, value in pages.items():
        if not isinstance(value, dict):
            raise FeedbackConfigError(
                "feedback aggregate contains an invalid page entry",
            )
        try:
            canonical_page_id = normalize_page_id(page_id)
        except ValueError as exc:
            raise FeedbackConfigError(
                "feedback aggregate contains an invalid page_id",
            ) from exc
        if canonical_page_id != page_id:
            raise FeedbackConfigError(
                "feedback aggregate page_id must already be canonicalized",
            )
        unknown_page = sorted(set(value) - allowed_page)
        if unknown_page:
            raise FeedbackConfigError(
                "feedback aggregate page entry contains unsupported field(s): "
                + ", ".join(unknown_page)
            )
        count = value.get("count")
        score = value.get("score")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > 9_007_199_254_740_991  # ruff: ignore[magic-value-comparison]
        ):
            raise FeedbackConfigError(
                "feedback aggregate count must be a JavaScript-safe non-negative integer"
            )
        if (
            isinstance(score, bool)
            or not isinstance(score, int)  # lint
            or abs(score)  # lint
            > 9_007_199_254_740_991  # ruff: ignore[magic-value-comparison]
        ):
            raise FeedbackConfigError(
                "feedback aggregate score must be a JavaScript-safe integer"
            )
        if abs(score) > count * 5:
            raise FeedbackConfigError(
                "feedback aggregate score is impossible for its rating count",
            )
        counts: dict[str, int] = {}
        for field in sorted(distribution_fields):
            item = value.get(field)
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                or item > 9_007_199_254_740_991  # ruff: ignore[magic-value-comparison]
            ):
                raise FeedbackConfigError(
                    f"feedback aggregate {field} must be a JavaScript-safe non-negative integer"
                )
            counts[field] = item
        if sum(counts.values()) != count:
            raise FeedbackConfigError(
                "feedback aggregate sign counts must sum exactly to count"
            )
        min_score = counts["positive_count"] - 5 * counts["negative_count"]
        max_score = 5 * counts["positive_count"] - counts["negative_count"]
        if score < min_score or score > max_score:
            raise FeedbackConfigError(
                "feedback aggregate score is impossible for its sign distribution"
            )
        result[page_id] = {"count": count, "score": score, **counts}
    metadata = {"contract": AGGREGATE_CONTRACT, "complete": complete}
    return (result, metadata) if return_metadata else result
