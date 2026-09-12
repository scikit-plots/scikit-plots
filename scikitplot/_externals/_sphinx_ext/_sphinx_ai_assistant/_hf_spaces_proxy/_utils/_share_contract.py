"""
Security contract for server-backed conversation shares.

The browser is untrusted.  Share requests carry structured conversation data and
an allowlisted representation id; callers never choose response MIME types or
submit rendered HTML for the server to host.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import math
import re
import secrets
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SHARE_SCHEMA_VERSION = "2.1"
SHARE_ACCEPTED_SCHEMA_VERSIONS = frozenset({"2.0", "2.1"})
SHARE_FORMATS: dict[str, tuple[str, str]] = {
    "html": ("text/html; charset=utf-8", ".html"),
    "json": ("application/json; charset=utf-8", ".json"),
    "txt": ("text/plain; charset=utf-8", ".txt"),
    "yaml": ("application/yaml", ".yaml"),
    "toml": ("application/toml", ".toml"),
}


def share_artifact_filename(fmt: str) -> str:
    """
    Return the stable human-facing filename for a Global Share artifact.

    The public read capability is intentionally excluded from the filename.
    """
    fmt = validate_share_format(fmt)
    _mime, ext = SHARE_FORMATS[fmt]
    return f"ai-conversation-global-share-{fmt}{ext}"


MAX_SHARE_RECORDS = 1000
MAX_SHARE_TEXT_CHARS = 200_000
MAX_SHARE_METADATA_CHARS = 2048
MAX_SHARE_RESOURCES_PER_MESSAGE = 512
MAX_SHARE_RESOURCES_TOTAL = 4096
MAX_SAFE_INTEGER = 9_007_199_254_740_991

_SHARE_ID_RE = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
)


class ShareValidationError(ValueError):
    """Raised when an untrusted share snapshot violates the public contract."""


def _bounded_string(
    value: Any, *, limit: int, field: str, nullable: bool = True
) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ShareValidationError(
            f"{field} must be a string" + (" or null" if nullable else "")
        )
    if len(value) > limit:
        raise ShareValidationError(f"{field} is too long")
    return value


def _bounded_int(value: Any, *, field: str, nullable: bool = True) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShareValidationError(
            f"{field} must be an integer" + (" or null" if nullable else "")
        )
    return value


def _safe_scalar(value: Any, *, field: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_SHARE_METADATA_CHARS:
            raise ShareValidationError(f"{field} is too long")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ShareValidationError(f"{field} must be a finite primitive value")


def sanitize_share_page_url(value: Any) -> str:
    """Return an HTTP(S) source URL without credentials, query, or fragment."""
    if not isinstance(value, str) or not value:
        return ""
    if len(value) > 8192:  # ruff: ignore[magic-value-comparison]
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return ""
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", "", ""))


_RESOURCE_KINDS = frozenset(
    {
        "text",
        "image",
        "vector_image",
        "audio",
        "video",
        "data",
        "file",
        "replay",
        "page",
        "pdf",
        "archive",
    }
)
_RESOURCE_DELIVERIES = frozenset({"context", "raw", "not_sent"})
_RESOURCE_INTENTS = frozenset({"", "auto", "raw", "extract", "context"})


def _resource_count(
    value: Any, *, field: str, maximum: int = MAX_SHARE_RESOURCES_PER_MESSAGE
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ShareValidationError(
            f"{field} must be an integer between 0 and {maximum}"
        )
    return value


def _resource_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ShareValidationError(f"{field} must be a boolean")
    return value


def _safe_relative_path(value: Any, *, field: str) -> str:
    text = _bounded_string(value, limit=1024, field=field) or ""
    if not text:
        return ""
    text = text.replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return ""
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if any(
        part == ".."  # lint
        or any(ord(ch) < 32 for ch in part)  # ruff: ignore[magic-value-comparison]
        for part in parts
    ):
        return ""
    return "/".join(parts)


def _safe_resource_name(value: Any, *, field: str, limit: int = 512) -> str:
    text = _bounded_string(value, limit=limit, field=field, nullable=False) or ""
    return (
        "".join(
            ch  # lint
            for ch in text  # lint
            if ord(ch) >= 32  # ruff: ignore[magic-value-comparison]
        )
        .replace("/", "_")
        .replace("\\", "_")[:limit]
    )


def _canonical_resource_item(
    raw: Any, record_index: int, item_index: int, *, allow_source_urls: bool
) -> dict[str, Any]:
    prefix = f"records[{record_index}].resources.items[{item_index}]"
    if not isinstance(raw, dict):
        raise ShareValidationError(f"{prefix} must be an object")
    kind = _bounded_string(
        raw.get("kind"), limit=32, field=f"{prefix}.kind", nullable=False
    )
    if kind not in _RESOURCE_KINDS:
        raise ShareValidationError(f"{prefix}.kind is not allowed")
    delivery = _bounded_string(
        raw.get("delivery"), limit=32, field=f"{prefix}.delivery", nullable=False
    )
    if delivery not in _RESOURCE_DELIVERIES:
        raise ShareValidationError(f"{prefix}.delivery is not allowed")
    intent = (
        _bounded_string(raw.get("intent"), limit=32, field=f"{prefix}.intent") or ""
    )
    if intent not in _RESOURCE_INTENTS:
        raise ShareValidationError(f"{prefix}.intent is not allowed")
    size = _bounded_int(raw.get("size"), field=f"{prefix}.size", nullable=False)
    line_count = _bounded_int(
        raw.get("lineCount"), field=f"{prefix}.lineCount", nullable=False
    )
    if size is None or size < 0 or size > MAX_SAFE_INTEGER:
        raise ShareValidationError(f"{prefix}.size is out of range")
    if (
        line_count is None  # lint
        or line_count < 0  # lint
        or line_count > 1_000_000  # ruff: ignore[magic-value-comparison]
    ):
        raise ShareValidationError(f"{prefix}.lineCount is out of range")
    badge = (
        _bounded_string(
            raw.get("badge"), limit=12, field=f"{prefix}.badge", nullable=False
        )
        or "FILE"
    )
    badge = re.sub(r"[^A-Za-z0-9+._-]", "", badge).upper()[:12] or "FILE"
    modality = (
        _bounded_string(raw.get("modality"), limit=32, field=f"{prefix}.modality")
        or kind
    )
    modality = re.sub(r"[^A-Za-z0-9_-]", "", modality)[:32]
    source_kind = (
        _bounded_string(raw.get("sourceKind"), limit=24, field=f"{prefix}.sourceKind")
        or ""
    )
    source_kind = re.sub(r"[^A-Za-z0-9_-]", "", source_kind)[:24]
    archive_name = (
        _bounded_string(
            raw.get("archiveName"), limit=512, field=f"{prefix}.archiveName"
        )
        or ""
    )
    if archive_name:
        archive_name = _safe_resource_name(archive_name, field=f"{prefix}.archiveName")
    item = {
        "name": _safe_resource_name(raw.get("name"), field=f"{prefix}.name"),
        "badge": badge,
        "kind": kind,
        "size": size,
        "lineCount": line_count,
        "included": _resource_bool(raw.get("included"), field=f"{prefix}.included"),
        "localOnly": _resource_bool(raw.get("localOnly"), field=f"{prefix}.localOnly"),
        "delivery": delivery,
        "modality": modality,
        "intent": intent,
        "replay": _resource_bool(raw.get("replay"), field=f"{prefix}.replay"),
        "boundedExcerpt": _resource_bool(
            raw.get("boundedExcerpt"), field=f"{prefix}.boundedExcerpt"
        ),
        "status": (
            _bounded_string(raw.get("status"), limit=96, field=f"{prefix}.status") or ""
        ),
        "type": (
            _bounded_string(raw.get("type"), limit=120, field=f"{prefix}.type") or ""
        ),
        "relativePath": _safe_relative_path(
            raw.get("relativePath"), field=f"{prefix}.relativePath"
        ),
        "sourceKind": source_kind,
        "archiveName": archive_name,
    }
    if kind == "page":
        role = (
            _bounded_string(
                raw.get("contextRole"), limit=16, field=f"{prefix}.contextRole"
            )
            or "pinned"
        )
        item["contextRole"] = "current" if role == "current" else "pinned"
        item["sourceUrl"] = (
            sanitize_share_page_url(raw.get("sourceUrl")) if allow_source_urls else ""
        )
    return item


def _resource_aggregates(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "includedCount": sum(1 for item in items if item["included"]),
        "localOnlyCount": sum(1 for item in items if item["localOnly"]),
        "contextCount": sum(1 for item in items if item["delivery"] == "context"),
        "rawCount": sum(1 for item in items if item["delivery"] == "raw"),
        "notSentCount": sum(1 for item in items if item["delivery"] == "not_sent"),
        "pageCount": sum(1 for item in items if item["kind"] == "page"),
        "replayCount": sum(
            1 for item in items if item["kind"] == "replay" or item["replay"]
        ),
        "totalBytes": min(MAX_SAFE_INTEGER, sum(item["size"] for item in items)),
    }


def _canonical_resources(
    raw: Any, record_index: int, *, allow_source_urls: bool
) -> dict[str, Any]:
    prefix = f"records[{record_index}].resources"
    if not isinstance(raw, dict):
        raise ShareValidationError(f"{prefix} must be an object")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise ShareValidationError(f"{prefix}.items must be an array")
    if len(raw_items) > MAX_SHARE_RESOURCES_PER_MESSAGE:
        raise ShareValidationError(f"{prefix}.items contains too many resources")
    items = [
        _canonical_resource_item(
            item, record_index, i, allow_source_urls=allow_source_urls
        )
        for i, item in enumerate(raw_items)
    ]
    derived = _resource_aggregates(items)
    total = _resource_count(
        raw.get("totalCount", len(items)), field=f"{prefix}.totalCount"
    )
    if total < len(items):
        raise ShareValidationError(f"{prefix}.totalCount cannot be smaller than items")

    def merged(name: str) -> int:
        value = _resource_count(raw.get(name, derived[name]), field=f"{prefix}.{name}")
        return max(derived[name], min(total, value))

    total_bytes = _bounded_int(
        raw.get("totalBytes", derived["totalBytes"]),
        field=f"{prefix}.totalBytes",
        nullable=False,
    )
    if total_bytes is None or total_bytes < 0 or total_bytes > MAX_SAFE_INTEGER:
        raise ShareValidationError(f"{prefix}.totalBytes is out of range")
    omitted = _resource_count(
        raw.get("omittedCount", max(0, total - len(items))),
        field=f"{prefix}.omittedCount",
    )
    omitted = max(omitted, total - len(items))
    complete = raw.get("complete")
    if not isinstance(complete, bool):
        raise ShareValidationError(f"{prefix}.complete must be a boolean")
    version = _resource_count(
        raw.get("version", 2), field=f"{prefix}.version", maximum=2
    )
    if version != 2:  # ruff: ignore[magic-value-comparison]
        raise ShareValidationError(f"{prefix}.version must be 2")
    return {
        "version": 2,
        "totalCount": total,
        "includedCount": merged("includedCount"),
        "localOnlyCount": merged("localOnlyCount"),
        "contextCount": merged("contextCount"),
        "rawCount": merged("rawCount"),
        "notSentCount": merged("notSentCount"),
        "pageCount": merged("pageCount"),
        "replayCount": merged("replayCount"),
        "totalBytes": max(derived["totalBytes"], total_bytes),
        "itemCount": len(items),
        "omittedCount": omitted,
        "complete": complete and omitted == 0 and total == len(items),
        "items": items,
    }


def _canonical_record(
    raw: Any, index: int, safe_page: str | None, session_id: str | None
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ShareValidationError(f"records[{index}] must be an object")
    role = raw.get("role")
    if role not in {"user", "assistant", "error"}:
        raise ShareValidationError(f"records[{index}].role is not allowed")
    text = _bounded_string(
        raw.get("text"),
        limit=MAX_SHARE_TEXT_CHARS,
        field=f"records[{index}].text",
        nullable=False,
    )
    turn_index = _bounded_int(
        raw.get("turn_index"), field=f"records[{index}].turn_index", nullable=False
    )
    message_index = _bounded_int(
        raw.get("message_index"),
        field=f"records[{index}].message_index",
        nullable=False,
    )
    ts = _bounded_int(raw.get("ts"), field=f"records[{index}].ts")
    ts_iso = _bounded_string(
        raw.get("ts_iso"), limit=128, field=f"records[{index}].ts_iso"
    )
    if raw.get("resources") is not None and role != "user":
        raise ShareValidationError(
            f"records[{index}].resources is allowed only for user messages"
        )

    return {
        "turn_index": turn_index,
        "message_index": message_index,
        "role": role,
        "text": text,
        "ts": ts,
        "ts_iso": ts_iso,
        "model_id": _bounded_string(
            raw.get("model_id"),
            limit=MAX_SHARE_METADATA_CHARS,
            field=f"records[{index}].model_id",
        ),
        "model_provider": _bounded_string(
            raw.get("model_provider"),
            limit=MAX_SHARE_METADATA_CHARS,
            field=f"records[{index}].model_provider",
        ),
        "model_name": _bounded_string(
            raw.get("model_name"),
            limit=MAX_SHARE_METADATA_CHARS,
            field=f"records[{index}].model_name",
        ),
        "feedback_rating_value": _safe_scalar(
            raw.get("feedback_rating_value"),
            field=f"records[{index}].feedback_rating_value",
        ),
        "feedback_rating_label": _bounded_string(
            raw.get("feedback_rating_label"),
            limit=MAX_SHARE_METADATA_CHARS,
            field=f"records[{index}].feedback_rating_label",
        ),
        "feedback_message": _bounded_string(
            raw.get("feedback_message"),
            limit=MAX_SHARE_TEXT_CHARS,
            field=f"records[{index}].feedback_message",
        ),
        "resources": (
            _canonical_resources(
                raw.get("resources"), index, allow_source_urls=bool(safe_page)
            )
            if raw.get("resources") is not None and role == "user"
            else None
        ),
        # Never trust duplicated per-record identity/page claims from the client;
        # bind them to the canonical session values reconstructed above.
        "session_id": session_id,
        "page_url": safe_page,
    }


def _build_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in records:
        if row["role"] == "user":
            current = {
                "turn_index": row["turn_index"],
                "user": {
                    "text": row["text"],
                    "ts": row["ts"],
                    "ts_iso": row["ts_iso"],
                    "resources": row.get("resources"),
                },
                "assistant": None,
            }
            turns.append(current)
        elif (
            row["role"] == "assistant"
            and current is not None
            and current["assistant"] is None
        ):
            current["assistant"] = {
                "text": row["text"],
                "ts": row["ts"],
                "ts_iso": row["ts_iso"],
                "model_id": row["model_id"],
                "model_provider": row["model_provider"],
                "model_name": row["model_name"],
                "feedback_rating_value": row["feedback_rating_value"],
                "feedback_rating_label": row["feedback_rating_label"],
                "feedback_message": row["feedback_message"],
            }
    return turns


def canonicalize_share_snapshot(raw: Any) -> dict[str, Any]:
    """Validate and reconstruct the allowlisted canonical schema-v2.1 Share snapshot."""
    if not isinstance(raw, dict):
        raise ShareValidationError("snapshot must be an object")
    if raw.get("schema_version") not in SHARE_ACCEPTED_SCHEMA_VERSIONS:
        raise ShareValidationError(
            "snapshot.schema_version must be one of: '2.0', '2.1'"
        )

    raw_session = raw.get("session")
    if not isinstance(raw_session, dict):
        raise ShareValidationError("snapshot.session must be an object")
    session_id = _bounded_string(raw_session.get("id"), limit=256, field="session.id")
    safe_page = sanitize_share_page_url(raw_session.get("page_url")) or None
    session = {
        "id": session_id,
        "page_url": safe_page,
        "page_title": _bounded_string(
            raw_session.get("page_title"), limit=2048, field="session.page_title"
        ),
        "assistant_name": (
            _bounded_string(
                raw_session.get("assistant_name"),
                limit=256,
                field="session.assistant_name",
            )
            or "AI Assistant"
        ),
        "exported_at": _bounded_int(
            raw_session.get("exported_at"), field="session.exported_at"
        ),
        "exported_at_iso": _bounded_string(
            raw_session.get("exported_at_iso"),
            limit=128,
            field="session.exported_at_iso",
        ),
    }

    raw_records = raw.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ShareValidationError("snapshot.records must be a non-empty array")
    if len(raw_records) > MAX_SHARE_RECORDS:
        raise ShareValidationError("snapshot.records contains too many messages")
    records = [
        _canonical_record(row, i, safe_page, session_id)
        for i, row in enumerate(raw_records)
    ]
    if (
        sum((row.get("resources") or {}).get("itemCount", 0) for row in records)
        > MAX_SHARE_RESOURCES_TOTAL
    ):
        raise ShareValidationError(
            "snapshot.records contains too many resource metadata rows"
        )

    # Never accept caller-supplied turns/unknown root data as trusted.  Turns are
    # a derived view of validated records and are rebuilt server-side.
    return {
        "schema_version": SHARE_SCHEMA_VERSION,
        "session": session,
        "turns": _build_turns(records),
        "records": records,
    }


def validate_share_format(value: Any) -> str:
    if not isinstance(value, str) or value not in SHARE_FORMATS:
        raise ShareValidationError("format must be one of: html, json, txt, yaml, toml")
    return value


def _resource_summary(manifest: dict[str, Any] | None) -> str:
    if not manifest or not manifest.get("totalCount"):
        return ""
    return (
        f"Resources used for this question: {manifest['totalCount']}"
        f" · context {manifest['contextCount']}"
        f" · raw {manifest['rawCount']}"
        f" · not-sent {manifest['notSentCount']}"
    )


def _resource_text_lines(manifest: dict[str, Any] | None) -> list[str]:
    if not manifest or not manifest.get("totalCount"):
        return []
    lines = [f"[{_resource_summary(manifest)}]"]
    for item in manifest.get("items") or []:
        line = f"- [{item.get('badge') or 'FILE'}] {item.get('name') or 'file'}"
        if item.get("status"):
            line += f" — {item['status']}"
        lines.append(line)
    if manifest.get("omittedCount"):
        count = manifest["omittedCount"]
        lines.append(
            f"- … {count} resource metadata row{'s' if count != 1 else ''} omitted from restored state"
        )
    return lines


def _resource_html(manifest: dict[str, Any] | None) -> str:
    if not manifest or not manifest.get("totalCount"):
        return ""
    rows: list[str] = []
    for item in manifest.get("items") or []:
        status = (
            f'<span class="resource-status">{html.escape(str(item.get("status") or ""))}</span>'
            if item.get("status")
            else ""
        )
        rows.append(
            '<li class="resource-card">'
            f'<span class="resource-badge">{html.escape(str(item.get("badge") or "FILE"))}</span>'
            f'<span class="resource-name">{html.escape(str(item.get("name") or "file"))}</span>{status}</li>'
        )
    if manifest.get("omittedCount"):
        count = manifest["omittedCount"]
        rows.append(
            f'<li class="resource-card omitted">… {count} resource metadata row{"s" if count != 1 else ""} omitted</li>'
        )
    return (
        '<section class="resources"><div class="resource-summary">'
        + html.escape(_resource_summary(manifest))
        + '</div><ul class="resource-list">'
        + "".join(rows)
        + "</ul></section>"
    )


def _render_html(snapshot: dict[str, Any]) -> str:
    session = snapshot["session"]
    assistant_name = html.escape(str(session.get("assistant_name") or "AI Assistant"))
    page_title = html.escape(str(session.get("page_title") or "Shared conversation"))
    page_url = str(session.get("page_url") or "")
    source = ""
    if page_url:
        escaped_url = html.escape(page_url, quote=True)
        source = f'<p class="source">Source: <a href="{escaped_url}" rel="noopener noreferrer">{escaped_url}</a></p>'

    messages: list[str] = []
    for row in snapshot["records"]:
        role = row["role"]
        label = (
            "You"
            if role == "user"
            else ("Error" if role == "error" else assistant_name)
        )
        text = html.escape(str(row.get("text") or ""))
        cls = (
            "user" if role == "user" else ("error" if role == "error" else "assistant")
        )
        meta_parts: list[str] = []
        if row.get("ts_iso"):
            meta_parts.append(html.escape(str(row["ts_iso"])))
        if row.get("model_name"):
            meta_parts.append(html.escape(str(row["model_name"])))
        if row.get("model_provider"):
            meta_parts.append(html.escape(str(row["model_provider"])))
        if (
            row.get("feedback_rating_label")
            or row.get("feedback_rating_value") is not None
        ):
            rating = str(
                row.get("feedback_rating_label") or row.get("feedback_rating_value")
            )
            if row.get("feedback_rating_value") is not None and row.get(
                "feedback_rating_label"
            ):
                rating += f" ({row['feedback_rating_value']})"
            meta_parts.append("Rating: " + html.escape(rating))
        if row.get("feedback_message"):
            meta_parts.append("Feedback: " + html.escape(str(row["feedback_message"])))
        meta = f'<div class="meta">{" · ".join(meta_parts)}</div>' if meta_parts else ""
        resources = _resource_html(row.get("resources")) if role == "user" else ""
        messages.append(
            f'<article class="msg {cls}"><div class="role">{label}</div>'
            f"{resources}<pre>{text}</pre>{meta}</article>"
        )

    # No scripts and no remote resources.  CSP on the HTTP response is the
    # primary policy; this meta tag protects downloaded/copied representations.
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Shared AI conversation</title><style>
:root{font-family:system-ui,sans-serif;color-scheme:light dark}body{margin:0;background:Canvas;color:CanvasText}.wrap{max-width:850px;margin:auto;padding:24px}.head{border-bottom:1px solid color-mix(in srgb,CanvasText 20%,transparent);padding-bottom:16px}.source{overflow-wrap:anywhere}.source a{color:inherit}.msg{margin:18px 0;padding:14px;border:1px solid color-mix(in srgb,CanvasText 18%,transparent);border-radius:12px}.msg.user{margin-left:10%}.msg.error{border-style:dashed}.role{font-weight:700;margin-bottom:8px}.resources{margin:0 0 10px}.resource-summary{font-size:.78rem;opacity:.7}.resource-list{list-style:none;padding:0;margin:6px 0;display:grid;gap:4px}.resource-card{display:flex;gap:6px;flex-wrap:wrap;font-size:.78rem}.resource-badge{font-weight:700}.resource-status{opacity:.65}.msg pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:0}.meta{opacity:.65;font-size:.8rem;margin-top:8px}
</style></head><body><main class="wrap"><header class="head"><h1>"""
        + assistant_name
        + " — Shared conversation</h1><p>"
        + page_title
        + "</p>"
        + source
        + "</header>"
        + "".join(messages)
        + "</main></body></html>"
    )


def _render_text(snapshot: dict[str, Any]) -> str:
    lines: list[str] = []
    session = snapshot["session"]
    lines.append(
        f"{session.get('assistant_name') or 'AI Assistant'} — Shared conversation"
    )
    lines.append(f"Schema: {snapshot.get('schema_version') or SHARE_SCHEMA_VERSION}")
    if session.get("page_title"):
        lines.append(f"Page title: {session['page_title']}")
    if session.get("page_url"):
        lines.append(f"Source: {session['page_url']}")
    if session.get("exported_at_iso"):
        lines.append(f"Exported: {session['exported_at_iso']}")
    lines.append("")
    for row in snapshot["records"]:
        role = row["role"]
        label = (
            "USER" if role == "user" else ("ERROR" if role == "error" else "ASSISTANT")
        )
        meta: list[str] = []
        if row.get("ts_iso"):
            meta.append(str(row["ts_iso"]))
        if role in {"assistant", "error"} and (
            row.get("model_name") or row.get("model_id")
        ):
            model = str(row.get("model_name") or row.get("model_id"))
            if row.get("model_provider"):
                model += f" · {row['model_provider']}"
            meta.append(model)
        lines.append(f"[{label}]" + (f"  [{' · '.join(meta)}]" if meta else ""))
        if role == "user":
            lines.extend(_resource_text_lines(row.get("resources")))
        if role in {"assistant", "error"} and (
            row.get("feedback_rating_label")
            or row.get("feedback_rating_value") is not None
        ):
            bits = []
            if row.get("feedback_rating_label"):
                bits.append(str(row["feedback_rating_label"]))
            if row.get("feedback_rating_value") is not None:
                bits.append(str(row["feedback_rating_value"]))
            lines.append(f"[Rating: {' · '.join(bits)}]")
        if role in {"assistant", "error"} and row.get("feedback_message"):
            lines.append(f"[Feedback: {row['feedback_message']}]")
        lines.extend([str(row.get("text") or ""), ""])
    return "\n".join(lines).rstrip() + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return "null"
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_value(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, list):
        if not value:
            return pad + "[]"
        rows: list[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                child = _yaml_value(item, indent + 2).splitlines()
                rows.append(pad + "- " + child[0][indent + 2 :])
                rows.extend(child[1:])
            else:
                rows.append(pad + "- " + _yaml_scalar(item))
        return "\n".join(rows)
    if isinstance(value, dict):
        if not value:
            return pad + "{}"
        rows = []
        for key, item in value.items():
            qkey = json.dumps(str(key), ensure_ascii=False)
            if isinstance(item, (dict, list)):
                rows.append(f"{pad}{qkey}:\n{_yaml_value(item, indent + 2)}")
            else:
                rows.append(f"{pad}{qkey}: {_yaml_scalar(item)}")
        return "\n".join(rows)
    return pad + _yaml_scalar(value)


def _render_yaml(snapshot: dict[str, Any]) -> str:
    return _yaml_value(snapshot) + "\n"


def _toml_scalar(value: Any) -> str | None:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return str(value)
    return None


def _toml_fields(
    lines: list[str], obj: dict[str, Any], omit: set[str] | None = None
) -> None:
    skipped = omit or set()
    for key, value in obj.items():
        if key in skipped or value is None:
            continue
        rendered = _toml_scalar(value)
        if rendered is not None:
            lines.append(f"{key} = {rendered}")


def _toml_resources(
    lines: list[str], table: str, manifest: dict[str, Any] | None
) -> None:
    if not manifest or not manifest.get("totalCount"):
        return
    lines.append(f"[{table}]")
    _toml_fields(lines, manifest, {"items"})
    for item in manifest.get("items") or []:
        lines.append(f"[[{table}.items]]")
        _toml_fields(lines, item)


def _render_toml(snapshot: dict[str, Any]) -> str:
    lines = [
        "# AI Assistant conversation export",
        "# schema v2.1 semantics: omitted optional values represent null",
        f"schema_version = {json.dumps(str(snapshot.get('schema_version') or SHARE_SCHEMA_VERSION), ensure_ascii=False)}",
        "",
        "[session]",
    ]
    _toml_fields(lines, snapshot.get("session") or {})
    for turn in snapshot.get("turns") or []:
        lines.extend(["", "[[turns]]"])
        if turn.get("turn_index") is not None:
            lines.append(f"turn_index = {turn['turn_index']}")
        if isinstance(turn.get("user"), dict):
            lines.append("[turns.user]")
            _toml_fields(lines, turn["user"], {"resources"})
            _toml_resources(
                lines, "turns.user.resources", turn["user"].get("resources")
            )
        if isinstance(turn.get("assistant"), dict):
            lines.append("[turns.assistant]")
            _toml_fields(lines, turn["assistant"])
    for record in snapshot.get("records") or []:
        lines.extend(["", "[[records]]"])
        _toml_fields(lines, record, {"resources"})
        _toml_resources(lines, "records.resources", record.get("resources"))
    return "\n".join(lines) + "\n"


def render_share(snapshot: dict[str, Any], fmt: str) -> tuple[str, str, str]:
    """Render a validated snapshot using a server-owned representation."""
    fmt = validate_share_format(fmt)
    mime, ext = SHARE_FORMATS[fmt]
    if fmt == "html":
        content = _render_html(snapshot)
    elif fmt == "json":
        content = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    elif fmt == "yaml":
        content = _render_yaml(snapshot)
    elif fmt == "toml":
        content = _render_toml(snapshot)
    else:
        content = _render_text(snapshot)
    return content, mime, ext


def render_share_viewer_shell(
    read_path: str = "/v1/share/read", download_path: str = "/v1/share/download"
) -> str:
    """
    Return the fixed-path public Share viewer.

    The public read capability remains in ``location.hash`` and is sent to the
    fixed read endpoint only in a JSON request body. The shell renders all
    conversation values with DOM ``textContent`` and never injects untrusted HTML.
    """
    read_path_json = json.dumps(str(read_path), ensure_ascii=False)
    download_path_json = json.dumps(str(download_path), ensure_ascii=False)
    template = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><title>Shared AI conversation</title><style>:root{font-family:system-ui,sans-serif;color-scheme:light dark}body{margin:0;background:Canvas;color:CanvasText}.wrap{max-width:850px;margin:auto;padding:24px}.head{border-bottom:1px solid currentColor;padding-bottom:16px}.source{overflow-wrap:anywhere}.source a{color:inherit}.msg{margin:18px 0;padding:14px;border:1px solid currentColor;border-radius:12px}.msg.user{margin-left:10%}.msg.error{border-style:dashed}.role{font-weight:700;margin-bottom:8px}.meta{margin-top:8px;font-size:.82rem;opacity:.72;overflow-wrap:anywhere}.resources{margin:0 0 10px;padding:10px;border:1px solid color-mix(in srgb,currentColor 28%,transparent);border-radius:9px}.resource-summary{font-size:.8rem;opacity:.76;margin-bottom:6px}.resource-list{display:grid;gap:6px}.resource-card{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;font-size:.82rem}.resource-badge{font-size:.7rem;font-weight:700;border:1px solid currentColor;border-radius:5px;padding:1px 5px}.resource-source{color:inherit}.feedback{margin-top:8px;padding:8px 10px;border-left:3px solid currentColor;white-space:pre-wrap;overflow-wrap:anywhere}.artifact{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin:0 0 18px;padding:10px 12px;border:1px solid currentColor;border-radius:10px}.artifact-name{font-size:.82rem;opacity:.8;overflow-wrap:anywhere}.download{font:inherit;padding:7px 11px;border:1px solid currentColor;border-radius:8px;background:Canvas;color:CanvasText;cursor:pointer}.download[disabled]{opacity:.55;cursor:wait}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:0}.error-note{border:1px dashed currentColor;padding:14px;border-radius:12px}</style></head><body><main id="app" class="wrap"><p>Loading shared conversation…</p></main><script>(()=>{'use strict';const app=document.getElementById('app');const fail=(m)=>{app.replaceChildren();const p=document.createElement('p');p.className='error-note';p.textContent=m;app.appendChild(p);};let raw=(location.hash||'').slice(1);if(raw.startsWith('share='))raw=raw.slice(6);try{raw=decodeURIComponent(raw)}catch(_e){}if(!/^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.test(raw)){fail('This Share link is invalid or incomplete.');return;}const readJson=async(r)=>{const max=4*1024*1024;const h=r.headers&&r.headers.get?r.headers.get('content-length'):null;if(h!=null&&String(h).trim()!==''){if(!/^\d+$/.test(String(h).trim())||Number(h)>max)throw new Error('Share response is too large.');}if(!r.body||typeof r.body.getReader!=='function'||typeof TextDecoder!=='function')throw new Error('Bounded Share reader unavailable.');const rd=r.body.getReader(),dec=new TextDecoder(),parts=[];let n=0;try{for(;;){const x=await rd.read();if(x.done)break;const v=x.value||new Uint8Array(0);n+=Number(v.byteLength||v.length||0);if(n>max)throw new Error('Share response is too large.');parts.push(dec.decode(v,{stream:true}));}parts.push(dec.decode());}catch(e){try{await rd.cancel()}catch(_e){}throw e;}finally{try{rd.releaseLock()}catch(_e){}}return JSON.parse(parts.join(''));};const readDownload=async(r)=>{const max=8*1024*1024;const h=r.headers&&r.headers.get?r.headers.get('content-length'):null;if(h!=null&&String(h).trim()!==''){if(!/^\d+$/.test(String(h).trim())||Number(h)>max)throw new Error('Shared artifact is too large to download.');}if(!r.body||typeof r.body.getReader!=='function'||typeof TextDecoder!=='function')throw new Error('Bounded Share download reader unavailable.');const rd=r.body.getReader(),dec=new TextDecoder(),parts=[];let n=0;try{for(;;){const x=await rd.read();if(x.done)break;const v=x.value||new Uint8Array(0);n+=Number(v.byteLength||v.length||0);if(n>max)throw new Error('Shared artifact is too large to download.');parts.push(dec.decode(v,{stream:true}));}parts.push(dec.decode());}catch(e){try{await rd.cancel()}catch(_e){}throw e;}finally{try{rd.releaseLock()}catch(_e){}}return parts.join('');};fetch(__READ_PATH__,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({shareId:raw}),cache:'no-store',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer'}).then(async r=>{if(!r.ok){throw new Error(r.status===410?'This Share has expired.':r.status===404?'This Share is unavailable.':'Could not load this Share.');}return await readJson(r);}).then(data=>{app.replaceChildren();const fmt=String(data.format||'txt').toLowerCase();const ext={json:'.json',html:'.html',txt:'.txt',yaml:'.yaml',toml:'.toml'}[fmt]||'.txt';const fallback='ai-conversation-global-share-'+fmt+ext;const requested=String(data.filename||'');const filename=/^ai-conversation-global-share-(?:json\.json|html\.html|txt\.txt|yaml\.yaml|toml\.toml)$/.test(requested)?requested:fallback;const bar=document.createElement('section');bar.className='artifact';bar.setAttribute('aria-label','Shared artifact');const artifactName=document.createElement('span');artifactName.className='artifact-name';artifactName.textContent='Global Share · '+fmt.toUpperCase()+' · '+filename;const download=document.createElement('button');download.type='button';download.className='download';download.textContent='Download '+fmt.toUpperCase();download.addEventListener('click',async()=>{if(download.disabled)return;download.disabled=true;const prior=download.textContent;download.textContent='Downloading…';try{const r=await fetch(__DOWNLOAD_PATH__,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({shareId:raw}),cache:'no-store',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer'});if(!r.ok)throw new Error(r.status===410?'This Share has expired.':r.status===404?'This Share is unavailable.':'Could not download this Share.');const text=await readDownload(r);const type=String((r.headers&&r.headers.get&&r.headers.get('content-type'))||data.mimeType||'text/plain;charset=utf-8');const blob=new Blob([text],{type});const objectUrl=URL.createObjectURL(blob);try{const a=document.createElement('a');a.href=objectUrl;a.download=filename;a.rel='noopener';a.style.display='none';document.body.appendChild(a);a.click();a.remove();}finally{setTimeout(()=>URL.revokeObjectURL(objectUrl),0);}}catch(e){fail(e&&e.message?e.message:'Could not download this Share.');return;}finally{download.disabled=false;download.textContent=prior;}});bar.append(artifactName,download);app.appendChild(bar);if(data.format==='html'&&data.snapshot&&data.snapshot.session&&Array.isArray(data.snapshot.records)){const snap=data.snapshot;const h=document.createElement('header');h.className='head';const h1=document.createElement('h1');h1.textContent=(snap.session.assistant_name||'AI Assistant')+' — Shared conversation';h.appendChild(h1);if(snap.session.page_title){const p=document.createElement('p');p.textContent=snap.session.page_title;h.appendChild(p);}if(snap.session.page_url){const p=document.createElement('p');p.className='source';p.append('Source: ');const a=document.createElement('a');a.href=snap.session.page_url;a.rel='noopener noreferrer';a.referrerPolicy='no-referrer';a.textContent=snap.session.page_url;p.appendChild(a);h.appendChild(p);}app.appendChild(h);const appendMeta=(article,row)=>{const bits=[];if(row.ts_iso)bits.push(String(row.ts_iso));const model=String(row.model_name||row.model_id||'');const provider=String(row.model_provider||'');if(model)bits.push(provider?model+' · '+provider:model);else if(provider)bits.push(provider);if(row.feedback_rating_label!=null||row.feedback_rating_value!=null){let rating='Rating: '+String(row.feedback_rating_label||'rated');if(row.feedback_rating_value!=null)rating+=' ('+String(row.feedback_rating_value)+')';bits.push(rating);}if(bits.length){const meta=document.createElement('div');meta.className='meta';meta.textContent=bits.join(' · ');article.appendChild(meta);}if(row.feedback_message){const note=document.createElement('div');note.className='feedback';note.textContent='Feedback: '+String(row.feedback_message);article.appendChild(note);}};const appendResources=(article,manifest)=>{if(!manifest||!Array.isArray(manifest.items)||Number(manifest.totalCount||0)<=0)return;const section=document.createElement('section');section.className='resources';section.setAttribute('aria-label','Resources used for this question');const summary=document.createElement('div');summary.className='resource-summary';summary.textContent='Resources: '+String(manifest.totalCount||manifest.items.length)+' · context '+String(manifest.contextCount||0)+' · raw '+String(manifest.rawCount||0)+' · not-sent '+String(manifest.notSentCount||0);section.appendChild(summary);const list=document.createElement('div');list.className='resource-list';for(const item of manifest.items){const card=document.createElement('div');card.className='resource-card';const badge=document.createElement('span');badge.className='resource-badge';badge.textContent=String(item.badge||'FILE');const label=document.createElement('span');label.textContent=String(item.name||'Resource')+(item.status?' — '+String(item.status):'');card.append(badge,label);const source=String(item.sourceUrl||'');if(/^https?:\/\//i.test(source)){const a=document.createElement('a');a.className='resource-source';a.href=source;a.rel='noopener noreferrer';a.referrerPolicy='no-referrer';a.textContent='Source';card.appendChild(a);}list.appendChild(card);}section.appendChild(list);article.appendChild(section);};for(const row of snap.records){const article=document.createElement('article');article.className='msg '+(row.role==='user'?'user':row.role==='error'?'error':'assistant');const role=document.createElement('div');role.className='role';role.textContent=row.role==='user'?'You':row.role==='error'?'Error':(snap.session.assistant_name||'AI Assistant');article.appendChild(role);if(row.role==='user')appendResources(article,row.resources);const pre=document.createElement('pre');pre.textContent=String(row.text||'');article.appendChild(pre);appendMeta(article,row);app.appendChild(article);}return;}const pre=document.createElement('pre');pre.textContent=String(data.content||'');app.appendChild(pre);}).catch(e=>fail(e&&e.message?e.message:'Could not load this Share.'));})();</script></body></html>"""
    return template.replace("__READ_PATH__", read_path_json).replace(
        "__DOWNLOAD_PATH__", download_path_json
    )


def generate_edit_token() -> str:
    return secrets.token_urlsafe(32)


def hash_edit_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_edit_token(token: str, expected_hash: str) -> bool:
    if not token or not expected_hash:
        return False
    candidate = hash_edit_token(token)
    return hmac.compare_digest(candidate, expected_hash)


def valid_share_id(value: str) -> bool:
    return bool(_SHARE_ID_RE.fullmatch(value or ""))
