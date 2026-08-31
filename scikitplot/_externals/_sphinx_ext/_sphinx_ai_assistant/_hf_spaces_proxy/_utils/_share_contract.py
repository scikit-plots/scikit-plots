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

SHARE_SCHEMA_VERSION = "2.0"
SHARE_FORMATS: dict[str, tuple[str, str]] = {
    "html": ("text/html; charset=utf-8", ".html"),
    "json": ("application/json; charset=utf-8", ".json"),
    "txt": ("text/plain; charset=utf-8", ".txt"),
    "yaml": ("application/yaml", ".yaml"),
    "toml": ("application/toml", ".toml"),
}
MAX_SHARE_RECORDS = 1000
MAX_SHARE_TEXT_CHARS = 200_000
MAX_SHARE_METADATA_CHARS = 2048

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


def _canonical_record(
    raw: Any, index: int, safe_page: str, session_id: str
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
                "user": {"text": row["text"], "ts": row["ts"], "ts_iso": row["ts_iso"]},
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
    """Validate and reconstruct the allowlisted schema-v2 share snapshot."""
    if not isinstance(raw, dict):
        raise ShareValidationError("snapshot must be an object")
    if raw.get("schema_version") != SHARE_SCHEMA_VERSION:
        raise ShareValidationError("snapshot.schema_version must be '2.0'")

    raw_session = raw.get("session")
    if not isinstance(raw_session, dict):
        raise ShareValidationError("snapshot.session must be an object")
    session_id = (
        _bounded_string(raw_session.get("id"), limit=256, field="session.id") or ""
    )
    safe_page = sanitize_share_page_url(raw_session.get("page_url"))
    session = {
        "id": session_id,
        "page_url": safe_page,
        "page_title": (
            _bounded_string(
                raw_session.get("page_title"), limit=2048, field="session.page_title"
            )
            or ""
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
        if row.get("model_name"):
            meta_parts.append(html.escape(str(row["model_name"])))
        if row.get("model_provider"):
            meta_parts.append(html.escape(str(row["model_provider"])))
        meta = f'<div class="meta">{" · ".join(meta_parts)}</div>' if meta_parts else ""
        messages.append(
            f'<article class="msg {cls}"><div class="role">{label}</div>'
            f"<pre>{text}</pre>{meta}</article>"
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
:root{font-family:system-ui,sans-serif;color-scheme:light dark}body{margin:0;background:Canvas;color:CanvasText}.wrap{max-width:850px;margin:auto;padding:24px}.head{border-bottom:1px solid color-mix(in srgb,CanvasText 20%,transparent);padding-bottom:16px}.source{overflow-wrap:anywhere}.source a{color:inherit}.msg{margin:18px 0;padding:14px;border:1px solid color-mix(in srgb,CanvasText 18%,transparent);border-radius:12px}.msg.user{margin-left:10%}.msg.error{border-style:dashed}.role{font-weight:700;margin-bottom:8px}.msg pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:0}.meta{opacity:.65;font-size:.8rem;margin-top:8px}
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
    if session.get("page_title"):
        lines.append(str(session["page_title"]))
    if session.get("page_url"):
        lines.append(f"Source: {session['page_url']}")
    lines.append("")
    for row in snapshot["records"]:
        role = row["role"]
        label = (
            "USER" if role == "user" else ("ERROR" if role == "error" else "ASSISTANT")
        )
        lines.extend([f"[{label}]", str(row.get("text") or ""), ""])
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


def _toml_fields(lines: list[str], obj: dict[str, Any]) -> None:
    for key, value in obj.items():
        if value is None:
            continue
        rendered = _toml_scalar(value)
        if rendered is not None:
            lines.append(f"{key} = {rendered}")


def _render_toml(snapshot: dict[str, Any]) -> str:
    lines = [
        "# AI Assistant conversation export",
        "# schema v2 semantics: omitted optional values represent null",
        f"schema_version = {json.dumps(str(snapshot.get('schema_version') or '2.0'), ensure_ascii=False)}",
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
            _toml_fields(lines, turn["user"])
        if isinstance(turn.get("assistant"), dict):
            lines.append("[turns.assistant]")
            _toml_fields(lines, turn["assistant"])
    for record in snapshot.get("records") or []:
        lines.extend(["", "[[records]]"])
        _toml_fields(lines, record)
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


def render_share_viewer_shell(read_path: str = "/v1/share/read") -> str:
    """
    Return the fixed-path public Share viewer.

    The public read capability remains in ``location.hash`` and is sent to the
    fixed read endpoint only in a JSON request body. The shell renders all
    conversation values with DOM ``textContent`` and never injects untrusted HTML.
    """
    read_path_json = json.dumps(str(read_path), ensure_ascii=False)
    template = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><title>Shared AI conversation</title><style>:root{font-family:system-ui,sans-serif;color-scheme:light dark}body{margin:0;background:Canvas;color:CanvasText}.wrap{max-width:850px;margin:auto;padding:24px}.head{border-bottom:1px solid currentColor;padding-bottom:16px}.source{overflow-wrap:anywhere}.source a{color:inherit}.msg{margin:18px 0;padding:14px;border:1px solid currentColor;border-radius:12px}.msg.user{margin-left:10%}.msg.error{border-style:dashed}.role{font-weight:700;margin-bottom:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:0}.error-note{border:1px dashed currentColor;padding:14px;border-radius:12px}</style></head><body><main id="app" class="wrap"><p>Loading shared conversation…</p></main><script>(()=>{'use strict';const app=document.getElementById('app');const fail=(m)=>{app.replaceChildren();const p=document.createElement('p');p.className='error-note';p.textContent=m;app.appendChild(p);};let raw=(location.hash||'').slice(1);if(raw.startsWith('share='))raw=raw.slice(6);try{raw=decodeURIComponent(raw)}catch(_e){}if(!/^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.test(raw)){fail('This Share link is invalid or incomplete.');return;}const readJson=async(r)=>{const max=4*1024*1024;const h=r.headers&&r.headers.get?r.headers.get('content-length'):null;if(h!=null&&String(h).trim()!==''){if(!/^\d+$/.test(String(h).trim())||Number(h)>max)throw new Error('Share response is too large.');}if(!r.body||typeof r.body.getReader!=='function'||typeof TextDecoder!=='function')throw new Error('Bounded Share reader unavailable.');const rd=r.body.getReader(),dec=new TextDecoder(),parts=[];let n=0;try{for(;;){const x=await rd.read();if(x.done)break;const v=x.value||new Uint8Array(0);n+=Number(v.byteLength||v.length||0);if(n>max)throw new Error('Share response is too large.');parts.push(dec.decode(v,{stream:true}));}parts.push(dec.decode());}catch(e){try{await rd.cancel()}catch(_e){}throw e;}finally{try{rd.releaseLock()}catch(_e){}}return JSON.parse(parts.join(''));};fetch(__READ_PATH__,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({shareId:raw}),cache:'no-store',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer'}).then(async r=>{if(!r.ok){throw new Error(r.status===410?'This Share has expired.':r.status===404?'This Share is unavailable.':'Could not load this Share.');}return await readJson(r);}).then(data=>{app.replaceChildren();if(data.format==='html'&&data.snapshot&&data.snapshot.session&&Array.isArray(data.snapshot.records)){const snap=data.snapshot;const h=document.createElement('header');h.className='head';const h1=document.createElement('h1');h1.textContent=(snap.session.assistant_name||'AI Assistant')+' — Shared conversation';h.appendChild(h1);if(snap.session.page_title){const p=document.createElement('p');p.textContent=snap.session.page_title;h.appendChild(p);}if(snap.session.page_url){const p=document.createElement('p');p.className='source';p.append('Source: ');const a=document.createElement('a');a.href=snap.session.page_url;a.rel='noopener noreferrer';a.referrerPolicy='no-referrer';a.textContent=snap.session.page_url;p.appendChild(a);h.appendChild(p);}app.appendChild(h);for(const row of snap.records){const article=document.createElement('article');article.className='msg '+(row.role==='user'?'user':row.role==='error'?'error':'assistant');const role=document.createElement('div');role.className='role';role.textContent=row.role==='user'?'You':row.role==='error'?'Error':(snap.session.assistant_name||'AI Assistant');const pre=document.createElement('pre');pre.textContent=String(row.text||'');article.append(role,pre);app.appendChild(article);}return;}const pre=document.createElement('pre');pre.textContent=String(data.content||'');app.appendChild(pre);}).catch(e=>fail(e&&e.message?e.message:'Could not load this Share.'));})();</script></body></html>"""
    return template.replace("__READ_PATH__", read_path_json)


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
