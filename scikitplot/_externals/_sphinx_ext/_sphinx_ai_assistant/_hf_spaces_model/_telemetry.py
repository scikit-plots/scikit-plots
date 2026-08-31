# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Privacy-safe logging helpers for bundled AI services.

The project is open source, so logging policy must remain safe even when an
attacker knows every redaction rule.  The primary control is data minimization:
callers log fixed event metadata, never request/conversation bodies.  The
helpers below are a defence-in-depth boundary for exception text and values
that reach logging through libraries or future code.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import TracebackType
from typing import Any

MAX_LOG_TEXT = 512
MAX_EXCEPTION_FRAMES = 12
MAX_EXCEPTION_MESSAGE = 256

# Keep these patterns deliberately high-confidence.  Detection is a fallback,
# not permission to log sensitive data in the first place.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN [^-\r\n]{1,64} PRIVATE KEY-----.*?-----END [^-\r\n]{1,64} PRIVATE KEY-----",
            re.IGNORECASE | re.DOTALL,
        ),
        "<private-key-redacted>",
    ),
    (re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE), "Bearer <credential-redacted>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{4,}\b"), "<credential-redacted>"),
    (re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{8,}\b"), "<credential-redacted>"),
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,})\b"),
        "<credential-redacted>",
    ),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<credential-redacted>"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
        "<credential-redacted>",
    ),
    (
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|token)\s*[:=]\s*[^\s,;&]+"
        ),
        "<credential-field-redacted>",
    ),
    (
        re.compile(r"\b[A-Za-z]:[\\/](?:[^\r\n\t ]+[\\/])*[^\r\n\t ]*"),
        "<local-path-redacted>",
    ),
    (re.compile(r"\bfile://[^\s\"'<>]+", re.IGNORECASE), "<local-url-redacted>"),
    (re.compile(r"\bhttps?://[^\s\"'<>]+", re.IGNORECASE), "<url-redacted>"),
    (
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "<email-redacted>",
    ),
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "<ip-redacted>",
    ),
)

# Field names that should not survive a structured-event helper even if the
# value happens not to match a known secret shape.
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "setcookie",
        "token",
        "edittoken",
        "shareid",
        "uuid",
        "sessionid",
        "conversationid",
        "query",
        "answer",
        "content",
        "body",
        "prompt",
        "messages",
        "feedbackmessage",
        "url",
        "pageurl",
        "email",
        "password",
        "secret",
        "apikey",
        "accesstoken",
    }
)


def sanitize_log_text(value: Any, *, max_chars: int = MAX_LOG_TEXT) -> str:
    """Return a bounded single logical log value with high-confidence redaction."""
    text = str(value or "")
    # Redact before converting control characters so multi-line secret shapes
    # (for example PEM private keys) are still recognized as one value.
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    # Prevent terminal/log forging while retaining a readable separator.
    text = text.replace("\x00", "<nul>").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > max_chars:
        text = text[:max_chars] + "…<truncated>"
    return text


def safe_exception_summary(
    exc_info: tuple[type[BaseException], BaseException, TracebackType | None] | None,
) -> dict[str, Any] | None:
    """Return a bounded stack summary without source lines or filesystem paths."""
    if not exc_info:
        return None
    exc_type, exc, tb = exc_info
    frames: list[dict[str, Any]] = []
    cur = tb
    while cur is not None:
        code = cur.tb_frame.f_code
        frames.append(
            {
                "file": Path(code.co_filename).name,
                "function": sanitize_log_text(code.co_name, max_chars=80),
                "line": int(cur.tb_lineno),
            }
        )
        cur = cur.tb_next
    if len(frames) > MAX_EXCEPTION_FRAMES:
        frames = frames[-MAX_EXCEPTION_FRAMES:]
    return {
        "type": sanitize_log_text(
            getattr(exc_type, "__name__", "Exception"), max_chars=80
        ),
        "message": sanitize_log_text(exc, max_chars=MAX_EXCEPTION_MESSAGE),
        "frames": frames,
    }


def safe_event_fields(fields: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional structured fields, dropping sensitive field names."""
    out: dict[str, Any] = {}
    for key, value in (fields or {}).items():
        name = str(key)
        if name.lower().replace("-", "").replace("_", "") in _SENSITIVE_FIELD_NAMES:
            continue
        if value is None or isinstance(value, (bool, int, float)):
            out[name] = value
        else:
            out[name] = sanitize_log_text(value, max_chars=160)
    return out


class PrivacyJsonFormatter(logging.Formatter):
    """Emit bounded JSON logs with sanitized exception metadata."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": sanitize_log_text(record.name, max_chars=80),
            "event": sanitize_log_text(record.getMessage()),
        }
        summary = safe_exception_summary(record.exc_info)
        if summary:
            payload["exception"] = summary
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_privacy_logging(*, level: int = logging.INFO) -> logging.Logger:
    """Install one root handler with privacy-safe JSON formatting."""
    handler = logging.StreamHandler()
    handler.setFormatter(PrivacyJsonFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(level)
    return logging.getLogger()
