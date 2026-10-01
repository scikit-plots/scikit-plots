# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_document_generation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded provider-neutral contract for one-shot AI Learn document generation."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

DOCUMENT_GENERATION_REQUEST_CONTRACT = "assistant.document-generation-request.v1"
DOCUMENT_GENERATION_RESPONSE_CONTRACT = "assistant.document-generation-response.v1"
DOCUMENT_GENERATION_CAPABILITY_VERSION = 1
MAX_DOCUMENT_GENERATION_REQUEST_BYTES = 128 * 1024
MAX_DOCUMENT_PROMPT_CHARS = 48_000
MAX_DOCUMENT_TITLE_CHARS = 200
MAX_DOCUMENT_OUTPUT_CHARS = 1_000_000
FORMATS = {
    "markdown": ("md", "text/markdown"),
    "rst": ("rst", "text/x-rst"),
    "text": ("txt", "text/plain"),
}


class DocumentGenerationError(ValueError):
    """Stable public request/response validation error."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class DocumentGenerationRequest:
    prompt: str
    selected_model: str
    format: str
    title: str


def _text(value: Any, *, field: str, maximum: int, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str) or len(value) > maximum:
        raise DocumentGenerationError("REQUEST_INVALID", f"{field} is invalid")
    if (
        any(  # lint
            (
                ord(ch) < 32  # ruff: ignore[magic-value-comparison]
                and ch not in "\n\r\t"
            )
            for ch in value
        )
        or "\x7f" in value
    ):
        raise DocumentGenerationError(
            "REQUEST_INVALID", f"{field} contains control characters"
        )
    value = value.strip()
    if required and not value:
        raise DocumentGenerationError("REQUEST_INVALID", f"{field} is required")
    return value


def parse_document_generation_request(raw: bytes) -> DocumentGenerationRequest:
    if len(raw) > MAX_DOCUMENT_GENERATION_REQUEST_BYTES:
        raise DocumentGenerationError("REQUEST_TOO_LARGE")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DocumentGenerationError(
            "REQUEST_INVALID",
            "request must be valid JSON",
        ) from exc
    if not isinstance(value, dict):
        raise DocumentGenerationError("REQUEST_INVALID", "request must be an object")
    allowed = {"contract", "prompt", "selected_model", "format", "title"}
    if set(value) - allowed:
        raise DocumentGenerationError(
            "REQUEST_INVALID",
            "request contains unsupported fields",
        )
    if value.get("contract") != DOCUMENT_GENERATION_REQUEST_CONTRACT:
        raise DocumentGenerationError(
            "REQUEST_INVALID", "unsupported document generation contract"
        )
    fmt = _text(
        value.get("format", "markdown"),
        field="format",
        maximum=32,
        required=True,
    ).lower()
    if fmt not in FORMATS:
        raise DocumentGenerationError("REQUEST_INVALID", "unsupported document format")
    return DocumentGenerationRequest(
        prompt=_text(
            value.get("prompt"),
            field="prompt",
            maximum=MAX_DOCUMENT_PROMPT_CHARS,
            required=True,
        ),
        selected_model=_text(
            value.get("selected_model"),
            field="selected_model",
            maximum=256,
        ),
        format=fmt,
        title=_text(
            value.get("title", "AI Learn document"),
            field="title",
            maximum=MAX_DOCUMENT_TITLE_CHARS,
        )
        or "AI Learn document",
    )


def build_document_prompt(
    request: DocumentGenerationRequest,
) -> str:
    syntax = {
        "markdown": "GitHub-flavored Markdown",
        "rst": "reStructuredText",
        "text": "plain text",
    }[request.format]
    return (
        "Create one self-contained learning document from the request below.\n"
        f"Output format: {syntax}.\n"
        "Return only the document body: no surrounding code fence, no preamble about the task, "
        "and no claim that the document was saved or published. Preserve uncertainty, distinguish "
        "source-grounded facts from synthesis, and do not invent citations.\n\n"
        "Untrusted document request:\n" + request.prompt
    )


def extract_text_completion(
    payload: Any,
) -> str:
    """Extract bounded text from common OpenAI-compatible non-stream responses."""
    text = ""
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = [
                        part["text"]
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    ]
                    text = "".join(parts)
        if not text and isinstance(payload.get("output_text"), str):
            text = payload["output_text"]
    text = text.strip()
    if not text:
        raise DocumentGenerationError(
            "UPSTREAM_RESPONSE_INVALID",
            "model returned no document text",
        )
    if len(text) > MAX_DOCUMENT_OUTPUT_CHARS:
        raise DocumentGenerationError(
            "UPSTREAM_RESPONSE_TOO_LARGE",
        )
    return text


def _filename(title: str, fmt: str) -> str:
    stem = (
        re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:80]
        or "ai-learn-document"
    )
    return f"{stem}.{FORMATS[fmt][0]}"


def build_document_response(
    request: DocumentGenerationRequest,
    content: str,
    *,
    selected_model: str,
) -> dict[str, Any]:
    if len(content) > MAX_DOCUMENT_OUTPUT_CHARS:
        raise DocumentGenerationError("UPSTREAM_RESPONSE_TOO_LARGE")
    raw = content.encode("utf-8")
    return {
        "contract": DOCUMENT_GENERATION_RESPONSE_CONTRACT,
        "document_id": "doc_" + uuid.uuid4().hex,
        "title": request.title,
        "format": request.format,
        "filename": _filename(request.title, request.format),
        "mime_type": FORMATS[request.format][1],
        "content": content,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "selected_model": selected_model,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "publication": "separate-reviewed-step",
    }
