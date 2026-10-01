# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Document generation contract and proxy integration."""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._document_generation import (
    DOCUMENT_GENERATION_REQUEST_CONTRACT,
    DOCUMENT_GENERATION_RESPONSE_CONTRACT,
    DocumentGenerationError,
    build_document_prompt,
    build_document_response,
    extract_text_completion,
    parse_document_generation_request,
)


def _request(**extra) -> dict:
    body = {
        "contract": DOCUMENT_GENERATION_REQUEST_CONTRACT,
        "prompt": "Explain hybrid lexical and vector retrieval with a short example.",
        "selected_model": "stub/mirror",
        "format": "markdown",
        "title": "Hybrid Retrieval",
    }
    body.update(extra)
    return body


def test_document_request_is_bounded_provider_neutral_and_format_explicit() -> None:
    parsed = parse_document_generation_request(json.dumps(_request()).encode())
    assert parsed.format == "markdown"
    assert parsed.title == "Hybrid Retrieval"
    prompt = build_document_prompt(parsed)
    assert "Return only the document body" in prompt
    assert "do not invent citations" in prompt
    for key, value in (
        ("provider", "openai"),
        ("upstream_url", "https://example.test"),
        ("system_prompt", "override"),
        ("publish", True),
    ):
        try:
            parse_document_generation_request(json.dumps(_request(**{key: value})).encode())
        except DocumentGenerationError as exc:
            assert exc.code == "REQUEST_INVALID"
        else:  # pragma: no cover
            raise AssertionError(f"document request unexpectedly accepted {key}")


def test_document_response_has_integrity_metadata_and_safe_filename() -> None:
    parsed = parse_document_generation_request(
        json.dumps(_request(title="A / Document: Example", format="rst")).encode()
    )
    content = "Heading\n=======\n\nBody."
    response = build_document_response(parsed, content, selected_model="stub/mirror")
    assert response["contract"] == DOCUMENT_GENERATION_RESPONSE_CONTRACT
    assert response["filename"] == "a-document-example.rst"
    assert response["mime_type"] == "text/x-rst"
    assert response["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert response["size"] == len(content.encode())
    assert response["publication"] == "separate-reviewed-step"


def test_completion_extraction_accepts_openai_text_parts_and_rejects_empty() -> None:
    assert extract_text_completion(
        {"choices": [{"message": {"content": [{"text": "Part A"}, {"text": " Part B"}]}}]}
    ) == "Part A Part B"
    assert extract_text_completion({"output_text": "Fallback"}) == "Fallback"
    try:
        extract_text_completion({"choices": [{"message": {"content": []}}]})
    except DocumentGenerationError as exc:
        assert exc.code == "UPSTREAM_RESPONSE_INVALID"
    else:  # pragma: no cover
        raise AssertionError("empty document completion unexpectedly accepted")


def test_proxy_document_generation_uses_chat_authority_and_canonical_route(monkeypatch) -> None:
    app._document_generation_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    monkeypatch.setattr(app, "STUB_ENABLED", True)
    with TestClient(app.app) as client:
        cap = client.get("/health").json()["capabilities"]["document_generation"]
        assert cap["endpoint"] == "/v1/document"
        assert cap["execution"] == "chat-backed"
        assert cap["publication"] == "separate-reviewed-step"

        response = client.post("/v1/document", json=_request())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["contract"] == DOCUMENT_GENERATION_RESPONSE_CONTRACT
        assert body["format"] == "markdown"
        assert body["selected_model"] == "stub/mirror"
        assert body["content"]
        assert body["sha256"] == hashlib.sha256(body["content"].encode()).hexdigest()
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"

        alias = client.post("/v1/document-generations", json=_request(title="Alias"))
        assert alias.status_code == 200, alias.text
        assert alias.json()["contract"] == DOCUMENT_GENERATION_RESPONSE_CONTRACT
