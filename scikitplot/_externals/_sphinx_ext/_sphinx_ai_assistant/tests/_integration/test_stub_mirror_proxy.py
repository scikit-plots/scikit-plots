"""Run 107-109 compatibility: Mirror shows client wire plus derived AI input."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import MAINTENANCE_ROOT, RUNTIME_ROOT

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = RUNTIME_ROOT
PROXY_DIR = ROOT / "_hf_spaces_proxy"
DEV_PROXY = MAINTENANCE_ROOT / "_maintenance" / "tools" / "dev_proxy.py"
if str(PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(PROXY_DIR))

os.environ.setdefault("HF_TOKEN", "")
app = importlib.import_module("app")
contract = importlib.import_module("_utils._chat_contract")


def _structured(*, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "contract": contract.CHAT_CONTRACT,
            "model": "stub/mirror",
            "user_message": "QUESTION-WITH-ONE-TURN-FILE\nAttachment: api.md\nFILE-BODY",
            "context": {
                "page_text": "CURRENT-PAGE-CONTEXT",
                "page_descriptor": "API Reference · https://docs.example.org/api",
            },
            "max_tokens": 1200,
            "stream": stream,
            "reasoning": {"effort": "high"},
        }
    ).encode()


@pytest.mark.asyncio
async def test_structured_mirror_reports_client_proxy_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "STUB_ENABLED", True)
    response = await app._stub_intercept(_structured(), {"Authorization": "Bearer hf_NEVER_ECHO_THIS_VALUE_123456789"})
    assert response is not None and response.status_code == 200
    doc = json.loads(response.body)
    text = doc["choices"][0]["message"]["content"]
    assert "Stub mirror · request-chain security inspector" in text
    assert "browser-sent system prompt: `not sent`" in text
    assert "Effective AI input after trusted server policy" in text
    assert contract.SERVER_SYSTEM_POLICY in text
    assert "Mirror stops here and does **not** forward it upstream." in text
    assert "CURRENT-PAGE-CONTEXT" in text
    assert "QUESTION-WITH-ONE-TURN-FILE" in text
    assert "FILE-BODY" in text
    assert "API Reference · https://docs.example.org/api" in text
    assert "raw HTTP body" in text
    assert "hf_NEVER_ECHO_THIS_VALUE_123456789" not in text
    assert doc["stub_report"]["upstream_called"] is False
    assert "effective_payload" not in doc["stub_report"]


@pytest.mark.asyncio
async def test_streaming_mirror_preserves_same_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "STUB_ENABLED", True)
    response = await app._stub_intercept(_structured(stream=True), {})
    assert response is not None
    pieces = []
    async for chunk in response.body_iterator:
        pieces.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    blob = "".join(pieces)
    deltas = []
    for frame in blob.split("\n\n"):
        if not frame.startswith("data: ") or frame == "data: [DONE]":
            continue
        doc = json.loads(frame[6:])
        delta = doc.get("choices", [{}])[0].get("delta", {}).get("content")
        if isinstance(delta, str):
            deltas.append(delta)
    text = "".join(deltas)
    assert "CURRENT-PAGE-CONTEXT" in text
    assert "QUESTION-WITH-ONE-TURN-FILE" in text
    assert "data: [DONE]" in blob


def test_hf_proxy_default_and_docs_are_synchronized() -> None:
    app_text = (PROXY_DIR / "app.py").read_text()
    readme = (PROXY_DIR / "README.md").read_text()
    assert '"true",' in app_text
    assert "| `STUB_ENABLED` | `true` |" in readme
    assert "stub/mirror" in readme


def test_dev_proxy_uses_same_default() -> None:
    text = DEV_PROXY.read_text()
    assert 'os.environ.get("STUB_ENABLED", "true")' in text
