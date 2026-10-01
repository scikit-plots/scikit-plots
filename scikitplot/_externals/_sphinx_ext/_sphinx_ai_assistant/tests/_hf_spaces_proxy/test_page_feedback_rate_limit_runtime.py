from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve()
PROXY = HERE.parents[2] / "_hf_spaces_proxy"


def _run_proxy_script(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROXY,
        check=True,
        capture_output=True,
        text=True,
    )


def test_page_feedback_first_attempt_without_retry_record_is_null_safe() -> None:
    _run_proxy_script(
        r'''
import asyncio
from starlette.requests import Request
import app

request = Request({
    "type": "http",
    "http_version": "1.1",
    "method": "POST",
    "scheme": "https",
    "path": "/v1/feedback",
    "raw_path": b"/v1/feedback",
    "query_string": b"",
    "headers": [],
    "client": ("127.0.0.1", 50000),
    "server": ("example.test", 443),
})

calls = 0
async def fake_consume(*args, **kwargs):
    global calls
    calls += 1
    return True, calls

app._consume_rate_limit = fake_consume
app._page_feedback_retry_rl.clear()
app._page_feedback_retry_last_prune = app._time.monotonic()

async def main():
    result = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    assert result == (True, 1), result
    assert calls == 1, calls
    assert len(app._page_feedback_retry_rl) == 1

asyncio.run(main())
'''
    )


def test_page_feedback_same_request_retries_reuse_admission_without_new_quota() -> None:
    _run_proxy_script(
        r'''
import asyncio
from starlette.requests import Request
import app

request = Request({
    "type": "http",
    "http_version": "1.1",
    "method": "POST",
    "scheme": "https",
    "path": "/v1/feedback",
    "raw_path": b"/v1/feedback",
    "query_string": b"",
    "headers": [],
    "client": ("127.0.0.1", 50000),
    "server": ("example.test", 443),
})

calls = 0
async def fake_consume(*args, **kwargs):
    global calls
    calls += 1
    return True, calls

app._consume_rate_limit = fake_consume
app._page_feedback_retry_rl.clear()
app._page_feedback_retry_last_prune = app._time.monotonic()

async def main():
    first = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    retry1 = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    retry2 = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    retry3 = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    blocked = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    assert first == (True, 1), first
    assert retry1 == (True, 1), retry1
    assert retry2 == (True, 2), retry2
    assert retry3 == (True, 3), retry3
    assert blocked == (False, 4), blocked
    assert calls == 1, calls

asyncio.run(main())
'''
    )


def test_page_feedback_expired_retry_record_returns_to_authoritative_quota() -> None:
    _run_proxy_script(
        r'''
import asyncio
from starlette.requests import Request
import app

request = Request({
    "type": "http",
    "http_version": "1.1",
    "method": "POST",
    "scheme": "https",
    "path": "/v1/feedback",
    "raw_path": b"/v1/feedback",
    "query_string": b"",
    "headers": [],
    "client": ("127.0.0.1", 50000),
    "server": ("example.test", 443),
})

calls = 0
async def fake_consume(*args, **kwargs):
    global calls
    calls += 1
    return True, calls

app._consume_rate_limit = fake_consume
app._page_feedback_retry_rl.clear()
identity = app._page_feedback_rate_identity(request)
now = app._time.monotonic()
app._page_feedback_retry_rl[(identity, "request-hash")] = (
    2,
    now - app._PAGE_FEEDBACK_RETRY_WINDOW_SECONDS - 1,
)
app._page_feedback_retry_last_prune = now

async def main():
    result = await app._consume_page_feedback_rate(request, request_hash="request-hash")
    assert result == (True, 1), result
    assert calls == 1, calls
    assert app._page_feedback_retry_rl[(identity, "request-hash")][0] == 0

asyncio.run(main())
'''
    )
