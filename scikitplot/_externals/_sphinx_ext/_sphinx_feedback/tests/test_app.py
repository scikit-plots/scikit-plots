from __future__ import annotations

import asyncio
import json

from _sphinx_ext._sphinx_feedback._service.app import (
    FeedbackASGIApp,
    parse_allowed_origins,
    parse_trusted_proxy_cidrs,
)
from _sphinx_ext._sphinx_feedback._service._config import load_service_config


def payload(*, rating=1, feedback_id="feedback-" + "a" * 48):
    return {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": "docs",
        "page_id": "guide/install",
        "feedback_id": feedback_id,
        "rating": rating,
        "mode": "quick",
        "contributor": {"display_name": ""},
    }


async def call(
    app, *, method="POST", body=b"", origin="", client="203.0.113.7", extra_headers=()
):
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"docs.example.org"), (b"content-type", b"application/json")]
    if origin:
        headers.append((b"origin", origin.encode("ascii")))
    headers.extend(extra_headers)
    scope = {
        "type": "http",
        "method": method,
        "path": "/v1/feedback",
        "scheme": "https",
        "headers": headers,
        "client": (client, 54321),
    }
    await app(scope, receive, send)
    start = next(item for item in sent if item["type"] == "http.response.start")
    body_msg = next(item for item in sent if item["type"] == "http.response.body")
    decoded = json.loads(body_msg.get("body") or b"{}") if body_msg.get("body") else {}
    return start, decoded


def sqlite_app(tmp_path, *, rate=20, origins=(), allowed_sites="", trusted_proxies=()):
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "sqlite",
            "FEEDBACK_SQLITE_PATH": str(tmp_path / "feedback.sqlite3"),
            "FEEDBACK_PAGE_RATE_LIMIT_PER_HOUR": str(rate),
            "FEEDBACK_ALLOWED_SITE_IDS": allowed_sites,
        }
    )
    return FeedbackASGIApp(
        cfg, allowed_origins=origins, trusted_proxy_cidrs=trusted_proxies
    )


def test_standalone_asgi_sqlite_accepts_and_replays(tmp_path):
    app = sqlite_app(tmp_path)
    raw = json.dumps(payload()).encode()
    first, first_body = asyncio.run(call(app, body=raw))
    second, second_body = asyncio.run(call(app, body=raw))
    assert first["status"] == 202
    assert first_body["status"] == "accepted"
    assert second["status"] == 202
    assert second_body["status"] == "replay"


def test_standalone_asgi_conflict_is_409(tmp_path):
    app = sqlite_app(tmp_path)
    asyncio.run(call(app, body=json.dumps(payload()).encode()))
    status, body = asyncio.run(call(app, body=json.dumps(payload(rating=-1)).encode()))
    assert status["status"] == 409
    assert "different feedback content" in body["detail"]


def test_invalid_request_does_not_consume_rate_limit(tmp_path):
    app = sqlite_app(tmp_path, rate=1)
    status, _ = asyncio.run(call(app, body=b'{"contract":"bad"}'))
    assert status["status"] == 422
    valid, _ = asyncio.run(call(app, body=json.dumps(payload()).encode()))
    assert valid["status"] == 202
    limited, _ = asyncio.run(
        call(
            app,
            body=json.dumps(payload(feedback_id="feedback-" + "b" * 48)).encode(),
        )
    )
    assert limited["status"] == 429


def test_local_rate_limiter_stores_only_hmac_pseudonym(tmp_path):
    app = sqlite_app(tmp_path, rate=1)
    asyncio.run(call(app, body=json.dumps(payload()).encode(), client="198.51.100.9"))
    keys = list(app._rate)
    assert len(keys) == 1
    assert keys[0] != "198.51.100.9"
    assert len(keys[0]) == 64
    assert "198.51.100.9" not in keys[0]


def test_exact_cors_allowlist_and_preflight(tmp_path):
    app = sqlite_app(tmp_path, origins=("https://docs.example.org",))
    start, body = asyncio.run(
        call(app, method="OPTIONS", origin="https://docs.example.org")
    )
    assert start["status"] == 204
    assert body == {}
    headers = dict(start["headers"])
    assert headers[b"access-control-allow-origin"] == b"https://docs.example.org"

    denied, _ = asyncio.run(
        call(app, method="OPTIONS", origin="https://evil.example.org")
    )
    assert denied["status"] == 403


def test_allowed_origins_reject_paths_queries_credentials_and_insecure_remote_http():
    for value in [
        "https://example.org/path",
        "https://example.org?x=1",
        "https://user@example.org",
        "http://example.org",
    ]:
        try:
            parse_allowed_origins(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    assert parse_allowed_origins("http://127.0.0.1:8000,https://example.org") == (
        "http://127.0.0.1:8000",
        "https://example.org",
    )


def test_duplicate_json_fields_are_rejected_before_rate_limit(tmp_path):
    app = sqlite_app(tmp_path, rate=1)
    raw = (
        b'{"contract":"page.feedback-request.v1","contract":"page.feedback-request.v1",'
        b'"action":"submit","site_id":"docs","page_id":"guide/install",'
        b'"feedback_id":"feedback-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"rating":1,"mode":"quick","contributor":{"display_name":""}}'
    )
    rejected, body = asyncio.run(call(app, body=raw))
    assert rejected["status"] == 422
    assert "duplicate JSON field" in body["detail"]

    valid, _ = asyncio.run(call(app, body=json.dumps(payload()).encode()))
    assert valid["status"] == 202


def test_post_rejects_non_json_media_type_before_rate_limit(tmp_path):
    app = sqlite_app(tmp_path, rate=1)

    async def raw_call():
        messages = [{"type": "http.request", "body": json.dumps(payload()).encode(), "more_body": False}]
        sent = []
        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}
        async def send(message):
            sent.append(message)
        scope = {
            "type": "http", "method": "POST", "path": "/v1/feedback", "scheme": "https",
            "headers": [(b"host", b"docs.example.org"), (b"content-type", b"text/plain")],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        return next(item for item in sent if item["type"] == "http.response.start")

    rejected = asyncio.run(raw_call())
    assert rejected["status"] == 415
    valid, _ = asyncio.run(call(app, body=json.dumps(payload()).encode()))
    assert valid["status"] == 202


def test_same_request_retry_does_not_consume_new_event_quota(tmp_path):
    app = sqlite_app(tmp_path, rate=1)
    raw = json.dumps(payload()).encode()
    first, first_body = asyncio.run(call(app, body=raw))
    replay, replay_body = asyncio.run(call(app, body=raw))
    new_event, _ = asyncio.run(
        call(
            app,
            body=json.dumps(payload(feedback_id="feedback-" + "b" * 48)).encode(),
        )
    )
    assert first["status"] == 202
    assert first_body["status"] == "accepted"
    assert replay["status"] == 202
    assert replay_body["status"] == "replay"
    assert new_event["status"] == 429


def test_same_request_retry_allowance_is_bounded(tmp_path):
    app = sqlite_app(tmp_path, rate=1)
    raw = json.dumps(payload()).encode()
    statuses = [asyncio.run(call(app, body=raw))[0]["status"] for _ in range(5)]
    assert statuses == [202, 202, 202, 202, 429]
    assert len(app._rate_retries) == 1
    ((identity, commitment), (retry_count, _started)), = app._rate_retries.items()
    assert len(identity) == 64
    assert len(commitment) == 64
    assert retry_count == 3


def test_allowed_origins_canonicalize_default_ports():
    assert parse_allowed_origins(
        "https://example.org:443,http://localhost:80,http://localhost:8000"
    ) == (
        "https://example.org",
        "http://localhost",
        "http://localhost:8000",
    )


def test_duplicate_security_singleton_headers_fail_closed(tmp_path):
    app = sqlite_app(tmp_path)

    async def raw_call():
        sent = []
        messages = [
            {
                "type": "http.request",
                "body": json.dumps(payload()).encode(),
                "more_body": False,
            }
        ]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-type", b"application/json"),
            ],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        return next(item for item in sent if item["type"] == "http.response.start")

    assert asyncio.run(raw_call())["status"] == 400


def test_declared_oversize_body_is_rejected_before_read(tmp_path):
    app = sqlite_app(tmp_path)

    async def raw_call():
        sent = []
        receive_called = False

        async def receive():
            nonlocal receive_called
            receive_called = True
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(app.config.max_body_bytes + 1).encode()),
            ],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        start = next(item for item in sent if item["type"] == "http.response.start")
        return start, receive_called

    start, receive_called = asyncio.run(raw_call())
    assert start["status"] == 413
    assert receive_called is False


def test_excessively_fragmented_request_is_bounded(tmp_path):
    app = sqlite_app(tmp_path)

    async def raw_call():
        sent = []
        # More than the adapter's bounded ASGI request-frame allowance, while
        # keeping the total byte count tiny.
        messages = [
            {"type": "http.request", "body": b"", "more_body": True}
            for _ in range(300)
        ]

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [(b"content-type", b"application/json")],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        return next(item for item in sent if item["type"] == "http.response.start")

    assert asyncio.run(raw_call())["status"] == 413


def test_response_broken_pipe_after_durable_write_is_suppressed_and_replayable(tmp_path):
    app = sqlite_app(tmp_path, rate=1)
    raw = json.dumps(payload()).encode()

    async def first_call_with_broken_pipe():
        messages = [{"type": "http.request", "body": raw, "more_body": False}]
        response_started = False

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                return
            raise BrokenPipeError("client disconnected after response start")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [(b"content-type", b"application/json")],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        return response_started

    assert asyncio.run(first_call_with_broken_pipe()) is True
    replay, body = asyncio.run(call(app, body=raw))
    assert replay["status"] == 202
    assert body["status"] == "replay"


def test_json_responses_disable_mime_sniffing(tmp_path):
    app = sqlite_app(tmp_path)
    start, _ = asyncio.run(call(app, body=json.dumps(payload()).encode()))
    headers = dict(start["headers"])
    assert headers[b"x-content-type-options"] == b"nosniff"


def test_content_length_mismatch_fails_closed_before_validation(tmp_path):
    app = sqlite_app(tmp_path)
    raw = json.dumps(payload()).encode()

    async def raw_call():
        sent = []
        messages = [{"type": "http.request", "body": raw, "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw) - 1).encode("ascii")),
            ],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        start = next(item for item in sent if item["type"] == "http.response.start")
        body = next(item for item in sent if item["type"] == "http.response.body")
        return start, json.loads(body["body"])

    start, body = asyncio.run(raw_call())
    assert start["status"] == 400
    assert "Content-Length does not match" in body["detail"]
    assert not app._rate


def test_malformed_asgi_body_frame_fails_closed_without_coercive_allocation(tmp_path):
    app = sqlite_app(tmp_path)

    async def raw_call():
        messages = [{"type": "http.request", "body": 10_000_000, "more_body": False}]
        sent = []

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [
                (b"host", b"docs.example.org"),
                (b"content-type", b"application/json"),
            ],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        start = next(item for item in sent if item["type"] == "http.response.start")
        body = next(item for item in sent if item["type"] == "http.response.body")
        return start, json.loads(body.get("body") or b"{}")

    start, body = asyncio.run(raw_call())
    assert start["status"] == 400
    assert body["detail"] == "Malformed page feedback request framing."


def test_unauthorized_site_is_rejected_before_rate_limit_state(tmp_path):
    app = sqlite_app(tmp_path, rate=1, allowed_sites="docs")
    foreign = payload()
    foreign["site_id"] = "foreign-docs"
    rejected, body = asyncio.run(call(app, body=json.dumps(foreign).encode()))
    assert rejected["status"] == 422
    assert "not authorized" in body["detail"]
    assert app._rate == {}
    assert app._rate_retries == {}

    accepted, _ = asyncio.run(call(app, body=json.dumps(payload()).encode()))
    assert accepted["status"] == 202

def test_trusted_proxy_cidrs_are_strict_and_canonical():
    assert parse_trusted_proxy_cidrs("10.0.0.0/8,2001:db8::/32") == (
        "10.0.0.0/8",
        "2001:db8::/32",
    )
    for value in [
        "10.1.2.3/8",
        "not-a-network",
        "0.0.0.0/0",
        "::/0",
        "0.0.0.0/1,128.0.0.0/1",
        "::/1,8000::/1",
    ]:
        try:
            parse_trusted_proxy_cidrs(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)


def test_untrusted_peer_cannot_spoof_forwarded_rate_identity(tmp_path):
    app = sqlite_app(tmp_path, rate=1, trusted_proxies=("10.0.0.0/8",))
    first, _ = asyncio.run(
        call(
            app,
            body=json.dumps(payload()).encode(),
            client="198.51.100.10",
            extra_headers=((b"x-forwarded-for", b"203.0.113.111"),),
        )
    )
    second_payload = payload(feedback_id="feedback-" + "b" * 48)
    second, _ = asyncio.run(
        call(
            app,
            body=json.dumps(second_payload).encode(),
            client="198.51.100.10",
            extra_headers=((b"x-forwarded-for", b"203.0.113.222"),),
        )
    )
    assert first["status"] == 202
    assert second["status"] == 429


def test_trusted_proxy_chain_uses_first_untrusted_hop_from_right(tmp_path):
    app = sqlite_app(
        tmp_path,
        rate=1,
        trusted_proxies=("10.0.0.0/8", "192.0.2.0/24"),
    )
    scope = {
        "headers": [(b"x-forwarded-for", b"198.51.100.7, 192.0.2.9")],
        "client": ("10.0.0.5", 443),
    }
    assert app._rate_address(scope) == "198.51.100.7"


def test_malformed_forwarded_chain_falls_back_to_trusted_peer(tmp_path):
    app = sqlite_app(tmp_path, trusted_proxies=("10.0.0.0/8",))
    scope = {
        "headers": [(b"x-forwarded-for", b"not-an-ip")],
        "client": ("10.0.0.5", 443),
    }
    assert app._rate_address(scope) == "10.0.0.5"


def test_malformed_asgi_header_frames_fail_closed(tmp_path):
    app = sqlite_app(tmp_path)

    async def raw_call(headers):
        sent = []
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": headers,
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        return next(item for item in sent if item["type"] == "http.response.start")

    assert asyncio.run(raw_call([("content-type", "application/json")]))["status"] == 400
    assert asyncio.run(
        raw_call([(b"content-type", b"application/json\r\nX: y")])
    )["status"] == 400


def test_content_length_requires_ascii_decimal_grammar(tmp_path):
    app = sqlite_app(tmp_path)

    async def raw_call(value):
        sent = []
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/feedback",
            "scheme": "https",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", value),
            ],
            "client": ("203.0.113.7", 54321),
        }
        await app(scope, receive, send)
        return next(item for item in sent if item["type"] == "http.response.start")

    assert asyncio.run(raw_call(b"+2"))["status"] == 400
    assert asyncio.run(raw_call(b" 2"))["status"] == 400
