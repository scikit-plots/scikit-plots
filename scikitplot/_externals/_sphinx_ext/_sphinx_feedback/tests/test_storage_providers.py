"""
Storage providers: local git, signed webhook, custom adapter, GitHub Enterprise.

Notes
-----
Every provider is driven through :class:`PageFeedbackService` where possible,
so receipt validation, conflict mapping and mirrors are exercised as in
production. Network providers use ``httpx.MockTransport``; nothing leaves the
machine. The git tests run the real ``git`` binary and skip without it.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import types

import httpx
import pytest

from _sphinx_ext._sphinx_feedback._contracts import (
    FeedbackConflictError,
    build_feedback_event,
    feedback_request_hash,
)
from _sphinx_ext._sphinx_feedback._service._config import (
    FeedbackServiceConfigError,
    StorageTarget,
    load_service_config,
    parse_storage_targets,
)
from _sphinx_ext._sphinx_feedback._service._core import (
    FeedbackServiceUnavailable,
    PageFeedbackService,
)
from _sphinx_ext._sphinx_feedback._service._git import event_relative_path
from _sphinx_ext._sphinx_feedback._service._github import _event_path
from _sphinx_ext._sphinx_feedback._service._webhook import (
    DELIVERY_CONTRACT,
    SIGNATURE_HEADER,
    verify,
)

SECRET = "webhook-signing-secret"  # noqa: S105 - test value


def request(feedback_id="feedback-" + "1" * 48, *, rating=1, comment=""):
    payload = {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": "docs",
        "page_id": "guide/install",
        "feedback_id": feedback_id,
        "rating": rating,
        "mode": "detailed" if comment else "quick",
        "contributor": {"display_name": ""},
    }
    if comment:
        payload["comment"] = comment
    return payload


# ---------------------------------------------------------------- config


def test_each_provider_accepts_only_its_own_keys():
    with pytest.raises(FeedbackServiceConfigError, match="does not accept: url"):
        parse_storage_targets(
            [
                {
                    "id": "gh",
                    "provider": "github",
                    "role": "primary",
                    "repo": "org/repo",
                    "url": "https://x.example",
                }
            ]
        )
    with pytest.raises(FeedbackServiceConfigError, match="does not accept: repo"):
        parse_storage_targets(
            [
                {
                    "id": "hook",
                    "provider": "webhook",
                    "role": "primary",
                    "url": "https://hook.example",
                    "repo": "org/repo",
                }
            ]
        )


def test_webhook_url_rules():
    def hook(url, **extra):
        return parse_storage_targets(
            [{"id": "hook", "provider": "webhook", "role": "primary", "url": url, **extra}]
        )[0]

    target = hook("https://feedback.example.workers.dev/ingest")
    assert target.token_env == ("FEEDBACK_WEBHOOK_TOKEN",)
    assert hook("http://127.0.0.1:8787/ingest").url == "http://127.0.0.1:8787/ingest"
    assert hook("http://ingest.internal/x", allow_http=True).allow_http is True
    for bad, extra in (
        ("http://ingest.internal/x", {}),
        ("http://ingest.example.org/x", {"allow_http": True}),
        ("https://u:p@hook.example/x", {}),
        ("https://hook.example/x?k=v", {}),
    ):
        with pytest.raises(FeedbackServiceConfigError):
            hook(bad, **extra)


def test_git_and_custom_target_rules(tmp_path):
    git = parse_storage_targets(
        [
            {
                "id": "git",
                "provider": "git",
                "role": "primary",
                "repository_path": str(tmp_path),
            }
        ]
    )[0]
    assert git.repository_path == str(tmp_path)
    with pytest.raises(FeedbackServiceConfigError, match="absolute"):
        parse_storage_targets(
            [{"id": "git", "provider": "git", "role": "primary", "repository_path": "rel"}]
        )
    with pytest.raises(FeedbackServiceConfigError, match=r"package\.module:callable"):
        parse_storage_targets(
            [{"id": "c", "provider": "custom", "role": "primary", "factory": "os.system"}]
        )
    with pytest.raises(FeedbackServiceConfigError, match="may not carry secrets"):
        parse_storage_targets(
            [
                {
                    "id": "c",
                    "provider": "custom",
                    "role": "primary",
                    "factory": "pkg.mod:make",
                    "options": {"api_token": "x"},
                }
            ]
        )


def test_review_modes_bind_provider_families(tmp_path):
    git_target = json.dumps(
        [{"id": "g", "provider": "git", "role": "primary", "repository_path": str(tmp_path)}]
    )
    hook_target = json.dumps(
        [{"id": "h", "provider": "webhook", "role": "primary", "url": "https://h.example"}]
    )
    assert load_service_config(
        {"FEEDBACK_REVIEW_MODE": "local", "FEEDBACK_STORAGE_TARGETS": git_target}
    ).primary.provider == "git"
    with pytest.raises(FeedbackServiceConfigError, match="no network requests"):
        load_service_config(
            {"FEEDBACK_REVIEW_MODE": "local", "FEEDBACK_STORAGE_TARGETS": hook_target}
        )
    with pytest.raises(FeedbackServiceConfigError, match="provider primary"):
        load_service_config(
            {"FEEDBACK_REVIEW_MODE": "provider-pr", "FEEDBACK_STORAGE_TARGETS": git_target}
        )
    assert load_service_config(
        {"FEEDBACK_REVIEW_MODE": "custom", "FEEDBACK_STORAGE_TARGETS": hook_target}
    ).primary.provider == "webhook"


def test_environment_shorthands(tmp_path):
    local = load_service_config(
        {"FEEDBACK_REVIEW_MODE": "local", "FEEDBACK_GIT_REPOSITORY_PATH": str(tmp_path)}
    )
    assert local.primary.provider == "git"
    custom = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "custom",
            "FEEDBACK_WEBHOOK_URL": "http://ingest.internal/feedback",
            "FEEDBACK_WEBHOOK_ALLOW_HTTP": "true",
        }
    )
    assert custom.primary.url == "http://ingest.internal/feedback"
    ghe = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "provider-pr",
            "FEEDBACK_GITHUB_REPOSITORY": "org/repo",
            "FEEDBACK_GITHUB_API_URL": "https://ghe.example/api/v3/",
        }
    )
    assert ghe.primary.api_url == "https://ghe.example/api/v3"
    with pytest.raises(FeedbackServiceConfigError, match="true or false"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "custom",
                "FEEDBACK_WEBHOOK_URL": "https://h.example",
                "FEEDBACK_WEBHOOK_ALLOW_HTTP": "maybe",
            }
        )


# ---------------------------------------------------------------- git

git_missing = shutil.which("git") is None


def _git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)  # noqa: S603,S607
    return path


@pytest.mark.skipif(git_missing, reason="git is not installed")
def test_git_provider_commits_once_replays_and_refuses_conflicts(tmp_path):
    repo = _git_repo(tmp_path / "feedback-repo")
    service = PageFeedbackService(
        load_service_config(
            {"FEEDBACK_REVIEW_MODE": "local", "FEEDBACK_GIT_REPOSITORY_PATH": str(repo)}
        )
    )
    first = asyncio.run(service.submit(request()))
    second = asyncio.run(service.submit(request()))
    assert (first["status"], second["status"]) == ("accepted", "replay")
    log = subprocess.run(  # noqa: S603,S607
        ["git", "-C", str(repo), "log", "--format=%s|%an"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert log == [f"feedback: {request()['feedback_id']}|sphinx-feedback"]
    with pytest.raises(FeedbackConflictError):
        asyncio.run(service.submit(request(rating=-1)))


@pytest.mark.skipif(git_missing, reason="git is not installed")
def test_git_layout_matches_the_github_layout(tmp_path):
    target = StorageTarget(id="t", label="t", provider="git", role="primary", feedback_path="fb")
    event = build_feedback_event(request())
    assert event_relative_path(target, event) == _event_path(target, event)


@pytest.mark.skipif(git_missing, reason="git is not installed")
def test_git_provider_refuses_a_path_inside_another_repository(tmp_path):
    repo = _git_repo(tmp_path / "outer")
    inner = repo / "sub"
    inner.mkdir()
    service = PageFeedbackService(
        load_service_config(
            {"FEEDBACK_REVIEW_MODE": "local", "FEEDBACK_GIT_REPOSITORY_PATH": str(inner)}
        )
    )
    with pytest.raises(FeedbackServiceUnavailable) as caught:
        asyncio.run(service.submit(request()))
    assert caught.value.code == "git_repository_unavailable"


# ---------------------------------------------------------------- webhook


class Receiver:
    """An idempotent receiver that checks signatures, like a Worker would."""

    def __init__(self, *, status=None, receipt=None):
        self.stored = {}
        self.status = status
        self.receipt = receipt
        self.bodies = []

    def __call__(self, http_request: httpx.Request) -> httpx.Response:
        body = http_request.content
        self.bodies.append(body)
        if not verify(SECRET, body, http_request.headers.get(SIGNATURE_HEADER, "")):
            return httpx.Response(401)
        if self.status is not None:
            return httpx.Response(self.status, json=self.receipt or {})
        envelope = json.loads(body)
        assert envelope["contract"] == DELIVERY_CONTRACT
        key, digest = envelope["feedback_id"], envelope["request_hash"]
        if key in self.stored and self.stored[key] != digest:
            return httpx.Response(409, json={"status": "conflict"})
        status = "replay" if key in self.stored else "accepted"
        self.stored[key] = digest
        return httpx.Response(
            202, json={"status": status, "feedback_id": key, "request_hash": digest}
        )


def _webhook_service(receiver, *, secret=SECRET, mirror=False):
    targets = [{"id": "hook", "provider": "webhook", "role": "primary", "url": "https://hook.example/in"}]
    env = {
        "FEEDBACK_REVIEW_MODE": "custom",
        "FEEDBACK_STORAGE_TARGETS": json.dumps(targets),
    }
    if secret:
        env["FEEDBACK_WEBHOOK_TOKEN"] = secret
    service = PageFeedbackService(load_service_config(env), credential_env=env)

    async def submit(payload):
        transport = httpx.MockTransport(receiver)
        async with httpx.AsyncClient(transport=transport) as client:
            return await service.submit(payload, http_client=client)

    return submit


def test_webhook_delivers_signed_idempotent_events():
    receiver = Receiver()
    submit = _webhook_service(receiver)
    assert asyncio.run(submit(request()))["status"] == "accepted"
    assert asyncio.run(submit(request()))["status"] == "replay"
    assert receiver.bodies[0] == receiver.bodies[1]  # canonical, retry-stable bytes
    with pytest.raises(FeedbackConflictError):
        asyncio.run(submit(request(rating=-1)))


@pytest.mark.parametrize(
    ("status", "receipt", "code"),
    [
        (302, {}, "webhook_rejected"),
        (500, {}, "webhook_rejected"),
        (202, {"status": "accepted", "feedback_id": "feedback-" + "9" * 48}, "webhook_receipt_invalid"),
    ],
)
def test_webhook_failures_are_stable_codes(status, receipt, code):
    submit = _webhook_service(Receiver(status=status, receipt=receipt))
    with pytest.raises(FeedbackServiceUnavailable) as caught:
        asyncio.run(submit(request()))
    assert caught.value.code == code


def test_webhook_without_secret_is_refused_before_sending():
    receiver = Receiver()
    submit = _webhook_service(receiver, secret="")
    with pytest.raises(FeedbackServiceUnavailable) as caught:
        asyncio.run(submit(request()))
    assert caught.value.code == "credential_missing"
    assert receiver.bodies == []


# ---------------------------------------------------------------- custom


def _install_adapter_module(monkeypatch, *, asynchronous=False, fail=False):
    module = types.ModuleType("feedback_test_adapter")
    store = {}

    class Adapter:
        def __init__(self, target, env):
            self.options = dict(target.options)

        def _put(self, *, event, request_hash):
            if fail:
                raise RuntimeError("database password=hunter2 refused")
            key = event["feedback"]["id"]
            if key in store and store[key] != request_hash:
                raise FeedbackConflictError("different content")
            status = "replay" if key in store else "accepted"
            store[key] = request_hash
            return {"status": status, "provider": "custom", "feedback_id": key, "request_hash": request_hash}

        if asynchronous:

            async def put(self, *, event, request_hash):
                return self._put(event=event, request_hash=request_hash)

        else:

            def put(self, *, event, request_hash):
                return self._put(event=event, request_hash=request_hash)

    module.make = lambda *, target, env: Adapter(target, env)
    monkeypatch.setitem(sys.modules, "feedback_test_adapter", module)
    return store


def _custom_service(factory="feedback_test_adapter:make"):
    targets = [
        {
            "id": "mine",
            "provider": "custom",
            "role": "primary",
            "factory": factory,
            "options": {"table": "feedback"},
        }
    ]
    return PageFeedbackService(
        load_service_config(
            {"FEEDBACK_REVIEW_MODE": "custom", "FEEDBACK_STORAGE_TARGETS": json.dumps(targets)}
        )
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_custom_adapter_sync_and_async(monkeypatch, asynchronous):
    store = _install_adapter_module(monkeypatch, asynchronous=asynchronous)
    service = _custom_service()
    assert asyncio.run(service.submit(request()))["status"] == "accepted"
    assert asyncio.run(service.submit(request()))["status"] == "replay"
    assert list(store) == [request()["feedback_id"]]
    with pytest.raises(FeedbackConflictError):
        asyncio.run(service.submit(request(rating=-1)))


def test_custom_adapter_failure_text_never_surfaces(monkeypatch):
    _install_adapter_module(monkeypatch, fail=True)
    with pytest.raises(FeedbackServiceUnavailable) as caught:
        asyncio.run(_custom_service().submit(request()))
    assert caught.value.code == "custom_provider_failed"
    assert "hunter2" not in str(caught.value)


def test_custom_factory_import_failure_stops_the_service_at_start():
    with pytest.raises(FeedbackServiceConfigError, match="cannot import"):
        _custom_service("feedback_no_such_module:make")


# ---------------------------------------------------------------- GitHub Enterprise


def test_github_enterprise_api_url_and_review_host(tmp_path):
    from test_github_provider import GitHubState  # noqa: PLC0415 - shared fake

    state = GitHubState()
    ghe_pull = "https://ghe.example/org/repo/pull/1"
    original = state.response

    def response(http_request):
        assert http_request.url.host == "ghe.example"
        assert http_request.url.path.startswith("/api/v3/repos/org/repo/")
        reply = original(http_request)
        if state.pull_url:
            state.pull_url = ghe_pull
        if http_request.url.path.endswith("/pulls") and reply.status_code < 300:
            if http_request.method == "POST":
                return httpx.Response(reply.status_code, json={"html_url": ghe_pull})
            return httpx.Response(200, json=[{"html_url": ghe_pull}] if state.pull_url else [])
        return reply

    env = {
        "FEEDBACK_REVIEW_MODE": "provider-pr",
        "FEEDBACK_GITHUB_REPOSITORY": "org/repo",
        "FEEDBACK_GITHUB_API_URL": "https://ghe.example/api/v3",
        "FEEDBACK_GITHUB_TOKEN": "server-only",
    }
    service = PageFeedbackService(load_service_config(env), credential_env=env)

    async def submit():
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            # The shared fake knows one event: feedback-ccc... on guide/install.
            return await service.submit(
                request("feedback-" + "c" * 48), http_client=client
            )

    assert asyncio.run(submit())["status"] == "accepted"


def test_github_enterprise_refuses_review_urls_on_another_host():
    from _sphinx_ext._sphinx_feedback._service._github import (  # noqa: PLC0415
        ProviderWriteError,
        _review_url,
        _web_authority,
    )

    web = _web_authority("https://ghe.example/api/v3")
    assert web == ("ghe.example", None)
    assert _web_authority("https://api.github.com") == ("github.com", None)
    with pytest.raises(ProviderWriteError):
        _review_url("https://github.com/org/repo/pull/1", repo="org/repo", web=web)


def test_comment_text_is_carried_in_the_signed_event():
    receiver = Receiver()
    submit = _webhook_service(receiver)
    asyncio.run(submit(request(comment="Clear page")))
    envelope = json.loads(receiver.bodies[0])
    assert envelope["request_hash"] == feedback_request_hash(request(comment="Clear page"))


def test_readme_storage_example_is_a_valid_configuration():
    from pathlib import Path  # noqa: PLC0415

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text("utf-8")
    section = readme.split("## Storage providers", 1)[1].split("\n## ", 1)[0]
    block = section.split("```json", 1)[1].split("```", 1)[0]
    targets = parse_storage_targets(block)
    assert [t.provider for t in targets] == ["github", "webhook", "custom"]
    config = load_service_config(
        {"FEEDBACK_REVIEW_MODE": "provider-pr", "FEEDBACK_STORAGE_TARGETS": block}
    )
    assert config.primary.api_url == "https://ghe.example/api/v3"


@pytest.mark.parametrize("value", ["feedback", 10, 2.5, True, 0])
def test_custom_options_accept_every_scalar(value):
    target = parse_storage_targets(
        [
            {
                "id": "c",
                "provider": "custom",
                "role": "primary",
                "factory": "pkg.mod:make",
                "options": {"setting": value},
            }
        ]
    )[0]
    assert target.options["setting"] == value


@pytest.mark.parametrize(
    "value", [["a"], {"k": "v"}, None, float("inf"), float("nan"), "x" * 1025]
)
def test_custom_options_refuse_non_scalars_and_oversize(value):
    # Regression: len() was taken before the type check, so an int raised
    # TypeError instead of a configuration error, and a list was not refused.
    with pytest.raises(FeedbackServiceConfigError, match="short scalars"):
        parse_storage_targets(
            [
                {
                    "id": "c",
                    "provider": "custom",
                    "role": "primary",
                    "factory": "pkg.mod:make",
                    "options": {"setting": value},
                }
            ]
        )
