from __future__ import annotations

import base64
import json

import httpx
import pytest

from _sphinx_ext._sphinx_feedback._contracts import (
    FeedbackConflictError,
    build_feedback_event,
    feedback_event_request_hash,
    page_digest,
    repository_event_bytes,
)
from _sphinx_ext._sphinx_feedback._service._config import StorageTarget
from _sphinx_ext._sphinx_feedback._service._github import (
    ProviderWriteError,
    _bounded_json,
    _headers,
    _review_page_text,
    _review_url,
    submit_github_review,
)


BASE_SHA = "a" * 40
BRANCH_SHA = "b" * 40
EVENT_PATH = (
    f"feedback/pages/{page_digest('docs', 'guide/install')}/"
    f"feedback-{'c' * 48}.json"
)


def event(rating=1):
    request = {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": "docs",
        "page_id": "guide/install",
        "feedback_id": "feedback-" + "c" * 48,
        "rating": rating,
        "mode": "quick",
        "contributor": {"display_name": ""},
    }
    return build_feedback_event(request)


def target():
    return StorageTarget(
        id="github-primary",
        label="GitHub",
        provider="github",
        role="primary",
        authority="feedback",
        repo="org/repo",
        branch="main",
        feedback_path="feedback",
        token_env=("FEEDBACK_GITHUB_TOKEN",),
        expose_links=True,
    )


class GitHubState:
    def __init__(self):
        self.branch_exists = False
        self.branch_ahead = 0
        self.content = None
        self.base_content = None
        self.pull_url = None
        self.extra_files = []
        self.put_status = 201
        self.pull_status = 201
        self.pull_race = False
        self.break_after_branch_create = False
        self.break_after_content_write = False
        self.content_race = False
        self.break_after_pull_create = False
        self.mutate_branch_after_pull = False
        self.calls = []

    def response(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path, str(request.url.query)))
        path = request.url.path
        method = request.method
        if method == "GET" and path.endswith("/git/ref/heads/main"):
            return httpx.Response(200, json={"object": {"sha": BASE_SHA}})
        if method == "POST" and path.endswith("/git/refs"):
            if self.branch_exists:
                return httpx.Response(422, json={"message": "Reference already exists"})
            self.branch_exists = True
            if self.break_after_branch_create:
                self.break_after_branch_create = False
                raise httpx.ReadError("broken pipe after branch create", request=request)
            return httpx.Response(201, json={"ref": "created"})
        if method == "GET" and "/git/ref/heads/feedback/" in path:
            if not self.branch_exists:
                return httpx.Response(404, json={"message": "Not Found"})
            sha = BRANCH_SHA if self.branch_ahead else BASE_SHA
            return httpx.Response(200, json={"object": {"sha": sha}})
        if method == "GET" and "/contents/" in path:
            ref = request.url.params.get("ref")
            content = self.base_content if ref == "main" else self.content
            if content is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(
                200,
                json={"content": base64.b64encode(content).decode("ascii")},
            )
        if method == "GET" and "/compare/" in path:
            files = []
            if self.branch_ahead:
                files = [{"filename": EVENT_PATH, "status": "added"}, *self.extra_files]
            return httpx.Response(
                200, json={"ahead_by": self.branch_ahead, "files": files}
            )
        if method == "PUT" and "/contents/" in path:
            if self.put_status not in {200, 201}:
                if self.content_race:
                    payload = json.loads(request.content)
                    self.content = base64.b64decode(payload["content"])
                    self.branch_ahead = 1
                return httpx.Response(self.put_status, json={"message": "write failed"})
            payload = json.loads(request.content)
            self.content = base64.b64decode(payload["content"])
            self.branch_ahead = 1
            if self.break_after_content_write:
                self.break_after_content_write = False
                raise httpx.ReadError("broken pipe after content write", request=request)
            return httpx.Response(self.put_status, json={"content": {"sha": BRANCH_SHA}})
        if method == "GET" and path.endswith("/pulls"):
            pulls = [{"html_url": self.pull_url}] if self.pull_url else []
            return httpx.Response(200, json=pulls)
        if method == "POST" and path.endswith("/pulls"):
            if self.pull_status not in {200, 201}:
                if self.pull_race:
                    self.pull_url = "https://github.com/org/repo/pull/1"
                return httpx.Response(self.pull_status, json={"message": "review failed"})
            self.pull_url = "https://github.com/org/repo/pull/1"
            if self.mutate_branch_after_pull:
                self.extra_files = [{"filename": "README.md", "status": "modified"}]
            if self.break_after_pull_create:
                self.break_after_pull_create = False
                raise httpx.ReadError("broken pipe after pull create", request=request)
            return httpx.Response(self.pull_status, json={"html_url": self.pull_url})
        return httpx.Response(500, json={"unexpected": [method, path]})


async def run(state, *, item=None, target_value=None, request_hash=None):
    item = item or event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(state.response)) as client:
        return await submit_github_review(
            client=client,
            target=target_value or target(),
            token="server-only-token",
            event=item,
            request_hash=request_hash or feedback_event_request_hash(item),
        )


@pytest.mark.asyncio
async def test_fresh_submission_establishes_event_and_review():
    state = GitHubState()
    receipt = await run(state)
    assert receipt["status"] == "accepted"
    assert state.content == repository_event_bytes(event())
    assert state.content.count(b"\n") > 1
    assert state.pull_url


@pytest.mark.asyncio
async def test_retry_recovers_branch_created_before_broken_pipe():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 0
    receipt = await run(state)
    assert receipt["status"] == "accepted"
    assert state.content is not None
    assert state.pull_url
    assert any("/compare/" in path for _, path, _ in state.calls)


@pytest.mark.asyncio
async def test_retry_after_commit_but_before_review_opens_missing_review():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = (json.dumps(event(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    receipt = await run(state)
    assert receipt["status"] == "accepted"
    assert state.pull_url


@pytest.mark.asyncio
async def test_existing_review_is_idempotent_replay():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = (json.dumps(event(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    state.pull_url = "https://github.com/org/repo/pull/1"
    receipt = await run(state)
    assert receipt["status"] == "replay"
    assert receipt["review_url"] == state.pull_url


@pytest.mark.asyncio
async def test_existing_pretty_review_is_idempotent_replay():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = repository_event_bytes(event())
    state.pull_url = "https://github.com/org/repo/pull/1"
    receipt = await run(state)
    assert receipt["status"] == "replay"
    assert receipt["review_url"] == state.pull_url


@pytest.mark.asyncio
async def test_existing_duplicate_key_json_fails_closed_even_if_last_value_matches():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = (
        b'{"contract":"page.feedback-event.v1","site_id":"docs",'
        b'"page_id":"guide/install","feedback":{"id":'
        b'"feedback-cccccccccccccccccccccccccccccccccccccccccccccccc",'
        b'"rating":-1,"rating":1,"mode":"quick","contributor":"Anonymous"}}\n'
    )
    with pytest.raises(FeedbackConflictError):
        await run(state)


@pytest.mark.asyncio
async def test_same_id_different_content_conflicts():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = (json.dumps(event(-1), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(FeedbackConflictError):
        await run(state, item=event(1))


@pytest.mark.asyncio
async def test_branch_with_unrelated_unique_commit_fails_closed():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = None
    with pytest.raises(FeedbackConflictError):
        await run(state)


@pytest.mark.asyncio
async def test_expected_event_plus_unrelated_branch_change_fails_closed():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 1
    state.content = (
        json.dumps(event(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    state.extra_files = [{"filename": "README.md", "status": "modified"}]
    with pytest.raises(FeedbackConflictError, match="unexpected repository changes"):
        await run(state)
    assert state.pull_url is None


@pytest.mark.asyncio
async def test_expected_event_on_multi_commit_branch_fails_closed():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 2
    state.content = (
        json.dumps(event(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    with pytest.raises(FeedbackConflictError, match="unexpected commit history"):
        await run(state)
    assert state.pull_url is None


@pytest.mark.asyncio
async def test_merged_review_replays_when_event_is_now_on_base():
    state = GitHubState()
    state.branch_exists = True
    state.branch_ahead = 0
    state.content = (
        json.dumps(event(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    state.base_content = state.content
    state.pull_url = "https://github.com/org/repo/pull/1"
    receipt = await run(state)
    assert receipt["status"] == "replay"


@pytest.mark.asyncio
async def test_pull_request_creation_race_is_reconciled():
    state = GitHubState()
    state.pull_status = 422
    state.pull_race = True
    receipt = await run(state)
    assert receipt["status"] == "replay"
    assert receipt["review_url"] == "https://github.com/org/repo/pull/1"


@pytest.mark.asyncio
async def test_explicit_content_failure_leaves_resumable_branch():
    state = GitHubState()
    state.put_status = 422
    with pytest.raises(ProviderWriteError) as caught:
        await run(state)
    assert caught.value.code == "content_write_failed"
    assert state.branch_exists
    assert state.branch_ahead == 0


@pytest.mark.asyncio
async def test_transport_exception_is_mapped_to_stable_error_code():
    def broken(_request):
        raise httpx.ConnectError("secret low-level detail")

    async with httpx.AsyncClient(transport=httpx.MockTransport(broken)) as client:
        with pytest.raises(ProviderWriteError) as caught:
            await submit_github_review(
                client=client,
                target=target(),
                token="server-only-token",
                event=event(),
                request_hash=feedback_event_request_hash(event()),
            )
    assert caught.value.code == "provider_transport_failed"
    assert "secret low-level detail" not in str(caught.value)


@pytest.mark.asyncio
async def test_broken_pipe_after_branch_creation_is_reconciled_in_same_submission():
    state = GitHubState()
    state.break_after_branch_create = True
    receipt = await run(state)
    assert receipt["status"] == "accepted"
    assert state.content is not None
    assert state.pull_url


@pytest.mark.asyncio
async def test_broken_pipe_after_content_commit_is_reconciled_in_same_submission():
    state = GitHubState()
    state.break_after_content_write = True
    receipt = await run(state)
    assert receipt["status"] == "accepted"
    assert state.content is not None
    assert state.pull_url


@pytest.mark.asyncio
async def test_broken_pipe_after_pull_creation_is_reconciled_as_replay():
    state = GitHubState()
    state.break_after_pull_create = True
    receipt = await run(state)
    assert receipt["status"] == "replay"
    assert receipt["review_url"] == state.pull_url


def test_provider_branch_keeps_full_192_bit_event_nonce():
    from _sphinx_ext._sphinx_feedback._service._github import _branch

    branch = _branch(event())
    assert branch == "feedback/" + "c" * 48


@pytest.mark.asyncio
async def test_concurrent_content_create_race_reconciles_exact_event():
    state = GitHubState()
    state.put_status = 422
    state.content_race = True
    receipt = await run(state)
    assert receipt["status"] == "accepted"
    assert state.content is not None
    assert state.pull_url


def test_review_url_is_pinned_to_public_github_host_and_repository():
    assert (
        _review_url("https://github.com/org/repo/pull/1", repo="org/repo")
        == "https://github.com/org/repo/pull/1"
    )
    for value in [
        "https://evil.example/org/repo/pull/1",
        "https://github.com.evil.example/x",
        "http://github.com/x",
        "https://github.com/other/repo/pull/1",
        "https://github.com/org/repo/issues/1",
        "https://github.com/org/repo/pull/1?utm=tracking",
    ]:
        with pytest.raises(ProviderWriteError, match="review URL|configured repository"):
            _review_url(value, repo="org/repo")


@pytest.mark.asyncio
async def test_provider_requests_never_follow_redirects_even_with_redirecting_shared_client():
    seen_hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "api.github.com":
            return httpx.Response(302, headers={"location": "https://evil.example/steal"})
        return httpx.Response(200, json={"leaked": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        with pytest.raises(ProviderWriteError):
            await submit_github_review(
                client=client,
                target=target(),
                token="server-only-token",
                event=event(),
                request_hash=feedback_event_request_hash(event()),
            )
    assert seen_hosts == ["api.github.com"]


@pytest.mark.asyncio
async def test_provider_response_fragmentation_is_bounded_independently_of_bytes():
    class TinyChunkStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(1025):
                yield b"x"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=TinyChunkStream(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderWriteError) as caught:
            await _bounded_json(
                client,
                "GET",
                "https://api.github.com/repos/org/repo",
                headers={"Authorization": "Bearer server-only-token"},
            )
    assert caught.value.code == "provider_response_too_fragmented"


@pytest.mark.asyncio
async def test_branch_mutation_during_review_open_fails_closed_before_success_receipt():
    state = GitHubState()
    state.mutate_branch_after_pull = True
    with pytest.raises(FeedbackConflictError, match="unexpected repository changes"):
        await run(state)
    assert state.pull_url == "https://github.com/org/repo/pull/1"


def test_provider_user_agent_uses_package_version():
    from _sphinx_ext._sphinx_feedback import __version__

    assert _headers("token")["User-Agent"] == f"sphinx-feedback/{__version__}"


def test_provider_review_page_display_cannot_inject_markdown_mentions():
    rendered = _review_page_text("guide/`x`-@org/team (draft)")
    assert rendered == "guide/%60x%60-%40org/team%20%28draft%29"
    assert "@" not in rendered
    assert "`" not in rendered


@pytest.mark.asyncio
async def test_provider_rejects_request_commitment_not_implied_by_durable_event():
    state = GitHubState()
    with pytest.raises(ProviderWriteError) as caught:
        await run(state, request_hash="d" * 64)
    assert caught.value.code == "request_commitment_mismatch"
    assert state.calls == []


@pytest.mark.asyncio
async def test_review_branch_may_never_equal_configured_base_branch():
    state = GitHubState()
    colliding = target()
    colliding = StorageTarget(
        id=colliding.id,
        label=colliding.label,
        provider=colliding.provider,
        role=colliding.role,
        authority=colliding.authority,
        repo=colliding.repo,
        branch="feedback/" + "c" * 48,
        feedback_path=colliding.feedback_path,
        token_env=colliding.token_env,
        expose_links=colliding.expose_links,
    )
    with pytest.raises(ProviderWriteError) as caught:
        await run(state, target_value=colliding)
    assert caught.value.code == "branch_namespace_conflict"
    assert state.calls == []
