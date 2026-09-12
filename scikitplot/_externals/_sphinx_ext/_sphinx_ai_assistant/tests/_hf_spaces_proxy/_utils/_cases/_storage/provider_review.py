from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app as proxy_app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _storage as st


def _target(provider: str, repo: str = "org/repo") -> st.StorageTarget:
    env = f"AI_RECORD_STORAGE_TOKEN_{provider.upper()}"
    import os
    os.environ[env] = "server-only-token"
    return st.StorageTarget(
        id=f"{provider}-primary", label=provider, provider=provider, role="primary",
        repo=repo, branch="main", token_env=env,
    )


def test_review_key_is_opaque_and_branch_safe():
    receipt = "user-visible-receipt-123"
    key = st.review_key_for(receipt)
    branch = st.review_branch_for(receipt)
    assert receipt not in key and receipt not in branch
    assert len(key) == 24
    assert branch == f"ai-contrib-{key}"
    assert st.review_title_for(receipt) == f"Dataset contribution {key}"


def test_github_opens_native_pull_request_on_isolated_branch():
    target = _target("github")
    seen = []
    branch = st.review_branch_for("receipt-github")

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, str(req.url), req.content))
        path = req.url.path
        if req.method == "GET" and "/git/ref/heads/" in path:
            return httpx.Response(200, json={"object": {"sha": "abc123"}})
        if req.method == "GET" and path.endswith("/pulls"):
            return httpx.Response(200, json=[])
        if req.method == "POST" and path.endswith("/git/refs"):
            return httpx.Response(201)
        if req.method == "PUT" and "/contents/" in path:
            body = json.loads(req.content)
            assert body["branch"] == branch
            return httpx.Response(201)
        if req.method == "POST" and path.endswith("/pulls"):
            body = json.loads(req.content)
            assert body["head"] == branch and body["base"] == "main"
            return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/org/repo/pull/7"})
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            r = await coord.open_contribution_review(
                receipt_id="receipt-github", content=b'{"trainingStatus":"eligible"}',
                commit_message="review", path_timestamp=1788138000,
            )
            assert r.provider == "github" and r.status == "open"
            assert r.review_url.endswith("/pull/7")
            assert r.review_branch == branch
            assert r.path.startswith("contributions/")
    asyncio.run(run())
    assert all(b"server-only-token" not in body for _, _, body in seen)


def test_gitlab_opens_native_merge_request():
    target = _target("gitlab", "group/subgroup/repo")
    branch = st.review_branch_for("receipt-gitlab")

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.method == "GET" and path.endswith("/merge_requests"):
            return httpx.Response(200, json=[])
        if req.method == "POST" and path.endswith("/repository/branches"):
            return httpx.Response(201)
        if req.method == "POST" and "/repository/files/" in path:
            body = json.loads(req.content)
            assert body["branch"] == branch
            return httpx.Response(201)
        if req.method == "POST" and path.endswith("/merge_requests"):
            body = json.loads(req.content)
            assert body["source_branch"] == branch and body["target_branch"] == "main"
            return httpx.Response(201, json={"iid": 12, "web_url": "https://gitlab.com/group/subgroup/repo/-/merge_requests/12"})
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            r = await coord.open_contribution_review(
                receipt_id="receipt-gitlab", content=b'{}', commit_message="review", path_timestamp=1788138000
            )
            assert r.provider == "gitlab" and r.review_id == "12" and r.status == "open"
    asyncio.run(run())


def test_bitbucket_opens_native_pull_request():
    target = _target("bitbucket")
    branch = st.review_branch_for("receipt-bitbucket")

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.method == "GET" and path.endswith("/pullrequests"):
            return httpx.Response(200, json={"values": []})
        if req.method == "POST" and path.endswith("/refs/branches"):
            return httpx.Response(201)
        if req.method == "POST" and path.endswith("/src"):
            return httpx.Response(201)
        if req.method == "POST" and path.endswith("/pullrequests"):
            body = json.loads(req.content)
            assert body["source"]["branch"]["name"] == branch
            assert body["destination"]["branch"]["name"] == "main"
            return httpx.Response(201, json={
                "id": 9, "links": {"html": {"href": "https://bitbucket.org/org/repo/pull-requests/9"}}
            })
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            r = await coord.open_contribution_review(
                receipt_id="receipt-bitbucket", content=b'{}', commit_message="review", path_timestamp=1788138000
            )
            assert r.provider == "bitbucket" and r.review_id == "9" and r.status == "open"
    asyncio.run(run())


def test_huggingface_uses_create_commit_create_pr(monkeypatch):
    target = _target("huggingface")
    calls = []

    class Op:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class Api:
        def __init__(self, token=None): self.token = token
        def get_repo_discussions(self, *a, **k): return iter(())
        def create_commit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(pr_url="https://huggingface.co/datasets/org/repo/discussions/5")

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = Api
    fake.CommitOperationAdd = Op
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    # Compatibility path: no huggingface_hub.utils._http in this focused fake.
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils._http", raising=False)

    async def run():
        coord = st.StorageCoordinator([target])
        r = await coord.open_contribution_review(
            receipt_id="receipt-hf", content=b'{}', commit_message="review", path_timestamp=1788138000
        )
        assert r.provider == "huggingface" and r.review_id == "5"
        assert calls[0]["create_pr"] is True
        assert calls[0]["revision"] == "main"
    asyncio.run(run())


def test_github_manual_merge_state_is_detected():
    target = _target("github")
    branch = st.review_branch_for("receipt-merged")

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/pulls"):
            return httpx.Response(200, json=[{
                "number": 4, "state": "closed", "merged_at": "2026-08-31T00:00:00Z",
                "draft": False, "html_url": "https://github.com/org/repo/pull/4",
                "head": {"ref": branch},
            }])
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            r = await coord.get_contribution_review("receipt-merged")
            assert r is not None and r.status == "merged"
    asyncio.run(run())


def _contribution_payload():
    return {
        "schemaVersion": 4, "consentFlag": True, "consentVersion": "2.0.0",
        "records": [{"recordType": "qa", "query": "Q", "answer": "A", "message": ""}],
    }


def test_app_provider_pr_mode_submits_eligible_future_main_bytes(monkeypatch):
    captured = {}
    target = SimpleNamespace(provider="github", branch="main")

    class FakeStorage:
        primary = target
        def set_client(self, client): pass
        async def initialize(self): pass
        async def close(self): pass
        def manifest(self): return {"targets": []}
        async def open_contribution_review(self, **kwargs):
            captured.update(kwargs)
            return st.ReviewReceipt(
                provider="github", target_id="github-primary", repo="org/repo",
                base_branch="main", review_branch=st.review_branch_for(kwargs["receipt_id"]),
                review_key=st.review_key_for(kwargs["receipt_id"]), review_id="8",
                review_url="https://github.com/org/repo/pull/8", status="open",
                record_id=st.record_id_for(kwargs["content"]), path="contributions/x.jsonl",
            )
        async def get_contribution_review(self, receipt_id, **kwargs): return None
        async def close_contribution_review(self, receipt_id, **kwargs): return "closed"
        async def merge_contribution_review(self, receipt_id, **kwargs): raise AssertionError("not used")
        async def remove_current_view(self, *a, **k): return {}

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "provider-pr")
    proxy_app._contrib_quarantine.clear()
    with TestClient(proxy_app.app) as client:
        res = client.post("/v1/contribute", json=_contribution_payload())
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "quarantined"
        assert body["reviewMode"] == "provider-pr"
        assert body["reviewProvider"] == "github"
        rows = [json.loads(line) for line in captured["content"].decode().splitlines()]
        assert rows[0]["trainingStatus"] == "eligible"
        assert proxy_app._contrib_quarantine[body["receiptId"]]["records"][0]["trainingStatus"] == "quarantined"


def test_provider_pr_pending_delete_closes_review_before_local_tombstone(monkeypatch):
    closed = []
    target = SimpleNamespace(provider="gitlab", branch="main")

    class FakeStorage:
        primary = target
        def set_client(self, client): pass
        async def initialize(self): pass
        async def close(self): pass
        def manifest(self): return {"targets": []}
        async def open_contribution_review(self, **kwargs):
            return st.ReviewReceipt(
                provider="gitlab", target_id="gitlab-primary", repo="org/repo", base_branch="main",
                review_branch=st.review_branch_for(kwargs["receipt_id"]), review_key=st.review_key_for(kwargs["receipt_id"]),
                review_id="1", review_url="https://gitlab.com/org/repo/-/merge_requests/1", status="open",
                record_id=st.record_id_for(kwargs["content"]), path="contributions/x.jsonl")
        async def get_contribution_review(self, receipt_id, **kwargs):
            return st.ReviewReceipt(
                provider="gitlab", target_id="gitlab-primary", repo="org/repo", base_branch="main",
                review_branch=st.review_branch_for(receipt_id), review_key=st.review_key_for(receipt_id),
                review_id="1", review_url="https://gitlab.com/org/repo/-/merge_requests/1", status="open",
                record_id="", path="contributions/x.jsonl")
        async def close_contribution_review(self, receipt_id, **kwargs): closed.append(receipt_id); return "closed"
        async def merge_contribution_review(self, receipt_id, **kwargs): raise AssertionError
        async def remove_current_view(self, *a, **k): return {}

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "provider-pr")
    proxy_app._contrib_quarantine.clear()
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_contribution_payload()).json()
        deleted = client.delete(
            f"/v1/contribute/{created['receiptId']}",
            headers={"X-Contribution-Delete-Token": created["deleteToken"]},
        )
        assert deleted.status_code == 200
        assert closed == [created["receiptId"]]
        assert proxy_app._contrib_quarantine[created["receiptId"]]["state"] == "deleted"


def test_manual_provider_ui_merge_ratchets_receipt_to_eligible(monkeypatch):
    state = {"review": "open"}
    target = SimpleNamespace(provider="github", branch="main")

    class FakeStorage:
        primary = target
        def set_client(self, client): pass
        async def initialize(self): pass
        async def close(self): pass
        def manifest(self): return {"targets": []}
        async def open_contribution_review(self, **kwargs):
            return st.ReviewReceipt(
                provider="github", target_id="github-primary", repo="org/repo", base_branch="main",
                review_branch=st.review_branch_for(kwargs["receipt_id"]), review_key=st.review_key_for(kwargs["receipt_id"]),
                review_id="2", review_url="https://github.com/org/repo/pull/2", status=state["review"],
                record_id=st.record_id_for(kwargs["content"]), path="contributions/final.jsonl")
        async def get_contribution_review(self, receipt_id, **kwargs):
            return st.ReviewReceipt(
                provider="github", target_id="github-primary", repo="org/repo", base_branch="main",
                review_branch=st.review_branch_for(receipt_id), review_key=st.review_key_for(receipt_id),
                review_id="2", review_url="https://github.com/org/repo/pull/2", status=state["review"],
                record_id="", path="contributions/final.jsonl")
        async def close_contribution_review(self, receipt_id, **kwargs): return "closed"
        async def merge_contribution_review(self, receipt_id, **kwargs): raise AssertionError
        async def remove_current_view(self, *a, **k): return {}

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "provider-pr")
    proxy_app._contrib_quarantine.clear()
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_contribution_payload()).json()
        state["review"] = "merged"
        status = client.get(
            f"/v1/contribute/{created['receiptId']}",
            headers={"X-Contribution-Delete-Token": created["deleteToken"]},
        )
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "eligible"
        assert body["trainingEligible"] is True
        assert proxy_app._contrib_quarantine[created["receiptId"]]["records"] == []
        assert proxy_app._contrib_quarantine[created["receiptId"]]["storage"]["review"]["status"] == "merged"


def test_closed_provider_review_remains_noneligible_and_reports_review_state(monkeypatch):
    target = SimpleNamespace(provider="bitbucket", branch="main")

    class FakeStorage:
        primary = target
        def set_client(self, client): pass
        async def initialize(self): pass
        async def close(self): pass
        def manifest(self): return {"targets": []}
        async def open_contribution_review(self, **kwargs):
            return st.ReviewReceipt(
                provider="bitbucket", target_id="bitbucket-primary", repo="org/repo", base_branch="main",
                review_branch=st.review_branch_for(kwargs["receipt_id"]), review_key=st.review_key_for(kwargs["receipt_id"]),
                review_id="3", review_url="https://bitbucket.org/org/repo/pull-requests/3", status="closed",
                record_id=st.record_id_for(kwargs["content"]), path="contributions/final.jsonl")
        async def get_contribution_review(self, receipt_id, **kwargs):
            return st.ReviewReceipt(
                provider="bitbucket", target_id="bitbucket-primary", repo="org/repo", base_branch="main",
                review_branch=st.review_branch_for(receipt_id), review_key=st.review_key_for(receipt_id),
                review_id="3", review_url="https://bitbucket.org/org/repo/pull-requests/3", status="closed",
                record_id="", path="contributions/final.jsonl")
        async def close_contribution_review(self, receipt_id, **kwargs): return "closed"
        async def merge_contribution_review(self, receipt_id, **kwargs): raise AssertionError
        async def remove_current_view(self, *a, **k): return {}

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "provider-pr")
    proxy_app._contrib_quarantine.clear()
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_contribution_payload()).json()
        status = client.get(
            f"/v1/contribute/{created['receiptId']}",
            headers={"X-Contribution-Delete-Token": created["deleteToken"]},
        )
        body = status.json()
        assert body["status"] == "quarantined"
        assert body["reviewStatus"] == "closed"
        assert body["trainingEligible"] is False


def test_gitlab_manual_merge_state_is_detected():
    target = _target("gitlab")
    branch = st.review_branch_for("receipt-gitlab-merged")

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/merge_requests"):
            return httpx.Response(200, json=[{
                "iid": 17, "state": "merged", "source_branch": branch,
                "target_branch": "main", "web_url": "https://gitlab.com/org/repo/-/merge_requests/17",
            }])
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            r = await coord.get_contribution_review("receipt-gitlab-merged")
            assert r is not None and r.status == "merged" and r.review_id == "17"
    asyncio.run(run())


def test_bitbucket_manual_merge_state_is_detected():
    target = _target("bitbucket")
    branch = st.review_branch_for("receipt-bitbucket-merged")

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/pullrequests"):
            state = req.url.params.get("state")
            if state == "MERGED":
                return httpx.Response(200, json={"values": [{
                    "id": 21, "state": "MERGED",
                    "source": {"branch": {"name": branch}},
                    "links": {"html": {"href": "https://bitbucket.org/org/repo/pull-requests/21"}},
                }]})
            return httpx.Response(200, json={"values": []})
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            r = await coord.get_contribution_review("receipt-bitbucket-merged")
            assert r is not None and r.status == "merged" and r.review_id == "21"
    asyncio.run(run())


def test_huggingface_manual_merge_state_is_detected(monkeypatch):
    target = _target("huggingface")
    title = st.review_title_for("receipt-hf-merged")

    class Api:
        def __init__(self, token=None): self.token = token
        def get_repo_discussions(self, *a, **k):
            return iter([SimpleNamespace(
                title=title, is_pull_request=True, num=23, status="merged"
            )])

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils._http", raising=False)

    async def run():
        coord = st.StorageCoordinator([target])
        r = await coord.get_contribution_review("receipt-hf-merged")
        assert r is not None and r.status == "merged" and r.review_id == "23"
    asyncio.run(run())
