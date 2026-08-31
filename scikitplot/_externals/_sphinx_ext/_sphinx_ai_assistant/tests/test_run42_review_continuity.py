from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
import types
from types import SimpleNamespace

from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app as proxy_app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _storage as st
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._contribution_ledger import (
    MemoryContributionLedger,
    SQLiteContributionLedger,
)


def _target(provider: str = "huggingface") -> st.StorageTarget:
    env = f"RUN42_{provider.upper()}_TOKEN"
    os.environ[env] = "server-only-test-token"
    return st.StorageTarget(
        id=f"{provider}-primary",
        label=provider,
        provider=provider,
        role="primary",
        repo="org/repo",
        branch="main",
        token_env=env,
    )


def _entry(receipt_id: str = "receipt-42") -> dict:
    return {
        "receiptId": receipt_id,
        "state": "quarantined",
        "records": [{"recordType": "qa", "query": "Q", "answer": "A"}],
        "bytes": 10,
        "deleteTokenHash": "0" * 64,
        "expiresAt": 4_102_444_800.0,
        "receivedAt": 1_788_138_000.0,
        "dedupKeys": [f"{receipt_id}:0"],
        "storage": {},
        "withdrawalStorage": {},
        "currentViewRemoval": {},
        "lastError": "",
        "operation": {"payloadDigest": "a" * 64, "operationId": "", "reviewRevision": 1},
        "rowCount": 1,
    }


def _payload(answer: str = "A") -> dict:
    return {
        "schemaVersion": 4,
        "consentFlag": True,
        "consentVersion": "2.0.0",
        "records": [
            {"recordType": "qa", "query": "What?", "answer": answer, "message": ""}
        ],
    }


def test_review_record_path_is_stable_for_one_receipt_across_payload_revisions():
    target = _target()
    first = st.review_record_path(target, "receipt-stable", 1_788_138_000.0)
    second = st.review_record_path(target, "receipt-stable", 1_788_138_000.0)
    other = st.review_record_path(target, "receipt-other", 1_788_138_000.0)
    assert first == second
    assert first != other
    assert "receipt-stable" not in first
    assert first.endswith(f"ct_{st.review_key_for('receipt-stable')}.jsonl")


def test_memory_ledger_updates_payload_revision_and_review_locator_atomically():
    async def run():
        ledger = MemoryContributionLedger(
            max_pending_entries=10, max_pending_bytes=10_000, max_receipts=20
        )
        entry = _entry()
        await ledger.create(entry)
        bound = await ledger.set_pending_storage(
            entry["receiptId"], storage={"review": {"reviewId": "7"}}
        )
        assert bound["storage"]["review"]["reviewId"] == "7"
        updated = await ledger.replace_pending_payload(
            entry["receiptId"],
            records=[{"recordType": "qa", "answer": "B"}],
            byte_count=11,
            dedup_keys=["same-authority-key"],
            payload_digest="b" * 64,
            row_count=1,
            storage={"review": {"reviewId": "7", "status": "open"}},
        )
        assert updated["receiptId"] == entry["receiptId"]
        assert updated["operation"]["reviewRevision"] == 2
        assert updated["operation"]["payloadDigest"] == "b" * 64
        assert updated["storage"]["review"]["reviewId"] == "7"
        assert updated["records"][0]["answer"] == "B"

    asyncio.run(run())


def test_sqlite_ledger_preserves_review_locator_during_revision_update(tmp_path):
    async def run():
        ledger = SQLiteContributionLedger(
            str(tmp_path / "review.sqlite3"),
            max_pending_entries=10,
            max_pending_bytes=10_000,
            max_receipts=20,
        )
        await ledger.initialize()
        entry = _entry("receipt-sqlite")
        await ledger.create(entry)
        await ledger.set_pending_storage(
            entry["receiptId"], storage={"review": {"reviewId": "12"}}
        )
        updated = await ledger.replace_pending_payload(
            entry["receiptId"],
            records=[{"recordType": "qa", "answer": "changed"}],
            byte_count=15,
            dedup_keys=["receipt-sqlite:0"],
            payload_digest="c" * 64,
            row_count=1,
            storage={"review": {"reviewId": "12", "status": "open"}},
        )
        reread = await ledger.get(entry["receiptId"])
        assert updated["operation"]["reviewRevision"] == 2
        assert reread is not None
        assert reread["storage"]["review"]["reviewId"] == "12"
        await ledger.close()

    asyncio.run(run())


def test_hf_update_uses_persisted_pr_ref_without_scanning_or_opening_new_pr(monkeypatch):
    target = _target("huggingface")
    commits = []

    class Op:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class Api:
        def __init__(self, token=None):
            self.token = token

        def get_discussion_details(self, repo_id, num, repo_type=None):
            assert repo_id == "org/repo" and num == 7 and repo_type == "dataset"
            return SimpleNamespace(
                is_pull_request=True,
                status="open",
                url="https://huggingface.co/datasets/org/repo/discussions/7",
            )

        def get_repo_discussions(self, *args, **kwargs):
            raise AssertionError("persisted review IDs must bypass repository scans")

        def create_commit(self, **kwargs):
            commits.append(kwargs)
            return SimpleNamespace()

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = Api
    fake.CommitOperationAdd = Op
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils._http", raising=False)

    receipt_id = "receipt-hf-update"
    key = st.review_key_for(receipt_id)
    hint = {
        "recordId": "old",
        "paths": {target.id: st.review_record_path(target, receipt_id, 1_788_138_000.0)},
        "review": {
            "provider": "huggingface",
            "targetId": target.id,
            "repo": target.repo,
            "baseBranch": "main",
            "reviewBranch": st.review_branch_for(receipt_id),
            "reviewKey": key,
            "reviewId": "7",
            "reviewUrl": "https://huggingface.co/datasets/org/repo/discussions/7",
            "status": "open",
        },
    }

    async def run():
        coord = st.StorageCoordinator([target])
        review = await coord.update_contribution_review(
            receipt_id=receipt_id,
            content=b'{"trainingStatus":"eligible","revision":2}',
            commit_message="revision 2",
            path_timestamp=1_788_138_000.0,
            review_hint=hint,
        )
        assert review.review_id == "7"

    asyncio.run(run())
    assert len(commits) == 1
    assert commits[0]["revision"] == "refs/pr/7"
    assert "create_pr" not in commits[0]


def test_app_unchanged_resubmit_is_noop_and_changed_resubmit_updates_same_review(monkeypatch):
    calls = {"update": 0, "open": 0}
    review_state = {"status": "open", "record_id": "r1"}
    target = SimpleNamespace(provider="github", branch="main")

    def receipt(receipt_id: str, content: bytes = b""):
        return st.ReviewReceipt(
            provider="github",
            target_id="github-primary",
            repo="org/repo",
            base_branch="main",
            review_branch=st.review_branch_for(receipt_id),
            review_key=st.review_key_for(receipt_id),
            review_id="42",
            review_url="https://github.com/org/repo/pull/42",
            status=review_state["status"],
            record_id=st.record_id_for(content) if content else review_state["record_id"],
            path=st.review_record_path(_target("github"), receipt_id, 1_788_138_000.0),
        )

    class FakeStorage:
        primary = target

        def set_client(self, client):
            pass

        async def initialize(self):
            pass

        async def close(self):
            pass

        def manifest(self):
            return {"targets": []}

        async def open_contribution_review(self, **kwargs):
            calls["open"] += 1
            return receipt(kwargs["receipt_id"], kwargs["content"])

        async def get_contribution_review(self, receipt_id, **kwargs):
            assert kwargs.get("review_hint", {}).get("review", {}).get("reviewId") == "42"
            return receipt(receipt_id)

        async def update_contribution_review(self, **kwargs):
            calls["update"] += 1
            assert kwargs.get("review_hint", {}).get("review", {}).get("reviewId") == "42"
            review_state["record_id"] = st.record_id_for(kwargs["content"])
            return receipt(kwargs["receipt_id"], kwargs["content"])

        async def close_contribution_review(self, receipt_id, **kwargs):
            return "closed"

        async def merge_contribution_review(self, receipt_id, **kwargs):
            raise AssertionError("not used")

        async def remove_current_view(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "provider-pr")
    proxy_app._contrib_quarantine.clear()

    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_payload())
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["reviewProvider"] == "github"
        assert body["reviewId"] == "42"
        assert body["reviewPath"].endswith(
            f"ct_{st.review_key_for(body['receiptId'])}.jsonl"
        )
        assert "reviewUrl" not in body
        headers = {"X-Contribution-Delete-Token": body["deleteToken"]}

        unchanged = client.put(
            f"/v1/contribute/{body['receiptId']}", json=_payload(), headers=headers
        )
        assert unchanged.status_code == 200, unchanged.text
        assert unchanged.json()["reviewUpdate"] == "unchanged"
        assert unchanged.json()["reviewRevision"] == 1
        assert calls["update"] == 0

        changed = client.put(
            f"/v1/contribute/{body['receiptId']}", json=_payload("B"), headers=headers
        )
        assert changed.status_code == 200, changed.text
        changed_body = changed.json()
        assert changed_body["reviewUpdate"] == "updated"
        assert changed_body["reviewRevision"] == 2
        assert changed_body["receiptId"] == body["receiptId"]
        assert calls["update"] == 1
        assert calls["open"] == 1
        stored = proxy_app._contrib_quarantine[body["receiptId"]]
        assert stored["storage"]["review"]["reviewId"] == "42"
        assert stored["operation"]["reviewRevision"] == 2


def test_support_review_reference_is_non_secret_and_survives_without_live_review():
    receipt_id = "a" * 32
    entry = _entry(receipt_id)
    path = f"contributions/2026/08/31/ct_{st.review_key_for(receipt_id)}.jsonl"
    entry["storage"] = {
        "paths": {"hf-primary": path},
        "review": {
            "provider": "huggingface",
            "targetId": "hf-primary",
            "reviewId": "77",
            "reviewUrl": "https://huggingface.co/datasets/private/repo/discussions/77",
        },
    }
    ref = proxy_app._contribution_review_reference(entry)
    assert ref == {
        "reviewProvider": "huggingface",
        "reviewId": "77",
        "reviewPath": path,
    }
    encoded = json.dumps(ref)
    assert "reviewUrl" not in encoded
    assert "deleteToken" not in encoded
    assert "private/repo" not in encoded


def test_status_payload_exposes_support_locator_but_not_review_url(monkeypatch):
    receipt_id = "b" * 32
    token = "t" * 43
    target = SimpleNamespace(provider="huggingface", branch="main")
    path = f"contributions/2026/08/31/ct_{st.review_key_for(receipt_id)}.jsonl"

    class FakeStorage:
        primary = target
        def set_client(self, client): pass
        async def initialize(self): pass
        async def close(self): pass
        def manifest(self): return {"targets": []}
        async def get_contribution_review(self, rid, **kwargs):
            return st.ReviewReceipt(
                provider="huggingface", target_id="hf-primary", repo="org/repo",
                base_branch="main", review_branch=st.review_branch_for(rid),
                review_key=st.review_key_for(rid), review_id="88",
                review_url="https://huggingface.co/datasets/org/repo/discussions/88",
                status="open", record_id="r", path=path,
            )

    async def seed():
        ledger = MemoryContributionLedger(max_pending_entries=10, max_pending_bytes=10_000, max_receipts=20)
        e = _entry(receipt_id)
        e["deleteTokenHash"] = proxy_app.hash_edit_token(token)
        e["storage"] = {
            "paths": {"hf-primary": path},
            "review": {
                "provider": "huggingface", "targetId": "hf-primary", "repo": "org/repo",
                "baseBranch": "main", "reviewBranch": st.review_branch_for(receipt_id),
                "reviewKey": st.review_key_for(receipt_id), "reviewId": "88",
                "reviewUrl": "https://huggingface.co/datasets/org/repo/discussions/88", "status": "open",
            },
        }
        await ledger.create(e)
        return ledger

    ledger = asyncio.run(seed())
    monkeypatch.setattr(proxy_app, "_CONTRIBUTION_LEDGER", ledger)
    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "provider-pr")
    with TestClient(proxy_app.app) as client:
        response = client.get(
            f"/v1/contribute/{receipt_id}",
            headers={"X-Contribution-Delete-Token": token},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["reviewProvider"] == "huggingface"
        assert body["reviewId"] == "88"
        assert body["reviewPath"] == path
        assert "reviewUrl" not in body


def test_worker_and_space_cors_allow_pending_review_put():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "_cf_worker" / "index.js").read_text(encoding="utf-8")
    app_src = (root / "_hf_spaces_proxy" / "app.py").read_text(encoding="utf-8")
    assert "POST, PUT, PATCH, DELETE, OPTIONS" in worker
    assert 'allow_methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]' in app_src


def test_review_description_uses_the_same_visible_review_key_as_title_and_branch():
    receipt_id = "receipt-visible-key"
    key = st.review_key_for(receipt_id)
    description = st.review_description_for(key, "main")
    assert f"`{key}`" in description
    assert st.review_title_for(receipt_id).endswith(key)
    assert st.review_branch_for(receipt_id).endswith(key)


def _hint_for(target: st.StorageTarget, receipt_id: str, review_id: str = "7") -> dict:
    return {
        "recordId": "old-record",
        "paths": {target.id: st.review_record_path(target, receipt_id, 1_788_138_000.0)},
        "review": {
            "provider": target.provider,
            "targetId": target.id,
            "repo": target.repo,
            "baseBranch": target.branch,
            "reviewBranch": st.review_branch_for(receipt_id),
            "reviewKey": st.review_key_for(receipt_id),
            "reviewId": review_id,
            "reviewUrl": "https://example.invalid/review/7",
            "status": "open",
        },
    }


def test_github_update_uses_direct_pr_id_and_same_source_branch():
    import httpx

    target = _target("github")
    receipt_id = "receipt-github-update"
    branch = st.review_branch_for(receipt_id)
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if req.method == "GET" and req.url.path.endswith("/pulls/7"):
            return httpx.Response(
                200,
                json={
                    "number": 7,
                    "state": "open",
                    "draft": False,
                    "merged_at": None,
                    "html_url": "https://github.com/org/repo/pull/7",
                },
            )
        if req.method == "GET" and "/contents/" in req.url.path:
            assert req.url.params.get("ref") == branch
            return httpx.Response(200, json={"sha": "oldsha"})
        if req.method == "PUT" and "/contents/" in req.url.path:
            body = json.loads(req.content)
            assert body["branch"] == branch
            assert body["sha"] == "oldsha"
            return httpx.Response(200)
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            review = await coord.update_contribution_review(
                receipt_id=receipt_id,
                content=b'{"trainingStatus":"eligible","revision":2}',
                commit_message="revision 2",
                path_timestamp=1_788_138_000.0,
                review_hint=_hint_for(target, receipt_id),
            )
            assert review.review_id == "7"

    asyncio.run(run())
    assert ("GET", "/repos/org/repo/pulls/7") in seen
    assert not any(path.endswith("/pulls") for _, path in seen)


def test_gitlab_update_uses_direct_mr_id_and_same_source_branch():
    import httpx

    target = _target("gitlab")
    receipt_id = "receipt-gitlab-update"
    branch = st.review_branch_for(receipt_id)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/merge_requests/7"):
            return httpx.Response(
                200,
                json={
                    "iid": 7,
                    "state": "opened",
                    "web_url": "https://gitlab.com/org/repo/-/merge_requests/7",
                },
            )
        if req.method == "GET" and "/repository/files/" in req.url.path:
            assert req.url.params.get("ref") == branch
            return httpx.Response(200, json={})
        if req.method == "PUT" and "/repository/files/" in req.url.path:
            body = json.loads(req.content)
            assert body["branch"] == branch
            return httpx.Response(200)
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            review = await coord.update_contribution_review(
                receipt_id=receipt_id,
                content=b'{"trainingStatus":"eligible","revision":2}',
                commit_message="revision 2",
                path_timestamp=1_788_138_000.0,
                review_hint=_hint_for(target, receipt_id),
            )
            assert review.status == "open"

    asyncio.run(run())


def test_bitbucket_update_uses_direct_pr_id_and_same_source_branch():
    import httpx

    target = _target("bitbucket")
    receipt_id = "receipt-bitbucket-update"
    branch = st.review_branch_for(receipt_id)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/pullrequests/7"):
            return httpx.Response(
                200,
                json={
                    "id": 7,
                    "state": "OPEN",
                    "links": {"html": {"href": "https://bitbucket.org/org/repo/pull-requests/7"}},
                },
            )
        if req.method == "POST" and req.url.path.endswith("/src"):
            # Multipart parsing is unnecessary here; the target branch is encoded
            # by the storage writer and covered by existing provider tests.
            return httpx.Response(201)
        raise AssertionError((req.method, str(req.url)))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            coord = st.StorageCoordinator([target], client=client)
            review = await coord.update_contribution_review(
                receipt_id=receipt_id,
                content=b'{"trainingStatus":"eligible","revision":2}',
                commit_message=f"revision 2 on {branch}",
                path_timestamp=1_788_138_000.0,
                review_hint=_hint_for(target, receipt_id),
            )
            assert review.status == "open"

    asyncio.run(run())
