from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app as proxy_app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _storage as st
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._contribution_ledger import MemoryContributionLedger


def _payload(*, value: int = 1, mode: str = "quick", note: str = "") -> dict:
    return {
        "schemaVersion": 1,
        "consentFlag": True,
        "consentVersion": "2.0.0",
        "trainingConsentFlag": True,
        "trainingConsentVersion": "1.0.0",
        "feedbackId": "feedback-event-1",
        "answerIndex": 2,
        "ratingValue": value,
        "ratingLabel": "helpful" if value > 0 else "not_helpful",
        "ratingTitle": "Helpful" if value > 0 else "Not helpful",
        "ratingMode": mode,
        "ratingScaleMin": -1 if mode == "quick" else -2,
        "ratingScaleMax": 1 if mode == "quick" else 2,
        "message": note,
        "query": "What does this page cover?",
        "answer": "It covers the documented API.",
        "model": {"provider": "huggingface", "model": "Qwen/test"},
        "page": "https://docs.example.test/page",
        "ts": 1_788_138_000_000,
    }


class _FakeStorage:
    def __init__(self):
        self.primary = SimpleNamespace(provider="huggingface", branch="main")
        self.open_calls = 0
        self.update_calls = 0
        self.close_calls = 0
        self.remove_calls = 0
        self.status = "open"
        self.paths: dict[str, str] = {}

    def set_client(self, client):
        pass

    async def initialize(self):
        pass

    async def close(self):
        pass

    def manifest(self):
        return {"targets": []}

    def primary_ready(self):
        return True

    def _receipt(self, receipt_id: str, content: bytes | None = None):
        path = self.paths.setdefault(
            receipt_id,
            f"feedback/2026/09/01/fb_{st.feedback_review_key_for(receipt_id)}.jsonl",
        )
        return st.ReviewReceipt(
            provider="huggingface",
            target_id="hf-primary",
            repo="org/feedback",
            base_branch="main",
            review_branch=st.feedback_review_branch_for(receipt_id),
            review_key=st.feedback_review_key_for(receipt_id),
            review_id="45",
            review_url="https://huggingface.co/datasets/org/feedback/discussions/45",
            status=self.status,
            record_id=st.record_id_for(content or b"feedback"),
            path=path,
        )

    async def open_feedback_review(self, **kwargs):
        self.open_calls += 1
        return self._receipt(kwargs["receipt_id"], kwargs["content"])

    async def update_feedback_review(self, **kwargs):
        self.update_calls += 1
        assert kwargs.get("review_hint", {}).get("review", {}).get("reviewId") == "45"
        return self._receipt(kwargs["receipt_id"], kwargs["content"])

    async def get_feedback_review(self, receipt_id, **kwargs):
        return self._receipt(receipt_id)

    async def close_feedback_review(self, receipt_id, **kwargs):
        self.close_calls += 1
        self.status = "closed"
        return "closed"

    async def remove_current_view(self, paths, **kwargs):
        self.remove_calls += 1
        return {"removed": sorted(paths.values())}


def _fresh_ledger():
    return MemoryContributionLedger(
        max_pending_entries=100,
        max_pending_bytes=5_000_000,
        max_receipts=200,
    )


def test_feedback_review_requires_its_own_explicit_consent(monkeypatch):
    storage = _FakeStorage()
    monkeypatch.setattr(proxy_app, "_STORAGE", storage)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER", _fresh_ledger())
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_READY", True)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR", "")
    monkeypatch.setattr(proxy_app, "FEEDBACK_REVIEW_MODE", "provider-pr")
    bad = _payload()
    bad["consentFlag"] = False
    with TestClient(proxy_app.app) as client:
        res = client.post("/v1/feedback/review", json=bad)
        assert res.status_code == 403
    assert storage.open_calls == 0


def test_feedback_review_requires_explicit_training_consent(monkeypatch):
    storage = _FakeStorage()
    monkeypatch.setattr(proxy_app, "_STORAGE", storage)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER", _fresh_ledger())
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_READY", True)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR", "")
    monkeypatch.setattr(proxy_app, "FEEDBACK_REVIEW_MODE", "provider-pr")
    bad = _payload()
    bad["trainingConsentFlag"] = False
    with TestClient(proxy_app.app) as client:
        res = client.post("/v1/feedback/review", json=bad)
        assert res.status_code == 403
    assert storage.open_calls == 0


def test_feedback_quality_ratio_normalizes_multilevel_scale():
    payload = _payload(value=1, mode="panel")
    record = proxy_app.normalize_feedback_review_record(
        payload, server_ts_ms=1_788_138_000_000, receipt_id="b" * 32
    )
    assert record["ratingScaleMin"] == -2.0
    assert record["ratingScaleMax"] == 2.0
    assert record["qualityScore"] == 0.75
    assert record["qualityPercent"] == 75.0


def test_feedback_review_create_update_noop_merge_and_withdraw_training_eligibility(monkeypatch):
    storage = _FakeStorage()
    ledger = _fresh_ledger()
    monkeypatch.setattr(proxy_app, "_STORAGE", storage)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER", ledger)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_READY", True)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR", "")
    monkeypatch.setattr(proxy_app, "FEEDBACK_REVIEW_MODE", "provider-pr")
    proxy_app._feedback_review_rl.clear()

    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/feedback/review", json=_payload())
        assert created.status_code == 200, created.text
        first = created.json()
        assert first["status"] == "in_review"
        assert first["trainingEligible"] is False
        assert first["feedbackReview"] is True
        assert first["reviewProvider"] == "huggingface"
        assert first["reviewId"] == "45"
        assert first["reviewPath"].startswith("feedback/")
        assert "/fb_" in first["reviewPath"]
        assert "reviewUrl" not in first
        token = first["deleteToken"]
        rid = first["receiptId"]
        headers = {"X-Feedback-Review-Token": token}
        assert storage.open_calls == 1

        unchanged = client.put(f"/v1/feedback/review/{rid}", json=_payload(), headers=headers)
        assert unchanged.status_code == 200, unchanged.text
        assert unchanged.json()["reviewUpdate"] == "unchanged"
        assert unchanged.json()["reviewRevision"] == 1
        assert storage.update_calls == 0

        changed = client.put(
            f"/v1/feedback/review/{rid}",
            json=_payload(value=-1, mode="panel", note="The example was unclear."),
            headers=headers,
        )
        assert changed.status_code == 200, changed.text
        assert changed.json()["reviewUpdate"] == "updated"
        assert changed.json()["reviewRevision"] == 2
        assert changed.json()["trainingEligible"] is False
        assert storage.update_calls == 1

        stored = asyncio.run(ledger.get(rid))
        assert stored is not None
        row = stored["records"][0]
        assert row["_source"] == "feedback"
        assert row["trainingStatus"] == "eligible"
        assert row["recordType"] == "qa"
        assert row["query"] == "What does this page cover?"
        assert row["answer"] == "It covers the documented API."
        assert row["message"] == "The example was unclear."
        assert row["ratingScaleMin"] == -2.0
        assert row["ratingScaleMax"] == 2.0
        assert row["qualityScore"] == 0.25
        assert row["qualityPercent"] == 25.0
        assert row["trainingConsentVersion"] == "1.0.0"

        # The PR carries future eligible bytes, but the public lifecycle stays
        # non-eligible until a maintainer actually merges the provider review.
        storage.status = "merged"
        status = client.get(f"/v1/feedback/review/{rid}", headers=headers)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "reviewed"
        assert status.json()["trainingEligible"] is True

        withdrawn = client.delete(f"/v1/feedback/review/{rid}", headers=headers)
        assert withdrawn.status_code == 202, withdrawn.text
        assert withdrawn.json()["status"] == "withdrawn"
        assert withdrawn.json()["trainingEligible"] is False
        assert storage.remove_calls == 1


def test_feedback_review_pending_withdraw_closes_provider_review(monkeypatch):
    storage = _FakeStorage()
    ledger = _fresh_ledger()
    monkeypatch.setattr(proxy_app, "_STORAGE", storage)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER", ledger)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_READY", True)
    monkeypatch.setattr(proxy_app, "_FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR", "")
    monkeypatch.setattr(proxy_app, "FEEDBACK_REVIEW_MODE", "provider-pr")
    proxy_app._feedback_review_rl.clear()
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/feedback/review", json=_payload()).json()
        headers = {"X-Feedback-Review-Token": created["deleteToken"]}
        res = client.delete(f"/v1/feedback/review/{created['receiptId']}", headers=headers)
        assert res.status_code == 200, res.text
        assert res.json()["status"] == "withdrawn"
    assert storage.close_calls == 1
    assert storage.remove_calls == 0


def test_feedback_normalizer_enters_training_builder_only_with_explicit_training_consent():
    record = proxy_app.normalize_feedback_review_record(
        _payload(), server_ts_ms=1_788_138_000_000, receipt_id="a" * 32
    )
    assert record["_source"] == "feedback"
    assert record["trainingStatus"] == "eligible"
    assert record["qualityScore"] == 1.0
    assert record["qualityPercent"] == 100.0
    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import deduplicate_dataset as dd
    clean = dd.deduplicate([record])
    assert clean == [record]


def _storage_target(provider: str) -> st.StorageTarget:
    import os
    env = f"RUN45_{provider.upper()}_TOKEN"
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


def test_feedback_review_path_and_metadata_are_stable_and_non_identifying():
    target = _storage_target("huggingface")
    receipt = "reader-private-receipt-value"
    first = st.feedback_review_record_path(target, receipt, 1_788_138_000.0)
    second = st.feedback_review_record_path(target, receipt, 1_788_139_000.0)
    key = st.feedback_review_key_for(receipt)
    assert first == second
    assert first.startswith("feedback/")
    assert first.endswith(f"fb_{key}.jsonl")
    assert receipt not in first
    assert st.feedback_review_branch_for(receipt) == f"ai-feedback-{key}"
    assert st.feedback_review_title_for(receipt) == f"Feedback review {key}"
    description = st.review_description_for(key, "main", review_kind="feedback")
    assert "Training eligible while this review is open: **No**" in description
    assert "Merge = approve this Q&A + quality signal" in description
    assert "same review" in description


def test_feedback_update_uses_persisted_review_locator_for_every_provider(monkeypatch):
    async def run(provider: str):
        target = _storage_target(provider)
        coord = st.StorageCoordinator([target])
        receipt_id = f"receipt-{provider}"
        review = st.ReviewReceipt(
            provider=provider,
            target_id=target.id,
            repo=target.repo,
            base_branch="main",
            review_branch=st.feedback_review_branch_for(receipt_id),
            review_key=st.feedback_review_key_for(receipt_id),
            review_id="7",
            review_url="https://example.invalid/review/7",
            status="open",
            record_id="old",
            path=st.feedback_review_record_path(target, receipt_id, 1_788_138_000.0),
        )
        calls = {"refresh": 0, "discover": 0, "update": 0}

        async def refresh(_target, hinted):
            calls["refresh"] += 1
            assert hinted is not None and hinted.review_id == "7"
            return review

        async def discover(*args, **kwargs):
            calls["discover"] += 1
            raise AssertionError("persisted feedback review IDs must bypass discovery scans")

        async def update(_target, *, review, path, content, message):
            calls["update"] += 1
            assert review.review_id == "7"
            assert path.endswith(f"fb_{st.feedback_review_key_for(receipt_id)}.jsonl")
            assert content == b'{"revision":2}'
            assert "revision 2" in message

        monkeypatch.setattr(coord, "_refresh_review_target", refresh)
        monkeypatch.setattr(coord, "_discover_review_target", discover)
        monkeypatch.setattr(coord, "_update_review_target", update)
        hint = {
            "review": {
                "provider": provider,
                "targetId": target.id,
                "repo": target.repo,
                "baseBranch": "main",
                "reviewBranch": st.feedback_review_branch_for(receipt_id),
                "reviewKey": st.feedback_review_key_for(receipt_id),
                "reviewId": "7",
                "reviewUrl": "https://example.invalid/review/7",
                "status": "open",
            }
        }
        updated = await coord.update_feedback_review(
            receipt_id=receipt_id,
            content=b'{"revision":2}',
            commit_message="feedback revision 2",
            path_timestamp=1_788_138_000.0,
            review_hint=hint,
        )
        assert updated.review_id == "7"
        assert calls == {"refresh": 1, "discover": 0, "update": 1}

    for provider in ("huggingface", "github", "gitlab", "bitbucket"):
        asyncio.run(run(provider))


def test_feedback_review_management_header_is_cors_allowlisted():
    with TestClient(proxy_app.app) as client:
        res = client.options(
            "/v1/feedback/review/abc",
            headers={
                "Origin": "https://scikit-plots.github.io",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "x-feedback-review-token,content-type",
            },
        )
        assert res.status_code == 200, res.text
        allowed = res.headers.get("access-control-allow-headers", "").lower()
        assert "x-feedback-review-token" in allowed
        assert "content-type" in allowed
