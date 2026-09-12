from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from pathlib import Path

ROOT = RUNTIME_ROOT
GUIDE = ROOT / "_hf_spaces_proxy" / "FEEDBACK_REVIEW_GUIDE.md"
README = ROOT / "README.md"
PROXY_README = ROOT / "_hf_spaces_proxy" / "README.md"
DATASET_GUIDE = ROOT / "_hf_spaces_proxy" / "DATASET_CONTRIBUTION_GUIDE.md"


def test_feedback_review_guide_covers_independent_control_planes_and_lifecycle():
    text = GUIDE.read_text(encoding="utf-8")
    required = (
        "Anonymous rating telemetry",
        "Share with maintainers",
        "POST /v1/feedback",
        "/v1/feedback/review",
        "FEEDBACK_REVIEW_MODE=provider-pr",
        "FEEDBACK_PERSIST_ENABLED=false",
        "one Q&A",
        "qualityScore",
        "training-eligible",
        "same review",
        "no-op",
        "Withdraw feedback",
        "Hugging Face",
        "GitHub",
        "GitLab",
        "Bitbucket Cloud",
    )
    for item in required:
        assert item in text


def test_feedback_docs_keep_telemetry_review_and_contribution_permissions_separate():
    guide = GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    dataset = DATASET_GUIDE.read_text(encoding="utf-8")
    assert "telemetry consent and feedback-review consent use different versioned keys" in guide
    assert "telemetry consent never authorizes" in readme.lower()
    assert "Maintainer feedback review" in dataset
    assert "Never" in dataset


def test_feedback_docs_explain_why_telemetry_can_look_like_noop():
    text = GUIDE.read_text(encoding="utf-8")
    assert "Why the switch may appear to have no repository effect" in text
    assert "Browser telemetry permission and server telemetry persistence are separate" in text
    assert "Server telemetry persistence" in text
    assert "Maintainer review readiness" in text
    assert "Endpoint Configuration no longer duplicates" in text


def test_proxy_readme_advertises_feedback_review_routes_and_config():
    text = PROXY_README.read_text(encoding="utf-8")
    for item in (
        "`POST` | `/v1/feedback/review`",
        "`PUT` | `/v1/feedback/review/{receipt}`",
        "`GET` | `/v1/feedback/review/{receipt}`",
        "`DELETE` | `/v1/feedback/review/{receipt}`",
        "`X-Feedback-Review-Token`",
        "`FEEDBACK_REVIEW_MODE`",
        "`FEEDBACK_REVIEW_LEDGER_BACKEND`",
        "`FEEDBACK_REVIEW_RATE_LIMIT_PER_HOUR`",
        "./FEEDBACK_REVIEW_GUIDE.md",
    ):
        assert item in text


def test_readme_exposes_shared_workspace_and_feedback_guide():
    text = README.read_text(encoding="utf-8")
    assert "## Feedback and maintainer review" in text
    assert "[ Feedback ] [ Dataset contribution ] [ Activity ]" in text
    assert "FEEDBACK_REVIEW_GUIDE.md" in text
    assert "training-eligible" in text
    assert "qualityScore" in text


def test_feedback_docs_define_cloud_merged_as_derived_not_authoritative():
    text = GUIDE.read_text(encoding="utf-8")
    for item in (
        "Cloud merged feedback view",
        "--feedback-review-cloud-merged",
        "ai-feedback-review-cloud-merged-jsonl-<UTC timestamp>.jsonl",
        "derived, not",
        "individual canonical provider feedback records",
        "--include-unreviewed",
    ):
        assert item in text
