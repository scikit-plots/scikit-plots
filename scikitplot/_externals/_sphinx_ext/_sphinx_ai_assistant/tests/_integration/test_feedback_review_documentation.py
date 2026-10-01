from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

ROOT = RUNTIME_ROOT
GUIDE = ROOT / "_hf_spaces_proxy" / "FEEDBACK_REVIEW_GUIDE.md"
README = ROOT / "README.md"
PROXY_README = ROOT / "_hf_spaces_proxy" / "README.md"
DATASET_GUIDE = ROOT / "_hf_spaces_proxy" / "DATASET_CONTRIBUTION_GUIDE.md"


def test_feedback_review_guide_defines_current_feedback_authorities():
    text = GUIDE.read_text(encoding="utf-8")
    required = (
        "AI Assistant local rating",
        "Generic documentation page feedback",
        "POST /v1/feedback",
        "/v1/feedback/review",
        "page.feedback-request.v1",
        "FEEDBACK_REVIEW_MODE",
        "one Q&A",
        "training eligibility",
        "Hugging Face",
        "GitHub",
        "GitLab",
        "Bitbucket",
    )
    for item in required:
        assert item in text
    assert "FEEDBACK_PERSIST_ENABLED" not in text
    assert "anonymous Assistant rating-telemetry transport" in text


def test_feedback_docs_keep_generic_review_and_contribution_authorities_separate():
    guide = GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    dataset = DATASET_GUIDE.read_text(encoding="utf-8")
    assert "never falls through to `/v1/feedback`" in guide
    assert "/v1/feedback/review" in readme
    assert "Maintainer feedback review" in dataset
    assert "Never" in dataset
    assert "FEEDBACK_PERSIST_ENABLED" not in dataset


def test_feedback_docs_explain_local_rating_and_explicit_review():
    text = GUIDE.read_text(encoding="utf-8")
    assert "A local Assistant rating does not create a repository review" in text
    assert "Local rating state is intentionally local" in text
    assert "Share with" in text and "maintainers" in text
    assert "no Assistant anonymous rating telemetry route" in text


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
    assert "FEEDBACK_PERSIST_ENABLED" not in text


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
        "provider-native review",
        "merge",
        "canonical branch",
        "Generic page feedback",
    ):
        assert item in text
