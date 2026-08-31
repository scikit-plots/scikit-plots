"""Documentation contract for the provider-native contribution workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "DATASET_CONTRIBUTION_GUIDE.md"
DEEP_GUIDE = ROOT / "_hf_spaces_proxy" / "DATASET_COLLECTION_GUIDANCE.md"
PROXY_README = ROOT / "_hf_spaces_proxy" / "README.md"


def test_dataset_contribution_guide_exists_and_covers_complete_lifecycle():
    text = GUIDE.read_text(encoding="utf-8")
    required = (
        "CONTRIBUTION_REVIEW_MODE=provider-pr",
        "Save private receipt",
        "Copy private withdrawal code",
        "Copy support reference",
        "Recover withdrawal access",
        "Check status",
        "Delete pending / withdraw training use",
        "trainingEligible = false",
        "APPROVED / ELIGIBLE",
        "NOT ACCEPTED",
        "WITHDRAWN",
        "CONTRIBUTION_LEDGER_BACKEND=sqlite",
        "CONTRIBUTION_LEDGER_BACKEND=redis",
        "RECORD_STORAGE_TARGETS",
        "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
    )
    for item in required:
        assert item in text


def test_dataset_guide_documents_every_supported_review_provider():
    text = GUIDE.read_text(encoding="utf-8")
    for provider in ("Hugging Face", "GitHub", "GitLab", "Bitbucket Cloud"):
        assert provider in text
    assert "Merge Request" in text
    assert "Pull Request" in text


def test_dataset_guide_preserves_one_primary_review_authority():
    text = GUIDE.read_text(encoding="utf-8")
    assert "Only the **Primary** decides eligibility" in text
    assert "Mirrors are not independent review" in text
    assert "does not synchronously fan" in text


def test_readme_has_beginner_path_to_dataset_and_proxy_guides():
    text = README.read_text(encoding="utf-8")
    assert "## Start here" in text
    assert "## Dataset contribution and review" in text
    assert "DATASET_CONTRIBUTION_GUIDE.md" in text
    assert "_hf_spaces_proxy/README.md" in text
    assert "CONTRIBUTION_REVIEW_MODE=provider-pr" in text


def test_deep_guide_no_longer_claims_review_token_is_only_approval_path():
    text = DEEP_GUIDE.read_text(encoding="utf-8")
    head = text[:9000]
    assert "**Recommended human-review mode:** `CONTRIBUTION_REVIEW_MODE=provider-pr`" in head
    assert "**Compatibility mode:** `CONTRIBUTION_REVIEW_MODE=ledger`" in head
    assert "review-token-only promotion" in head
    assert "**Guide version:** 4.0" in text


def test_proxy_readme_routes_operators_to_the_right_guide_depth():
    text = PROXY_README.read_text(encoding="utf-8")
    assert "../DATASET_CONTRIBUTION_GUIDE.md" in text
    assert "DATASET_COLLECTION_GUIDANCE.md" in text
    assert "CONTRIBUTION_REVIEW_MODE" in text


def test_documentation_examples_reference_secret_names_not_secret_values():
    combined = "\n".join(
        p.read_text(encoding="utf-8") for p in (README, GUIDE, DEEP_GUIDE, PROXY_README)
    )
    # Provider token values must stay placeholders; topology may expose only env names.
    assert '"token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY"' in combined
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in combined
    assert "github_pat_" not in combined


def test_dataset_docs_explain_review_continuity_and_duplicate_suppression():
    guide = GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    proxy = PROXY_README.read_text(encoding="utf-8")
    for text in (guide, readme, proxy):
        assert "same PR/MR" in text or "same review" in text
    assert 'reviewUpdate="unchanged"' in guide
    assert "Update existing review" in guide
    assert "refs/pr/<number>" in guide
    assert "direct review lookup" in guide
    assert "global content deduplication" in guide
    assert "`PUT` | `/v1/contribute/{receipt}`" in proxy


def test_dataset_docs_explain_portable_withdrawal_and_nonsecret_support_fallback():
    guide = GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    proxy = PROXY_README.read_text(encoding="utf-8")
    for text in (guide, readme, proxy):
        assert "private withdrawal code" in text.lower()
        assert "support reference" in text.lower()
    assert "aicm2." in guide
    assert "ct_<stable-review-key>.jsonl" in guide
    assert "does not extend server-side retention" in guide
    assert "reviewProvider" in proxy
    assert "reviewId" in proxy
    assert "reviewPath" in proxy
    assert "Provider review URLs" in proxy
