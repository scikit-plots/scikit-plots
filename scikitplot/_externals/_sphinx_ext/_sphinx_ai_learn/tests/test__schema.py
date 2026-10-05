"""Normalized data-contract, graph-integrity, and draft semantics tests."""
from copy import deepcopy

import pytest

from _sphinx_ext._sphinx_ai_learn import _schema as schema


def catalog():
    return {
        "contract": schema.CATALOG_CONTRACT,
        "revision": "test-1",
        "subjects": [
            {
                "id": "topic-a",
                "kind": "topic",
                "title": "A",
                "sections": [
                    {
                        "id": "summary",
                        "title": "Summary",
                        "body": "<script>not executed</script>",
                        "citations": [
                            {"source_id": "source-a", "locator": "Section 1"}
                        ],
                    }
                ],
            },
            {
                "id": "source-a",
                "kind": "source",
                "title": "Source",
                "url": "https://example.org/a",
            },
        ],
    }


def draft():
    return {
        "contract": schema.CONTRIBUTION_CONTRACT,
        "draft_id": "draft-a",
        "base_revision": "empty",
        "subject": {
            "id": "topic-custom",
            "kind": "topic",
            "title": "A new title",
            "sections": [],
        },
        "section": {
            "id": "custom-section",
            "title": "My question",
            "body": "My answer",
            "citations": [],
        },
        "authorship": "human",
    }


def test_independent_custom_topic_draft():
    result = schema.validate_contribution(draft())
    assert result["subject"]["id"] == "topic-custom"
    assert result["section"]["id"] == "custom-section"
    assert "trainingStatus" not in result


def test_normalization_does_not_mutate_input():
    value = catalog()
    before = deepcopy(value)
    result = schema.validate_catalog(value)
    result["subjects"][0]["sections"][0]["body"] = "changed"
    assert value == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(contract="learn.catalog.v999"),
        lambda x: x["subjects"].append(x["subjects"][0]),
        lambda x: x["subjects"][0].update(id="../escape"),
        lambda x: x["subjects"][0].update(related=["missing"]),
        lambda x: x["subjects"][0]["sections"][0]["citations"][0].update(
            source_id="topic-a"
        ),
        lambda x: x["subjects"][0]["sections"].append(
            x["subjects"][0]["sections"][0]
        ),
        lambda x: x["subjects"][0].update(title="A\nB"),
        lambda x: x["subjects"][0].update(trainingStatus="eligible"),
    ],
)
def test_rejects_invalid_catalog_graph(mutate):
    value = catalog()
    mutate(value)
    with pytest.raises(schema.LearnValidationError):
        schema.validate_catalog(value)




def test_unknown_related_error_names_subject_and_targets():
    value = catalog()
    value["subjects"][0]["related"] = ["missing-b", "missing-a"]
    with pytest.raises(
        schema.LearnValidationError,
        match=r"subject\.related: topic-a -> unknown target missing-b[\s\S]*subject\.related: topic-a -> unknown target missing-a",
    ):
        schema.validate_catalog(value)

def test_only_current_normalized_catalog_contract_is_accepted():
    for old in ("learn.catalog.v1", "learn.catalog.v2"):
        value = catalog()
        value["contract"] = old
        with pytest.raises(schema.LearnValidationError, match="unsupported version"):
            schema.validate_catalog(value)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://example.org",
        "https://name:pass@example.org",
        "https://example.org:99999",
        "https://exa mple.org",
        "https://example.org/\nsecret",
    ],
)
def test_rejects_active_or_credentialed_urls(url):
    with pytest.raises(schema.LearnValidationError):
        schema.public_url(url)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trainingStatus", "eligible"),
        ("consentFlag", True),
        ("status", "published"),
        ("repository", "attacker/repo"),
    ],
)
def test_drafts_cannot_grant_training_or_publication(field, value):
    value_draft = draft()
    value_draft[field] = value
    with pytest.raises(schema.LearnValidationError):
        schema.validate_contribution(value_draft)


def test_revision_and_identity_survive_title_change():
    value = draft()
    first = schema.validate_contribution(value)
    value["subject"]["title"] = "A better custom title"
    second = schema.validate_contribution(value)
    assert first["subject"]["id"] == second["subject"]["id"]
    assert first["draft_id"] == second["draft_id"]


def test_optional_authors_are_bounded_plain_text_metadata():
    value = catalog()
    value["subjects"][0]["authors"] = ["Ask Plaat"]
    assert schema.validate_catalog(value)["subjects"][0]["authors"] == ["Ask Plaat"]
    value["subjects"][0]["authors"] = ["Ask Plaat", "Ask Plaat"]
    with pytest.raises(schema.LearnValidationError, match="duplicate display names"):
        schema.validate_catalog(value)
    value["subjects"][0]["authors"] = ["Ask Plaat", "ask plaat"]
    with pytest.raises(schema.LearnValidationError, match="duplicate display names"):
        schema.validate_catalog(value)
    value["subjects"][0]["authors"] = ["x" * 81]
    with pytest.raises(schema.LearnValidationError, match="invalid text length"):
        schema.validate_catalog(value)




def test_section_contributors_are_bounded_plain_text_public_credit():
    value = catalog()
    section = value["subjects"][0]["sections"][0]
    section["contributors"] = ["Ada", "DataFox"]
    normalized = schema.validate_catalog(value)
    assert normalized["subjects"][0]["sections"][0]["contributors"] == [
        "Ada",
        "DataFox",
    ]
    bad = deepcopy(value)
    bad["subjects"][0]["sections"][0]["contributors"] = ["Ada", "ada"]
    with pytest.raises(schema.LearnValidationError, match="duplicate display names"):
        schema.validate_catalog(bad)
    bad = deepcopy(value)
    bad["subjects"][0]["sections"][0]["contributors"] = ["x" * 81]
    with pytest.raises(schema.LearnValidationError):
        schema.validate_catalog(bad)


def test_section_generation_ledger_requires_active_projection_without_embedded_feedback():
    value = catalog()
    section = value["subjects"][0]["sections"][0]
    section["contributors"] = ["Ada"]
    section["active_generation_id"] = "generation-aaaaaaaaaaaaaaaa"
    section["generations"] = [
        {
            "id": "generation-aaaaaaaaaaaaaaaa",
            "created_at": "2026-09-27T03:00:00Z",
            "body": section["body"],
            "citations": deepcopy(section["citations"]),
            "links": [],
            "contributors": ["Ada"],
        }
    ]
    normalized = schema.validate_catalog(value)["subjects"][0]["sections"][0]
    assert normalized["active_generation_id"] == "generation-aaaaaaaaaaaaaaaa"
    assert "feedback" not in normalized["generations"][0]

    bad = deepcopy(value)
    bad["subjects"][0]["sections"][0]["generations"][0]["feedback"] = []
    with pytest.raises(schema.LearnValidationError, match="unexpected"):
        schema.validate_catalog(bad)

    bad = deepcopy(value)
    bad["subjects"][0]["sections"][0]["body"] = "projection drift"
    with pytest.raises(schema.LearnValidationError, match="must match active generation"):
        schema.validate_catalog(bad)


def test_social_links_and_trending_metrics_are_bounded():
    value = catalog()
    value["subjects"][0]["metrics"] = {"hackernews": 210, "github": 0}
    value["subjects"][0]["sections"].append(
        {
            "id": "hackernews",
            "title": "HackerNews",
            "body": "",
            "links": [
                {
                    "title": "Dream-RSI",
                    "url": "https://news.ycombinator.com/item?id=49726955",
                    "meta": "(210 points, 51 comments)",
                }
            ],
        }
    )
    result = schema.validate_catalog(value)["subjects"][0]
    assert result["metrics"]["hackernews"] == 210
    assert result["sections"][-1]["links"][0]["url"].startswith(
        "https://news.ycombinator.com/"
    )
    bad = deepcopy(value)
    bad["subjects"][0]["metrics"]["hackernews"] = -1
    with pytest.raises(schema.LearnValidationError, match="non-negative"):
        schema.validate_catalog(bad)
    bad = deepcopy(value)
    bad["subjects"][0]["sections"][-1]["links"][0]["url"] = "javascript:alert(1)"
    with pytest.raises(schema.LearnValidationError):
        schema.validate_catalog(bad)


def test_audio_accepts_only_durable_local_mp3_or_wav():
    value = {
        "contract": schema.CATALOG_CONTRACT,
        "revision": "audio-1",
        "subjects": [
            {
                "id": "audio-a",
                "kind": "audio",
                "title": "Audio",
                "created_at": "2026-09-22T00:00:00Z",
                "media": {
                    "type": "audio",
                    "src": "/_static/learn/audio/a.mp3",
                    "mime_type": "audio/mpeg",
                    "duration_seconds": 123,
                },
            }
        ],
    }
    row = schema.validate_catalog(value)["subjects"][0]
    assert row["media"]["src"] == "/_static/learn/audio/a.mp3"
    assert row["media"]["duration_seconds"] == 123.0
    for src, mime in [
        ("https://example.org/a.mp3", "audio/mpeg"),
        ("/_static/../secret.mp3", "audio/mpeg"),
        ("/_static/audio/a.ogg", "audio/ogg"),
        ("/_static/audio/a.wav", "audio/mpeg"),
    ]:
        bad = deepcopy(value)
        bad["subjects"][0]["media"].update(src=src, mime_type=mime)
        with pytest.raises(schema.LearnValidationError):
            schema.validate_catalog(bad)


def test_document_accepts_only_passive_local_assets():
    value = {
        "contract": schema.CATALOG_CONTRACT,
        "revision": "document-1",
        "subjects": [
            {
                "id": "document-a",
                "kind": "document",
                "title": "Document",
                "created_at": "2026-09-22T00:00:00Z",
                "media": {
                    "type": "document",
                    "src": "/_static/learn/documents/a.md",
                    "mime_type": "text/markdown",
                },
            }
        ],
    }
    row = schema.validate_catalog(value)["subjects"][0]
    assert row["media"] == {
        "type": "document",
        "src": "/_static/learn/documents/a.md",
        "mime_type": "text/markdown",
    }

    accepted = [
        ("/_static/learn/documents/a.rst", "text/x-rst"),
        ("/_static/learn/documents/a.txt", "text/plain"),
        ("/_static/learn/documents/a.pdf", "application/pdf"),
    ]
    for src, mime in accepted:
        candidate = deepcopy(value)
        candidate["subjects"][0]["media"].update(src=src, mime_type=mime)
        assert schema.validate_catalog(candidate)["subjects"][0]["media"]["src"] == src

    rejected = [
        ("https://example.org/a.md", "text/markdown"),
        ("/_static/../secret.md", "text/markdown"),
        ("/_static/learn/documents/a.html", "text/html"),
        ("/_static/learn/documents/a.md", "text/plain"),
    ]
    for src, mime in rejected:
        candidate = deepcopy(value)
        candidate["subjects"][0]["media"].update(src=src, mime_type=mime)
        with pytest.raises(schema.LearnValidationError):
            schema.validate_catalog(candidate)
