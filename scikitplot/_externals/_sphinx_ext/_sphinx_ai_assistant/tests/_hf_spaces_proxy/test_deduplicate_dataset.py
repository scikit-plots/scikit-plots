from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = RUNTIME_ROOT
PROXY = ROOT / "_hf_spaces_proxy"
sys.path.insert(0, str(PROXY))

import deduplicate_dataset as dd


def _row(*, key="c:0", source="feedback", ts=1, action="rate"):
    return {
        "schemaVersion": 2,
        "conversationId": "c",
        "answerIndex": 0,
        "action": action,
        "_dedup_key": key,
        "_source": source,
        "_ts": ts,
        "trainingStatus": "eligible" if source == "contribution" else "telemetry",
    }


def _write(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode() + b"\n"
    path.write_bytes(data)
    return data


def test_legacy_feedback_snapshot_is_excluded_from_training_by_default(tmp_path):
    _write(tmp_path / "feedback" / "1.jsonl", [_row()])
    out = tmp_path / "clean.jsonl"
    assert dd.main(["--local-dir", str(tmp_path), "--output", str(out)]) == 0
    assert out.exists()
    assert len(out.read_text().splitlines()) == 0


def test_legacy_repo_id_parser_still_defaults_to_huggingface():
    args = dd._build_parser().parse_args(["--repo-id", "org/data"])
    source = dd._direct_source(args)
    assert source.provider == "huggingface"
    assert source.repo == "org/data"


def test_storage_config_primary_is_default(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_PRIMARY", "hf_test")
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR", "gh_test")
    raw = json.dumps([
        {
            "id": "hf-primary",
            "provider": "huggingface",
            "role": "primary",
            "repo": "org/data",
            "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
            "token_type": "fine-grained",
        },
        {
            "id": "github-mirror",
            "provider": "github",
            "role": "mirror",
            "repo": "org/records",
            "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR",
        },
    ])
    sources = dd._storage_sources(raw, target_id=None, all_targets=False)
    assert [s.id for s in sources] == ["hf-primary"]
    assert sources[0].token == "hf_test"


def test_storage_config_can_select_mirror(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_PRIMARY", "hf_test")
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR", "gh_test")
    raw = json.dumps([
        {
            "id": "hf-primary",
            "provider": "huggingface",
            "role": "primary",
            "repo": "org/data",
            "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
        },
        {
            "id": "github-mirror",
            "provider": "github",
            "role": "mirror",
            "repo": "org/records",
            "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR",
        },
    ])
    sources = dd._storage_sources(raw, target_id="github-mirror", all_targets=False)
    assert [s.provider for s in sources] == ["github"]


def test_all_targets_suppresses_identical_canonical_mirror_file(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    rows = [_row()]
    data = _write(a / "feedback/2026/08/28/fb_aaaaaaaaaaaaaaaaaaaaaaaa.jsonl", rows)
    (b / "feedback/2026/08/28").mkdir(parents=True)
    (b / "feedback/2026/08/28/fb_aaaaaaaaaaaaaaaaaaaaaaaa.jsonl").write_bytes(data)
    s1 = dd.DatasetSource("a", "huggingface", "org/a")
    s2 = dd.DatasetSource("b", "github", "org/b")
    records, stats = dd.load_sources_records([(s1, a), (s2, b)], merge_mirrors=True)
    assert len(records) == 1
    assert stats.mirrored_files_suppressed == 1


def test_all_targets_fails_closed_on_same_record_id_different_bytes(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    name = "fb_bbbbbbbbbbbbbbbbbbbbbbbb.jsonl"
    _write(a / "feedback/2026/08/28" / name, [_row(ts=1)])
    _write(b / "feedback/2026/08/28" / name, [_row(ts=2)])
    s1 = dd.DatasetSource("a", "huggingface", "org/a")
    s2 = dd.DatasetSource("b", "github", "org/b")
    with pytest.raises(dd.DatasetMirrorConflict):
        dd.load_sources_records([(s1, a), (s2, b)], merge_mirrors=True)


def test_all_targets_suppresses_exact_legacy_record_across_different_files(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(a / "feedback/100.jsonl", [_row()])
    # Same record, intentionally different JSON bytes/spacing so file-hash
    # suppression does not short-circuit the exact-record fallback.
    (b / "feedback").mkdir(parents=True)
    (b / "feedback/200.jsonl").write_text(
        json.dumps(_row(), sort_keys=False, indent=2).replace("\n", " ") + "\n",
        encoding="utf-8",
    )
    s1 = dd.DatasetSource("a", "huggingface", "org/a")
    s2 = dd.DatasetSource("b", "github", "org/b")
    records, stats = dd.load_sources_records([(s1, a), (s2, b)], merge_mirrors=True)
    assert len(records) == 1
    assert stats.exact_records_suppressed == 1


def test_contribution_still_beats_feedback():
    clean = dd.deduplicate([
        _row(source="feedback", ts=100),
        _row(source="contribution", ts=50),
    ])
    assert len(clean) == 1
    assert clean[0]["_source"] == "contribution"


def test_tar_extractor_rejects_parent_traversal(tmp_path):
    import io
    import tarfile

    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("../escape.jsonl")
        payload = b"{}\n"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    with pytest.raises(dd.DatasetSourceError) as exc:
        dd._safe_extract_tar(archive, tmp_path / "extract", max_extract_bytes=1024 * 1024)
    assert exc.value.code == "ARCHIVE_PATH"


def test_token_env_preferred_in_direct_mode(monkeypatch):
    monkeypatch.setenv("MY_DATASET_TOKEN", "secret")
    args = dd._build_parser().parse_args([
        "--provider", "github",
        "--repo-id", "org/repo",
        "--token-env", "MY_DATASET_TOKEN",
        "--token", "legacy",
    ])
    assert dd._direct_source(args).token == "secret"


def _review_row(*, feedback_id: str, ts: int, answer_index: int = 0, source: str = "feedback", status: str = "eligible") -> dict:
    return {
        "schemaVersion": 5,
        "_source": source,
        "_ts": ts,
        "_dedup_key": f"receipt-{feedback_id}:feedback",
        "conversationId": None,
        "feedbackId": feedback_id,
        "feedbackChainId": feedback_id,
        "recordType": "qa",
        "answerIndex": answer_index,
        "action": "review",
        "prevFeedbackId": None,
        "prevFeedbackIds": [],
        "editCount": 0,
        "status": "active",
        "trainingStatus": status,
        "ratingValue": 1,
        "ratingSlug": "helpful",
        "ratingTitle": "Helpful",
        "ratingMode": "quick",
        "ratingScaleMin": -1,
        "ratingScaleMax": 1,
        "qualityScore": 1,
        "qualityPercent": 100,
        "message": "",
        "query": f"q-{feedback_id}",
        "answer": f"a-{feedback_id}",
        "messages": None,
        "model": {"id": "m", "provider": "p", "model": "p/m"},
        "modelEvidence": "client_selected",
        "page": "https://example.test/docs",
        "consentVersion": "2.0.0",
        "trainingConsentVersion": "1.0.0",
        "ts": ts,
        "feedbackReview": True,
    }


def _contribution_row(
    *,
    key: str,
    ts: int,
    record_type: str = "qa",
    answer_index: int | None = 0,
    status: str = "eligible",
    action: str = "rate",
) -> dict:
    row = {
        "schemaVersion": 5,
        "_source": "contribution",
        "_ts": ts,
        "_dedup_key": key,
        "conversationId": None,
        "feedbackId": "fb-" + key if record_type == "qa" else None,
        "feedbackChainId": "fb-" + key if record_type == "qa" else None,
        "recordType": record_type,
        "answerIndex": answer_index if record_type == "qa" else None,
        "action": action,
        "prevFeedbackId": None,
        "prevFeedbackIds": [],
        "editCount": 0,
        "status": "active" if action == "rate" else "withdrawn",
        "trainingStatus": status,
        "ratingValue": 1 if record_type == "qa" else None,
        "ratingSlug": "helpful" if record_type == "qa" else None,
        "ratingTitle": "Helpful" if record_type == "qa" else None,
        "ratingMode": "quick" if record_type == "qa" else None,
        "ratingScaleMin": None,
        "ratingScaleMax": None,
        "qualityScore": None,
        "qualityPercent": None,
        "message": "",
        "query": "q-" + key if record_type == "qa" else "",
        "answer": "a-" + key if record_type == "qa" else "",
        "messages": (
            [
                {"role": "user", "content": "q", "ts": ts - 1},
                {"role": "assistant", "content": "a", "ts": ts, "model": None, "feedback": None},
            ]
            if record_type == "conversation"
            else None
        ),
        "model": None,
        "modelEvidence": None,
        "page": "https://example.test/docs",
        "consentVersion": "2.0.0",
        "trainingConsentVersion": None,
        "ts": ts,
    }
    return row


def test_feedback_review_cloud_merged_is_filtered_and_deterministically_ordered():
    later = _review_row(feedback_id="b", ts=20, answer_index=1)
    earlier = _review_row(feedback_id="a", ts=10, answer_index=0)
    contribution = _review_row(feedback_id="c", ts=5, source="contribution")
    telemetry = _review_row(feedback_id="d", ts=3, status="telemetry")
    not_review = dict(_review_row(feedback_id="e", ts=4), feedbackReview=False)

    merged = dd.feedback_review_cloud_merged(
        [later, contribution, telemetry, not_review, earlier]
    )

    assert [row["feedbackId"] for row in merged] == ["a", "b"]
    assert all(row["_source"] == "feedback" for row in merged)
    assert all(row["trainingStatus"] == "eligible" for row in merged)
    assert all(row["feedbackReview"] is True for row in merged)


def test_feedback_review_cloud_merged_writer_binds_bytes_with_manifest(tmp_path):
    out = tmp_path / "ai-feedback-review-cloud-merged-jsonl-test.jsonl"
    result = dd.write_feedback_review_cloud_merged(
        [_review_row(feedback_id="b", ts=20), _review_row(feedback_id="a", ts=10)],
        out,
        source_description="github:primary",
    )

    assert out.read_bytes() == dd._jsonl_bytes(result["records"])
    manifest_path = Path(result["manifestPath"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    assert manifest["lifecycleRole"] == "cloud-merged"
    assert manifest["representation"] == "jsonl"
    assert manifest["derived"] is True
    assert manifest["authoritative"] is False
    assert manifest["authority"] == "individual canonical provider feedback records"
    assert manifest["recordCount"] == 2
    assert manifest["contentSha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["filename"] == out.name
    assert manifest["source"] == "github:primary"


def test_feedback_review_cloud_merged_default_filename_is_role_and_format_explicit(monkeypatch):
    monkeypatch.setattr(dd.time, "time", lambda: 1788790000.0)
    name = dd._feedback_review_cloud_merged_filename()
    assert name.startswith("ai-feedback-review-cloud-merged-jsonl-")
    assert name.endswith(".jsonl")
    assert "receipt" not in name
    assert "feedbackId" not in name


def test_feedback_review_cloud_merged_cli_uses_role_filename_and_manifest(tmp_path, monkeypatch):
    _write(
        tmp_path / "feedback" / "2026" / "09" / "07" / "fb_a.jsonl",
        [_review_row(feedback_id="a", ts=10)],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dd.time, "time", lambda: 1788790000.0)

    assert dd.main(["--local-dir", str(tmp_path), "--feedback-review-cloud-merged"]) == 0

    outputs = list(tmp_path.glob("ai-feedback-review-cloud-merged-jsonl-*.jsonl"))
    assert len(outputs) == 1
    rows = [json.loads(line) for line in outputs[0].read_text(encoding="utf-8").splitlines()]
    assert [row["feedbackId"] for row in rows] == ["a"]
    manifest = outputs[0].with_name(outputs[0].name + ".manifest.json")
    assert manifest.exists()


def test_feedback_review_cloud_merged_rejects_unreviewed_audit_mode(tmp_path):
    with pytest.raises(SystemExit):
        dd.main([
            "--local-dir",
            str(tmp_path),
            "--feedback-review-cloud-merged",
            "--include-unreviewed",
        ])


def test_local_snapshot_does_not_reingest_derived_cloud_merged_feedback(tmp_path):
    authoritative = _review_row(feedback_id="a", ts=10)
    _write(tmp_path / "feedback" / "fb_a.jsonl", [authoritative])
    derived = tmp_path / "ai-feedback-review-cloud-merged-jsonl-2026-09-07-12-00-00.jsonl"
    _write(derived, [authoritative, _review_row(feedback_id="b", ts=11)])

    records = dd.load_all_records(tmp_path)

    assert [row.get("feedbackId") for row in records] == ["a"]


def test_feedback_review_cloud_merged_bytes_do_not_depend_on_input_order(tmp_path):
    a = _review_row(feedback_id="a", ts=10)
    b = _review_row(feedback_id="b", ts=20)
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    dd.write_feedback_review_cloud_merged([b, a], one, source_description="test")
    dd.write_feedback_review_cloud_merged([a, b], two, source_description="test")
    assert one.read_bytes() == two.read_bytes()

def test_contribution_cloud_merged_is_filtered_and_deterministically_ordered():
    later = _contribution_row(key="receipt-b:1", ts=20, answer_index=1)
    earlier = _contribution_row(key="receipt-a:0", ts=10, answer_index=0)
    conversation = _contribution_row(
        key="receipt-c:conversation", ts=15, record_type="conversation", answer_index=None
    )
    quarantined = _contribution_row(
        key="receipt-q:0", ts=5, status="quarantined"
    )
    feedback = _review_row(feedback_id="f", ts=4)

    merged = dd.contribution_cloud_merged(
        [later, feedback, quarantined, conversation, earlier]
    )

    assert [row["_dedup_key"] for row in merged] == [
        "receipt-a:0",
        "receipt-c:conversation",
        "receipt-b:1",
    ]
    assert all(row["_source"] == "contribution" for row in merged)
    assert all(row["trainingStatus"] == "eligible" for row in merged)
    assert {row["recordType"] for row in merged} == {"qa", "conversation"}


def test_contribution_cloud_merged_respects_withdrawal_lifecycle():
    eligible = _contribution_row(key="receipt-a:0", ts=10)
    withdrawn = _contribution_row(
        key="receipt-a:0",
        ts=20,
        record_type="qa",
        answer_index=0,
        status="withdrawn",
        action="withdraw",
    )

    assert dd.contribution_cloud_merged([eligible, withdrawn]) == []


def test_contribution_cloud_merged_writer_binds_bytes_and_scope_counts(tmp_path):
    out = tmp_path / "ai-contribution-cloud-merged-jsonl-test.jsonl"
    rows = [
        _contribution_row(key="receipt-a:0", ts=10),
        _contribution_row(
            key="receipt-c:conversation", ts=15, record_type="conversation", answer_index=None
        ),
    ]
    result = dd.write_contribution_cloud_merged(
        rows, out, source_description="github:primary"
    )

    assert out.read_bytes() == dd._jsonl_bytes(result["records"])
    manifest_path = Path(result["manifestPath"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    assert manifest["artifactFamily"] == "ai-contribution"
    assert manifest["lifecycleRole"] == "cloud-merged"
    assert manifest["representation"] == "jsonl"
    assert manifest["derived"] is True
    assert manifest["authoritative"] is False
    assert manifest["authority"] == "individual canonical provider contribution records"
    assert manifest["recordCount"] == 2
    assert manifest["qaRecordCount"] == 1
    assert manifest["conversationRecordCount"] == 1
    assert manifest["contentSha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert "_dedup_key" in manifest["sourceIdentityFields"]


def test_contribution_cloud_merged_default_filename_is_role_and_format_explicit(monkeypatch):
    monkeypatch.setattr(dd.time, "time", lambda: 1788790000.0)
    name = dd._contribution_cloud_merged_filename()
    assert name.startswith("ai-contribution-cloud-merged-jsonl-")
    assert name.endswith(".jsonl")
    assert "receipt" not in name
    assert "delete" not in name


def test_contribution_cloud_merged_cli_uses_role_filename_and_manifest(tmp_path, monkeypatch):
    _write(
        tmp_path / "contributions" / "2026" / "09" / "07" / "ct_a.jsonl",
        [_contribution_row(key="receipt-a:0", ts=10)],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dd.time, "time", lambda: 1788790000.0)

    assert dd.main(["--local-dir", str(tmp_path), "--contribution-cloud-merged"]) == 0

    outputs = list(tmp_path.glob("ai-contribution-cloud-merged-jsonl-*.jsonl"))
    assert len(outputs) == 1
    rows = [json.loads(line) for line in outputs[0].read_text(encoding="utf-8").splitlines()]
    assert [row["_dedup_key"] for row in rows] == ["receipt-a:0"]
    assert outputs[0].with_name(outputs[0].name + ".manifest.json").exists()


def test_contribution_cloud_merged_rejects_unreviewed_audit_mode(tmp_path):
    with pytest.raises(SystemExit):
        dd.main([
            "--local-dir",
            str(tmp_path),
            "--contribution-cloud-merged",
            "--include-unreviewed",
        ])


def test_cloud_merged_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        dd.main([
            "--local-dir",
            str(tmp_path),
            "--feedback-review-cloud-merged",
            "--contribution-cloud-merged",
        ])


def test_local_snapshot_does_not_reingest_derived_cloud_merged_contribution(tmp_path):
    authoritative = _contribution_row(key="receipt-a:0", ts=10)
    _write(tmp_path / "contributions" / "ct_a.jsonl", [authoritative])
    derived = tmp_path / "ai-contribution-cloud-merged-jsonl-2026-09-07-12-00-00.jsonl"
    _write(derived, [authoritative, _contribution_row(key="receipt-b:0", ts=11)])

    records = dd.load_all_records(tmp_path)

    assert [row.get("_dedup_key") for row in records] == ["receipt-a:0"]


def test_provider_source_iterator_skips_derived_cloud_merged_artifacts(tmp_path):
    source = dd.DatasetSource("primary", "github", "org/repo")
    authoritative = tmp_path / "contributions" / "ct_a.jsonl"
    derived = tmp_path / "contributions" / "ai-contribution-cloud-merged-jsonl-2026-09-07-12-00-00.jsonl"
    _write(authoritative, [_contribution_row(key="receipt-a:0", ts=10)])
    _write(derived, [_contribution_row(key="receipt-b:0", ts=11)])

    files = [logical for logical, _ in dd._iter_source_files(tmp_path, source)]

    assert files == ["contributions/ct_a.jsonl"]



def test_contribution_cloud_merged_reapplies_privacy_boundary_to_legacy_rows():
    qa = _contribution_row(key="receipt-a:0", ts=10)
    qa["page"] = "https://user:pass@example.test/docs/page?token=SECRET#frag"
    qa["model"] = {
        "id": {"secret": "nested"},
        "provider": "huggingface",
        "model": "org/model",
        "label": "private label",
        "endpoint": "https://user:pass@internal.example/v1?token=SECRET#frag",
        "info_url": "https://private.example/info",
        "description": "operator-only",
        "default": True,
    }
    conversation = _contribution_row(
        key="receipt-c:conversation", ts=15, record_type="conversation", answer_index=None
    )
    conversation["page"] = "file:///private/work/docs/index.html?secret=x#frag"
    conversation["messages"][1]["model"] = {
        "id": "m1",
        "provider": "openai",
        "model": "gpt-test",
        "endpoint": "https://internal.example/v1?token=SECRET",
        "description": "private",
    }

    merged = dd.contribution_cloud_merged([conversation, qa])
    by_type = {row["recordType"]: row for row in merged}

    assert by_type["qa"]["page"] == "https://example.test/docs/page"
    assert by_type["qa"]["model"]["id"] is None
    assert by_type["qa"]["model"]["provider"] == "huggingface"
    assert by_type["qa"]["model"]["model"] == "org/model"
    assert by_type["qa"]["model"]["endpoint"] is None
    assert by_type["qa"]["model"]["description"] is None
    assert by_type["conversation"]["page"] == ""
    message_model = by_type["conversation"]["messages"][1]["model"]
    assert message_model["id"] == "m1"
    assert message_model["endpoint"] is None
    assert message_model["description"] is None


def test_custom_named_cloud_merged_artifact_is_skipped_when_manifest_binds_bytes(tmp_path):
    authoritative = _contribution_row(key="receipt-a:0", ts=10)
    _write(tmp_path / "contributions" / "ct_a.jsonl", [authoritative])
    custom = tmp_path / "analysis-export.jsonl"
    dd.write_contribution_cloud_merged(
        [authoritative, _contribution_row(key="receipt-b:0", ts=11)],
        custom,
        source_description="test",
    )

    records = dd.load_all_records(tmp_path)

    assert [row.get("_dedup_key") for row in records] == ["receipt-a:0"]


def test_tampered_or_unbound_derived_manifest_cannot_hide_jsonl_input(tmp_path):
    row = _contribution_row(key="receipt-a:0", ts=10)
    custom = tmp_path / "analysis-export.jsonl"
    dd.write_contribution_cloud_merged([row], custom, source_description="test")
    custom.write_bytes(custom.read_bytes() + dd._jsonl_bytes([_contribution_row(key="receipt-b:0", ts=11)]))

    records = dd.load_all_records(tmp_path)

    assert {r.get("_dedup_key") for r in records} == {"receipt-a:0", "receipt-b:0"}


def test_atomic_derived_write_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch):
    out = tmp_path / "merged.jsonl"
    out.write_bytes(b"original\n")

    def fail_replace(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(dd.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        dd._atomic_write_bytes(out, b"replacement\n")

    assert out.read_bytes() == b"original\n"
    assert list(tmp_path.glob(".merged.jsonl.*.tmp")) == []

def test_contribution_cloud_merged_bytes_do_not_depend_on_input_order(tmp_path):
    a = _contribution_row(key="receipt-a:0", ts=10)
    b = _contribution_row(
        key="receipt-c:conversation", ts=15, record_type="conversation", answer_index=None
    )
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    dd.write_contribution_cloud_merged([b, a], one, source_description="test")
    dd.write_contribution_cloud_merged([a, b], two, source_description="test")
    assert one.read_bytes() == two.read_bytes()
