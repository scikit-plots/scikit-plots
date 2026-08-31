from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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
