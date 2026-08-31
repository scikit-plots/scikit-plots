from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_hf_spaces_proxy"))

from _utils._storage import (
    StorageCoordinator,
    StorageTarget,
    StorageWriteError,
    canonical_record_path,
    load_storage_targets,
    public_links,
    record_id_for,
)


def test_legacy_hf_target_accepts_fine_grained(monkeypatch):
    token = "hf_" + "a" * 60
    targets = load_storage_targets(
        "",
        legacy_repo="scikit-plots/ai-assistant-contributions",
        legacy_token=token,
        legacy_token_type="fine-grained",
    )
    assert len(targets) == 1
    target = targets[0]
    assert target.provider == "huggingface"
    assert target.role == "primary"
    assert target.token_type == "fine-grained"
    assert target.token == token


def test_multi_provider_targets_parse_and_paths(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF", "hf_test")
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GH", "gh_test")
    raw = json.dumps(
        [
            {
                "id": "hf-primary",
                "provider": "huggingface",
                "role": "primary",
                "repo": "org/dataset",
                "branch": "main",
                "token_env": "AI_RECORD_STORAGE_TOKEN_HF",
                "token_type": "fine-grained",
                "paths": {"feedback": "ratings", "contributions": "training/records"},
            },
            {
                "id": "github-mirror",
                "provider": "github",
                "role": "mirror",
                "repo": "org/repo",
                "token_env": "AI_RECORD_STORAGE_TOKEN_GH",
            },
        ]
    )
    targets = load_storage_targets(raw)
    assert [t.role for t in targets] == ["primary", "mirror"]
    assert targets[0].feedback_path == "ratings"
    assert targets[0].contributions_path == "training/records"
    assert public_links(targets[0])["feedback"].endswith("/tree/main/ratings")
    assert public_links(targets[1])["contributions"].endswith("/tree/main/contributions")


def test_token_env_is_restricted():
    raw = json.dumps(
        [
            {
                "id": "bad",
                "provider": "github",
                "role": "primary",
                "repo": "org/repo",
                "token_env": "HOME",
            }
        ]
    )
    with pytest.raises(ValueError):
        load_storage_targets(raw)


def test_read_hf_token_is_blocked_before_dispatch(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_READ", "hf_read")
    target = StorageTarget(
        id="hf-primary",
        label="HF",
        provider="huggingface",
        role="primary",
        repo="org/dataset",
        token_env="AI_RECORD_STORAGE_TOKEN_HF_READ",
        token_type="read",
    )
    coord = StorageCoordinator([target])

    called = False

    async def _dispatch(*args, **kwargs):
        nonlocal called
        called = True

    coord._dispatch = _dispatch  # type: ignore[method-assign]
    with pytest.raises(StorageWriteError) as exc:
        asyncio.run(coord.write(kind="feedback", content=b"{}", commit_message="test"))
    assert exc.value.code == "READ_TOKEN"
    assert not called


def test_primary_success_mirror_failure_degrades_only_mirror(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_A", "a")
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_B", "b")
    primary = StorageTarget(
        id="primary",
        label="Primary",
        provider="github",
        role="primary",
        repo="org/repo",
        token_env="AI_RECORD_STORAGE_TOKEN_A",
    )
    mirror = StorageTarget(
        id="mirror",
        label="Mirror",
        provider="gitlab",
        role="mirror",
        repo="org/repo",
        token_env="AI_RECORD_STORAGE_TOKEN_B",
    )
    coord = StorageCoordinator([primary, mirror])

    async def _dispatch(target, *args, **kwargs):
        if target.id == "mirror":
            raise StorageWriteError("MIRROR_FAIL")

    coord._dispatch = _dispatch  # type: ignore[method-assign]
    receipt = asyncio.run(coord.write(kind="feedback", content=b'{"x":1}', commit_message="test"))
    assert receipt.accepted is True
    assert receipt.primary == "primary"
    assert receipt.mirrors == {"mirror": "degraded"}


def test_record_id_and_path_are_stable():
    content = b'{"x":1}'
    rid = record_id_for(content)
    assert rid == record_id_for(content)
    target = StorageTarget(
        id="hf-primary",
        label="HF",
        provider="huggingface",
        role="primary",
        repo="org/dataset",
        token_env="AI_RECORD_STORAGE_TOKEN_UNUSED",
    )
    path = canonical_record_path(target, "feedback", rid, now=1767225600.0)
    assert path.startswith("feedback/2026/01/01/fb_")
    assert path.endswith(".jsonl")


def test_manifest_reports_hf_three_token_types(monkeypatch):
    for token_type, expected in [
        ("fine-grained", "unverified"),
        ("read", "denied-read-token"),
        ("write", "broad-write"),
    ]:
        env = f"AI_RECORD_STORAGE_TOKEN_{token_type.upper().replace('-', '_')}"
        monkeypatch.setenv(env, "hf_" + "x" * 60)
        target = StorageTarget(
            id="hf-primary",
            label="HF",
            provider="huggingface",
            role="primary",
            repo="org/dataset",
            token_env=env,
            token_type=token_type,
        )
        coord = StorageCoordinator([target])
        # Avoid network/installed-version dependency: emulate the capability
        # result that initialize() maps into the public manifest.
        coord._state[target.id].write_capability = expected
        manifest = coord.manifest()
        token = manifest["targets"][0]["token"]
        assert token["type"] == token_type
        assert token["write_capability"] == expected


def test_proxy_dockerfile_copies_storage_module():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "_hf_spaces_proxy" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=1000:1000 _utils ./_utils" in dockerfile
    assert (root / "_hf_spaces_proxy" / "_utils" / "_storage.py").is_file()


def test_gitlab_nested_namespace_is_supported(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GL", "gl")
    raw = json.dumps([{
        "id": "gitlab-primary",
        "provider": "gitlab",
        "role": "primary",
        "repo": "org/subgroup/records",
        "token_env": "AI_RECORD_STORAGE_TOKEN_GL",
    }])
    target = load_storage_targets(raw)[0]
    assert target.repo == "org/subgroup/records"
    assert "/org/subgroup/records" in public_links(target)["root"]


def test_github_adapter_uses_contents_api(monkeypatch):
    import httpx

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GH2", "secret-gh")
    seen = {}

    async def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"content": {"sha": "x"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = StorageTarget(
        id="github-primary", label="GH", provider="github", role="primary",
        repo="org/repo", token_env="AI_RECORD_STORAGE_TOKEN_GH2",
    )
    coord = StorageCoordinator([target], client=client)
    asyncio.run(coord.write(kind="feedback", content=b"{}", commit_message="test"))
    asyncio.run(client.aclose())
    assert seen["method"] == "PUT"
    assert "/repos/org/repo/contents/feedback/" in seen["url"]
    assert seen["auth"] == "Bearer secret-gh"


def test_gitlab_adapter_supports_nested_project(monkeypatch):
    import httpx

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GL2", "secret-gl")
    seen = {}

    async def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("private-token")
        return httpx.Response(201, json={"file_path": "ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = StorageTarget(
        id="gitlab-primary", label="GL", provider="gitlab", role="primary",
        repo="org/sub/repo", token_env="AI_RECORD_STORAGE_TOKEN_GL2",
        api_base="https://gitlab.example/api/v4",
    )
    coord = StorageCoordinator([target], client=client)
    asyncio.run(coord.write(kind="contributions", content=b"{}", commit_message="test"))
    asyncio.run(client.aclose())
    assert seen["method"] == "POST"
    assert "/projects/org%2Fsub%2Frepo/repository/files/" in seen["url"]
    assert seen["auth"] == "secret-gl"


def test_bitbucket_adapter_uses_source_commit_api(monkeypatch):
    import httpx

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_BB", "secret-bb")
    seen = {}

    async def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"hash": "abc"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = StorageTarget(
        id="bitbucket-primary", label="BB", provider="bitbucket", role="primary",
        repo="workspace/repo", token_env="AI_RECORD_STORAGE_TOKEN_BB",
    )
    coord = StorageCoordinator([target], client=client)
    asyncio.run(coord.write(kind="feedback", content=b"{}", commit_message="test"))
    asyncio.run(client.aclose())
    assert seen["method"] == "POST"
    assert "/2.0/repositories/workspace/repo/src" in seen["url"]
    assert seen["auth"] == "Bearer secret-bb"


def test_hf_fine_grained_repo_write_preflight_verifies(monkeypatch):
    import types

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_VERIFY", "hf_" + "v" * 60)

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def auth_check(self, repo_id, repo_type=None, token=None, write=False):
            assert repo_id == "org/dataset"
            assert repo_type == "dataset"
            assert write is True
            return None

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    target = StorageTarget(
        id="hf-primary", label="HF", provider="huggingface", role="primary",
        repo="org/dataset", token_env="AI_RECORD_STORAGE_TOKEN_HF_VERIFY",
        token_type="fine-grained",
    )
    coord = StorageCoordinator([target])
    assert asyncio.run(coord._hf_write_capability(target)) == "verified"


def test_hf_fine_grained_explicit_permission_denial_is_blocked(monkeypatch):
    import types

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_DENIED", "hf_" + "d" * 60)

    class Denied(Exception):
        def __init__(self):
            super().__init__("private provider message must not escape")
            self.response = types.SimpleNamespace(status_code=403)

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def auth_check(self, repo_id, repo_type=None, token=None, write=False):
            raise Denied()

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    target = StorageTarget(
        id="hf-primary", label="HF", provider="huggingface", role="primary",
        repo="org/dataset", token_env="AI_RECORD_STORAGE_TOKEN_HF_DENIED",
        token_type="fine-grained",
    )
    coord = StorageCoordinator([target])
    assert asyncio.run(coord._hf_write_capability(target)) == "denied"


def test_app_prefers_neutral_hf_dataset_token_name():
    root = Path(__file__).resolve().parents[1]
    source = (root / "_hf_spaces_proxy" / "app.py").read_text(encoding="utf-8")
    assert 'os.environ.get("HF_DATASET_TOKEN", "")' in source
    assert "HF_DATASET_TOKEN_EXPLICIT or HF_WRITE_TOKEN or HF_TOKEN" in source
    assert "legacy_token=HF_DATASET_TOKEN" in source
    assert '"hf_dataset_token_type": "unknown"' in source


def test_shared_env_loader_prefers_hf_dataset_token(monkeypatch):
    from _utils._shared_logic import load_proxy_env

    monkeypatch.setenv("HF_TOKEN", "hf_inference")
    monkeypatch.setenv("HF_TOKEN_TYPE", "read")
    monkeypatch.setenv("HF_WRITE_TOKEN", "hf_legacy")
    monkeypatch.setenv("HF_WRITE_TOKEN_TYPE", "write")
    monkeypatch.setenv("HF_DATASET_TOKEN", "hf_dataset")
    monkeypatch.setenv("HF_DATASET_TOKEN_TYPE", "fine-grained")
    cfg = load_proxy_env()
    assert cfg["hf_dataset_token"] == "hf_dataset"
    assert cfg["hf_dataset_token_type"] == "fine-grained"
    assert cfg["hf_write_token"] == "hf_legacy"


def test_multisource_config_suppresses_legacy_hf_token_warning_path():
    root = Path(__file__).resolve().parents[1]
    source = (root / "_hf_spaces_proxy" / "app.py").read_text(encoding="utf-8")
    assert "_LEGACY_HF_STORAGE_ACTIVE: bool = bool(" in source
    assert "TRAINING_DATASET_REPO and not RECORD_STORAGE_TARGETS" in source
    assert 'TRAINING_DATASET_REPO if _LEGACY_HF_STORAGE_ACTIVE else ""' in source
    assert "if _LEGACY_HF_STORAGE_ACTIVE and not HF_DATASET_TOKEN:" in source

def test_proxy_requirements_enable_modern_hf_write_preflight():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "_hf_spaces_proxy" / "requirements.txt").read_text(encoding="utf-8")
    assert "huggingface_hub==1.16.1" in requirements
    assert "huggingface_hub~=0.23.0" not in requirements


def test_dataset_collection_guide_matches_secret_variable_contract():
    root = Path(__file__).resolve().parents[1]
    guide = (root / "_hf_spaces_proxy" / "DATASET_COLLECTION_GUIDANCE.md").read_text(encoding="utf-8")
    assert "`RECORD_STORAGE_TARGETS` | **Variable**" in guide
    assert "`AI_RECORD_STORAGE_TOKEN_HF_PRIMARY` | **Secret**" in guide
    assert "`TRAINING_DATASET_REPO` | **Variable**" in guide
    assert '"token_type": "fine-grained"' in guide
    assert "you do not need" in guide and "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY_TYPE" in guide
    assert "Primary only" in guide
    assert "--all-targets" in guide



def test_proxy_dockerfile_copies_contribution_lifecycle_ledger():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "_hf_spaces_proxy" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=1000:1000 _utils ./_utils" in dockerfile
    assert (root / "_hf_spaces_proxy" / "_utils" / "_contribution_ledger.py").is_file()


def test_github_current_view_removal_uses_contents_delete(monkeypatch):
    import httpx

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GH_DEL", "secret-gh")
    seen = []

    async def handler(request):
        seen.append((request.method, str(request.url)))
        if request.method == "GET":
            return httpx.Response(200, json={"sha": "abc123"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"commit": {"sha": "commit"}})
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = StorageTarget(
        id="github-primary", label="GH", provider="github", role="primary",
        repo="org/repo", token_env="AI_RECORD_STORAGE_TOKEN_GH_DEL",
    )
    coord = StorageCoordinator([target], client=client)
    result = asyncio.run(coord.remove_current_view(
        {"github-primary": "contributions/2026/08/29/ct_deadbeef.jsonl"},
        record_id="record-id",
    ))
    asyncio.run(client.aclose())
    assert result == {"github-primary": "removed-current-view"}
    assert [m for m, _ in seen] == ["GET", "DELETE"]
    assert "record-id" in coord._suppressed_retry_record_ids


def test_gitlab_current_view_removal_uses_repository_file_delete(monkeypatch):
    import httpx

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_GL_DEL", "secret-gl")
    seen = {}

    async def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("private-token")
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = StorageTarget(
        id="gitlab-primary", label="GL", provider="gitlab", role="primary",
        repo="org/sub/repo", token_env="AI_RECORD_STORAGE_TOKEN_GL_DEL",
        api_base="https://gitlab.example/api/v4",
    )
    coord = StorageCoordinator([target], client=client)
    result = asyncio.run(coord.remove_current_view(
        {"gitlab-primary": "contributions/x.jsonl"}, record_id="rid"
    ))
    asyncio.run(client.aclose())
    assert result == {"gitlab-primary": "removed-current-view"}
    assert seen["method"] == "DELETE"
    assert "/projects/org%2Fsub%2Frepo/repository/files/contributions%2Fx.jsonl" in seen["url"]
    assert seen["auth"] == "secret-gl"


def test_bitbucket_current_view_removal_uses_files_parameter(monkeypatch):
    import httpx

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_BB_DEL", "secret-bb")
    seen = []

    async def handler(request):
        body = (await request.aread()).decode("utf-8", errors="replace")
        seen.append((request.method, str(request.url), body))
        if request.method == "GET":
            return httpx.Response(200, content=b"existing")
        if request.method == "POST":
            return httpx.Response(201, json={"hash": "new"})
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = StorageTarget(
        id="bb-primary", label="BB", provider="bitbucket", role="primary",
        repo="workspace/repo", token_env="AI_RECORD_STORAGE_TOKEN_BB_DEL",
    )
    coord = StorageCoordinator([target], client=client)
    result = asyncio.run(coord.remove_current_view(
        {"bb-primary": "contributions/x.jsonl"}, record_id="rid"
    ))
    asyncio.run(client.aclose())
    assert result == {"bb-primary": "removed-current-view"}
    assert seen[0][0] == "GET" and seen[1][0] == "POST"
    assert "files=%2Fcontributions%2Fx.jsonl" in seen[1][2]


def test_huggingface_current_view_removal_uses_commit_operation_delete(monkeypatch):
    import types

    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_DEL", "hf_secret")
    captured = {}

    class FakeDelete:
        def __init__(self, *, path_in_repo):
            self.path_in_repo = path_in_repo

    class FakeApi:
        def __init__(self, *, token):
            captured["token"] = token

        def create_commit(self, **kwargs):
            captured.update(kwargs)
            return object()

    fake = types.SimpleNamespace(CommitOperationDelete=FakeDelete, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    target = StorageTarget(
        id="hf-primary", label="HF", provider="huggingface", role="primary",
        repo="org/dataset", token_env="AI_RECORD_STORAGE_TOKEN_HF_DEL",
        token_type="fine-grained",
    )
    coord = StorageCoordinator([target])
    result = asyncio.run(coord.remove_current_view(
        {"hf-primary": "contributions/x.jsonl"}, record_id="rid"
    ))
    assert result == {"hf-primary": "removed-current-view"}
    assert captured["token"] == "hf_secret"
    assert captured["repo_id"] == "org/dataset"
    assert captured["repo_type"] == "dataset"
    assert captured["operations"][0].path_in_repo == "contributions/x.jsonl"


def test_storage_path_timestamp_makes_replay_path_stable(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_STABLE", "hf_secret")
    target = StorageTarget(
        id="hf-primary", label="HF", provider="huggingface", role="primary",
        repo="org/dataset", token_env="AI_RECORD_STORAGE_TOKEN_HF_STABLE",
        token_type="fine-grained",
    )
    coord = StorageCoordinator([target])
    seen = []

    async def fake_dispatch(_target, path, content, message):
        seen.append((path, bytes(content), message))

    coord._dispatch = fake_dispatch

    async def run():
        one = await coord.write(
            kind="contributions", content=b"same-reviewed-row", commit_message="one",
            path_timestamp=1_700_000_000.0,
        )
        two = await coord.write(
            kind="contributions", content=b"same-reviewed-row", commit_message="two",
            path_timestamp=1_700_000_000.0,
        )
        return one, two

    one, two = asyncio.run(run())
    assert one.record_id == two.record_id
    assert one.paths == two.paths
    assert seen[0][0] == seen[1][0]


def test_withdrawal_suppression_is_rechecked_inside_target_lock(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_TOKEN_HF_SUPPRESS", "hf_secret")
    target = StorageTarget(
        id="hf-primary", label="HF", provider="huggingface", role="primary",
        repo="org/dataset", token_env="AI_RECORD_STORAGE_TOKEN_HF_SUPPRESS",
        token_type="fine-grained",
    )
    coord = StorageCoordinator([target])
    coord._suppressed_retry_record_ids.add("withdrawn-rid")

    async def should_not_dispatch(*args, **kwargs):
        raise AssertionError("withdrawn retry reached provider dispatch")

    coord._dispatch = should_not_dispatch

    async def run():
        with pytest.raises(StorageWriteError) as exc:
            await coord._write_target(
                target, "contributions", "withdrawn-rid", b"eligible-row", "retry",
                path="contributions/2026/08/29/ct_withdrawn-rid.jsonl",
            )
        assert exc.value.code == "WITHDRAWN"

    asyncio.run(run())
