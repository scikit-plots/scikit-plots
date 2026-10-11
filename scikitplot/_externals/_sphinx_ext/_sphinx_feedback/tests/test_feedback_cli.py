"""Generators: snapshot from reviewed events, and conf/env snippets."""

from __future__ import annotations

import json
from types import SimpleNamespace

from _sphinx_ext._sphinx_feedback._cli import main
from _sphinx_ext._sphinx_feedback._config import validate_config
from _sphinx_ext._sphinx_feedback._contracts import (
    build_feedback_event,
    feedback_event_request_hash,
    page_digest,
    repository_event_bytes,
)
from _sphinx_ext._sphinx_feedback._service._sqlite import SQLiteFeedbackStore


def _event(site, page, digit, rating):
    return build_feedback_event(
        {
            "contract": "page.feedback-request.v1",
            "action": "submit",
            "site_id": site,
            "page_id": page,
            "feedback_id": "feedback-" + digit * 48,
            "rating": rating,
            "mode": "quick" if rating in (-1, 1) else "detailed",
            "contributor": {"display_name": ""},
        }
    )


def _write_event(root, event):
    folder = root / "feedback" / "pages" / page_digest(event["site_id"], event["page_id"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{event['feedback']['id']}.json").write_bytes(repository_event_bytes(event))


def test_aggregate_reads_provider_layout_and_sqlite_and_skips_other_sites(tmp_path, capsys):
    events_root = tmp_path / "review-repo"
    _write_event(events_root, _event("scikit-plots", "index", "1", 1))
    _write_event(events_root, _event("scikit-plots", "index", "2", -1))
    _write_event(events_root, _event("scikit-plots-learn", "index", "3", 1))
    (events_root / "feedback" / "unrelated.json").write_text("{}", encoding="utf-8")
    store = SQLiteFeedbackStore(tmp_path / "f.sqlite3")
    extra = _event("scikit-plots", "api", "4", 0)
    store.put(
        feedback_id=extra["feedback"]["id"],
        request_hash=feedback_event_request_hash(extra),
        event=extra,
    )
    out = tmp_path / "docs" / "_feedback" / "aggregate.json"
    status = main(
        [
            "aggregate",
            "--site-id", "scikit-plots",
            "--out", str(out),
            "--events", str(events_root),
            "--sqlite", str(tmp_path / "f.sqlite3"),
            "--complete",
        ]
    )
    assert status == 0
    snapshot = json.loads(out.read_text(encoding="utf-8"))
    assert snapshot["complete"] is True
    assert snapshot["site_id"] == "scikit-plots"
    assert snapshot["pages"]["index"]["count"] == 2
    assert snapshot["pages"]["api"]["neutral_count"] == 1
    assert "other_sites_skipped=1" in capsys.readouterr().out


def test_aggregate_is_partial_unless_certified(tmp_path):
    root = tmp_path / "events"
    _write_event(root, _event("docs", "index", "5", 1))
    out = tmp_path / "agg.json"
    assert main(["aggregate", "--site-id", "docs", "--out", str(out), "--events", str(root)]) == 0
    assert "complete" not in json.loads(out.read_text(encoding="utf-8"))


def test_aggregate_refuses_a_corrupt_event_file(tmp_path, capsys):
    root = tmp_path / "events"
    bad = root / "pages" / ("a" * 24)
    bad.mkdir(parents=True)
    (bad / ("feedback-" + "6" * 48 + ".json")).write_text('{"not": "an event"}', "utf-8")
    status = main(["aggregate", "--site-id", "docs", "--out", str(tmp_path / "x.json"), "--events", str(root)])
    assert status == 2
    assert "invalid feedback event file" in capsys.readouterr().err


def test_init_prints_settings_the_build_accepts(capsys):
    assert (
        main(
            [
                "init",
                "--site-id", "intranet-docs",
                "--endpoint", "http://feedback.internal:8080/v1/feedback",
                "--policy", "private-network",
                "--mode", "local",
                "--site-origin", "http://docs.internal:8000",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "FEEDBACK_ORIGIN_POLICY=private-network" in out
    assert "FEEDBACK_ALLOWED_ORIGINS=http://docs.internal:8000" in out
    namespace: dict = {"extensions": []}
    conf = out.split("\n\n")[0]
    exec(conf, namespace)  # noqa: S102 - generated, test-local text
    values = {k: v for k, v in namespace.items() if k.startswith("feedback_")}
    values["feedback_aggregate_file"] = ""
    normalized = validate_config(SimpleNamespace(**values))
    assert normalized["endpoint"] == "http://feedback.internal:8080/v1/feedback"


def test_init_refuses_an_endpoint_the_policy_forbids(capsys):
    status = main(["init", "--site-id", "docs", "--endpoint", "http://feedback.internal/v1"])
    assert status == 2
    assert "private-network" in capsys.readouterr().err


def test_check_provider_runs_read_only_lookups_against_each_forge(tmp_path):
    import asyncio  # noqa: PLC0415

    import httpx  # noqa: PLC0415
    from test_forge_providers import BitbucketFake, GitLabFake, Repo  # noqa: PLC0415

    from _sphinx_ext._sphinx_feedback._cli import check_targets  # noqa: PLC0415
    from _sphinx_ext._sphinx_feedback._service._config import load_service_config  # noqa: PLC0415

    gitlab_repo, bitbucket_repo = Repo(), Repo()
    gitlab, bitbucket = GitLabFake(gitlab_repo), BitbucketFake(bitbucket_repo)
    targets = [
        {"id": "gl", "provider": "gitlab", "role": "primary", "repo": GitLabFake.project},
        {"id": "bb", "provider": "bitbucket", "role": "mirror", "repo": BitbucketFake.project},
        {"id": "hook", "provider": "webhook", "role": "mirror", "url": "https://h.example/x"},
    ]
    env = {
        "FEEDBACK_REVIEW_MODE": "provider-pr",
        "FEEDBACK_STORAGE_TARGETS": json.dumps(targets),
        "FEEDBACK_GITLAB_TOKEN": "t",
        "FEEDBACK_BITBUCKET_TOKEN": "t",
    }

    def route(request):
        return (gitlab if request.url.host == "gitlab.com" else bitbucket)(request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
            return await check_targets(load_service_config(env), env, client=client)

    results = asyncio.run(run())
    assert [(r[0], r[1]) for r in results] == [("gl", "ok"), ("bb", "ok"), ("hook", "skip")]
    writes = [c for c in gitlab.calls + bitbucket.calls if c[0] != "GET"]
    assert writes == []
