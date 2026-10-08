from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from _sphinx_ext._sphinx_feedback._aggregate import write_aggregate
from _sphinx_ext._sphinx_feedback._contracts import (
    FeedbackConflictError,
    FeedbackValidationError,
    build_feedback_event,
    feedback_request_hash,
)
from _sphinx_ext._sphinx_feedback._service._config import (
    FeedbackServiceConfigError,
    StorageTarget,
    load_service_config,
    parse_storage_targets,
)
from _sphinx_ext._sphinx_feedback._service._core import (
    FeedbackServiceUnavailable,
    PageFeedbackService,
)
from _sphinx_ext._sphinx_feedback._service._sqlite import SQLiteFeedbackStore


def request(feedback_id="feedback-" + "1" * 48, *, site_id="docs", page_id="guide/install", rating=1):
    return {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": site_id,
        "page_id": page_id,
        "feedback_id": feedback_id,
        "rating": rating,
        "mode": "quick",
        "contributor": {"display_name": ""},
    }


def target(**overrides):
    data = {
        "id": "primary",
        "label": "Primary",
        "authority": "feedback",
        "provider": "github",
        "role": "primary",
        "repo": "org/repo",
        "branch": "main",
        "paths": {"feedback": "feedback"},
        "token_env": ["FEEDBACK_GITHUB_TOKEN"],
        "expose_links": True,
    }
    data.update(overrides)
    return data


def test_disabled_service_is_default_and_has_no_targets():
    cfg = load_service_config({})
    assert cfg.review_mode == "disabled"
    assert cfg.targets == ()




def test_disabled_mode_accepts_explicit_empty_target_json():
    cfg = load_service_config(
        {"FEEDBACK_REVIEW_MODE": "disabled", "FEEDBACK_STORAGE_TARGETS": "[]"}
    )
    assert cfg.targets == ()


def test_storage_target_json_rejects_duplicate_fields():
    raw = (
        '[{"id":"primary","id":"shadow","label":"Primary",'
        '"authority":"feedback","provider":"github","role":"primary",'
        '"repo":"org/repo","branch":"main"}]'
    )
    with pytest.raises(FeedbackServiceConfigError, match="duplicate JSON field"):
        parse_storage_targets(raw)


def test_sqlite_target_rejects_network_only_configuration():
    base = {
        "id": "primary",
        "label": "SQLite",
        "authority": "feedback",
        "provider": "sqlite",
        "role": "primary",
        "database": "feedback.sqlite3",
    }
    for extra in [
        {"repo": "org/repo"},
        {"branch": "main"},
        {"paths": {"feedback": "feedback"}},
        {"token_env": ["FEEDBACK_GITHUB_TOKEN"]},
        {"expose_links": True},
    ]:
        with pytest.raises(FeedbackServiceConfigError):
            parse_storage_targets([{**base, **extra}])


def test_sqlite_review_mode_is_strictly_local_only(tmp_path):
    raw = json.dumps(
        [
            {
                "id": "primary",
                "label": "SQLite",
                "authority": "feedback",
                "provider": "sqlite",
                "role": "primary",
                "database": str(tmp_path / "feedback.db"),
            },
            target(id="github-mirror", role="mirror"),
        ]
    )
    with pytest.raises(FeedbackServiceConfigError, match="local-only"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "sqlite",
                "FEEDBACK_STORAGE_TARGETS": raw,
            }
        )


def test_invalid_credential_value_fails_closed_without_transport_call():
    parsed = parse_storage_targets(
        [target(token_env=["FEEDBACK_GITHUB_TOKEN"])]
    )[0]
    for invalid in ["bad\nvalue", " token", "token ", "tökén", "x" * 4097]:
        with pytest.raises(FeedbackServiceConfigError, match="credential value"):
            parsed.resolve_token({"FEEDBACK_GITHUB_TOKEN": invalid})


def test_sqlite_mode_builds_local_primary(tmp_path):
    cfg = load_service_config({"FEEDBACK_REVIEW_MODE": "sqlite", "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db")})
    assert cfg.primary.provider == "sqlite"


def test_github_fallback_token_chain_is_server_side():
    parsed = parse_storage_targets([target(token_env=None)])[0]
    assert parsed.token_env == (
        "FEEDBACK_GITHUB_TOKEN",
        "GITHUB_TOKEN",
        "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR",
    )


def test_token_resolution_obeys_declared_order():
    parsed = parse_storage_targets([target(token_env=["FEEDBACK_GITHUB_TOKEN", "GITHUB_TOKEN"])])[0]
    token, name = parsed.resolve_token({"GITHUB_TOKEN": "second", "FEEDBACK_GITHUB_TOKEN": "first"})
    assert (token, name) == ("first", "FEEDBACK_GITHUB_TOKEN")


@pytest.mark.parametrize("env_name", ["AWS_SECRET_ACCESS_KEY", "HOME", "PATH"])
def test_unrelated_process_secret_names_are_rejected(env_name):
    with pytest.raises(FeedbackServiceConfigError, match="allowlist"):
        parse_storage_targets([target(token_env=[env_name])])


def test_literal_secret_fields_are_rejected():
    raw = target()
    raw["token"] = "secret"
    with pytest.raises(FeedbackServiceConfigError):
        parse_storage_targets([raw])


def test_unknown_target_and_path_fields_fail_closed():
    raw = target()
    raw["telemetry"] = True
    with pytest.raises(FeedbackServiceConfigError, match="unsupported field"):
        parse_storage_targets([raw])
    raw = target(paths={"feedback": "feedback", "other": "x"})
    with pytest.raises(FeedbackServiceConfigError, match="unsupported field"):
        parse_storage_targets([raw])


def test_authority_must_be_feedback():
    with pytest.raises(FeedbackServiceConfigError, match="authority"):
        parse_storage_targets([target(authority="records")])


def test_expose_links_requires_real_boolean():
    with pytest.raises(FeedbackServiceConfigError, match="boolean"):
        parse_storage_targets([target(expose_links="false")])


def test_topology_requires_exactly_one_primary():
    with pytest.raises(FeedbackServiceConfigError):
        parse_storage_targets([target(role="mirror")])
    with pytest.raises(FeedbackServiceConfigError):
        parse_storage_targets([target(id="a"), target(id="b")])


def test_storage_target_count_is_bounded():
    values = [target(id=f"t{i}", role="primary" if i == 0 else "mirror") for i in range(6)]
    with pytest.raises(FeedbackServiceConfigError):
        parse_storage_targets(values)


@pytest.mark.parametrize(
    "name,value",
    [
        ("FEEDBACK_PAGE_MAX_BODY_BYTES", "not-an-int"),
        ("FEEDBACK_PAGE_MAX_BODY_BYTES", "999999"),
        ("FEEDBACK_PAGE_RATE_LIMIT_PER_HOUR", "0"),
        ("FEEDBACK_PAGE_RATE_LIMIT_PER_HOUR", "241"),
        ("FEEDBACK_MIRROR_TIMEOUT_SECONDS", "0.1"),
        ("FEEDBACK_MIRROR_TIMEOUT_SECONDS", "16"),
    ],
)
def test_security_limits_fail_closed_instead_of_silent_clamping(name, value):
    with pytest.raises(FeedbackServiceConfigError):
        load_service_config({"FEEDBACK_REVIEW_MODE": "sqlite", name: value})


def test_sqlite_accept_replay_and_conflict(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    req = request()
    event = build_feedback_event(req)
    digest = feedback_request_hash(req)
    assert store.put(feedback_id=req["feedback_id"], request_hash=digest, event=event)["status"] == "accepted"
    assert store.put(feedback_id=req["feedback_id"], request_hash=digest, event=event)["status"] == "replay"
    changed = request(rating=-1)
    with pytest.raises(FeedbackConflictError):
        store.put(
            feedback_id=req["feedback_id"],
            request_hash=feedback_request_hash(changed),
            event=build_feedback_event(changed),
        )


def test_sqlite_concurrent_same_event_is_exactly_once(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    req = request()
    event = build_feedback_event(req)
    digest = feedback_request_hash(req)

    def put_once():
        return store.put(feedback_id=req["feedback_id"], request_hash=digest, event=event)["status"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(lambda _: put_once(), range(4)))
    assert statuses.count("accepted") == 1
    assert statuses.count("replay") == 3


def test_sqlite_concurrent_first_open_across_store_instances_is_exactly_once(tmp_path):
    path = tmp_path / "feedback.db"
    req = request()
    event = build_feedback_event(req)
    digest = feedback_request_hash(req)

    def put_once():
        store = SQLiteFeedbackStore(path)
        return store.put(
            feedback_id=req["feedback_id"], request_hash=digest, event=event
        )["status"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: put_once(), range(8)))
    assert statuses.count("accepted") == 1
    assert statuses.count("replay") == 7


def test_sqlite_schema_has_no_participant_identity_or_timestamp_columns(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    req = request()
    store.put(feedback_id=req["feedback_id"], request_hash=feedback_request_hash(req), event=build_feedback_event(req))
    with sqlite3.connect(store.path) as conn:
        names = [row[1] for row in conn.execute("PRAGMA table_info(feedback_events)")]
    assert names == ["feedback_id", "request_hash", "site_id", "page_id", "event_json"]


def test_sqlite_aggregate_is_site_scoped(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    for index, site in enumerate(["docs-a", "docs-b"]):
        req = request(feedback_id="feedback-" + str(index + 1) * 48, site_id=site, rating=index * 2 - 1)
        store.put(feedback_id=req["feedback_id"], request_hash=feedback_request_hash(req), event=build_feedback_event(req))
    assert store.aggregate(site_id="docs-a") == {
        "guide/install": {
            "count": 1, "score": -1, "positive_count": 0, "negative_count": 1, "neutral_count": 0
        }
    }
    assert store.aggregate(site_id="docs-b") == {
        "guide/install": {
            "count": 1, "score": 1, "positive_count": 1, "negative_count": 0, "neutral_count": 0
        }
    }


def test_offline_aggregate_is_deterministic_site_scoped_and_atomic(tmp_path):
    rows = [build_feedback_event(request()), build_feedback_event(request("feedback-" + "2" * 48, rating=-1))]
    path = tmp_path / "aggregate.json"
    write_aggregate(path, rows, site_id="docs")
    first = path.read_bytes()
    write_aggregate(path, reversed(rows), site_id="docs")
    second = path.read_bytes()
    assert first == second
    data = json.loads(first)
    assert data["site_id"] == "docs"
    assert data["contract"] == "page.feedback-aggregate.v3"
    assert data["pages"]["guide/install"] == {
        "count": 2, "score": 0, "positive_count": 1, "negative_count": 1, "neutral_count": 0
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_offline_aggregate_can_explicitly_certify_complete_snapshot(tmp_path):
    path = tmp_path / "aggregate-complete.json"
    write_aggregate(path, [], site_id="docs", complete_snapshot=True)
    data = json.loads(path.read_bytes())
    assert data == {
        "complete": True,
        "contract": "page.feedback-aggregate.v3",
        "pages": {},
        "site_id": "docs",
    }


def test_offline_aggregate_complete_snapshot_flag_must_be_boolean(tmp_path):
    with pytest.raises(ValueError, match="complete_snapshot must be a boolean"):
        write_aggregate(tmp_path / "aggregate.json", [], site_id="docs", complete_snapshot="yes")


def test_offline_aggregate_rejects_mixed_sites(tmp_path):
    rows = [build_feedback_event(request(site_id="other"))]
    with pytest.raises(ValueError, match="site_id"):
        write_aggregate(tmp_path / "a.json", rows, site_id="docs")




def test_offline_aggregate_rejects_invalid_site_id_even_when_empty(tmp_path):
    with pytest.raises(ValueError, match="site_id"):
        write_aggregate(tmp_path / "aggregate.json", [], site_id="../docs")


def test_sqlite_aggregate_rejects_invalid_site_id(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    with pytest.raises(ValueError, match="site_id"):
        store.aggregate(site_id="../docs")


def test_page_feedback_service_sqlite_roundtrip(tmp_path):
    cfg = load_service_config({"FEEDBACK_REVIEW_MODE": "sqlite", "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db")})
    service = PageFeedbackService(cfg)
    first = asyncio.run(service.submit(request()))
    second = asyncio.run(service.submit(request()))
    assert first["status"] == "accepted"
    assert second["status"] == "replay"
    assert first["contract"] == "page.feedback-receipt.v1"


def test_feedback_token_allowlist_rejects_non_token_feedback_env():
    with pytest.raises(FeedbackServiceConfigError, match="allowlist"):
        parse_storage_targets([target(token_env=["FEEDBACK_GITHUB_REPOSITORY"])])


@pytest.mark.parametrize("branch", ["main/", "a//b", "a.lock", "a/b.lock", "a..b"])
def test_unsafe_git_branches_rejected(branch):
    with pytest.raises(FeedbackServiceConfigError, match="branch"):
        parse_storage_targets([target(branch=branch)])


def test_provider_shorthand_is_validated_not_trusted():
    with pytest.raises(FeedbackServiceConfigError, match="owner/repo"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "provider-pr",
                "FEEDBACK_GITHUB_REPOSITORY": "not a repo",
            }
        )


def test_unimplemented_provider_fails_at_service_configuration_boundary():
    raw = json.dumps(
        [
            target(
                provider="huggingface",
                repo="org/repo",
                token_env=["FEEDBACK_HF_TOKEN"],
            )
        ]
    )
    with pytest.raises(FeedbackServiceConfigError, match="not implemented"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "provider-pr",
                "FEEDBACK_STORAGE_TARGETS": raw,
            }
        )


def test_offline_aggregate_deduplicates_exact_event_and_rejects_conflict(tmp_path):
    first = build_feedback_event(request())
    path = tmp_path / "aggregate.json"
    write_aggregate(path, [first, dict(first)], site_id="docs")
    data = json.loads(path.read_bytes())
    assert data["pages"]["guide/install"] == {
        "count": 1, "score": 1, "positive_count": 1, "negative_count": 0, "neutral_count": 0
    }

    conflict = json.loads(json.dumps(first))
    conflict["feedback"]["rating"] = -1
    with pytest.raises(ValueError, match="conflicting durable content"):
        write_aggregate(path, [first, conflict], site_id="docs")


def test_primary_provider_receipt_status_is_fail_closed(tmp_path, monkeypatch):
    cfg = load_service_config(
        {"FEEDBACK_REVIEW_MODE": "sqlite", "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db")}
    )
    service = PageFeedbackService(cfg)

    async def invalid_write(*args, **kwargs):
        return {"status": "maybe"}

    monkeypatch.setattr(service, "_write_target", invalid_write)
    with pytest.raises(FeedbackServiceUnavailable, match="invalid receipt"):
        asyncio.run(service.submit(request()))


def test_mirrors_run_concurrently_and_timeout_as_degraded(tmp_path):
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "sqlite",
            "FEEDBACK_SQLITE_PATH": str(tmp_path / "primary.db"),
            "FEEDBACK_MIRROR_TIMEOUT_SECONDS": "0.5",
            "FEEDBACK_STORAGE_TARGETS": json.dumps(
                [
                    {
                        "id": "primary", "label": "Primary", "authority": "feedback",
                        "provider": "sqlite", "role": "primary",
                        "database": str(tmp_path / "primary.db"),
                    },
                    {
                        "id": "mirror-a", "label": "A", "authority": "feedback",
                        "provider": "sqlite", "role": "mirror",
                        "database": str(tmp_path / "a.db"),
                    },
                    {
                        "id": "mirror-b", "label": "B", "authority": "feedback",
                        "provider": "sqlite", "role": "mirror",
                        "database": str(tmp_path / "b.db"),
                    },
                ]
            ),
        }
    )

    class SlowMirrors(PageFeedbackService):
        async def _write_target(self, target, **kwargs):
            if target.role == "mirror":
                await asyncio.sleep(1.0)
                return {"status": "accepted", "feedback_id": kwargs["event"]["feedback"]["id"]}
            return await super()._write_target(target, **kwargs)

    service = SlowMirrors(cfg)
    loop = asyncio.new_event_loop()
    try:
        started = loop.time()
        receipt = loop.run_until_complete(service.submit(request()))
        elapsed = loop.time() - started
    finally:
        loop.close()
    assert elapsed < 0.99
    assert receipt["status"] == "accepted"
    assert receipt["mirrors"] == {"mirror-a": "degraded", "mirror-b": "degraded"}


def test_aggregate_writer_rejects_output_over_loader_limit_without_replacing_file(tmp_path, monkeypatch):
    import _sphinx_ext._sphinx_feedback._aggregate as aggregate_module

    path = tmp_path / "aggregate.json"
    path.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(aggregate_module, "MAX_AGGREGATE_BYTES", 100)
    with pytest.raises(ValueError, match="exceeds 2 MiB"):
        aggregate_module.write_aggregate(path, [build_feedback_event(request())], site_id="docs")
    assert path.read_text(encoding="utf-8") == "existing\n"


def test_storage_targets_pathological_integer_is_wrapped_as_config_error():
    raw = '[{"id":"primary","label":"Primary","authority":"feedback",' \
          '"provider":"sqlite","role":"primary","database":"feedback.db",' \
          '"unexpected":' + "9" * 5000 + '}]'
    with pytest.raises(FeedbackServiceConfigError, match="not valid JSON"):
        parse_storage_targets(raw)


def test_invalid_mirror_receipt_is_reported_degraded(tmp_path, monkeypatch):
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "sqlite",
            "FEEDBACK_STORAGE_TARGETS": json.dumps(
                [
                    {
                        "id": "primary", "label": "Primary", "authority": "feedback",
                        "provider": "sqlite", "role": "primary",
                        "database": str(tmp_path / "primary.db"),
                    },
                    {
                        "id": "mirror", "label": "Mirror", "authority": "feedback",
                        "provider": "sqlite", "role": "mirror",
                        "database": str(tmp_path / "mirror.db"),
                    },
                ]
            ),
        }
    )
    service = PageFeedbackService(cfg)
    original = service._write_target

    async def controlled(target, **kwargs):
        if target.role == "mirror":
            return {"status": "maybe"}
        return await original(target, **kwargs)

    monkeypatch.setattr(service, "_write_target", controlled)
    receipt = asyncio.run(service.submit(request()))
    assert receipt["status"] == "accepted"
    assert receipt["mirrors"] == {"mirror": "degraded"}


def test_storage_target_json_has_explicit_size_bound():
    raw = '[{"id":"primary","label":"' + ("x" * (64 * 1024)) + '"}]'
    with pytest.raises(FeedbackServiceConfigError, match="64 KiB"):
        parse_storage_targets(raw)


def test_github_environment_shorthand_does_not_expose_review_link_by_default():
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "provider-pr",
            "FEEDBACK_GITHUB_REPOSITORY": "org/repo",
        }
    )
    assert cfg.primary is not None
    assert cfg.primary.provider == "github"
    assert cfg.primary.expose_links is False


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"id": 7}, "id must be a string"),
        ({"provider": 7}, "provider must be a string"),
        ({"role": 7}, "role must be a string"),
        ({"label": 7}, "label must be a string"),
        ({"branch": 7}, "branch must be a string"),
        ({"database": 7}, "database must be a string"),
        ({"paths": []}, "paths must be an object"),
        ({"paths": ""}, "paths must be an object"),
        ({"paths": {"feedback": 7}}, "feedback path must be a string"),
    ],
)
def test_storage_target_json_types_fail_closed_instead_of_coercing(patch, message):
    raw = target()
    raw.update(patch)
    with pytest.raises(FeedbackServiceConfigError, match=message):
        parse_storage_targets([raw])


def test_literal_secret_rejection_is_explicit_before_unknown_field_handling():
    raw = target()
    raw["token"] = "server-secret"
    with pytest.raises(FeedbackServiceConfigError, match="literal secrets"):
        parse_storage_targets([raw])


def test_storage_target_json_rejects_invalid_unicode_before_json_parse():
    with pytest.raises(FeedbackServiceConfigError, match="invalid Unicode"):
        parse_storage_targets("[\ud800]")


def test_aggregate_writer_is_content_idempotent_and_does_not_replace_equal_bytes(tmp_path, monkeypatch):
    import _sphinx_ext._sphinx_feedback._aggregate as aggregate_module

    path = tmp_path / "aggregate.json"
    rows = [build_feedback_event(request())]
    aggregate_module.write_aggregate(path, rows, site_id="docs")
    original = path.read_bytes()

    def forbidden_replace(*_args, **_kwargs):
        raise AssertionError("byte-identical aggregate must not be replaced")

    monkeypatch.setattr(aggregate_module.os, "replace", forbidden_replace)
    aggregate_module.write_aggregate(path, rows, site_id="docs")
    assert path.read_bytes() == original


def test_aggregate_writer_refuses_symlink_target(tmp_path):
    target = tmp_path / "aggregate.json"
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        write_aggregate(target, [build_feedback_event(request())], site_id="docs")
    assert outside.read_text(encoding="utf-8") == "outside"


def test_sqlite_events_validate_site_id_and_fail_closed_on_noncanonical_tamper(tmp_path):
    path = tmp_path / "feedback.db"
    store = SQLiteFeedbackStore(path)
    req = request()
    event = build_feedback_event(req)
    store.put(
        feedback_id=req["feedback_id"],
        request_hash=feedback_request_hash(req),
        event=event,
    )
    with pytest.raises(ValueError):
        store.events(site_id="../bad")

    with store._connect() as conn:
        raw = conn.execute(
            "SELECT event_json FROM feedback_events WHERE feedback_id = ?",
            (req["feedback_id"],),
        ).fetchone()[0]
        # Whitespace-only semantic rewrite is still a canonical-integrity failure.
        conn.execute(
            "UPDATE feedback_events SET event_json = ? WHERE feedback_id = ?",
            (" " + raw, req["feedback_id"]),
        )
    with pytest.raises(sqlite3.DatabaseError, match="not canonical"):
        store.events(site_id="docs")


def test_server_side_site_allowlist_rejects_foreign_site_before_storage(tmp_path):
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "sqlite",
            "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db"),
            "FEEDBACK_ALLOWED_SITE_IDS": "docs,api-docs",
        }
    )
    service = PageFeedbackService(cfg)
    assert cfg.allowed_site_ids == ("docs", "api-docs")
    with pytest.raises(FeedbackValidationError, match="not authorized"):
        asyncio.run(service.submit(request(site_id="foreign-docs")))
    assert not (tmp_path / "f.db").exists()


def test_allowed_site_ids_are_strict_bounded_and_canonical():
    for value in ["docs,,api", "bad/site", ",docs", "docs,"]:
        with pytest.raises(FeedbackServiceConfigError):
            load_service_config(
                {
                    "FEEDBACK_REVIEW_MODE": "sqlite",
                    "FEEDBACK_ALLOWED_SITE_IDS": value,
                }
            )


def test_token_env_entries_must_be_strings():
    bad = target(token_env=["FEEDBACK_GITHUB_TOKEN", 7])
    with pytest.raises(FeedbackServiceConfigError, match="must be environment-variable name strings"):
        parse_storage_targets([bad])


def test_credential_mapping_values_must_already_be_strings():
    parsed = parse_storage_targets([target()])[0]
    with pytest.raises(FeedbackServiceConfigError, match="must be a string"):
        parsed.resolve_token({"FEEDBACK_GITHUB_TOKEN": 123})


def test_sqlite_rejects_request_commitment_that_does_not_match_event(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    req = request()
    event = build_feedback_event(req)
    with pytest.raises(ValueError, match="does not match the durable feedback event"):
        store.put(
            feedback_id=req["feedback_id"],
            request_hash="0" * 64,
            event=event,
        )


def test_sqlite_readback_rejects_tampered_index_metadata_and_request_hash(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    req = request()
    event = build_feedback_event(req)
    digest = feedback_request_hash(req)
    store.put(feedback_id=req["feedback_id"], request_hash=digest, event=event)

    for column, value in [
        ("site_id", "other-site"),
        ("page_id", "other/page"),
        ("request_hash", "f" * 64),
    ]:
        with store._connect() as conn:
            conn.execute(
                f"UPDATE feedback_events SET {column} = ? WHERE feedback_id = ?",
                (value, req["feedback_id"]),
            )
        with pytest.raises(sqlite3.DatabaseError, match="metadata does not match"):
            store.events()
        with store._connect() as conn:
            conn.execute(
                "DELETE FROM feedback_events WHERE feedback_id = ?",
                (req["feedback_id"],),
            )
        store.put(feedback_id=req["feedback_id"], request_hash=digest, event=event)


def test_optional_page_authority_manifest_binds_page_and_revision(tmp_path):
    manifest = tmp_path / "authority.json"
    manifest.write_text(
        json.dumps(
            {
                "contract": "page.feedback-authority.v1",
                "sites": {
                    "docs": {
                        "guide/install": "rev-42",
                        "guide/other": "",
                    }
                },
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "sqlite",
            "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db"),
            "FEEDBACK_ALLOWED_SITE_IDS": "docs",
            "FEEDBACK_PAGE_AUTHORITY_FILE": str(manifest),
        }
    )
    service = PageFeedbackService(cfg)

    approved = request(page_id="guide/install")
    approved["page_revision"] = "rev-42"
    assert asyncio.run(service.submit(approved))["status"] == "accepted"

    wrong_revision = request(
        feedback_id="feedback-" + "2" * 48,
        page_id="guide/install",
    )
    wrong_revision["page_revision"] = "rev-41"
    with pytest.raises(FeedbackValidationError, match="page_revision"):
        asyncio.run(service.submit(wrong_revision))

    unknown_page = request(
        feedback_id="feedback-" + "3" * 48,
        page_id="guide/missing",
    )
    with pytest.raises(FeedbackValidationError, match="page_id is not authorized"):
        asyncio.run(service.submit(unknown_page))

    revision_optional = request(
        feedback_id="feedback-" + "4" * 48,
        page_id="guide/other",
    )
    revision_optional["page_revision"] = "any-public-revision"
    assert asyncio.run(service.submit(revision_optional))["status"] == "accepted"


def test_page_authority_manifest_rejects_duplicate_noncanonical_and_oversized_shapes(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"contract":"page.feedback-authority.v1","sites":{"docs":{},"docs":{}}}',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackServiceConfigError, match="duplicate JSON field"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "sqlite",
                "FEEDBACK_PAGE_AUTHORITY_FILE": str(duplicate),
            }
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(
            {
                "contract": "page.feedback-authority.v1",
                "sites": {"docs": {"guide/install": " rev-42 "}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FeedbackServiceConfigError, match="already be canonicalized"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "sqlite",
                "FEEDBACK_PAGE_AUTHORITY_FILE": str(noncanonical),
            }
        )

    wrong_contract = tmp_path / "wrong.json"
    wrong_contract.write_text('{"contract":"other.v1","sites":{}}', encoding="utf-8")
    with pytest.raises(FeedbackServiceConfigError, match="unsupported contract"):
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "sqlite",
                "FEEDBACK_PAGE_AUTHORITY_FILE": str(wrong_contract),
            }
        )


def test_explicit_empty_page_authority_manifest_denies_all_pages(tmp_path):
    manifest = tmp_path / "empty-authority.json"
    manifest.write_text(
        '{"contract":"page.feedback-authority.v1","sites":{}}',
        encoding="utf-8",
    )
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "sqlite",
            "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db"),
            "FEEDBACK_PAGE_AUTHORITY_FILE": str(manifest),
        }
    )
    assert cfg.page_authority == ()
    service = PageFeedbackService(cfg)
    with pytest.raises(FeedbackValidationError, match="page_id is not authorized"):
        asyncio.run(service.submit(request()))


def test_network_provider_rejects_explicit_empty_credential_alias_list():
    with pytest.raises(FeedbackServiceConfigError, match="requires at least one token_env"):
        parse_storage_targets([target(token_env=[])])


@pytest.mark.parametrize(
    "bad_receipt",
    [
        {"status": "accepted", "provider": "sqlite", "feedback_id": "feedback-" + "9" * 48, "request_hash": "0" * 64},
        {"status": "accepted", "provider": "github", "feedback_id": "feedback-" + "1" * 48, "request_hash": "0" * 64},
        {"status": "accepted", "provider": "sqlite", "feedback_id": "feedback-" + "1" * 48, "request_hash": "f" * 64},
    ],
)
def test_primary_provider_receipt_must_match_provider_event_and_commitment(tmp_path, monkeypatch, bad_receipt):
    cfg = load_service_config(
        {"FEEDBACK_REVIEW_MODE": "sqlite", "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.db")}
    )
    service = PageFeedbackService(cfg)

    async def invalid_write(*args, **kwargs):
        return bad_receipt

    monkeypatch.setattr(service, "_write_target", invalid_write)
    with pytest.raises(FeedbackServiceUnavailable, match="invalid receipt"):
        asyncio.run(service.submit(request()))


def test_github_base_feedback_namespace_is_rejected_at_configuration_time():
    with pytest.raises(FeedbackServiceConfigError, match="review namespace"):
        parse_storage_targets([target(branch="feedback")])


def test_mirror_children_are_cancelled_and_joined_with_parent(tmp_path, monkeypatch):
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "provider-pr",
            "FEEDBACK_STORAGE_TARGETS": json.dumps(
                [
                    target(id="primary", role="primary"),
                    target(id="mirror", role="mirror", repo="org/mirror"),
                ]
            ),
            "FEEDBACK_GITHUB_TOKEN": "token",
            "FEEDBACK_MIRROR_TIMEOUT_SECONDS": "15",
        }
    )
    service = PageFeedbackService(cfg, credential_env={"FEEDBACK_GITHUB_TOKEN": "token"})
    mirror_started = asyncio.Event()
    mirror_cancelled = asyncio.Event()

    async def fake_write(target_obj, *, request, event, request_hash, http_client=None):
        if target_obj.role == "primary":
            return {
                "status": "accepted",
                "provider": "github",
                "feedback_id": event["feedback"]["id"],
                "request_hash": request_hash,
            }
        mirror_started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            mirror_cancelled.set()
            raise

    monkeypatch.setattr(service, "_write_target", fake_write)

    async def scenario():
        task = asyncio.create_task(service.submit(request()))
        await asyncio.wait_for(mirror_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(mirror_cancelled.wait(), timeout=1)

    asyncio.run(scenario())


def test_sqlite_aggregate_can_filter_exact_page_revision(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    for index, revision in enumerate(["rev-1", "rev-2", "rev-2"]):
        req = request(feedback_id="feedback-" + f"{index + 1:x}" * 48)
        req["page_revision"] = revision
        event = build_feedback_event(req)
        store.put(
            feedback_id=req["feedback_id"],
            request_hash=feedback_request_hash(req),
            event=event,
        )
    assert store.aggregate(site_id="docs") == {
        "guide/install": {
            "count": 3, "score": 3, "positive_count": 3, "negative_count": 0, "neutral_count": 0
        }
    }
    assert store.aggregate(site_id="docs", page_revision="rev-2") == {
        "guide/install": {
            "count": 2, "score": 2, "positive_count": 2, "negative_count": 0, "neutral_count": 0
        }
    }


def test_revision_scoped_aggregate_writer_filters_old_events(tmp_path):
    first = request(feedback_id="feedback-" + "a" * 48)
    first["page_revision"] = "rev-1"
    second = request(feedback_id="feedback-" + "b" * 48)
    second["page_revision"] = "rev-2"
    path = tmp_path / "aggregate.json"
    write_aggregate(
        path,
        [build_feedback_event(first), build_feedback_event(second)],
        site_id="docs",
        page_revision="rev-2",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "contract": "page.feedback-aggregate.v3",
        "page_revision": "rev-2",
        "pages": {
            "guide/install": {
                "count": 1, "score": 1, "positive_count": 1, "negative_count": 0, "neutral_count": 0
            }
        },
        "site_id": "docs",
    }


def test_independently_cancelled_mirror_is_degraded_not_primary_failure(monkeypatch):
    cfg = load_service_config(
        {
            "FEEDBACK_REVIEW_MODE": "provider-pr",
            "FEEDBACK_STORAGE_TARGETS": json.dumps(
                [
                    target(id="primary", role="primary"),
                    target(id="mirror", role="mirror", repo="org/mirror"),
                ]
            ),
            "FEEDBACK_GITHUB_TOKEN": "token",
        }
    )
    service = PageFeedbackService(cfg, credential_env={"FEEDBACK_GITHUB_TOKEN": "token"})

    async def fake_write(target_obj, *, request, event, request_hash, http_client=None):
        if target_obj.role == "mirror":
            raise asyncio.CancelledError()
        return {
            "status": "accepted",
            "provider": "github",
            "feedback_id": event["feedback"]["id"],
            "request_hash": request_hash,
        }

    monkeypatch.setattr(service, "_write_target", fake_write)
    receipt = asyncio.run(service.submit(request()))
    assert receipt["status"] == "accepted"
    assert receipt["mirrors"] == {"mirror": "degraded"}


def test_sqlite_aggregate_tracks_positive_negative_and_neutral_independently(tmp_path):
    store = SQLiteFeedbackStore(tmp_path / "feedback.db")
    for index, rating in enumerate([5, 1, -5, -2, 0]):
        req = request(feedback_id="feedback-" + format(index + 1, "x") * 48, rating=rating)
        if rating not in {-1, 1}:
            req["mode"] = "detailed"
        event = build_feedback_event(req)
        store.put(
            feedback_id=req["feedback_id"],
            request_hash=feedback_request_hash(req),
            event=event,
        )
    assert store.aggregate(site_id="docs")["guide/install"] == {
        "count": 5,
        "score": -1,
        "positive_count": 2,
        "negative_count": 2,
        "neutral_count": 1,
    }
