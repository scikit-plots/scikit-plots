"""Tests for side-effect-free downloader dispatch planning."""

from __future__ import annotations

import socket

import pytest

from scikitplot.corpus import AnyDownloader, DownloadPlan, DownloadPolicy


def _boom(*args, **kwargs):
    raise AssertionError("download planning must not touch DNS/network")


def test_single_plan_is_side_effect_free_and_matches_specialist(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    downloader = AnyDownloader.from_policy(
        "https://github.com/org/repo/blob/main/data.csv",
        DownloadPolicy.constrained(),
        github_token="secret-token",
    )
    plan = downloader.plan()
    assert isinstance(plan, DownloadPlan)
    assert plan.downloader == "GitHubDownloader"
    assert plan.github_token_configured is True
    assert plan.timeout == 15.0
    assert plan.max_bytes == 25 * 1024 * 1024
    payload = plan.to_dict()
    assert "secret-token" not in repr(payload)
    assert payload["block_private_ips"] is True


def test_batch_plan_preserves_input_order_and_service_fields(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    downloader = AnyDownloader(
        [
            "https://www.youtube.com/watch?v=abc123",
            "https://example.com/report.pdf",
        ],
        youtube_mode="transcript",
        youtube_language="tr",
        headers=[None, {"X-Trace": "private-value"}],
    )
    plans = downloader.plan_all()
    assert [item.index for item in plans] == [0, 1]
    assert plans[0].downloader == "YouTubeDownloader"
    assert plans[0].youtube_language == "tr"
    assert plans[0].max_retries is None
    assert plans[1].downloader == "WebDownloader"
    assert plans[1].headers_configured is True
    assert plans[1].max_retries == 3
    assert "private-value" not in repr([item.to_dict() for item in plans])


def test_plan_and_runtime_specialist_share_one_resolver(monkeypatch) -> None:
    # Building the specialist must still not perform DNS; actual download does.
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    urls = [
        "https://drive.google.com/file/d/123/view",
        "https://github.com/org/repo/raw/main/data.csv",
        "https://example.com/data.csv",
    ]
    downloader = AnyDownloader(urls)
    plans = downloader.plan_all()
    specialists = [type(downloader._build_specialist(i)).__name__ for i in range(3)]
    assert specialists == [plan.downloader for plan in plans]


def test_plan_does_not_create_temp_output_directory(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    downloader = AnyDownloader("https://example.com/data.csv")
    assert downloader._tmp_dir is None
    plan = downloader.plan()
    assert plan.output_path is None
    assert downloader._tmp_dir is None
