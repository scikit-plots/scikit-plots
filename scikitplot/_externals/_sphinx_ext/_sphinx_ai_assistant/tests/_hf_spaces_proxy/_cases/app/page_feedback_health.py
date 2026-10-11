"""
``/health`` says whether the Space accepts page feedback from each default site.

A redeploy or a ``FEEDBACK_ALLOWED_SITE_IDS`` change on the Space is checked
from a browser with ``GET /health``; custom site IDs are counted, never listed.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import replace

from fastapi.testclient import TestClient

from ...._paths import RUNTIME_ROOT

PROXY = RUNTIME_ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy = importlib.import_module("app")


def test_health_reports_both_default_sites_allowed_by_default():
    with TestClient(proxy.app) as client:
        status = client.get("/health").json()["page_feedback"]
    assert status["default_site_ids_allowed"] == {
        "scikit-plots-learn": True,
        "scikit-plots": True,
    }
    assert status["any_site_id"] is False
    assert status["review_mode"] == proxy.PAGE_FEEDBACK_SERVICE_CONFIG.review_mode


def test_health_counts_but_never_lists_custom_site_ids(monkeypatch):
    narrowed = replace(
        proxy.PAGE_FEEDBACK_SERVICE_CONFIG,
        allowed_site_ids=("scikit-plots-learn", "private-intranet-docs"),
    )
    monkeypatch.setattr(proxy, "PAGE_FEEDBACK_SERVICE_CONFIG", narrowed)
    with TestClient(proxy.app) as client:
        body = client.get("/health").json()
    status = body["page_feedback"]
    assert status["default_site_ids_allowed"] == {
        "scikit-plots-learn": True,
        "scikit-plots": False,
    }
    assert status["allowed_site_id_count"] == 2
    assert "private-intranet-docs" not in str(body)


def test_health_reports_any_site_when_the_allowlist_is_blank(monkeypatch):
    monkeypatch.setattr(
        proxy,
        "PAGE_FEEDBACK_SERVICE_CONFIG",
        replace(proxy.PAGE_FEEDBACK_SERVICE_CONFIG, allowed_site_ids=()),
    )
    with TestClient(proxy.app) as client:
        status = client.get("/health").json()["page_feedback"]
    assert status["any_site_id"] is True
    assert status["allowed_site_id_count"] is None
    assert all(status["default_site_ids_allowed"].values())
