"""Service logs: every outcome is logged, and no participant data ever is."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from .._contracts import (
    FeedbackConflictError,
    FeedbackValidationError,
)
from .._service._config import load_service_config
from .._service._core import PageFeedbackService


# scikitplot/_externals/_sphinx_ext/_sphinx_feedback/tests/test_service_logging.py
# logger = __import__("logging").getLogger(__name__)
LOGGER_NAME = __name__.rsplit(r".", 2)[0]
logger = logging.getLogger(LOGGER_NAME)

COMMENT = "PRIVATE-COMMENT-TEXT"
CREDIT = "Private Credit Name"


def request(rating=1):
    return {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": "docs",
        "page_id": "guide/install",
        "feedback_id": "feedback-" + "4" * 48,
        "rating": rating,
        "mode": "detailed",
        "comment": COMMENT,
        "contributor": {"display_name": CREDIT},
    }


def _everything(records) -> str:
    return "\n".join(
        f"{r.getMessage()} {json.dumps(getattr(r, 'feedback', {}), sort_keys=True)}"
        for r in records
    )


def test_outcomes_are_logged_without_comment_or_credit(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    service = PageFeedbackService(
        load_service_config(
            {
                "FEEDBACK_REVIEW_MODE": "sqlite",
                "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.sqlite3"),
                "FEEDBACK_ALLOWED_SITE_IDS": "docs",
            }
        )
    )
    asyncio.run(service.submit(request()))
    asyncio.run(service.submit(request()))
    with pytest.raises(FeedbackConflictError):
        asyncio.run(service.submit(request(rating=-1)))
    with pytest.raises(FeedbackValidationError):
        asyncio.run(service.submit({**request(), "site_id": "elsewhere"}))
    events = [getattr(r, "feedback", {}).get("event") for r in caplog.records]
    assert events == ["service_ready", "stored", "stored", "conflict", "rejected"]
    rejected = caplog.records[-1].feedback
    assert rejected["code"] == "site_not_allowed"
    text = _everything(caplog.records)
    assert COMMENT not in text
    assert CREDIT not in text


def test_failure_detail_stays_at_debug(monkeypatch, caplog):
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    module = types.ModuleType("feedback_log_adapter")

    class Failing:
        def put(self, *, event, request_hash):
            raise RuntimeError("password=hunter2")

    module.make = lambda *, target, env: Failing()
    monkeypatch.setitem(sys.modules, "feedback_log_adapter", module)
    targets = [{"id": "c", "provider": "custom", "role": "primary", "factory": "feedback_log_adapter:make"}]
    service = PageFeedbackService(
        load_service_config(
            {"FEEDBACK_REVIEW_MODE": "custom", "FEEDBACK_STORAGE_TARGETS": json.dumps(targets)}
        )
    )
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    with pytest.raises(Exception):  # noqa: B017,PT011 - asserted via the log below
        asyncio.run(service.submit(request()))
    assert caplog.records[-1].feedback["code"] == "custom_provider_failed"
    assert "hunter2" not in _everything(caplog.records)
    assert all(r.exc_info is None for r in caplog.records)
