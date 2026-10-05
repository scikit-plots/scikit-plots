# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Package contract owned by :mod:`_hf_spaces_proxy._page_feedback._service.__init__`."""
from __future__ import annotations

from ....._hf_spaces_proxy._page_feedback import _service as package

EXPECTED_PUBLIC = [
    "FeedbackServiceConfig",
    "FeedbackServiceConfigError",
    "FeedbackServiceUnavailable",
    "PageFeedbackService",
    "StorageTarget",
    "load_service_config",
    "parse_storage_targets",
]


def test_public_surface_is_exactly_the_service_api() -> None:
    assert list(package.__all__) == EXPECTED_PUBLIC
    for name in EXPECTED_PUBLIC:
        assert getattr(package, name) is not None


def test_errors_are_exceptions_and_the_service_is_a_class() -> None:
    assert issubclass(package.FeedbackServiceConfigError, Exception)
    assert issubclass(package.FeedbackServiceUnavailable, Exception)
    assert isinstance(package.PageFeedbackService, type)


def test_the_asgi_adapter_is_not_imported_with_the_package() -> None:
    """``app`` reads its configuration from the environment when imported."""
    assert "app" not in package.__all__
    assert not hasattr(package, "app")
