"""Server-only generic page-feedback service API."""

from ._config import (
    FeedbackServiceConfig,
    FeedbackServiceConfigError,
    StorageTarget,
    load_service_config,
    parse_storage_targets,
)
from ._core import FeedbackServiceUnavailable, PageFeedbackService

__all__ = [
    "FeedbackServiceConfig",
    "FeedbackServiceConfigError",
    "FeedbackServiceUnavailable",
    "PageFeedbackService",
    "StorageTarget",
    "load_service_config",
    "parse_storage_targets",
]
