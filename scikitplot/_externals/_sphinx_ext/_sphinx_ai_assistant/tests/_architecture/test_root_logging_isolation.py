"""
The service's logging configuration stays inside this test package.

Notes
-----
**Developer notes.** Importing the proxy application replaces the root
logger's handlers with one whose filter rewrites URLs in log records in
place. ``conftest.py`` takes that configuration off the root logger when
collection ends and installs it for this package's tests only. These tests
pin both halves: it is in force here, and the removal leaves pytest's own
handlers and earlier handlers alone.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from .. import conftest


def test_the_held_handlers_are_installed_while_this_package_runs():
    root = logging.getLogger()
    for handler in conftest._held_service_handlers():
        assert handler in root.handlers


def test_removal_spares_earlier_handlers_and_pytests_own():
    root = logging.getLogger()
    earlier, late = logging.NullHandler(), logging.NullHandler()
    root.addHandler(earlier)
    outside = list(root.handlers)
    pytest_owned = [h for h in outside if type(h).__module__.startswith("_pytest.")]
    root.addHandler(late)
    try:
        conftest._remove_handlers_added_since(outside)
        assert late not in root.handlers
        assert earlier in root.handlers
        assert all(handler in root.handlers for handler in pytest_owned)
    finally:
        root.removeHandler(earlier)
        root.removeHandler(late)


def test_a_handler_attached_by_pytest_after_the_snapshot_is_left_alone():
    from _pytest.logging import LogCaptureHandler

    root = logging.getLogger()
    outside = list(root.handlers)
    capture = LogCaptureHandler()
    root.addHandler(capture)
    try:
        conftest._remove_handlers_added_since(outside)
        assert capture in root.handlers
    finally:
        root.removeHandler(capture)


def test_the_fixture_is_automatic_and_covers_the_whole_package():
    tree = ast.parse(Path(conftest.__file__).read_text(encoding="utf-8"))
    (function,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_service_logging_inside_this_package"
    ]
    (decorator,) = function.decorator_list
    keywords = {item.arg: item.value.value for item in decorator.keywords}
    assert keywords == {"scope": "package", "autouse": True}


def test_the_root_logger_is_recorded_before_any_service_import():
    source = Path(conftest.__file__).read_text(encoding="utf-8")
    assert source.index("_ROOT_LOGGING_AT_IMPORT: dict") < source.index("def _bootstrap_submodule")
