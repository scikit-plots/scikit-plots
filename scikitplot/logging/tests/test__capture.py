# scikitplot/logging/tests/test__capture.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of ``scikitplot/logging/_capture.py``, the body of ``captured_logger``.

Each test builds its own two-level hierarchy whose parent does not propagate,
as ``"scikitplot"`` does not, under a name no other test uses (Rule 55), and
calls the factory with the real ``caplog``. The fixture registered in the
conftests is this same function, so these tests hold in a run of
``scikit-plots-skinny`` alone, which ships no conftest that registers it.
"""

from __future__ import annotations

import logging
import uuid

import pytest

from .. import _capture


@pytest.fixture
def hierarchy():
    """
    Yield ``(parent, child)``: the parent does not propagate, as ``"scikitplot"``.

    Both are left with no handler, no level and default propagation.
    """
    name = f"skplt_capture_probe_{uuid.uuid4().hex}"
    parent = logging.getLogger(name)
    child = logging.getLogger(f"{name}.module")
    parent.propagate = False
    try:
        yield parent, child
    finally:
        for logger in (parent, child):
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False


@pytest.fixture
def capture(caplog):
    """The context-manager factory, as the registered fixture returns it."""
    return _capture.captured_logger(caplog)


def _state(logger, handler):
    return (
        logger.level,
        logger.propagate,
        logger.disabled,
        list(logger.handlers),
        handler.level,
        logging.root.manager.disable,
    )


class TestWhatIsCaptured:
    def test_a_child_of_a_logger_that_does_not_propagate_is_captured_once(
        self, capture, caplog, hierarchy
    ):
        _, child = hierarchy
        with capture(logging.WARNING, child):
            child.warning("one")
        assert [r.getMessage() for r in caplog.records] == ["one"]

    def test_records_below_the_level_are_not_captured(self, capture, caplog, hierarchy):
        _, child = hierarchy
        with capture(logging.WARNING, child):
            child.info("below")
            child.error("above")
        assert [r.levelno for r in caplog.records] == [logging.ERROR]

    def test_a_lower_level_than_the_handlers_is_captured(
        self, capture, caplog, hierarchy
    ):
        _, child = hierarchy
        with capture(logging.DEBUG, child):
            child.debug("detail")
        assert [r.getMessage() for r in caplog.records] == ["detail"]

    def test_records_from_before_the_block_are_cleared(
        self, capture, caplog, hierarchy
    ):
        _, child = hierarchy
        logging.getLogger().warning("earlier, at the root")
        assert caplog.records
        with capture(logging.WARNING, child):
            pass
        assert caplog.records == []

    def test_a_handler_already_on_the_logger_is_not_added_twice_or_removed(
        self, capture, caplog, hierarchy
    ):
        # pytest 9.1 attaches caplog.handler to a logger that does not
        # propagate when capture starts; the parent here is such a logger.
        # pytest 9.1 may have attached its own handlers here already, so the
        # handlers are compared with what they were, not with a fixed list.
        parent, _ = hierarchy
        parent.addHandler(caplog.handler)
        before = list(parent.handlers)
        with capture(logging.WARNING, parent):
            assert parent.handlers.count(caplog.handler) == 1
            parent.warning("once")
        assert [r.getMessage() for r in caplog.records] == ["once"]
        assert parent.handlers == before

    def test_a_disabled_logger_is_seen_while_captured(self, capture, caplog, hierarchy):
        _, child = hierarchy
        child.disabled = True
        with capture(logging.WARNING, child):
            child.warning("seen")
        assert [r.getMessage() for r in caplog.records] == ["seen"]
        assert child.disabled is True

    def test_logging_disable_is_lifted_and_put_back(self, capture, caplog, hierarchy):
        _, child = hierarchy
        logging.disable(logging.CRITICAL)
        try:
            with capture(logging.WARNING, child):
                child.warning("seen")
            assert logging.root.manager.disable == logging.CRITICAL
        finally:
            logging.disable(logging.NOTSET)
        assert [r.getMessage() for r in caplog.records] == ["seen"]

    def test_a_negative_assertion_inside_fails_when_the_code_logs(
        self, capture, caplog, hierarchy
    ):
        # Rule 153: the capture a "nothing logged" test uses must see records.
        _, child = hierarchy
        with capture(logging.WARNING, child):
            child.warning("probe")
        assert caplog.records != []


class TestWhatIsRestored:
    def test_every_change_is_undone_on_exit(self, capture, caplog, hierarchy):
        _, child = hierarchy
        child.setLevel(logging.ERROR)
        before = _state(child, caplog.handler)
        with capture(logging.DEBUG, child) as logger:
            assert logger is child
            assert logger.propagate is False
            assert caplog.handler in logger.handlers
            assert logger.level == logging.DEBUG == caplog.handler.level
        assert _state(child, caplog.handler) == before

    def test_every_change_is_undone_when_the_block_raises(
        self, capture, caplog, hierarchy
    ):
        _, child = hierarchy
        before = _state(child, caplog.handler)
        with pytest.raises(RuntimeError, match="inside"), capture(logging.INFO, child):
            raise RuntimeError("inside")
        assert _state(child, caplog.handler) == before

    def test_a_refused_call_changes_nothing(self, capture, caplog, hierarchy):
        _, child = hierarchy
        before = _state(child, caplog.handler)
        with pytest.raises(ValueError, match="unknown"), capture("LOUD", child):
            pass  # pragma: no cover - not reached
        assert _state(child, caplog.handler) == before


class TestArguments:
    def test_a_logger_may_be_given_by_its_name(self, capture, caplog, hierarchy):
        _, child = hierarchy
        with capture(logging.WARNING, child.name) as logger:
            child.warning("by name")
        assert logger is child
        assert [r.getMessage() for r in caplog.records] == ["by name"]

    @pytest.mark.parametrize("level", ["warning", "WARNING", "Warn", 30])
    def test_a_level_may_be_a_number_or_a_registered_name(
        self, capture, hierarchy, level
    ):
        _, child = hierarchy
        with capture(level, child) as logger:
            assert logger.level == logging.WARNING

    @pytest.mark.parametrize("level", [True, 30.0, None, [30]])
    def test_a_level_of_another_type_is_a_type_error(self, capture, hierarchy, level):
        _, child = hierarchy
        with pytest.raises(TypeError, match="level must be"), capture(level, child):
            pass  # pragma: no cover - not reached

    @pytest.mark.parametrize("level", ["LOUD", "", -1])
    def test_an_unknown_or_negative_level_is_a_value_error(
        self, capture, hierarchy, level
    ):
        _, child = hierarchy
        with pytest.raises(ValueError, match="level"), capture(level, child):
            pass  # pragma: no cover - not reached

    @pytest.mark.parametrize(
        "logger", [None, 1, logging.LoggerAdapter(logging.getLogger("x"), {})]
    )
    def test_a_logger_of_another_type_is_a_type_error(self, capture, logger):
        with pytest.raises(TypeError, match="logger must be"), capture(
            logging.WARNING, logger
        ):
            pass  # pragma: no cover - not reached

    @pytest.mark.parametrize("logger", ["", logging.getLogger()])
    def test_the_root_logger_is_refused(self, capture, logger):
        with pytest.raises(ValueError, match="root logger"), capture(
            logging.WARNING, logger
        ):
            pass  # pragma: no cover - not reached


def test_the_module_imports_no_pytest():
    """The module ships in scikit-plots-skinny, whose import check has no pytest."""
    import ast  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    tree = ast.parse(pathlib.Path(_capture.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "pytest" not in imported
    assert "_pytest" not in imported
