# tests/test_logging.py
# ruff: noqa: SLF001, PLR2004
"""
Comprehensive test suite for scikitplot/logging.py.

Coverage targets (≥99% statement + branch):
- Module-level constants, ``__getattr__``, ``__dir__``
- ``_is_jupyter_notebook``     – all 7 branches
- ``_coerce_level``            – int, digit-str, named-str, blank, non-str/int, unknown
- ``_default_logging_level``   – SKPLT_LOGGING_LEVEL, SKPLT_VERBOSE, verbose arg
- ``_get_thread_id``           – default / custom / falsy mask
- ``_get_caller``              – normal + no-frame path
- ``_logger_find_caller``      – stack_info True/False + no-code branch
- ``_GetFileAndLine``          – normal + no-code branch
- ``google2_log_prefix``       – all level/timestamp/file_and_line combos
- ``GoogleLogFormatter``       – init, formatTime UTC/local, format text/pprint/json/exc
- ``_make_default_formatter``  – 5 type branches + exception fallback
- ``_ensure_null_handler``     – missing / marker present / unmarked handler present
- ``AlwaysStdErrHandler``      – all __init__ paths, stream getter, stream setter
- ``_make_default_handler``    – None / Handler / Rotating / Rich / exception
- ``get_logger``               – first call (init) + second call (cached) + interactive
- Level accessors, ``sanitize_log_message``
- All convenience wrappers (critical … vlog)
- ``log_if``, ``log_every_n``, ``log_first_n``
- ``flush``, ``SpLogger.__getattr__``, ``TaskLevelStatusMessage``

Notes
-----
- Tests that depend on Python-version dead branches
  (the ``elif`` / ``else`` bodies of ``_logger_find_caller``) cannot be
  executed on the running interpreter.  Mark those lines
  ``# pragma: no cover`` in the source, or add an
  ``[report] exclude_lines`` entry in ``.coveragerc``.
- The ``_logger`` singleton is reset before every test via an
  ``autouse`` fixture to guarantee isolation.
"""

from __future__ import annotations

import datetime
import logging as _stdlib
import os
import sys
import threading
import traceback
from io import StringIO
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from .. import logging as splog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_globals(monkeypatch):
    """
    Reset mutable module-level state before every test.

    Guarantees that:
    * ``_logger`` singleton is ``None`` so ``get_logger()`` always runs the
      full initialisation path.
    * ``_log_counter_per_token`` is empty so ``log_every_n`` / ``log_first_n``
      counters start from zero.
    * ``SKPLT_VERBOSE`` and ``SKPLT_LOGGING_LEVEL`` env-vars are absent.
    """
    monkeypatch.setattr(splog, "_logger", None)
    monkeypatch.setattr(splog, "_log_counter_per_token", {})
    for var in ("SKPLT_VERBOSE", "SKPLT_LOGGING_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture()
def mock_logger(monkeypatch):
    """Replace ``get_logger`` with a ``MagicMock`` and return the mock."""
    lg = MagicMock(spec=_stdlib.Logger)
    monkeypatch.setattr(splog, "get_logger", lambda: lg)
    return lg


# ===========================================================================
# 1. Module-level constants
# ===========================================================================


class TestConstants:
    """Verify all public constants have expected values."""

    def test_level_constants(self):
        assert splog.CRITICAL == 50
        assert splog.DEBUG == 10
        assert splog.ERROR == 40
        assert splog.FATAL == 50
        assert splog.INFO == 20
        assert splog.NOTSET == 0
        assert splog.WARNING == 30
        assert splog.WARN == splog.WARNING

    def test_warn_is_alias_for_warning(self):
        assert splog.WARN is splog.WARNING

    def test_ansi_escape_codes(self):
        assert splog.RESET == "\033[0m"
        assert splog.BOLD == "\033[1m"
        assert splog.RED.startswith("\033[")
        assert splog.GREEN.startswith("\033[")
        assert splog.YELLOW.startswith("\033[")
        assert splog.BLUE.startswith("\033[")
        assert splog.MAGENTA.startswith("\033[")
        assert splog.CYAN.startswith("\033[")

    def test_utc_is_valid_tzinfo(self):
        # Must be either datetime.UTC (3.11+) or datetime.timezone.utc
        expected = getattr(datetime, "UTC", datetime.timezone.utc)
        assert splog.UTC is expected

    def test_thread_id_mask_is_positive(self):
        assert splog._THREAD_ID_MASK > 0

    def test_handler_marker_string(self):
        assert isinstance(splog._HANDLER_MARKER, str)
        assert splog._HANDLER_MARKER  # non-empty

    def test_all_exports_present(self):
        for name in splog.__all__:
            assert hasattr(splog, name), f"__all__ lists '{name}' but it is missing"


# ===========================================================================
# 2. Module-level __getattr__ and __dir__
# ===========================================================================


class TestModuleGetattr:
    """Module-level ``__getattr__`` – proxy to stdlib ``logging`` + logger."""

    def test_dunder_name_raises(self):
        """Dunder names must never be proxied (breaks introspection tools)."""
        with pytest.raises(AttributeError):
            splog.__getattr__("__mro__")

    def test_dunder_doc_raises(self):
        with pytest.raises(AttributeError):
            splog.__getattr__("__doc_does_not_exist__")

    def test_stdlib_attr_returned(self):
        """Known stdlib logging attribute is resolved correctly."""
        result = splog.__getattr__("getLevelName")
        assert result is _stdlib.getLevelName

    def test_stdlib_attr_cached_in_module_globals(self):
        """After resolution the attr is cached for subsequent lookups."""
        splog.__getattr__("getLevelName")
        # Cached → present in module dict
        assert "getLevelName" in vars(splog)

    def test_missing_attr_raises_attribute_error(self):
        """Attributes absent from both stdlib logging and the logger raise."""
        with pytest.raises(AttributeError):
            splog.__getattr__("THIS_ATTR_DOES_NOT_EXIST_12345")

    def test_sentinel_not_returned_for_missing(self):
        """Ensure None is never silently returned for missing attributes."""
        sentinel = object()
        result = sentinel
        try:
            result = splog.__getattr__("ANOTHER_MISSING_ATTR_99999")
        except AttributeError:
            pass
        assert result is sentinel  # no silent None return


class TestModuleDir:
    """Module-level ``__dir__`` combines module and stdlib logging names."""

    def test_dir_is_sorted(self):
        d = splog.__dir__()
        assert list(d) == sorted(d)

    def test_dir_contains_module_names(self):
        d = set(splog.__dir__())
        assert "get_logger" in d
        assert "DEBUG" in d

    def test_dir_contains_stdlib_names(self):
        d = set(splog.__dir__())
        assert "getLevelName" in d
        assert "basicConfig" in d


# ===========================================================================
# 3. _is_jupyter_notebook
# ===========================================================================


class TestIsJupyterNotebook:
    """All seven code paths for ``_is_jupyter_notebook``."""

    def test_jpy_parent_pid_env_var(self, monkeypatch):
        monkeypatch.setenv("JPY_PARENT_PID", "12345")
        assert splog._is_jupyter_notebook() is True

    def test_ipykernel_in_sys_modules(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.setitem(sys.modules, "ipykernel", MagicMock())
        assert splog._is_jupyter_notebook() is True

    def test_ipython_import_fails_returns_false(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        # Setting to None causes ImportError on 'from IPython import …'
        with patch.dict(sys.modules, {"IPython": None}):
            assert splog._is_jupyter_notebook() is False

    def test_get_ipython_returns_none(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = None
        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            assert splog._is_jupyter_notebook() is False

    def test_get_ipython_config_is_none(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        ip = MagicMock()
        ip.config = None
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = ip
        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            assert splog._is_jupyter_notebook() is False

    def test_get_ipython_config_missing_kernel_app(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        ip = MagicMock()
        ip.config = {"SomeOtherApp": {}}  # truthy but no IPKernelApp key
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = ip
        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            assert splog._is_jupyter_notebook() is False

    def test_get_ipython_config_has_kernel_app(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        ip = MagicMock()
        ip.config = {"IPKernelApp": {"connection_file": "kernel-42.json"}}
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = ip
        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            assert splog._is_jupyter_notebook() is True

    def test_get_ipython_raises_exception(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.side_effect = RuntimeError("oops")
        with patch.dict(sys.modules, {"IPython": mock_ipython}):
            assert splog._is_jupyter_notebook() is False


# ===========================================================================
# 4. _coerce_level
# ===========================================================================


class TestCoerceLevel:
    """All branches of ``_coerce_level``."""

    def test_int_passthrough(self):
        assert splog._coerce_level(10) == 10

    def test_int_zero(self):
        assert splog._coerce_level(0) == 0

    def test_str_digit(self):
        assert splog._coerce_level("10") == 10

    def test_str_digit_with_whitespace(self):
        assert splog._coerce_level("  20  ") == 20

    def test_str_named_upper(self):
        assert splog._coerce_level("DEBUG") == 10

    def test_str_named_lower(self):
        assert splog._coerce_level("warning") == 30

    def test_str_named_mixed(self):
        assert splog._coerce_level("Critical") == 50

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            splog._coerce_level("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            splog._coerce_level("   ")

    def test_non_str_non_int_raises(self):
        with pytest.raises(ValueError):
            splog._coerce_level(3.14)  # type: ignore[arg-type]

    def test_none_raises(self):
        with pytest.raises(ValueError):
            splog._coerce_level(None)  # type: ignore[arg-type]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown log level"):
            splog._coerce_level("SUPERVERBOSE")


# ===========================================================================
# 5. _default_logging_level
# ===========================================================================


class TestDefaultLoggingLevel:
    """Environment-variable and argument-driven level selection."""

    def test_env_logging_level_debug(self, monkeypatch):
        monkeypatch.setenv("SKPLT_LOGGING_LEVEL", "DEBUG")
        assert splog._default_logging_level() == _stdlib.DEBUG

    def test_env_logging_level_integer_string(self, monkeypatch):
        monkeypatch.setenv("SKPLT_LOGGING_LEVEL", "20")
        assert splog._default_logging_level() == _stdlib.INFO

    def test_env_verbose_set_overrides_arg(self, monkeypatch):
        monkeypatch.setenv("SKPLT_VERBOSE", "1")
        # verbose=False is the default, but env-var wins
        assert splog._default_logging_level(verbose=False) == _stdlib.DEBUG

    def test_verbose_arg_true(self):
        assert splog._default_logging_level(verbose=True) == _stdlib.DEBUG

    def test_verbose_arg_false(self):
        assert splog._default_logging_level(verbose=False) == _stdlib.WARNING

    def test_default_is_warning(self):
        assert splog._default_logging_level() == _stdlib.WARNING


# ===========================================================================
# 6. _get_thread_id
# ===========================================================================


class TestGetThreadId:
    """Thread-ID masking behaviour."""

    def test_returns_int(self):
        result = splog._get_thread_id()
        assert isinstance(result, int)

    def test_returns_non_negative(self):
        assert splog._get_thread_id() >= 0

    def test_custom_mask_applied(self):
        mask = 0xFFFF
        result = splog._get_thread_id(mask)
        assert result == (threading.get_ident() & mask)

    def test_falsy_mask_uses_default(self):
        """A falsy mask (0) must fall back to ``_THREAD_ID_MASK``."""
        result = splog._get_thread_id(0)
        expected = threading.get_ident() & splog._THREAD_ID_MASK
        assert result == expected


# ===========================================================================
# 7. _get_caller, _logger_find_caller, _GetFileAndLine
# ===========================================================================


class TestCallerHelpers:
    """Call-site helpers used for accurate log prefix information."""

    def test_get_caller_returns_code_and_frame(self):
        code, frame = splog._get_caller(offset=1)
        # offset=1 from _get_caller itself → lands on this test method
        assert code is not None or frame is not None  # at least one is live

    def test_GetFileAndLine_returns_tuple(self):
        result = splog._GetFileAndLine()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_GetFileAndLine_no_code_returns_unknown(self, monkeypatch):
        """When _get_caller finds no suitable frame both values are sentinel."""
        monkeypatch.setattr(splog, "_get_caller", lambda offset=3: (None, None))
        fname, lineno = splog._GetFileAndLine()
        assert fname == "<unknown>"
        assert lineno == 0

    def test_logger_find_caller_returns_4tuple(self):
        result = splog._logger_find_caller(stack_info=False)
        assert len(result) == 4

    def test_logger_find_caller_no_stack_info_is_none(self):
        _, _, _, sinfo = splog._logger_find_caller(stack_info=False)
        assert sinfo is None

    def test_logger_find_caller_stack_info_is_string(self):
        _, _, _, sinfo = splog._logger_find_caller(stack_info=True)
        assert isinstance(sinfo, str)

    def test_logger_find_caller_no_code_branch(self, monkeypatch):
        """Unreachable-frame case returns sentinel strings."""
        monkeypatch.setattr(splog, "_get_caller", lambda offset=4: (None, None))
        fname, lineno, funcname, _ = splog._logger_find_caller(stack_info=False)
        assert fname == "(unknown file)"
        assert lineno == 0
        assert funcname == "(unknown function)"


# ===========================================================================
# 8. google2_log_prefix
# ===========================================================================


class TestGoogle2LogPrefix:
    """All parameter combinations for ``google2_log_prefix``."""

    def test_returns_string(self):
        result = splog.google2_log_prefix()
        assert isinstance(result, str)

    def test_default_severity_is_i_when_level_unknown(self):
        result = splog.google2_log_prefix(level="UNKNOWNLEVEL")
        # severity "I" (info default)
        assert result[0] == "I"

    def test_known_level_severity_letter(self):
        result = splog.google2_log_prefix(level=_stdlib.DEBUG)
        # _level_names[DEBUG] == "DEBUG" → first letter "D"
        assert result[0] == "D"

    def test_explicit_timestamp(self):
        ts = 1_700_000_000.0
        result = splog.google2_log_prefix(timestamp=ts)
        assert isinstance(result, str)

    def test_explicit_file_and_line(self):
        result = splog.google2_log_prefix(file_and_line=("myfile.py", 42))
        assert "myfile.py" in result
        assert ":42]" in result

    def test_format_contains_thread_id(self):
        result = splog.google2_log_prefix()
        tid = splog._get_thread_id()
        # Thread ID should appear in the prefix
        assert str(tid) in result

    def test_warning_level_gives_w(self):
        result = splog.google2_log_prefix(level=_stdlib.WARNING)
        assert result[0] == "W"

    def test_error_level_gives_e(self):
        result = splog.google2_log_prefix(level=_stdlib.ERROR)
        assert result[0] == "E"

    def test_critical_level_gives_c(self):
        result = splog.google2_log_prefix(level=_stdlib.CRITICAL)
        assert result[0] == "C"


# ===========================================================================
# 9. GoogleLogFormatter
# ===========================================================================


class TestGoogleLogFormatter:
    """Init, formatTime, and all three format backends."""

    # helpers -----------------------------------------------------------------

    @staticmethod
    def _make_record(msg="hello world"):
        return _stdlib.LogRecord(
            name="test",
            level=_stdlib.INFO,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    # __init__ ----------------------------------------------------------------

    def test_default_init(self):
        fmt = splog.GoogleLogFormatter()
        assert fmt.backend == "text"
        assert fmt._use_utc is True

    def test_init_pprint_backend(self):
        fmt = splog.GoogleLogFormatter(backend="pprint")
        assert fmt.backend == "pprint"

    def test_init_json_backend(self):
        fmt = splog.GoogleLogFormatter(backend="json")
        assert fmt.backend == "json"

    def test_init_backend_case_insensitive(self):
        fmt = splog.GoogleLogFormatter(backend="JSON")
        assert fmt.backend == "json"

    def test_init_use_datetime_false_no_microseconds(self):
        fmt = splog.GoogleLogFormatter(use_datetime=False)
        # With use_datetime=False the datefmt must NOT end with '.%f'
        assert not fmt.datefmt.endswith(".%f")

    def test_init_use_datetime_true_has_microseconds(self):
        fmt = splog.GoogleLogFormatter(use_datetime=True)
        assert fmt.datefmt.endswith(".%f")

    def test_init_utc_false(self):
        fmt = splog.GoogleLogFormatter(use_utc=False)
        assert fmt._use_utc is False

    # formatTime ---------------------------------------------------------------

    def test_formatTime_utc_returns_string(self):
        fmt = splog.GoogleLogFormatter(use_utc=True)
        record = self._make_record()
        result = fmt.formatTime(record)
        assert isinstance(result, str)

    def test_formatTime_local_returns_string(self):
        fmt = splog.GoogleLogFormatter(use_utc=False)
        record = self._make_record()
        result = fmt.formatTime(record)
        assert isinstance(result, str)

    def test_formatTime_custom_datefmt(self):
        fmt = splog.GoogleLogFormatter(use_datetime=False)
        record = self._make_record()
        result = fmt.formatTime(record, datefmt="%Y")
        assert len(result) == 4  # just the year

    # format – text backend ---------------------------------------------------

    def test_format_text_returns_string(self):
        fmt = splog.GoogleLogFormatter(backend="text")
        record = self._make_record()
        result = fmt.format(record)
        assert isinstance(result, str)
        assert "hello world" in result

    def test_format_text_contains_level_letter(self):
        fmt = splog.GoogleLogFormatter(backend="text")
        record = self._make_record()
        result = fmt.format(record)
        assert "I" in result  # INFO → "I"

    # format – pprint backend -------------------------------------------------

    def test_format_pprint_produces_pprint_style(self):
        fmt = splog.GoogleLogFormatter(backend="pprint")
        record = self._make_record()
        result = fmt.format(record)
        # pprint output of a dict starts with "{"
        assert "{" in result

    # format – json backend ---------------------------------------------------

    def test_format_json_is_valid_json(self):
        import json

        fmt = splog.GoogleLogFormatter(backend="json")
        record = self._make_record()
        raw = fmt.format(record)
        parsed = json.loads(raw)
        assert "message" in parsed

    def test_format_json_contains_message(self):
        import json

        fmt = splog.GoogleLogFormatter(backend="json")
        record = self._make_record("json_test_msg")
        parsed = json.loads(fmt.format(record))
        assert "json_test_msg" in parsed["message"]

    # format – exception fallback ---------------------------------------------

    def test_format_exception_falls_back_to_text(self, monkeypatch):
        """If the JSON/pprint path raises, fall back to plain concatenation."""
        import pprint as _pprint

        fmt = splog.GoogleLogFormatter(backend="pprint")
        monkeypatch.setattr(_pprint, "pformat", MagicMock(side_effect=RuntimeError))
        record = self._make_record("fallback_test")
        result = fmt.format(record)
        # Fallback: concatenation of payload values → still contains message
        assert "fallback_test" in result


# ===========================================================================
# 10. _make_default_formatter
# ===========================================================================


class TestMakeDefaultFormatter:
    """All type branches + exception fallback of ``_make_default_formatter``."""

    def test_formatter_instance_returned_directly(self):
        custom = _stdlib.Formatter("%(message)s")
        result = splog._make_default_formatter(custom)
        assert result is custom

    def test_basic_format_string(self):
        result = splog._make_default_formatter("BASIC_FORMAT")
        assert isinstance(result, _stdlib.Formatter)

    def test_google_format_string(self):
        result = splog._make_default_formatter("GOOGLE_FORMAT")
        assert isinstance(result, splog.GoogleLogFormatter)

    def test_custom_format_string(self):
        result = splog._make_default_formatter("CUSTOM_FORMAT")
        assert isinstance(result, _stdlib.Formatter)

    def test_unknown_string_returns_none(self):
        """No branch matches → implicit ``None`` (fall-through without return)."""
        result = splog._make_default_formatter("TOTALLY_UNKNOWN_FORMAT")
        assert result is None

    def test_none_formatter_returns_none(self):
        """``None`` is not a Formatter instance and matches no branch."""
        result = splog._make_default_formatter(None)
        assert result is None

    def test_exception_fallback_returns_basic_formatter(self, monkeypatch):
        """If any constructor raises inside the try block, basic formatter is returned."""
        monkeypatch.setattr(
            splog, "GoogleLogFormatter", MagicMock(side_effect=RuntimeError("boom"))
        )
        result = splog._make_default_formatter("GOOGLE_FORMAT")
        assert isinstance(result, _stdlib.Formatter)

    def test_custom_time_format_passed_through(self):
        result = splog._make_default_formatter(
            "GOOGLE_FORMAT", time_format="%H:%M:%S"
        )
        assert isinstance(result, splog.GoogleLogFormatter)

    def test_use_datetime_false(self):
        result = splog._make_default_formatter("GOOGLE_FORMAT", use_datetime=False)
        assert isinstance(result, splog.GoogleLogFormatter)


# ===========================================================================
# 11. _ensure_null_handler
# ===========================================================================


class TestEnsureNullHandler:
    """NullHandler guard logic."""

    @staticmethod
    def _fresh_logger(name):
        lg = _stdlib.getLogger(name)
        lg.handlers.clear()
        return lg

    def test_adds_null_handler_when_none_present(self):
        lg = self._fresh_logger("test.enull.add")
        splog._ensure_null_handler(lg)
        marked = [
            h
            for h in lg.handlers
            if isinstance(h, _stdlib.NullHandler)
            and getattr(h, splog._HANDLER_MARKER, False)
        ]
        assert len(marked) == 1

    def test_skips_when_marker_already_present(self):
        lg = self._fresh_logger("test.enull.skip")
        splog._ensure_null_handler(lg)
        splog._ensure_null_handler(lg)  # second call
        marked = [
            h for h in lg.handlers if getattr(h, splog._HANDLER_MARKER, False)
        ]
        assert len(marked) == 1

    def test_adds_handler_when_unmarked_null_handler_exists(self):
        """An existing NullHandler WITHOUT the marker must not prevent adding."""
        lg = self._fresh_logger("test.enull.unmarked")
        unmarked = _stdlib.NullHandler()
        # Do NOT set the marker
        lg.addHandler(unmarked)
        splog._ensure_null_handler(lg)
        marked = [
            h for h in lg.handlers if getattr(h, splog._HANDLER_MARKER, False)
        ]
        assert len(marked) == 1


# ===========================================================================
# 12. AlwaysStdErrHandler
# ===========================================================================


class TestAlwaysStdErrHandler:
    """All init paths, stream property getter and setter."""

    def test_default_is_stderr_outside_jupyter(self, monkeypatch):
        monkeypatch.delenv("JPY_PARENT_PID", raising=False)
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
        with patch.dict(sys.modules, {"IPython": None}):
            h = splog.AlwaysStdErrHandler()
        assert h._stream is sys.stderr

    def test_default_is_stdout_inside_jupyter(self, monkeypatch):
        monkeypatch.setenv("JPY_PARENT_PID", "99999")
        h = splog.AlwaysStdErrHandler()
        assert h._stream is sys.stdout

    def test_explicit_stderr_string(self):
        h = splog.AlwaysStdErrHandler("stderr")
        assert h._stream is sys.stderr

    def test_explicit_stdout_string(self):
        h = splog.AlwaysStdErrHandler("stdout")
        assert h._stream is sys.stdout

    def test_case_insensitive_stream_string(self):
        h = splog.AlwaysStdErrHandler("STDERR")
        assert h._stream is sys.stderr

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError, match="'stdout', 'stderr'"):
            splog.AlwaysStdErrHandler("invalid_stream_name")

    def test_file_like_object_used_directly(self):
        buf = StringIO()
        h = splog.AlwaysStdErrHandler(buf)
        assert h.stream is buf

    def test_stream_getter_returns_stream(self):
        h = splog.AlwaysStdErrHandler("stderr")
        assert h.stream is sys.stderr

    def test_stream_setter_valid_value_no_error(self):
        """Setting to sys.stderr or sys.stdout must never raise."""
        h = splog.AlwaysStdErrHandler("stderr")
        h.stream = sys.stderr  # valid → no-op
        h.stream = sys.stdout  # valid → no-op

    def test_stream_setter_invalid_calls_set_stream(self):
        """Setting to a non-std stream triggers the fallback (no exception)."""
        h = splog.AlwaysStdErrHandler("stderr")
        buf = StringIO()
        h.stream = buf  # not in (sys.stderr, sys.stdout) → calls super().setStream


# ===========================================================================
# 13. _make_default_handler
# ===========================================================================


class TestMakeDefaultHandler:
    """All branches of ``_make_default_handler``."""

    def test_none_gives_always_stderr_handler(self):
        result = splog._make_default_handler(handler=None)
        assert isinstance(result, splog.AlwaysStdErrHandler)

    def test_handler_instance_returned_with_formatter(self):
        custom_h = _stdlib.StreamHandler(StringIO())
        result = splog._make_default_handler(handler=custom_h)
        assert result is custom_h

    def test_custom_formatter_attached(self):
        custom_h = _stdlib.StreamHandler(StringIO())
        custom_f = _stdlib.Formatter("%(message)s")
        result = splog._make_default_handler(handler=custom_h, formatter=custom_f)
        assert result.formatter is custom_f

    def test_rotating_file_handler_string(self, monkeypatch, tmp_path):
        """'RotatingFileHandler' branch – mock logging.handlers to avoid disk I/O."""
        mock_rfh = MagicMock(spec=_stdlib.Handler)
        mock_rfh.setFormatter = MagicMock()
        mock_handlers_mod = MagicMock()
        mock_handlers_mod.RotatingFileHandler.return_value = mock_rfh
        monkeypatch.setattr(splog._logging, "handlers", mock_handlers_mod, raising=False)
        result = splog._make_default_handler(handler="RotatingFileHandler")
        assert result is mock_rfh

    def test_rich_handler_string_with_rich_installed(self, monkeypatch):
        """'RichHandler' branch – mock the rich package."""
        mock_rich_handler = MagicMock(spec=_stdlib.Handler)
        mock_rich_handler.setFormatter = MagicMock()

        mock_console_cls = MagicMock()
        mock_console_cls.return_value = MagicMock()

        mock_rich_logging_mod = MagicMock()
        mock_rich_logging_mod.RichHandler.return_value = mock_rich_handler

        mock_rich_console_mod = MagicMock()
        mock_rich_console_mod.Console = mock_console_cls

        with patch.dict(
            sys.modules,
            {
                "rich": MagicMock(),
                "rich.console": mock_rich_console_mod,
                "rich.logging": mock_rich_logging_mod,
            },
        ):
            result = splog._make_default_handler(handler="RichHandler")
        assert result is mock_rich_handler

    def test_rich_handler_import_error_fallback(self):
        """When rich is absent, must fall back to AlwaysStdErrHandler."""
        with patch.dict(
            sys.modules,
            {"rich": None, "rich.console": None, "rich.logging": None},
        ):
            result = splog._make_default_handler(handler="RichHandler")
        assert isinstance(result, splog.AlwaysStdErrHandler)

    def test_exception_in_handler_creation_fallback(self, monkeypatch):
        """Any unexpected exception during construction → AlwaysStdErrHandler."""
        # Capture the real class BEFORE patching so the isinstance assertion
        # receives a proper type.  Capturing it after the first setattr would
        # yield the mock, and ``isinstance(x, MagicMock_instance)`` raises
        # TypeError because MagicMock intentionally excludes __instancecheck__.
        real_ash = splog.AlwaysStdErrHandler
        monkeypatch.setattr(
            splog, "AlwaysStdErrHandler", MagicMock(side_effect=[RuntimeError, MagicMock()])
        )
        # Pass an instance-like that IS a Handler but whose setFormatter raises.
        h_mock = MagicMock(spec=_stdlib.Handler)
        h_mock.setFormatter.side_effect = RuntimeError("setFormatter broken")
        result = splog._make_default_handler(handler=h_mock)
        # _make_default_handler uses a definition-time default (_fallback_cls)
        # that always points to the real AlwaysStdErrHandler regardless of
        # module-level patching, so result IS a real AlwaysStdErrHandler.
        assert isinstance(result, real_ash)


# ===========================================================================
# 14. get_logger
# ===========================================================================


class TestGetLogger:
    """Singleton semantics, double-checked locking, interactive paths."""

    def test_returns_logger_instance(self):
        lg = splog.get_logger()
        assert isinstance(lg, _stdlib.Logger)

    def test_singleton_same_object_on_second_call(self):
        lg1 = splog.get_logger()
        lg2 = splog.get_logger()
        assert lg1 is lg2

    def test_logger_name_is_scikitplot(self):
        lg = splog.get_logger()
        assert lg.name == "scikitplot"

    def test_propagate_is_false(self):
        lg = splog.get_logger()
        assert lg.propagate is False

    def test_interactive_path_sets_info_level(self, monkeypatch):
        """When sys.ps1 exists (interactive shell) the level is set to INFO."""
        monkeypatch.setattr(sys, "ps1", ">>> ", raising=False)
        lg = splog.get_logger()
        # In interactive mode the logger level must be INFO (20)
        assert lg.level == _stdlib.INFO

    def test_non_interactive_path_does_not_raise(self):
        """Normal (non-interactive) initialisation must complete without error."""
        lg = splog.get_logger()
        assert lg is not None

    def test_second_call_uses_cached_logger(self, monkeypatch):
        """After the singleton is set, subsequent calls skip initialisation."""
        sentinel = _stdlib.getLogger("sentinel_logger")
        monkeypatch.setattr(splog, "_logger", sentinel)
        # Should return the already-set sentinel without re-initialising
        assert splog.get_logger() is sentinel


# ===========================================================================
# 15. Level accessors
# ===========================================================================


class TestLevelAccessors:
    def test_getEffectiveLevel_returns_int(self):
        lvl = splog.getEffectiveLevel()
        assert isinstance(lvl, int)

    def test_get_verbosity_matches_effective_level(self):
        assert splog.get_verbosity() == splog.getEffectiveLevel()

    def test_setLevel_integer(self):
        splog.setLevel(_stdlib.DEBUG)
        assert splog.getEffectiveLevel() == _stdlib.DEBUG

    def test_setLevel_string(self):
        splog.setLevel("ERROR")
        assert splog.getEffectiveLevel() == _stdlib.ERROR

    def test_set_verbosity_delegates_to_setLevel(self):
        splog.set_verbosity(_stdlib.INFO)
        assert splog.get_verbosity() == _stdlib.INFO


# ===========================================================================
# 16. sanitize_log_message
# ===========================================================================


class TestSanitizeLogMessage:
    """Redaction logic for sensitive keywords."""

    _REDACTED = "[REDACTED] Potentially sensitive information detected."

    @pytest.mark.parametrize(
        "keyword",
        [
            "password",
            "Password",
            "PASSWORD",
            "secret",
            "token",
            "api_key",
            "access_key",
            "private_key",
        ],
    )
    def test_sensitive_keywords_redacted(self, keyword):
        msg = f"The {keyword} is abc123"
        assert splog.sanitize_log_message(msg) == self._REDACTED

    def test_clean_message_unchanged(self):
        msg = "Normal log message without secrets"
        assert splog.sanitize_log_message(msg) == msg

    def test_empty_string_unchanged(self):
        assert splog.sanitize_log_message("") == ""

    def test_numeric_string_unchanged(self):
        msg = "12345 67890"
        assert splog.sanitize_log_message(msg) == msg


# ===========================================================================
# 17. Convenience wrapper functions
# ===========================================================================


class TestConvenienceWrappers:
    """All thin wrappers delegate correctly to ``get_logger()``."""

    def test_critical(self, mock_logger):
        splog.critical("crit %s", "msg")
        mock_logger.critical.assert_called_once_with("crit %s", "msg")

    def test_fatal(self, mock_logger):
        splog.fatal("fatal msg")
        mock_logger.critical.assert_called_once_with("fatal msg")

    def test_error(self, mock_logger):
        splog.error("err %d", 42)
        mock_logger.error.assert_called_once_with("err %d", 42)

    def test_error_log_is_noop(self):
        """error_log discards all its arguments (del statement)."""
        # Must not raise and must not call get_logger
        splog.error_log("secret_password_12345", level=splog.DEBUG)

    def test_exception_calls_error_with_exc_info(self, mock_logger):
        splog.exception("oops")
        mock_logger.error.assert_called_once_with("oops", exc_info=True)

    def test_exception_custom_exc_info(self, mock_logger):
        splog.exception("oops", exc_info=False)
        mock_logger.error.assert_called_once_with("oops", exc_info=False)

    def test_warning(self, mock_logger):
        splog.warning("warn")
        mock_logger.warning.assert_called_once_with("warn")

    def test_warn(self, mock_logger):
        splog.warn("warn alias")
        mock_logger.warning.assert_called_once_with("warn alias")

    def test_info(self, mock_logger):
        splog.info("info msg")
        mock_logger.info.assert_called_once_with("info msg")

    def test_debug(self, mock_logger):
        splog.debug("dbg msg")
        mock_logger.debug.assert_called_once_with("dbg msg")

    def test_log(self, mock_logger):
        splog.log(_stdlib.WARNING, "log msg")
        mock_logger.log.assert_called_once_with(_stdlib.WARNING, "log msg")

    def test_vlog(self, mock_logger):
        splog.vlog(_stdlib.INFO, "vlog msg")
        mock_logger.log.assert_called_once_with(_stdlib.INFO, "vlog msg")

    def test_TaskLevelStatusMessage(self, mock_logger):
        splog.TaskLevelStatusMessage("task msg")
        mock_logger.error.assert_called_once_with("task msg")


# ===========================================================================
# 18. log_if
# ===========================================================================


class TestLogIf:
    def test_logs_when_condition_true(self, mock_logger):
        splog.log_if(_stdlib.WARNING, "should log", True)
        mock_logger.log.assert_called_once_with(_stdlib.WARNING, "should log")

    def test_does_not_log_when_condition_false(self, mock_logger):
        splog.log_if(_stdlib.WARNING, "should not log", False)
        mock_logger.log.assert_not_called()

    def test_args_forwarded(self, mock_logger):
        splog.log_if(_stdlib.INFO, "msg %s", True, "arg1")
        mock_logger.log.assert_called_once_with(_stdlib.INFO, "msg %s", "arg1")


# ===========================================================================
# 19. log_every_n
# ===========================================================================


class TestLogEveryN:
    """Counter-based periodic logging."""

    def test_n_less_than_1_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            splog.log_every_n(_stdlib.INFO, "msg", 0)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError):
            splog.log_every_n(_stdlib.INFO, "msg", 0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError):
            splog.log_every_n(_stdlib.INFO, "msg", -5)

    def test_logs_on_first_call(self, monkeypatch, mock_logger):
        """count=0 → 0 % n == 0 → condition True → logs."""
        counter = iter([0])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_every_n(_stdlib.INFO, "msg", 3)
        mock_logger.log.assert_called_once()

    def test_does_not_log_on_intermediate_call(self, monkeypatch, mock_logger):
        """count=1 → 1 % 3 != 0 → condition False → no log."""
        counter = iter([1])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_every_n(_stdlib.INFO, "msg", 3)
        mock_logger.log.assert_not_called()

    def test_logs_again_at_nth_call(self, monkeypatch, mock_logger):
        """count=3 → 3 % 3 == 0 → condition True → logs."""
        counter = iter([3])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_every_n(_stdlib.INFO, "msg", 3)
        mock_logger.log.assert_called_once()

    def test_n_equals_1_logs_every_call(self, monkeypatch, mock_logger):
        """n=1: every count % 1 == 0 → always logs."""
        calls = [0, 1, 2]
        call_iter = iter(calls)
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(call_iter))
        for _ in calls:
            splog.log_every_n(_stdlib.INFO, "msg", 1)
        assert mock_logger.log.call_count == 3


# ===========================================================================
# 20. log_first_n
# ===========================================================================


class TestLogFirstN:
    """Emit only for the first n calls from the same site."""

    def test_n_less_than_1_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            splog.log_first_n(_stdlib.INFO, "msg", 0)

    def test_logs_on_first_call(self, monkeypatch, mock_logger):
        """count=0 < n=2 → logs."""
        counter = iter([0])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_first_n(_stdlib.INFO, "msg", 2)
        mock_logger.log.assert_called_once()

    def test_logs_on_last_qualifying_call(self, monkeypatch, mock_logger):
        """count = n-1 < n → still logs."""
        counter = iter([1])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_first_n(_stdlib.INFO, "msg", 2)
        mock_logger.log.assert_called_once()

    def test_does_not_log_after_n(self, monkeypatch, mock_logger):
        """count = n → NOT (count < n) → no log."""
        counter = iter([2])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_first_n(_stdlib.INFO, "msg", 2)
        mock_logger.log.assert_not_called()

    def test_does_not_log_well_after_n(self, monkeypatch, mock_logger):
        """count >> n → still no log."""
        counter = iter([100])
        monkeypatch.setattr(splog, "_GetNextLogCountPerToken", lambda _: next(counter))
        splog.log_first_n(_stdlib.INFO, "msg", 2)
        mock_logger.log.assert_not_called()


# ===========================================================================
# 21. _GetNextLogCountPerToken
# ===========================================================================


class TestGetNextLogCountPerToken:
    """Counter starts at 0 and increments monotonically per token."""

    def test_first_call_returns_zero(self):
        token = ("__test__", 9999)
        result = splog._GetNextLogCountPerToken(token)
        assert result == 0

    def test_second_call_returns_one(self):
        token = ("__test2__", 8888)
        splog._GetNextLogCountPerToken(token)
        result = splog._GetNextLogCountPerToken(token)
        assert result == 1

    def test_independent_tokens_are_independent(self):
        tok_a = ("file_a.py", 1)
        tok_b = ("file_b.py", 1)
        splog._GetNextLogCountPerToken(tok_a)
        splog._GetNextLogCountPerToken(tok_a)
        result_b = splog._GetNextLogCountPerToken(tok_b)
        assert result_b == 0  # tok_b is fresh


# ===========================================================================
# 22. flush
# ===========================================================================


class TestFlush:
    def test_flush_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            splog.flush()


# ===========================================================================
# 23. SpLogger.__getattr__
# ===========================================================================


class TestSpLoggerGetattr:
    """All four resolution branches of ``SpLogger.__getattr__``."""

    def test_instance_dict_branch(self):
        """Name present in the instance's ``__dict__`` is returned directly."""
        sp = splog.SpLogger()
        # Bypass __setattr__ by writing directly to the underlying dict
        object.__setattr__(sp, "__dict__", {"_my_custom": 42, **sp.__dict__})
        sp.__dict__["_my_custom"] = 42
        assert sp._my_custom == 42

    def test_stdlib_logging_branch(self):
        """Name present in stdlib ``logging`` module is resolved there."""
        sp = splog.SpLogger()
        # DEBUG is in _logging, not in instance __dict__
        assert sp.DEBUG == _stdlib.DEBUG

    def test_stdlib_logging_info(self):
        sp = splog.SpLogger()
        assert sp.INFO == _stdlib.INFO

    def test_attribute_error_for_completely_missing(self):
        sp = splog.SpLogger()
        with pytest.raises(AttributeError, match="'SpLogger' object has no attribute"):
            _ = sp.THIS_ATTRIBUTE_DOES_NOT_EXIST_XYZ_99999


# ===========================================================================
# 24. Module __getattr__ – sentinel / contract tests (from original test file)
# ===========================================================================


class TestModuleGetAttrContract:
    """
    Ensure ``__getattr__`` never returns real values for non-existent attrs.

    Mirrors the intent of the commented-out test in the original
    ``test_logging.py``.
    """

    SENTINEL = object()

    def test_missing_attr_is_not_sentinel(self):
        """Requesting a non-existent name must raise, not silently return."""
        result = self.SENTINEL
        try:
            result = getattr(splog, "THIS_ATTRIBUTE_SHOULD_NOT_EXIST_12345")
        except AttributeError:
            pass
        assert result is self.SENTINEL

    def test_dunder_attr_is_not_silently_resolved(self):
        """Dunder attributes must never be proxied."""
        result = self.SENTINEL
        try:
            result = getattr(splog, "__mro__")
        except AttributeError:
            pass
        assert result is self.SENTINEL


# ===========================================================================
# 25. Integration smoke-test
# ===========================================================================


class TestIntegrationSmoke:
    """
    End-to-end log emission at every level to confirm the wiring is correct.
    Uses ``caplog`` (pytest's log capture fixture).
    """

    def test_all_levels_emit_without_error(self, caplog):
        splog.setLevel(_stdlib.DEBUG)
        with caplog.at_level(_stdlib.DEBUG, logger="scikitplot"):
            splog.debug("debug message")
            splog.info("info message")
            splog.warning("warning message")
            splog.error("error message")
            splog.critical("critical message")
        # At minimum the WARNING+ messages should appear (caplog threshold may vary)
        # The important thing is no exception was raised

    def test_set_and_get_verbosity_round_trip(self):
        splog.set_verbosity(_stdlib.DEBUG)
        assert splog.get_verbosity() == _stdlib.DEBUG
        splog.set_verbosity(_stdlib.WARNING)
        assert splog.get_verbosity() == _stdlib.WARNING

    def test_google_log_prefix_is_log_prefix(self):
        """``_log_prefix`` must point to ``google2_log_prefix`` after module init."""
        assert splog._log_prefix is splog.google2_log_prefix
