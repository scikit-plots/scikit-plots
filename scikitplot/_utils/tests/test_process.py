# scikitplot/_utils/tests/test_process.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.process`.

Coverage map
------------
ShellCommandException         from_completed_process: returncode,
                               args, stdout, stderr, no stdout/stderr -> TestShellCommandException
_remove_inaccessible_python_path  removes inaccessible paths,
                                   keeps accessible, missing key     -> TestRemoveInaccessiblePythonPath
_exec_cmd                      illegal kwarg (text), extra_env+env conflict,
                               capture+stream conflict, synchronous=False,
                               throw_on_error with nonzero, zero exit,
                               stream_output path, stdout/stderr conflict,
                               cmd list stringification              -> TestExecCmd
_join_commands                 POSIX bash -c output, Windows cmd /c  -> TestJoinCommands
cache_return_value_per_process caches per process, invalidates in child,
                               rejects kwargs                        -> TestCacheReturnValuePerProcess

Run standalone::

    python -m unittest scikitplot._utils.tests.test_process -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
import unittest.mock as mock

from ..process import (
    ShellCommandException,
    _exec_cmd,
    _join_commands,
    _remove_inaccessible_python_path,
    cache_return_value_per_process,
)


# ===========================================================================
# ShellCommandException
# ===========================================================================


class TestShellCommandException(unittest.TestCase):
    """ShellCommandException.from_completed_process must build a readable message."""

    def _make_process(self, returncode=1, args="echo hi", stdout=None, stderr=None):
        return subprocess.CompletedProcess(
            args=args,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    # -- basic construction --

    def test_is_exception(self):
        self.assertTrue(issubclass(ShellCommandException, Exception))

    def test_from_completed_process_returns_instance(self):
        proc = self._make_process()
        exc = ShellCommandException.from_completed_process(proc)
        self.assertIsInstance(exc, ShellCommandException)

    # -- message content --

    def test_message_contains_returncode(self):
        proc = self._make_process(returncode=42)
        exc = ShellCommandException.from_completed_process(proc)
        self.assertIn("42", str(exc))

    def test_message_contains_command(self):
        proc = self._make_process(args=["ls", "-la"])
        exc = ShellCommandException.from_completed_process(proc)
        self.assertIn("ls", str(exc))

    def test_message_includes_stdout_when_present(self):
        proc = self._make_process(stdout="some output text")
        exc = ShellCommandException.from_completed_process(proc)
        self.assertIn("some output text", str(exc))

    def test_message_includes_stderr_when_present(self):
        proc = self._make_process(stderr="error text here")
        exc = ShellCommandException.from_completed_process(proc)
        self.assertIn("error text here", str(exc))

    def test_message_excludes_stdout_label_when_none(self):
        proc = self._make_process(stdout=None, stderr=None)
        msg = str(ShellCommandException.from_completed_process(proc))
        self.assertNotIn("STDOUT", msg)

    def test_message_excludes_stderr_label_when_none(self):
        proc = self._make_process(stdout=None, stderr=None)
        msg = str(ShellCommandException.from_completed_process(proc))
        self.assertNotIn("STDERR", msg)

    def test_message_includes_stdout_label_when_present(self):
        proc = self._make_process(stdout="output")
        msg = str(ShellCommandException.from_completed_process(proc))
        self.assertIn("STDOUT", msg)

    def test_message_includes_stderr_label_when_present(self):
        proc = self._make_process(stderr="error")
        msg = str(ShellCommandException.from_completed_process(proc))
        self.assertIn("STDERR", msg)

    def test_can_be_raised_and_caught(self):
        proc = self._make_process()
        with self.assertRaises(ShellCommandException):
            raise ShellCommandException.from_completed_process(proc)


# ===========================================================================
# _remove_inaccessible_python_path
# ===========================================================================


class TestRemoveInaccessiblePythonPath(unittest.TestCase):
    """_remove_inaccessible_python_path must filter inaccessible paths from PYTHONPATH."""

    def test_removes_inaccessible_path(self):
        """Paths not accessible via os.R_OK must be removed."""
        env = {"PYTHONPATH": "/accessible:/inaccessible"}

        def fake_access(path, mode):
            return path == "/accessible"

        with mock.patch("scikitplot._utils.process._os.access", side_effect=fake_access):
            result = _remove_inaccessible_python_path(env)

        self.assertEqual(result["PYTHONPATH"], "/accessible")

    def test_keeps_all_accessible_paths(self):
        """When all paths are accessible, PYTHONPATH must be unchanged."""
        env = {"PYTHONPATH": "/a:/b:/c"}

        with mock.patch("scikitplot._utils.process._os.access", return_value=True):
            result = _remove_inaccessible_python_path(env)

        self.assertEqual(result["PYTHONPATH"], "/a:/b:/c")

    def test_all_inaccessible_gives_empty_pythonpath(self):
        """When all paths are inaccessible, PYTHONPATH must become an empty string."""
        env = {"PYTHONPATH": "/bad1:/bad2"}

        with mock.patch("scikitplot._utils.process._os.access", return_value=False):
            result = _remove_inaccessible_python_path(env)

        self.assertEqual(result["PYTHONPATH"], "")

    def test_missing_pythonpath_key_returns_env_unchanged(self):
        """When PYTHONPATH is absent, the function must return the env dict unmodified."""
        env = {"OTHER_VAR": "value"}
        result = _remove_inaccessible_python_path(env)
        self.assertEqual(result, {"OTHER_VAR": "value"})

    def test_returns_same_dict(self):
        """The function must mutate and return the same dict object."""
        env = {"PYTHONPATH": "/path"}
        with mock.patch("scikitplot._utils.process._os.access", return_value=True):
            result = _remove_inaccessible_python_path(env)
        self.assertIs(result, env)


# ===========================================================================
# _exec_cmd
# ===========================================================================


class TestExecCmd(unittest.TestCase):
    """_exec_cmd must correctly wrap subprocess.Popen with the given options."""

    def _run_true(self, **kwargs):
        """Run 'true' (always exits 0) or an echo command."""
        cmd = ["true"] if sys.platform != "win32" else ["cmd", "/c", "exit", "0"]
        return _exec_cmd(cmd, **kwargs)

    # -- illegal kwargs --

    def test_raises_on_text_kwarg(self):
        """Passing text= as a kwarg must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _exec_cmd(["echo", "hi"], text=False)
        self.assertIn("text", str(ctx.exception))

    # -- extra_env + env conflict --

    def test_raises_when_extra_env_and_env_both_given(self):
        """Specifying both extra_env and env= must raise ValueError."""
        with self.assertRaises(ValueError):
            _exec_cmd(["echo"], extra_env={"A": "1"}, env={"B": "2"})

    # -- capture + stream conflict --

    def test_raises_when_capture_and_stream_output(self):
        """Setting capture_output=True and stream_output=True together must raise."""
        with self.assertRaises(ValueError):
            _exec_cmd(["echo"], capture_output=True, stream_output=True)

    # -- stdout/stderr kwarg + capture_output conflict --

    def test_raises_when_stdout_kwarg_with_capture_output(self):
        """Providing stdout= in kwargs alongside capture_output must raise ValueError."""
        with self.assertRaises(ValueError):
            _exec_cmd(["echo"], capture_output=True, stdout=subprocess.PIPE)

    def test_raises_when_stderr_kwarg_with_capture_output(self):
        """Providing stderr= in kwargs alongside capture_output must raise ValueError."""
        with self.assertRaises(ValueError):
            _exec_cmd(["echo"], capture_output=True, stderr=subprocess.PIPE)

    # -- successful synchronous execution --

    @unittest.skipIf(sys.platform == "win32", "Uses POSIX commands")
    def test_successful_exit_returns_completed_process(self):
        """A command that exits 0 must return a CompletedProcess instance."""
        result = _exec_cmd(["true"])
        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertEqual(result.returncode, 0)

    @unittest.skipIf(sys.platform == "win32", "Uses POSIX commands")
    def test_capture_output_true_collects_stdout(self):
        """With capture_output=True, stdout must be captured in the result."""
        result = _exec_cmd(["echo", "hello"], capture_output=True)
        self.assertIn("hello", result.stdout)

    # -- throw_on_error --

    @unittest.skipIf(sys.platform == "win32", "Uses POSIX commands")
    def test_nonzero_exit_raises_shell_command_exception(self):
        """A nonzero exit code must raise ShellCommandException by default."""
        with self.assertRaises(ShellCommandException):
            _exec_cmd(["false"])

    @unittest.skipIf(sys.platform == "win32", "Uses POSIX commands")
    def test_nonzero_exit_no_raise_when_throw_false(self):
        """With throw_on_error=False, a nonzero exit must not raise."""
        result = _exec_cmd(["false"], throw_on_error=False)
        self.assertNotEqual(result.returncode, 0)

    # -- synchronous=False returns Popen --

    @unittest.skipIf(sys.platform == "win32", "Uses POSIX commands")
    def test_asynchronous_returns_popen(self):
        """synchronous=False must return a Popen instance, not CompletedProcess."""
        # Why capture_output=False is the right fix, not closing pipes manually
        proc = _exec_cmd(["sleep", "0.01"], synchronous=False, capture_output=False)
        try:
            self.assertIsInstance(proc, subprocess.Popen)
        finally:
            proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    # -- cmd list stringification --

    def test_cmd_list_elements_are_stringified(self):
        """Path-like objects in cmd list must be stringified before Popen."""
        import pathlib

        captured_cmd = []

        def fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            m = mock.MagicMock()
            m.communicate.return_value = ("", "")
            m.poll.return_value = 0
            m.args = cmd
            m.stdout = None
            m.stderr = None
            return m

        with mock.patch("scikitplot._utils.process._subprocess.Popen", side_effect=fake_popen):
            _exec_cmd(["echo", pathlib.Path("myfile.txt")], capture_output=False)

        for element in captured_cmd:
            self.assertIsInstance(element, str)

    # -- extra_env merges into environment --

    @unittest.skipIf(sys.platform == "win32", "Uses POSIX commands")
    def test_extra_env_appears_in_child_environment(self):
        """Environment variables from extra_env must be visible in the child process."""
        result = _exec_cmd(
            ["env"],
            extra_env={"SCIKITPLOT_TEST_VAR": "hello_test_123"},
            capture_output=True,
        )
        self.assertIn("SCIKITPLOT_TEST_VAR", result.stdout)


# ===========================================================================
# _join_commands
# ===========================================================================


class TestJoinCommands(unittest.TestCase):
    """_join_commands must produce a shell-runnable command list."""

    @unittest.skipIf(sys.platform == "win32", "POSIX only")
    def test_posix_uses_bash(self):
        """On POSIX, the result must start with ['bash', '-c']."""
        result = _join_commands("echo a", "echo b")
        self.assertEqual(result[:2], ["bash", "-c"])

    @unittest.skipIf(sys.platform == "win32", "POSIX only")
    def test_posix_joins_with_double_ampersand(self):
        """On POSIX, commands must be joined with ' && '."""
        result = _join_commands("echo a", "echo b")
        self.assertIn("&&", result[2])

    def test_windows_uses_cmd(self):
        """On Windows (mocked), the result must start with ['cmd', '/c']."""
        with mock.patch("scikitplot._utils.process.is_windows", return_value=True):
            result = _join_commands("echo a", "echo b")
        self.assertEqual(result[:2], ["cmd", "/c"])

    def test_windows_joins_with_ampersand(self):
        """On Windows (mocked), commands must be joined with ' & '."""
        with mock.patch("scikitplot._utils.process.is_windows", return_value=True):
            result = _join_commands("echo a", "echo b")
        self.assertIn("&", result[2])

    def test_single_command_no_separator(self):
        """A single command must not have a separator appended."""
        with mock.patch("scikitplot._utils.process.is_windows", return_value=False):
            result = _join_commands("echo a")
        self.assertNotIn("&&", result[2])

    def test_result_is_list(self):
        """_join_commands must return a list."""
        result = _join_commands("a", "b")
        self.assertIsInstance(result, list)

    def test_commands_are_stringified(self):
        """Non-string commands must be converted to str."""
        with mock.patch("scikitplot._utils.process.is_windows", return_value=False):
            result = _join_commands(42, "echo b")
        self.assertIsInstance(result[2], str)


# ===========================================================================
# cache_return_value_per_process
# ===========================================================================


class TestCacheReturnValuePerProcess(unittest.TestCase):
    """cache_return_value_per_process must cache within a process and bust on fork."""

    def setUp(self):
        # Clear the global cache before each test to avoid cross-test pollution
        from scikitplot._utils.process import _per_process_value_cache_map
        _per_process_value_cache_map.clear()

    # -- basic caching --

    def test_caches_return_value(self):
        """A decorated function must return the same object on repeated calls."""
        call_count = {"n": 0}

        @cache_return_value_per_process
        def fn(x):
            call_count["n"] += 1
            return x * 2

        result1 = fn(5)
        result2 = fn(5)
        self.assertEqual(result1, result2)
        self.assertEqual(call_count["n"], 1)  # called only once

    def test_different_args_not_cached(self):
        """Different arguments must produce separate cache entries."""
        call_count = {"n": 0}

        @cache_return_value_per_process
        def fn(x):
            call_count["n"] += 1
            return x

        fn(1)
        fn(2)
        self.assertEqual(call_count["n"], 2)

    def test_returns_correct_value(self):
        """The cached value must equal the original return value."""
        @cache_return_value_per_process
        def fn(x):
            return x ** 2

        self.assertEqual(fn(7), 49)

    # -- kwargs not allowed --

    def test_raises_on_keyword_args(self):
        """Calling with keyword arguments must raise ValueError."""
        @cache_return_value_per_process
        def fn(x):
            return x

        with self.assertRaises(ValueError) as ctx:
            fn(x=5)
        self.assertIn("key-word", str(ctx.exception).lower().replace("-", "-"))

    # -- process ID invalidation --

    def test_cache_invalidated_when_pid_changes(self):
        """If the PID differs from cached PID, the function must be re-called."""
        call_count = {"n": 0}
        original_pid = os.getpid()

        @cache_return_value_per_process
        def fn(x):
            call_count["n"] += 1
            return x

        fn(3)  # first call, caches with original_pid
        call_count["n"] = 0  # reset counter

        # Simulate a different PID (as if in a forked child)
        with mock.patch("scikitplot._utils.process._os.getpid", return_value=original_pid + 9999):
            fn(3)  # same args, different pid -> should re-execute

        self.assertEqual(call_count["n"], 1)

    # -- functools.wraps preserves name --

    def test_wrapped_function_preserves_name(self):
        """The decorator must preserve the wrapped function's __name__."""
        @cache_return_value_per_process
        def my_special_function(x):
            return x

        self.assertEqual(my_special_function.__name__, "my_special_function")

    # -- same pid reuses cache --

    def test_same_pid_uses_cached_value(self):
        """With the same PID, the cached value must be returned without re-calling."""
        call_count = {"n": 0}

        @cache_return_value_per_process
        def fn(x):
            call_count["n"] += 1
            return x + 1

        fn(10)
        fn(10)  # second call, same pid -> should hit cache
        self.assertEqual(call_count["n"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
