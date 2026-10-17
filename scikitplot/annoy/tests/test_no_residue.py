"""
The residue guard of ``conftest.py`` reports what a test leaves (Rule 30).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from scikitplot.annoy.tests.conftest import HERE, snapshot, writes_nothing_into


class TestContextManager:
    def test_a_new_file_is_reported(self, tmp_path):
        (tmp_path / "old.ann").write_bytes(b"")
        guard = writes_nothing_into(str(tmp_path))
        with pytest.raises(
            AssertionError, match=r"during this module: \['new.ann'\]"
        ), guard:
            (tmp_path / "new.ann").write_bytes(b"")

    def test_a_rewritten_file_is_reported(self, tmp_path):
        # The size and the modification time are set apart explicitly: a
        # file system's time resolution cannot make the test pass by chance.
        fixture = tmp_path / "fixture.tree"
        fixture.write_bytes(b"1234")
        os.utime(fixture, ns=(10**18, 10**18))
        guard = writes_nothing_into(str(tmp_path))
        with pytest.raises(
            AssertionError, match=r"during this module: \['fixture.tree'\]"
        ), guard:
            os.utime(fixture, ns=(10**18, 10**18 + 1))

    def test_reading_is_not_a_change(self, tmp_path):
        fixture = tmp_path / "fixture.tree"
        fixture.write_bytes(b"1234")
        with writes_nothing_into(str(tmp_path)):
            assert fixture.read_bytes() == b"1234"

    def test_removals_and_the_interpreter_cache_are_not_residue(self, tmp_path):
        (tmp_path / "old.ann").write_bytes(b"")
        with writes_nothing_into(str(tmp_path)):
            (tmp_path / "old.ann").unlink()
            (tmp_path / "__pycache__").mkdir()
        assert "__pycache__" not in snapshot(str(tmp_path))

    def test_an_error_inside_propagates_unchanged(self, tmp_path):
        def write_then_fail():
            with writes_nothing_into(str(tmp_path)):
                (tmp_path / "new.ann").write_bytes(b"")
                raise KeyError("inner")

        with pytest.raises(KeyError, match="inner"):
            write_then_fail()


def _run_suite(directory, modules):
    """Run pytest on a copy of the guard and ``modules``; return its report."""
    shutil.copy(os.path.join(HERE, "conftest.py"), directory / "conftest.py")
    for name, source in modules.items():
        (directory / name).write_text(textwrap.dedent(source), encoding="utf-8")
    # Hermetic: no configuration, no plugin of the outer environment.
    (directory / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    env.pop("PYTEST_ADDOPTS", None)
    done = subprocess.run(  # noqa: S603 - our interpreter, our arguments
        [
            sys.executable,
            "-m",
            "pytest",
            ".",
            "-q",
            "-rE",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(directory),  # node ids relative to the run's own root
        timeout=120,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


class TestFixtures:
    """The autouse fixtures, end to end, in a pytest run of their own."""

    def test_a_writing_module_errors_at_its_teardown(self, tmp_path):
        code, report = _run_suite(
            tmp_path,
            {
                "test_clean.py": (
                    "def test_ok(tmp_path):\n    (tmp_path / 'x.ann').write_bytes(b'')\n"
                ),
                "test_dirty.py": (
                    """
                    import os
                    def test_a():
                        open(os.path.join(os.path.dirname(__file__), 'x.ann'), 'wb').close()
                    def test_b():
                        pass
                """
                ),
            },
        )
        assert code == 1, report
        assert "ERROR test_dirty.py::test_b" in report
        assert "during this module: ['x.ann']" in report
        assert "3 passed, 1 error" in report
        assert "outside any module" not in report  # reported once

    def test_a_write_at_import_time_errors_the_session(self, tmp_path):
        code, report = _run_suite(
            tmp_path,
            {
                "test_imports.py": (
                    """
                    import os
                    open(os.path.join(os.path.dirname(__file__), 'at_import.ann'), 'wb').close()
                    def test_a():
                        pass
                """
                ),
            },
        )
        assert code == 1, report
        assert (
            "outside any module's tests (at import time): ['at_import.ann']" in report
        )
        assert "1 passed, 1 error" in report

    def test_a_clean_suite_passes(self, tmp_path):
        code, report = _run_suite(
            tmp_path, {"test_clean.py": "def test_ok():\n    pass\n"}
        )
        assert code == 0, report

    def test_editing_a_test_source_during_the_run_is_not_residue(self, tmp_path):
        # Saving a test file while the suite runs is the developer's edit,
        # not a write by a test: a Python source counts by presence only.
        code, report = _run_suite(
            tmp_path,
            {
                "test_edit.py": (
                    """
                    import os
                    def test_a():
                        with open(__file__, 'a', encoding='utf-8') as f:
                            f.write('# edited\\n')
                """
                ),
            },
        )
        assert code == 0, report


def test_the_committed_fixture_is_present():
    # Nine tests read it; a checkout without it fails them one by one.
    assert os.path.isfile(os.path.join(HERE, "test64.tree"))
