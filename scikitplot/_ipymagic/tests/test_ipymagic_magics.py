# scikitplot/_ipymagic/tests/test_ipymagic_magics.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The magics in a real IPython shell, and in a real Jupyter kernel."""

from __future__ import annotations

import os
import sys
import warnings

import pytest

pytest.importorskip("IPython")

from IPython.testing import globalipapp  # noqa: E402

from ... import _ipymagic as ipymagic  # noqa: E402
from .._results import CommandResult, ProfileResult  # noqa: E402


@pytest.fixture
def ip():
    shell = get_ipython()
    shell.run_line_magic("load_ext", "scikitplot._ipymagic")
    yield shell
    shell.run_line_magic("unload_ext", "scikitplot._ipymagic")


def get_ipython():
    """Return the one test shell (``start_ipython`` runs once per process)."""
    globalipapp.start_ipython()
    return globalipapp.get_ipython()


def _run(shell, source):
    result = shell.run_cell(source, store_history=False)
    return result


class TestMagics:
    def test_registered_and_removed(self, ip):
        magics = ip.magics_manager.magics
        assert {"scikitplot", "scikitplot_config", "scikitplot_profile"} <= set(
            magics["line"])
        assert {"scikitplot", "scikitplot_profile"} <= set(magics["cell"])
        ip.run_line_magic("unload_ext", "scikitplot._ipymagic")
        assert "scikitplot" not in ip.magics_manager.magics["line"]
        assert "%scikitplot" not in ip.strdispatchers["complete_command"].strs
        ip.run_line_magic("load_ext", "scikitplot._ipymagic")

    def test_command_returns_a_result(self, ip):
        outcome = _run(ip, "r = %scikitplot info --format json\nr")
        assert outcome.success
        assert isinstance(outcome.result, CommandResult)
        assert outcome.result.data["scikitplot"]["version"]

    def test_variables_expand(self, ip):
        ip.user_ns["fmt"] = "json"
        outcome = _run(ip, "%scikitplot info --format {fmt}")
        assert outcome.result.data is not None

    def test_cell_is_stdin(self, ip, capsys):
        outcome = _run(ip, "%%scikitplot cleanprompt redact\nmail me at alice@example.com\n")
        assert outcome.success and outcome.result.mode == "subprocess"
        assert "alice@example.com" not in outcome.result.stdout
        assert outcome.result.stdout.strip()

    def test_usage_error_is_two_lines(self, ip, capsys):
        outcome = _run(ip, "%scikitplot evn")
        assert not outcome.success
        shown = capsys.readouterr()
        text = shown.out + shown.err
        assert "did you mean 'env'?" in text
        assert "Traceback" not in text

    def test_config_and_raise(self, ip):
        outcome = _run(ip, "%scikitplot_config raise_on_error=true")
        assert outcome.result.raise_on_error is True
        try:
            failed = _run(ip, "%scikitplot info --format xml")
            assert type(failed.error_in_exec).__name__ == "CommandFailedError"
            bad = _run(ip, "%scikitplot_config colour=red")
            assert not bad.success
            kept = _run(ip, "%scikitplot_config")
            assert kept.result.raise_on_error is True
        finally:
            _run(ip, "%scikitplot_config raise_on_error=false")

    def test_profile_line_and_cell(self, ip):
        ip.user_ns["n"] = 200
        line = _run(ip, "%scikitplot_profile -s ncalls -l 3 sum(i * i for i in range(n))")
        assert isinstance(line.result, ProfileResult) and len(line.result.rows) == 3
        cell = _run(ip, "%%scikitplot_profile -l 5\ndef g(k):\n    return k * 2\nx = [g(i) for i in range(50)]\n")
        assert cell.success and ip.user_ns["x"][-1] == 98
        assert any(r.function == "g" and r.ncalls == 50 for r in cell.result.rows)

    def test_profile_errors(self, ip):
        assert not _run(ip, "%scikitplot_profile -s name 1 + 1").success
        assert not _run(ip, "%scikitplot_profile -l many 1 + 1").success
        assert not _run(ip, "%scikitplot_profile").success
        assert not _run(ip, "%%scikitplot_profile 1 + 1\n2 + 2\n").success
        raised = _run(ip, "%%scikitplot_profile\nx = 1\nraise KeyError('boom')\n")
        assert isinstance(raised.error_in_exec, KeyError)

    def test_profile_handles_ipython_syntax(self, ip):
        outcome = _run(ip, "%%scikitplot_profile\nvalue = %scikitplot info --format json\n")
        assert outcome.success and ip.user_ns["value"].ok

    def test_completion(self, ip):
        from IPython.core.completer import provisionalcompleter  # noqa: PLC0415

        def texts(line):
            with provisionalcompleter():
                return {c.text for c in ip.Completer.completions(line, len(line))}

        assert "cleanprompt" in texts("%scikitplot cle")
        assert "json" in texts("%scikitplot info --format ")

    def test_snakeviz_missing_runs_nothing(self, ip, monkeypatch):
        # Regression: the code ran (side effects) before snakeviz was checked.
        monkeypatch.setitem(sys.modules, "snakeviz", None)
        ip.user_ns.pop("ran", None)
        outcome = _run(ip, "%scikitplot_profile --snakeviz ran = 1")
        assert isinstance(outcome.error_in_exec, ipymagic.CapabilityMissingError)
        assert "ran" not in ip.user_ns

    def test_snakeviz_failure_still_shows_the_profile(self, ip):
        from unittest import mock  # noqa: PLC0415

        from .. import _magics  # noqa: PLC0415

        failure = ipymagic.IpyMagicError("no", code="snakeviz_start")
        with mock.patch.object(_magics, "require_snakeviz", autospec=True), \
             mock.patch.object(_magics, "start_snakeviz", autospec=True,
                               side_effect=failure), \
             mock.patch.object(_magics, "display", autospec=True) as shown:
            outcome = _run(ip, "%scikitplot_profile --snakeviz sum(range(9))")
        assert outcome.error_in_exec is failure
        assert isinstance(shown.call_args_list[0].args[0], ProfileResult)

    def test_quoted_dump_path(self, ip, tmp_path):
        target = tmp_path / "my run.prof"
        ip.user_ns["target"] = str(target)
        assert _run(ip, '%scikitplot_profile -D "{target}" sum(range(9))').success
        assert target.is_file()

    def test_stop_snakeviz(self, ip, capsys):
        assert _run(ip, "%scikitplot_profile --stop-snakeviz").success
        assert "stopped 0 snakeviz server(s)" in capsys.readouterr().out


def test_load_ext_scikitplot_and_the_deprecated_name():
    shell = get_ipython()
    shell.run_line_magic("load_ext", "scikitplot")
    try:
        assert "scikitplot_profile" in shell.magics_manager.magics["line"]
    finally:
        shell.run_line_magic("unload_ext", "scikitplot")
    from ... import _magic  # noqa: PLC0415

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _magic.load_ipython_extension(shell)
    try:
        assert any(issubclass(w.category, FutureWarning) for w in seen)
        assert "scikitplot" in shell.magics_manager.magics["line"]
    finally:
        _magic.unload_ipython_extension(shell)


def test_deprecation_is_visible_under_default_filters():
    # Regression: a DeprecationWarning attributed to IPython's extension
    # manager is hidden by Python's default filters, so nobody saw it.
    import subprocess  # noqa: PLC0415

    code = ("import warnings, scikitplot._magic as m\n"
            "class Shell:\n"
            "    pass\n"
            "import scikitplot._ipymagic as p\n"
            "p.load_ipython_extension = lambda ip: None\n"
            "m.load_ipython_extension(Shell())\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, check=False,
                          env={**os.environ, "PYTHONWARNINGS": ""})
    assert done.returncode == 0, done.stderr
    assert "FutureWarning" in done.stderr and "load_ext scikitplot" in done.stderr


def test_extension_names_share_one_instance():
    shell = get_ipython()
    shell.run_line_magic("load_ext", "scikitplot._ipymagic")
    try:
        shell.run_line_magic("scikitplot_config", "timeout=7")
        shell.run_line_magic("load_ext", "scikitplot")  # keeps the instance
        assert shell.run_line_magic("scikitplot_config", "").timeout == 7
        shell.run_line_magic("unload_ext", "scikitplot")
        assert "scikitplot" not in shell.magics_manager.magics["line"]
        # Regression: IPython still listed this name as loaded, so loading
        # it again said "already loaded" and left the magics missing.
        shell.run_line_magic("load_ext", "scikitplot._ipymagic")
        assert "scikitplot" in shell.magics_manager.magics["line"]
    finally:
        shell.run_line_magic("unload_ext", "scikitplot._ipymagic")


def test_importing_the_package_imports_no_ipython():
    import subprocess  # noqa: PLC0415

    code = (f"import sys, {ipymagic.__name__} as m\n"
            "m.run_command, m.CommandResult, m.profile_code, m.complete\n"
            "print(sorted(k for k in sys.modules if k.split('.')[0] in "
            "('IPython', 'snakeviz', 'tornado')))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_real_kernel_runs_every_magic(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    # The kernel is a new process: let it import the scikitplot under test
    # (a source checkout is not on its path otherwise).
    import scikitplot  # noqa: PLC0415

    root = os.path.dirname(os.path.dirname(os.path.abspath(scikitplot.__file__)))
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", root if not path else root + os.pathsep + path)
    cells = [
        "%load_ext scikitplot",
        "r = %scikitplot info --format json\nassert r.ok and r.data['python']",
        "%scikitplot env plan -n kernel-test --target pyodide --package six",
        "%%scikitplot cleanprompt redact\ncall me at +1 202 555 0143\n",
        "%scikitplot_config timeout=60",
        "p = %scikitplot_profile -l 2 sorted(range(1000), reverse=True)\nassert p.rows",
        "%scikitplot evn",
    ]
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(c) for c in cells])
    client = nbclient.NotebookClient(notebook, timeout=120, kernel_name="python3",
                                     allow_errors=True,
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    outputs = [c.outputs for c in notebook.cells]
    for index in range(6):
        assert not any(o.output_type == "error" for o in outputs[index]), outputs[index]
    plan_text = "".join(o.get("text", "") for o in outputs[2])
    assert "micropip" in plan_text
    error = [o for o in outputs[6] if o.output_type == "error"][0]
    assert error.ename == "MagicUsageError"
    assert any("did you mean 'env'?" in line for line in error.traceback)
