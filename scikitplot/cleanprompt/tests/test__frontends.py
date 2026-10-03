"""
Tests for :mod:`scikitplot.cleanprompt._frontends`.

Notes
-----
**Developer notes.** :class:`TestParity` is the point of this module. Two
frontends that agree on a handful of hand-picked cases prove nothing; these
drive every declared command through both and compare exit status and stdout
byte for byte, so a divergence cannot survive a test run.
"""

from __future__ import annotations

import io
import json

import pytest

from .._cli import COMMANDS, DESCRIPTION, EPILOG, PROG, main
from .._frontends import (
    FRONTENDS,
    build_argparse,
    is_click_available,
    load_runner,
    namespace_for,
    run_argparse,
    select_frontend,
)
from .._spec import Command, Param

CLICK = is_click_available()
needs_click = pytest.mark.skipif(not CLICK, reason="click is not installed")


def run(args, frontend, stdin=""):
    """Run the CLI through one frontend and return (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = main(
        args, stdin=io.StringIO(stdin), stdout=out, stderr=err, frontend=frontend
    )
    return code, out.getvalue(), err.getvalue()


class TestSelection:
    """Choosing a frontend."""

    def test_explicit_wins(self):
        assert select_frontend("argparse") == "argparse"

    def test_unknown_explicit_is_refused(self):
        with pytest.raises(ValueError, match="unknown frontend"):
            select_frontend("qt")

    def test_env_var_is_honoured(self, monkeypatch):
        monkeypatch.setenv("CLEANPROMPT_CLI_FRONTEND", "argparse")
        assert select_frontend() == "argparse"

    def test_project_env_var_is_honoured(self, monkeypatch):
        monkeypatch.delenv("CLEANPROMPT_CLI_FRONTEND", raising=False)
        monkeypatch.setenv("SCIKITPLOT_CLI_FRONTEND", "argparse")
        assert select_frontend() == "argparse"

    def test_submodule_var_beats_project_var(self, monkeypatch):
        monkeypatch.setenv("SCIKITPLOT_CLI_FRONTEND", "click")
        monkeypatch.setenv("CLEANPROMPT_CLI_FRONTEND", "argparse")
        assert select_frontend() == "argparse"

    def test_garbage_env_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv("CLEANPROMPT_CLI_FRONTEND", "nonsense")
        assert select_frontend() in FRONTENDS

    def test_asking_for_missing_click_falls_back(self, monkeypatch):
        """An inherited preference must not make a working command fail."""
        from .. import _frontends

        monkeypatch.setenv("CLEANPROMPT_CLI_FRONTEND", "click")
        monkeypatch.setattr(_frontends, "is_click_available", lambda: False)
        assert _frontends.select_frontend() == "argparse"

    def test_default_without_click_is_argparse(self, monkeypatch):
        from .. import _frontends

        for var in ("CLEANPROMPT_CLI_FRONTEND", "SCIKITPLOT_CLI_FRONTEND"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(_frontends, "is_click_available", lambda: False)
        assert _frontends.select_frontend() == "argparse"

    def test_availability_probe_does_not_import_click(self):
        import sys

        before = "click" in sys.modules
        is_click_available()
        assert ("click" in sys.modules) == before


class TestLoadRunner:
    """
    Choosing a frontend that will actually import.

    Notes
    -----
    **Developer notes.** ``select_frontend`` answers which frontend is wanted;
    ``load_runner`` answers which one works. They differ when click is present
    but unimportable, which took the whole CLI down — including ``--help`` —
    until this existed.
    """

    def test_returns_a_runner_and_its_name(self):
        runner, name = load_runner()
        assert callable(runner)
        assert name in FRONTENDS

    def test_argparse_is_honoured(self):
        assert load_runner("argparse") == (run_argparse, "argparse")

    @needs_click
    def test_click_is_honoured_when_it_imports(self):
        assert load_runner("click")[1] == "click"

    def test_unimportable_click_falls_back(self, monkeypatch):
        """Present-but-broken is not available; the base tier must survive."""
        import builtins

        from .. import _frontends

        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] == "click":
                raise ImportError("simulated broken click install")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(_frontends, "is_click_available", lambda: True)
        monkeypatch.setattr(builtins, "__import__", blocked)
        assert _frontends.load_runner("click")[1] == "argparse"

    def test_unknown_preference_is_refused(self):
        with pytest.raises(ValueError, match="unknown frontend"):
            load_runner("qt")


class TestNamespace:
    """Normalising parsed values into one handler shape."""

    def test_defaults_fill_missing_values(self):
        command = Command(
            name="x",
            summary="s.",
            handler="h",
            params=(Param(dest="a", flags=("--a",), default="fallback"),),
        )
        assert namespace_for(command, {}).a == "fallback"

    def test_none_means_not_supplied(self):
        """Click reports unsupplied options as None; argparse applies defaults."""
        command = Command(
            name="x",
            summary="s.",
            handler="h",
            params=(Param(dest="a", flags=("--a",), default="fallback"),),
        )
        assert namespace_for(command, {"a": None}).a == "fallback"

    def test_tuples_become_lists(self):
        command = Command(
            name="x",
            summary="s.",
            handler="h",
            params=(Param(dest="a", flags=("--a",), multiple=True),),
        )
        assert namespace_for(command, {"a": ("x", "y")}).a == ["x", "y"]

    def test_command_name_is_recorded(self):
        command = Command(name="x", summary="s.", handler="h")
        assert namespace_for(command, {}).command == "x"


class TestArgparseRendering:
    """The argparse view of the surface."""

    def test_every_command_is_present(self):
        parser = build_argparse(COMMANDS, PROG, DESCRIPTION, EPILOG)
        for command in COMMANDS:
            assert parser.parse_args([command.name, *_required(command)])

    def test_aliases_are_present(self):
        parser = build_argparse(COMMANDS, PROG, DESCRIPTION, EPILOG)
        for command in COMMANDS:
            for alias in command.aliases:
                assert parser.parse_args([alias, *_required(command)])

    def test_fresh_parser_each_call(self):
        assert build_argparse(
            COMMANDS, PROG, DESCRIPTION, EPILOG
        ) is not build_argparse(COMMANDS, PROG, DESCRIPTION, EPILOG)


def _required(command):
    """Return placeholder values for a command's required options and positionals."""
    out = []
    for param in command.params:
        if param.kind == "argument" and not param.multiple:
            out.append("x")
        elif param.required:
            out.extend([param.flags[0], "x"])
    return out


class TestParity:
    """The two frontends must be indistinguishable."""

    CASES = [
        ["kinds", "--format", "json"],
        ["doctor", "--format", "json"],
        ["doctor", "--format", "text"],
        ["kinds", "--format", "text"],
        ["docker"],
        ["docker", "--port", "9000", "--with-ner"],
    ]

    @needs_click
    @pytest.mark.parametrize("args", CASES)
    def test_exit_status_and_stdout_match(self, args):
        code_a, out_a, _ = run(args, "argparse")
        code_c, out_c, _ = run(args, "click")
        assert code_a == code_c
        assert out_a == out_c

    @needs_click
    @pytest.mark.parametrize(
        "extra",
        [
            ["--kinds", "EMAIL"],
            ["--profile", "strict"],
            ["--hide", "Acme"],
            ["--allow", "keepme"],
            ["--ignore-case"],
            ["--word-boundary"],
            ["--suggest-min-tokens", "2"],
            ["--no-suggest"],
            ["--reveal"],
        ],
    )
    def test_detection_options_match(self, extra, tmp_path):
        source = tmp_path / "t.txt"
        source.write_text("Ada Lovelace mailed ada@example.com", encoding="utf-8")
        args = ["inspect", "--in", str(source), "--format", "json", *extra]
        code_a, out_a, _ = run(args, "argparse")
        code_c, out_c, _ = run(args, "click")
        assert (code_a, out_a) == (code_c, out_c)

    @needs_click
    @pytest.mark.parametrize(
        "extra",
        [
            ["--hide", "Ada", "--hide", "Lovelace"],
            ["--allow", "one", "--allow", "two"],
            ["--kinds", "EMAIL", "--kinds", "URL"],
        ],
    )
    def test_repeatable_options_match(self, extra, tmp_path):
        """
        Repeatable options must take the same spelling in both frontends.

        Notes
        -----
        **Developer notes.** This is a regression. argparse rendered these as
        ``nargs="+"`` and accepted ``--hide A B``, while click rendered them as
        repeatable and rejected it, so the same command line worked or failed
        depending on which library happened to be installed. argparse now
        conforms to click, which is the shape both can express.
        """
        source = tmp_path / "t.txt"
        source.write_text("Ada Lovelace mailed ada@example.com", encoding="utf-8")
        args = ["inspect", "--in", str(source), "--format", "json", *extra]
        code_a, out_a, _ = run(args, "argparse")
        code_c, out_c, _ = run(args, "click")
        assert code_a == 0
        assert (code_a, out_a) == (code_c, out_c)

    @needs_click
    def test_unused_repeatable_option_gives_the_same_namespace(self, tmp_path):
        """Click yields () and argparse None; a handler must not see both."""
        source = tmp_path / "t.txt"
        source.write_text("mail ada@example.com", encoding="utf-8")
        args = ["inspect", "--in", str(source), "--format", "json"]
        assert run(args, "argparse")[1] == run(args, "click")[1]

    @needs_click
    def test_scan_exit_codes_match(self, tmp_path):
        source = tmp_path / "t.txt"
        source.write_text("mail ada@example.com", encoding="utf-8")
        args = ["scan", "--in", str(source), "--format", "json"]
        assert run(args, "argparse")[0] == run(args, "click")[0] == 3

    @needs_click
    def test_redact_and_restore_match(self, tmp_path):
        text = "Ada mailed ada@example.com and called +1 555 010 4477"
        source = tmp_path / "t.txt"
        source.write_text(text, encoding="utf-8")
        outputs = []
        for frontend in ("argparse", "click"):
            vault = tmp_path / "{0}.json".format(frontend)
            _, redacted, _ = run(
                [
                    "redact",
                    "--in",
                    str(source),
                    "--vault",
                    str(vault),
                    "--quiet",
                    "--color",
                    "never",
                ],
                frontend,
            )
            reply = tmp_path / "{0}.txt".format(frontend)
            reply.write_text(redacted.rstrip("\n"), encoding="utf-8")
            _, restored, _ = run(
                ["restore", "--in", str(reply), "--vault", str(vault), "--quiet"],
                frontend,
            )
            outputs.append((redacted, restored.rstrip("\n")))
        assert outputs[0] == outputs[1]
        assert outputs[0][1] == text

    @needs_click
    def test_version_matches(self):
        assert run(["--version"], "argparse") == run(["--version"], "click")

    @needs_click
    def test_unknown_command_is_a_usage_error_in_both(self):
        assert run(["nope"], "argparse")[0] == run(["nope"], "click")[0] == 2

    @needs_click
    def test_no_command_is_a_usage_error_in_both(self):
        assert run([], "argparse")[0] == run([], "click")[0] == 2

    @needs_click
    def test_help_exits_zero_in_both(self):
        assert run(["--help"], "argparse")[0] == run(["--help"], "click")[0] == 0

    @needs_click
    @pytest.mark.parametrize("command", [c.name for c in COMMANDS])
    def test_every_command_help_exits_zero_in_both(self, command):
        assert run([command, "--help"], "argparse")[0] == 0
        assert run([command, "--help"], "click")[0] == 0


class TestFrontendOverride:
    """``--frontend`` is read before the parser it selects."""

    def test_flag_form(self):
        code, out, _ = run(
            ["--frontend", "argparse", "kinds", "--format", "json"], None
        )
        assert code == 0
        assert json.loads(out)["kinds"]

    def test_equals_form(self):
        out, err = io.StringIO(), io.StringIO()
        code = main(
            ["--frontend=argparse", "kinds", "--format", "json"],
            stdout=out,
            stderr=err,
        )
        assert code == 0

    def test_unknown_value_is_a_usage_error(self):
        out, err = io.StringIO(), io.StringIO()
        code = main(["--frontend", "qt", "kinds"], stdout=out, stderr=err)
        assert code == 2
        assert "unknown frontend" in err.getvalue()

    def test_environment_selection_end_to_end(self, monkeypatch):
        monkeypatch.setenv("CLEANPROMPT_CLI_FRONTEND", "argparse")
        out, err = io.StringIO(), io.StringIO()
        assert main(["kinds", "--format", "json"], stdout=out, stderr=err) == 0
