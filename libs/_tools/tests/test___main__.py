# libs/_tools/tests/test___main__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of ``libs/_tools/__main__.py``: the command line and its exit codes."""

from __future__ import annotations

import pytest

from .. import __main__ as cli
from .. import generate, registry, staging

MAP = staging.load_distributions(staging.repo_root())


class TestParser:
    def test_a_command_is_required(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main([])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize(
        "argv", [["list"], ["generate"], ["check"], ["unstage"], ["build"], ["verify"]]
    )
    def test_every_command_parses_without_arguments(self, argv):
        assert callable(cli.build_parser().parse_args(argv).run)

    def test_verify_options(self):
        args = cli.build_parser().parse_args(
            ["verify", "scikit-plots-mcp", "--python", "3.8", "--python", "3.13",
             "--skip-build", "--skip-tests", "--report", "r.json", "--outdir", "out"]
        )
        assert args.names == ["scikit-plots-mcp"]
        assert args.python == ["3.8", "3.13"]
        assert (args.skip_build, args.skip_tests) == (True, True)
        assert (args.report, args.outdir) == ("r.json", "out")

    def test_build_defaults(self):
        args = cli.build_parser().parse_args(["build"])
        assert (args.names, args.outdir) == ([], "dist/libs")


class TestCommands:
    def test_list_shows_every_distribution(self, capsys):
        assert cli.main(["list"]) == 0
        out = capsys.readouterr().out
        for dist in MAP.DISTRIBUTIONS:
            assert dist.name in out
            assert f"libs/{registry.directory_of(dist.name)}" in out

    def test_check_passes_on_the_committed_files(self, capsys):
        assert cli.main(["check"]) == 0
        assert "up to date" in capsys.readouterr().out

    def test_check_fails_and_names_the_stale_files(self, monkeypatch, capsys):
        monkeypatch.setattr(generate, "check", lambda: ["libs/skinny/pyproject.toml"])
        assert cli.main(["check"]) == 1
        err = capsys.readouterr().err
        assert "stale: libs/skinny/pyproject.toml" in err
        assert "python -m libs._tools generate" in err

    def test_generate_reports_when_nothing_changed(self, monkeypatch, capsys):
        monkeypatch.setattr(generate, "generate", lambda: [])
        assert cli.main(["generate"]) == 0
        assert "Up to date." in capsys.readouterr().out

    def test_generate_lists_what_it_wrote(self, monkeypatch, capsys):
        monkeypatch.setattr(generate, "generate", lambda: ["libs/README.md"])
        assert cli.main(["generate"]) == 0
        out = capsys.readouterr().out
        assert "wrote libs/README.md" in out and "1 file(s) written." in out

    def test_an_unknown_distribution_is_a_usage_error_not_a_traceback(self, capsys):
        assert cli.main(["stage", "scikit-plots-nope"]) == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ") and "scikit-plots-nope" in err

    def test_inconsistent_inputs_are_a_usage_error(self, monkeypatch, capsys):
        def refuse():
            raise ValueError("the two registries disagree")

        monkeypatch.setattr(generate, "generate", refuse)
        assert cli.main(["generate"]) == 2
        assert "error: the two registries disagree" in capsys.readouterr().err

    def test_stage_then_unstage_leaves_the_lib_directory_clean(self, capsys):
        root = staging.repo_root()
        lib = root / "libs" / "rank-bm25"
        before = sorted(p.name for p in lib.iterdir())
        try:
            assert cli.main(["stage", "scikit_plots_rank_bm25"]) == 0
            assert (lib / "scikitplot" / "rank_bm25" / "__init__.py").is_file()
            assert not (lib / "scikitplot" / "__init__.py").exists()
        finally:
            assert cli.main(["unstage", "scikit-plots-rank-bm25"]) == 0
        assert sorted(p.name for p in lib.iterdir()) == before
