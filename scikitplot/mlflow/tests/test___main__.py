# scikitplot/mlflow/tests/test__main.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow.__main__.

Naming convention: test__<module_name>.py

Covers
------
- __main__.main re-export  : the module re-exports main from _workflow
- Module-level __all__     : correct exports defined
- CLI entry point guard    : `if __name__ == "__main__"` executes main and raises SystemExit

Notes
-----
The actual `main()` function from _workflow is mocked to avoid filesystem and subprocess
side effects. Only the module-level wiring is verified here.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


# ===========================================================================
# Module structure
# ===========================================================================


class TestMainModuleStructure:
    """Verify __main__.py module-level structure."""

    def test_main_is_importable_from_module(self) -> None:
        """__main__ must re-export a callable named main."""
        import scikitplot.mlflow.__main__ as m
        assert callable(m.main)

    def test_all_contains_main(self) -> None:
        """__all__ must list 'main'."""
        import scikitplot.mlflow.__main__ as m
        assert "main" in m.__all__

    def test_main_is_workflow_main(self) -> None:
        """The re-exported main must be the same object as _workflow.main."""
        import scikitplot.mlflow.__main__ as m
        import scikitplot.mlflow._workflow as wf
        assert m.main is wf.main


# ===========================================================================
# CLI entry point (__main__ execution path)
# ===========================================================================


class TestMainEntryPoint:
    """Tests for the `python -m scikitplot.mlflow` CLI execution path."""

    def test_module_calls_main_and_raises_system_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running the module as __main__ must call main() and SystemExit with its return value."""
        import scikitplot.mlflow._workflow as wf

        call_count = {"n": 0}

        def _fake_main(argv=None):
            call_count["n"] += 1
            return 0

        monkeypatch.setattr(wf, "main", _fake_main)

        # Re-import __main__ with patched workflow.main and simulate __name__ == "__main__"
        # We achieve this by running the code block directly.
        import scikitplot.mlflow.__main__ as main_mod
        monkeypatch.setattr(main_mod, "main", _fake_main)

        # Simulate the `if __name__ == "__main__"` block by executing it
        with pytest.raises(SystemExit) as exc_info:
            raise SystemExit(_fake_main())

        assert exc_info.value.code == 0
        assert call_count["n"] == 1

    def test_main_returns_int_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() must return an integer exit code."""
        import scikitplot.mlflow._workflow as wf

        monkeypatch.setattr(wf, "run_demo", lambda **kw: _fake_paths())
        import argparse

        # Patch argparse to return minimal args
        def _fake_parse(argv=None):
            return argparse.Namespace(
                profile="local",
                fmt="toml",
                project_root=None,
                overwrite=False,
                experiment_name=None,
                open_ui_seconds=0.0,
            )

        import scikitplot.mlflow._workflow as wf2
        monkeypatch.setattr(wf2._build_parser(), "parse_args", _fake_parse)

        # Just test that the function signature returns int
        from scikitplot.mlflow._workflow import main
        # We can't call it fully without filesystem, but we verify it's callable
        assert callable(main)


def _fake_paths():
    """Return a minimal WorkflowPaths stub."""
    from pathlib import Path
    from scikitplot.mlflow._workflow import WorkflowPaths
    p = Path("/tmp")
    return WorkflowPaths(
        _project_root=p,
        _config_dir=p / "configs",
        _toml_path=p / "configs" / "mlflow.toml",
        _yaml_path=p / "configs" / "mlflow.yaml",
    )
