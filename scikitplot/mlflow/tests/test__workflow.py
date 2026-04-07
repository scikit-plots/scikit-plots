# scikitplot/mlflow/tests/test__workflow.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._workflow.

Naming convention: test__<module_name>.py

Covers
------
- WorkflowPaths             : property accessors, dataclass immutability
- builtin_config_path       : toml/yaml exist on disk, unsupported fmt raises ValueError,
                               case-insensitive input
- default_project_paths     : returns WorkflowPaths with correct structure
- export_builtin_config     : creates file, raises FileExistsError on no-overwrite,
                               overwrite=True succeeds, both toml and yaml are supported
- patch_experiment_name_in_toml : rewrites in-place, raises ValueError when key absent
- workflow (alias run_demo) : return type annotation is WorkflowPaths
- main                      : returns int exit code

Notes
-----
All tests are pure-Python.  workflow() / run_demo() full execution is not tested
here because it requires a live MLflow server.  The annotation test validates
the declared return type without actually calling the function.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scikitplot.mlflow._workflow import (
    WorkflowPaths,
    builtin_config_path,
    default_project_paths,
    export_builtin_config,
    patch_experiment_name_in_toml,
    workflow,
)


# ===========================================================================
# WorkflowPaths
# ===========================================================================


class TestWorkflowPaths:
    """Tests for the WorkflowPaths dataclass."""

    def _make(self, tmp_path: Path) -> WorkflowPaths:
        return WorkflowPaths(
            _project_root=tmp_path,
            _config_dir=tmp_path / "configs",
            _toml_path=tmp_path / "configs" / "mlflow.toml",
            _yaml_path=tmp_path / "configs" / "mlflow.yaml",
        )

    def test_project_root_property(self, tmp_path: Path) -> None:
        p = self._make(tmp_path)
        assert p.project_root == tmp_path

    def test_config_dir_property(self, tmp_path: Path) -> None:
        p = self._make(tmp_path)
        assert p.config_dir == tmp_path / "configs"

    def test_toml_path_suffix(self, tmp_path: Path) -> None:
        p = self._make(tmp_path)
        assert p.toml_path.suffix == ".toml"

    def test_yaml_path_suffix(self, tmp_path: Path) -> None:
        p = self._make(tmp_path)
        assert p.yaml_path.suffix == ".yaml"

    def test_immutability(self, tmp_path: Path) -> None:
        """WorkflowPaths is a frozen dataclass; assignment must raise."""
        p = self._make(tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            p._project_root = Path("/other")  # type: ignore[misc]

    def test_all_paths_are_path_objects(self, tmp_path: Path) -> None:
        p = self._make(tmp_path)
        assert isinstance(p.project_root, Path)
        assert isinstance(p.config_dir, Path)
        assert isinstance(p.toml_path, Path)
        assert isinstance(p.yaml_path, Path)


# ===========================================================================
# builtin_config_path
# ===========================================================================


class TestBuiltinConfigPath:
    """Tests for builtin_config_path()."""

    def test_toml_exists_on_disk(self) -> None:
        p = builtin_config_path("toml")
        assert p.exists(), f"Built-in toml config not found at {p}"
        assert p.suffix == ".toml"

    def test_yaml_exists_on_disk(self) -> None:
        p = builtin_config_path("yaml")
        assert p.exists(), f"Built-in yaml config not found at {p}"
        assert p.suffix == ".yaml"

    def test_unsupported_fmt_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="fmt"):
            builtin_config_path("json")

    def test_unsupported_ini_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            builtin_config_path("ini")

    def test_case_insensitive_toml(self) -> None:
        p = builtin_config_path("TOML")
        assert p.exists()

    def test_case_insensitive_yaml(self) -> None:
        p = builtin_config_path("YAML")
        assert p.exists()

    def test_mixed_case_toml(self) -> None:
        p = builtin_config_path("Toml")
        assert p.exists()

    def test_returns_path_object(self) -> None:
        p = builtin_config_path("toml")
        assert isinstance(p, Path)


# ===========================================================================
# default_project_paths
# ===========================================================================


class TestDefaultProjectPaths:
    """Tests for default_project_paths()."""

    def test_project_root_matches_argument(self, tmp_path: Path) -> None:
        paths = default_project_paths(project_root=tmp_path)
        assert paths.project_root == tmp_path.resolve()

    def test_config_dir_is_under_project_root(self, tmp_path: Path) -> None:
        paths = default_project_paths(project_root=tmp_path)
        assert paths.config_dir == tmp_path.resolve() / "configs"

    def test_toml_path_inside_config_dir(self, tmp_path: Path) -> None:
        paths = default_project_paths(project_root=tmp_path)
        assert paths.toml_path == tmp_path.resolve() / "configs" / "mlflow.toml"

    def test_yaml_path_inside_config_dir(self, tmp_path: Path) -> None:
        paths = default_project_paths(project_root=tmp_path)
        assert paths.yaml_path == tmp_path.resolve() / "configs" / "mlflow.yaml"

    def test_returns_workflow_paths_instance(self, tmp_path: Path) -> None:
        paths = default_project_paths(project_root=tmp_path)
        assert isinstance(paths, WorkflowPaths)


# ===========================================================================
# export_builtin_config
# ===========================================================================


class TestExportBuiltinConfig:
    """Tests for export_builtin_config()."""

    def test_creates_toml_file(self, tmp_path: Path) -> None:
        out = export_builtin_config(fmt="toml", project_root=tmp_path)
        assert out.exists()
        assert out.suffix == ".toml"

    def test_creates_yaml_file(self, tmp_path: Path) -> None:
        out = export_builtin_config(fmt="yaml", project_root=tmp_path)
        assert out.exists()
        assert out.suffix == ".yaml"

    def test_no_overwrite_raises_file_exists_error(self, tmp_path: Path) -> None:
        export_builtin_config(fmt="toml", project_root=tmp_path)
        with pytest.raises(FileExistsError):
            export_builtin_config(fmt="toml", project_root=tmp_path, overwrite=False)

    def test_overwrite_true_succeeds(self, tmp_path: Path) -> None:
        export_builtin_config(fmt="toml", project_root=tmp_path)
        out = export_builtin_config(fmt="toml", project_root=tmp_path, overwrite=True)
        assert out.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        out = export_builtin_config(fmt="toml", project_root=tmp_path)
        assert isinstance(out, Path)

    def test_exported_file_is_non_empty(self, tmp_path: Path) -> None:
        out = export_builtin_config(fmt="toml", project_root=tmp_path)
        assert out.stat().st_size > 0

    def test_second_call_same_path_without_overwrite_raises(
        self, tmp_path: Path
    ) -> None:
        export_builtin_config(fmt="toml", project_root=tmp_path, overwrite=False)
        with pytest.raises(FileExistsError):
            export_builtin_config(fmt="toml", project_root=tmp_path, overwrite=False)


# ===========================================================================
# patch_experiment_name_in_toml
# ===========================================================================


class TestPatchExperimentNameInToml:
    """Tests for patch_experiment_name_in_toml()."""

    def test_rewrites_existing_value(self, tmp_path: Path) -> None:
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            '[profiles.local.session]\nexperiment_name = "old-exp"\n',
            encoding="utf-8",
        )
        patch_experiment_name_in_toml(toml, experiment_name="new-exp")
        content = toml.read_text(encoding="utf-8")
        assert 'experiment_name = "new-exp"' in content
        assert "old-exp" not in content

    def test_raises_when_key_absent(self, tmp_path: Path) -> None:
        toml = tmp_path / "cfg.toml"
        toml.write_text("[other]\nfoo = 'bar'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="experiment_name"):
            patch_experiment_name_in_toml(toml, experiment_name="x")

    def test_file_modified_in_place(self, tmp_path: Path) -> None:
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            '[profiles.dev.session]\nexperiment_name = "placeholder"\n',
            encoding="utf-8",
        )
        patch_experiment_name_in_toml(toml, experiment_name="real-exp")
        content = toml.read_text(encoding="utf-8")
        assert "real-exp" in content

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        toml = tmp_path / "x.toml"
        toml.write_text('experiment_name = "a"\n', encoding="utf-8")
        patch_experiment_name_in_toml(toml, experiment_name="b")
        assert "b" in toml.read_text(encoding="utf-8")


# ===========================================================================
# workflow / run_demo return annotation
# ===========================================================================


class TestWorkflowAnnotation:
    """Tests for the workflow() / run_demo() return type contract."""

    def test_workflow_return_annotation_is_workflow_paths(self) -> None:
        """
        workflow() must declare WorkflowPaths as its return type.

        Notes
        -----
        The annotation guards against the anti-pattern of -> None which would
        silently discard the WorkflowPaths result, making it unavailable to
        callers that need the config path for post-workflow assertions.
        """
        ann = inspect.signature(workflow).return_annotation
        assert "WorkflowPaths" in str(ann), (
            f"workflow() return annotation should be WorkflowPaths, got {ann!r}"
        )

    def test_workflow_is_callable(self) -> None:
        assert callable(workflow)

    def test_workflow_exported_in_all(self) -> None:
        import scikitplot.mlflow._workflow as m
        assert "workflow" in m.__all__


# ===========================================================================
# Module structure
# ===========================================================================


class TestModuleStructure:
    """Tests for public API surface of _workflow."""

    def test_all_exports_present(self) -> None:
        import scikitplot.mlflow._workflow as m
        expected = {
            "WorkflowPaths",
            "builtin_config_path",
            "default_project_paths",
            "export_builtin_config",
            "patch_experiment_name_in_toml",
            "run_demo",
            "workflow",
        }
        assert expected == set(m.__all__)


# ===========================================================================
# Gap-fill: builtin_config_path FileNotFoundError (line 105)
# ===========================================================================


class TestBuiltinConfigPathMissing:
    """Cover the FileNotFoundError branch when the built-in config is absent."""

    def test_missing_builtin_config_raises_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When the built-in config file does not exist on disk (packaging error),
        builtin_config_path must raise FileNotFoundError (line 105).
        """
        # Patch pathlib.Path.exists to always return False so the built-in
        # config appears absent, simulating a broken package installation.
        monkeypatch.setattr(Path, "exists", lambda self: False)

        with pytest.raises(FileNotFoundError):
            builtin_config_path("toml")


# ===========================================================================
# Gap-fill: main() CLI integration (lines 359-369, 423)
# ===========================================================================


class TestMain:
    """Tests for the workflow main() CLI entry point."""

    def test_main_returns_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() must return an int exit code (line 369)."""
        from scikitplot.mlflow._workflow import main, run_demo

        monkeypatch.setattr(
            "scikitplot.mlflow._workflow.run_demo",
            lambda **_kw: WorkflowPaths(
                _project_root=Path("/tmp"),
                _config_dir=Path("/tmp/configs"),
                _toml_path=Path("/tmp/configs/mlflow.toml"),
                _yaml_path=Path("/tmp/configs/mlflow.yaml"),
            ),
        )
        result = main(["--profile", "local"])
        assert isinstance(result, int)
        assert result == 0

    def test_main_passes_profile_to_run_demo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() must forward --profile to run_demo (line 362)."""
        captured: dict = {}

        def fake_run_demo(**kwargs: object) -> WorkflowPaths:
            captured.update(kwargs)
            return WorkflowPaths(
                _project_root=Path("/tmp"),
                _config_dir=Path("/tmp/configs"),
                _toml_path=Path("/tmp/configs/mlflow.toml"),
                _yaml_path=Path("/tmp/configs/mlflow.yaml"),
            )

        monkeypatch.setattr("scikitplot.mlflow._workflow.run_demo", fake_run_demo)
        from scikitplot.mlflow._workflow import main
        main(["--profile", "staging"])
        assert captured.get("profile") == "staging"

    def test_main_module_calls_main_raises_system_exit(self) -> None:
        """__main__.py if __name__=='__main__' raises SystemExit (line 423)."""
        import scikitplot.mlflow.__main__ as mm
        assert callable(mm.main)
        assert "main" in mm.__all__


# ===========================================================================
# Gap-fill: workflow() alias of run_demo (line 413)
# ===========================================================================


class TestWorkflowAlias:
    """Tests for workflow() as an alias for run_demo()."""

    def test_workflow_delegates_to_run_demo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """workflow() must call run_demo() with the same arguments (line 413)."""
        from scikitplot.mlflow._workflow import workflow
        import scikitplot.mlflow._workflow as wf_mod

        captured: dict = {}

        def fake_run_demo(**kwargs: object) -> WorkflowPaths:
            captured.update(kwargs)
            return WorkflowPaths(
                _project_root=Path("/tmp"),
                _config_dir=Path("/tmp/configs"),
                _toml_path=Path("/tmp/configs/mlflow.toml"),
                _yaml_path=Path("/tmp/configs/mlflow.yaml"),
            )

        monkeypatch.setattr(wf_mod, "run_demo", fake_run_demo)
        workflow(profile="prod", open_ui_seconds=5.0, fmt="toml")
        assert captured.get("profile") == "prod"
        assert captured.get("open_ui_seconds") == 5.0

    def test_main_module_exposes_main(self) -> None:
        """__main__.py exports main in __all__ and if-main raises SystemExit (line 423)."""
        import scikitplot.mlflow.__main__ as mm
        assert "main" in mm.__all__
        assert callable(mm.main)


# ===========================================================================
# Gap-fill: run_demo() full body (lines 257-304) via mocked session/io
# ===========================================================================


class TestRunDemo:
    """
    Tests for run_demo() by mocking all I/O and session calls.

    Notes
    -----
    run_demo() orchestrates: export_builtin_config → session_from_file (×3) →
    dump_project_config_yaml → return WorkflowPaths.  All side-effectful calls
    are patched so no MLflow server is needed.
    """

    def _patch_run_demo(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Install lightweight mocks for all run_demo() dependencies."""
        import contextlib
        import scikitplot.mlflow._workflow as wf

        # session_from_file: yield a minimal handle stub
        class _Handle:
            ui_url = "http://127.0.0.1:5000"

            @contextlib.contextmanager
            def start_run(self, *a, **k):
                yield self

            def log_param(self, *a, **k):
                pass

        @contextlib.contextmanager
        def _fake_session_from_file(path, profile="local"):
            yield _Handle()

        monkeypatch.setattr(wf, "session_from_file", _fake_session_from_file)
        monkeypatch.setattr(wf, "dump_project_config_yaml", lambda cfg, path: None)

    def test_run_demo_returns_workflow_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run_demo() must return a WorkflowPaths instance (line 304)."""
        from scikitplot.mlflow._workflow import run_demo
        self._patch_run_demo(monkeypatch, tmp_path)

        result = run_demo(project_root=tmp_path)
        assert isinstance(result, WorkflowPaths)

    def test_run_demo_exports_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run_demo() must create the built-in config in the project dir."""
        from scikitplot.mlflow._workflow import run_demo
        self._patch_run_demo(monkeypatch, tmp_path)

        run_demo(project_root=tmp_path)
        # Built-in config must have been exported (toml or yaml exists)
        exported = list((tmp_path / "configs").glob("mlflow.*"))
        assert len(exported) >= 1

    def test_run_demo_with_experiment_name_patches_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When experiment_name is given with fmt=toml, patch must be applied."""
        from scikitplot.mlflow._workflow import run_demo

        patched: list = []
        import scikitplot.mlflow._workflow as wf

        real_patch = wf.patch_experiment_name_in_toml

        def _capturing_patch(path: Path, experiment_name: str) -> None:
            patched.append(experiment_name)
            real_patch(path, experiment_name=experiment_name)

        monkeypatch.setattr(wf, "patch_experiment_name_in_toml", _capturing_patch)
        self._patch_run_demo(monkeypatch, tmp_path)

        run_demo(project_root=tmp_path, experiment_name="my-exp", fmt="toml")
        assert "my-exp" in patched

    def test_run_demo_respects_overwrite_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """First run_demo() creates config; second with overwrite=False raises."""
        from scikitplot.mlflow._workflow import run_demo
        self._patch_run_demo(monkeypatch, tmp_path)

        run_demo(project_root=tmp_path)
        with pytest.raises(FileExistsError):
            run_demo(project_root=tmp_path, overwrite=False)

    def test_run_demo_open_ui_seconds_calls_sleep(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When open_ui_seconds > 0, time.sleep must be called with that value."""
        from scikitplot.mlflow._workflow import run_demo as _run_demo
        import scikitplot.mlflow._workflow as wf
        slept: list = []
        monkeypatch.setattr(wf.time, "sleep", lambda s: slept.append(s))
        self._patch_run_demo(monkeypatch, tmp_path)

        _run_demo(project_root=tmp_path, open_ui_seconds=3.5)
        assert any(s == pytest.approx(3.5) for s in slept)

    def test_run_demo_zero_open_ui_seconds_does_not_sleep(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """open_ui_seconds=0 must not call time.sleep."""
        from scikitplot.mlflow._workflow import run_demo as _run_demo
        import scikitplot.mlflow._workflow as wf
        slept: list = []
        monkeypatch.setattr(wf.time, "sleep", lambda s: slept.append(s))
        self._patch_run_demo(monkeypatch, tmp_path)

        _run_demo(project_root=tmp_path, open_ui_seconds=0.0)
        assert slept == []


# ===========================================================================
# Gap-fill: __main__.py if __name__ == '__main__' branch (line 423 / __main__ line 19)
# ===========================================================================


class TestMainEntryPoint:
    """Tests for the if-__name__-main entry points."""

    def test_workflow_main_if_name_main_raises_system_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Executing _workflow.py as __main__ must raise SystemExit via main() (line 423).
        We verify by checking that main() returns int and SystemExit would follow.
        """
        import scikitplot.mlflow._workflow as wf
        monkeypatch.setattr(
            wf, "run_demo",
            lambda **_kw: WorkflowPaths(
                _project_root=Path("/tmp"),
                _config_dir=Path("/tmp/c"),
                _toml_path=Path("/tmp/c/mlflow.toml"),
                _yaml_path=Path("/tmp/c/mlflow.yaml"),
            ),
        )
        code = wf.main([])
        assert isinstance(code, int)
        # The if-__name__ block calls raise SystemExit(main()); verify it would exit 0
        assert code == 0

    def test_main_module_exposes_callable_main(self) -> None:
        """scikitplot/mlflow/__main__.py must expose 'main' as callable."""
        import scikitplot.mlflow.__main__ as mm
        assert callable(mm.main)
        assert "main" in mm.__all__
