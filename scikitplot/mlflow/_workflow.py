# scikitplot/mlflow/_workflow.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_workflow.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ._project import dump_project_config_yaml, find_project_root, load_project_config
from ._project_session import session_from_file


@dataclass(frozen=True)
class WorkflowPaths:
    """
    Standardized project config paths used by the workflow.

    Attributes
    ----------
    project_root : pathlib.Path
        Project root directory.
    config_dir : pathlib.Path
        Configuration directory (typically `<project_root>/configs`).
    toml_path : pathlib.Path
        TOML config path.
    yaml_path : pathlib.Path
        YAML config path.
    """

    _project_root: Path
    _config_dir: Path
    _toml_path: Path
    _yaml_path: Path

    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return self._project_root

    @property
    def config_dir(self) -> Path:
        """Configuration directory (typically `<project_root>/configs`)."""
        return self._config_dir

    @property
    def toml_path(self) -> Path:
        """TOML config path."""
        return self._toml_path

    @property
    def yaml_path(self) -> Path:
        """YAML config path."""
        return self._yaml_path


def builtin_config_path(fmt: str = "toml") -> Path:
    """
    Return the path to the built-in demo config shipped with the package.

    Parameters
    ----------
    fmt : {"toml", "yaml"}, default="toml"
        Which demo config to return.

    Returns
    -------
    pathlib.Path
        Path to the built-in config file.

    Raises
    ------
    ValueError
        If `fmt` is not supported.
    FileNotFoundError
        If the built-in config file is missing (packaging error).
    """
    fmt_norm = fmt.strip().lower()
    if fmt_norm not in {"toml", "yaml"}:
        raise ValueError(f"fmt must be 'toml' or 'yaml', got {fmt!r}.")
    name = "mlflow.toml" if fmt_norm == "toml" else "mlflow.yaml"
    p = Path(__file__).resolve().parent / "_configs" / name
    if not p.exists():
        raise FileNotFoundError(f"Built-in demo config not found: {p}")
    return p


def default_project_paths(*, project_root: Path | None = None) -> WorkflowPaths:
    """
    Compute standard config file paths for a project.

    Attributes
    ----------
    project_root : pathlib.Path or None, default=None
        If None, uses :func:`scikitplot.mlflow.project.find_project_root`.

    Returns
    -------
    WorkflowPaths
        Standardized paths.
    """
    root = (project_root or find_project_root()).resolve()
    config_dir = root / "configs"
    return WorkflowPaths(
        _project_root=root,
        _config_dir=config_dir,
        _toml_path=config_dir / "mlflow.toml",
        _yaml_path=config_dir / "mlflow.yaml",
    )


def export_builtin_config(
    *,
    fmt: str = "toml",
    dest_path: Path | None = None,
    project_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Export the built-in demo config into the current project.

    Parameters
    ----------
    fmt : {"toml", "yaml"}, default="toml"
        Which built-in config format to export.
    dest_path : pathlib.Path or None, default=None
        Destination config path. If None, uses `<project_root>/configs/mlflow.<fmt>`.
    project_root : pathlib.Path or None, default=None
        Project root (used when dest_path is None).
    overwrite : bool, default=False
        If False, raise when destination exists.

    Returns
    -------
    pathlib.Path
        Destination path written.

    Raises
    ------
    FileExistsError
        If destination exists and overwrite is False.
    """
    src = builtin_config_path(fmt)
    paths = default_project_paths(project_root=project_root)
    if dest_path is None:
        dest_path = (
            paths.toml_path if fmt.strip().lower() == "toml" else paths.yaml_path
        )

    dest_path = dest_path.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Config already exists: {dest_path} (use overwrite=True)."
        )

    dest_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def patch_experiment_name_in_toml(path: Path, *, experiment_name: str) -> None:
    """
    Patch `experiment_name = ...` in a TOML config deterministically.

    Parameters
    ----------
    path : pathlib.Path
        TOML config path.
    experiment_name : str
        New experiment name.

    Raises
    ------
    ValueError
        If the TOML file does not contain an experiment_name assignment.
    """
    txt = path.read_text(encoding="utf-8")
    new = re.sub(
        r'(?m)^(\s*experiment_name\s*=\s*)".*?"\s*$',
        lambda m: f'{m.group(1)}"{experiment_name}"',
        txt,
        count=1,
    )
    if new == txt:
        raise ValueError(f"Could not patch experiment_name in TOML: {path}")
    path.write_text(new, encoding="utf-8")


def run_demo(
    *,
    profile: str = "local",
    project_root: Path | None = None,
    open_ui_seconds: float = 10.0,
    experiment_name: str | None = None,
    fmt: str = "toml",
    overwrite: bool = False,
) -> WorkflowPaths:
    """
    Run a beginner-friendly end-to-end demo workflow.

    The workflow demonstrates:
    1) Export demo config shipped in the library into *your* project.
    2) Optionally customize the config (experiment name).
    3) Start a session and log a "train" run.
    4) Export project config to YAML (backup / editable format).
    5) Re-open session from YAML and keep UI open briefly.
    6) Start a "predict" run.

    Parameters
    ----------
    profile : str, default="local"
        Profile name in the config file.
    project_root : pathlib.Path or None, default=None
        Project root. If None, auto-detected.
    open_ui_seconds : float, default=10.0
        How long to keep the UI reachable (sleep) in the "UI check" step.
        Set to 0 to skip sleeping.
    experiment_name : str or None, default=None
        If provided, patch the exported config to use this experiment name.
    fmt : {"toml", "yaml"}, default="toml"
        Which built-in config format to export initially.
    overwrite : bool, default=False
        Overwrite existing config files in the project.

    Returns
    -------
    WorkflowPaths
        Paths used in the workflow.

    Raises
    ------
    ImportError
        If MLflow is not installed when attempting to start a session.
    """
    paths = default_project_paths(project_root=project_root)

    # 1) Export built-in config into project.
    exported = export_builtin_config(
        fmt=fmt, project_root=paths.project_root, overwrite=overwrite
    )

    # 2) Optional config customization (beginner-friendly).
    if experiment_name is not None:
        if exported.suffix.lower() == ".toml":
            patch_experiment_name_in_toml(exported, experiment_name=experiment_name)
        else:
            try:
                import yaml  # type: ignore[]  # noqa: PLC0415
            except Exception as e:
                raise ImportError(
                    "YAML customization requires PyYAML. Install via: pip install pyyaml"
                ) from e
            data = yaml.safe_load(exported.read_text(encoding="utf-8")) or {}
            prof = data.get("profiles", {}).get(profile, {})
            sess = prof.get("session", {}) or {}
            sess["experiment_name"] = experiment_name
            data.setdefault("profiles", {}).setdefault(profile, {}).setdefault(
                "session", {}
            ).update(sess)
            exported.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    # 3) Train step.
    with session_from_file(exported, profile=profile) as mlflow:  # noqa: SIM117
        with mlflow.start_run():
            mlflow.log_param("phase", "train")

    # 4) Export to YAML in the project for easy editing / backup.
    cfg = load_project_config(exported, profile=profile)
    dump_project_config_yaml(cfg, paths.yaml_path)

    # 5) UI check using YAML.
    with session_from_file(paths.yaml_path, profile=profile) as mlflow:
        print(f"🌐 Open MLflow UI: {mlflow.ui_url}")  # noqa: T201
        if open_ui_seconds > 0:
            time.sleep(open_ui_seconds)

    # 6) Predict step.
    with session_from_file(paths.yaml_path, profile=profile) as mlflow:  # noqa: SIM117
        with mlflow.start_run(run_name="predict"):
            mlflow.log_param("phase", "predict")

    return paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scikitplot.mlflow",
        description="Beginner-friendly MLflow workflow demo for scikitplot.mlflow.",
    )
    p.add_argument(
        "--profile", default="local", help="Profile name in config (default: local)."
    )
    p.add_argument(
        "--fmt",
        default="toml",
        choices=["toml", "yaml"],
        help="Which built-in config format to export first.",
    )
    p.add_argument(
        "--project-root",
        default=None,
        help="Project root directory. If omitted, auto-detected via pyproject.toml/.git.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configs in <project_root>/configs.",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Optional experiment name to patch into the exported config.",
    )
    p.add_argument(
        "--open-ui-seconds",
        type=float,
        default=10.0,
        help="Seconds to keep the UI reachable in the demo (default: 10). Use 0 to skip.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Command-line arguments (excluding program name). If None, uses sys.argv[1:].

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    args = _build_parser().parse_args(argv)
    pr = Path(args.project_root).resolve() if args.project_root else None
    run_demo(
        profile=args.profile,
        project_root=pr,
        open_ui_seconds=float(args.open_ui_seconds),
        experiment_name=args.experiment_name,
        fmt=args.fmt,
        overwrite=bool(args.overwrite),
    )
    return 0


def workflow(
    *,
    profile: str = "local",
    open_ui_seconds: float = 0.0,
    experiment_name: str | None = None,
    fmt: str = "toml",
    overwrite: bool = False,
) -> None:
    """
    Run the built-in end-to-end MLflow workflow demo.

    This is a small, newbie-friendly helper that:

    1. Exports the library's built-in demo config (TOML or YAML) into your project.
    2. Runs a small "train" logging run.
    3. Optionally keeps the UI open for inspection.
    4. Runs a small "predict" logging run.

    Parameters
    ----------
    profile : str, default="local"
        Profile name inside the project config.
    open_ui_seconds : float, default=0.0
        If > 0, sleeps for this many seconds while the session is open, printing ``ui_url``.
    experiment_name : str or None, default=None
        If provided, patches the exported config to use this experiment name.
    fmt : {"toml", "yaml"}, default="toml"
        Which built-in demo config format to export.
    overwrite : bool, default=False
        Whether to overwrite existing project config files.

    Returns
    -------
    None

    See Also
    --------
    run_demo
        The implementation used by the CLI entry point.
    """
    return run_demo(
        profile=profile,
        open_ui_seconds=open_ui_seconds,
        experiment_name=experiment_name,
        fmt=fmt,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
