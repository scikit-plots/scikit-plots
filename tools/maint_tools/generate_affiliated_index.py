# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the affiliated/partial-distributions documentation index.

The page is registry-driven.  Package ownership comes from
``scikitplot/_distributions.py`` and packaging facts come from
``libs/_tools/registry.py``.  The prose around the generated registry lives in
``docs/source/affiliated/index.rst.in``.

This helper deliberately loads those files without importing ``scikitplot`` so
it remains useful before the project is built or installed.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

_MARKER = ".. GENERATED_PARTIAL_DISTRIBUTIONS"


def repo_root() -> Path:
    """Return the repository root containing this maintenance helper."""
    root = Path(__file__).resolve().parents[2]
    required = (
        root / "scikitplot" / "_distributions.py",
        root / "libs" / "_tools" / "registry.py",
        root / "docs" / "source" / "affiliated" / "index.rst.in",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "not a complete scikit-plots checkout; missing: "
            + ", ".join(str(path) for path in missing)
        )
    return root


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(root: Path):
    distributions = _load_module(
        root / "scikitplot" / "_distributions.py", "_skplt_affiliated_distributions"
    )
    registry = _load_module(root / "libs" / "_tools" / "registry.py", "_skplt_lib_registry")
    packages = registry.by_distribution()

    declared = [dist.name for dist in distributions.DISTRIBUTIONS]
    configured = list(packages)
    if declared != configured:
        raise ValueError(
            "partial-distribution registries disagree: "
            f"ownership={declared!r}, packaging={configured!r}"
        )
    return distributions, registry, packages


def _python_requirement(package) -> str:
    return package.requires_python or "same as scikit-plots core"


def _build_kind(package) -> str:
    return "compiled" if package.extensions else "pure Python"


def _surfaces(dist) -> tuple[str, ...]:
    names: list[str] = []
    if "__init__.py" in dist.files:
        names.append("scikitplot")
    for tree in dist.trees:
        names.append("scikitplot." + tree.replace("/", "."))
    return tuple(names)


def _render_registry(root: Path) -> str:
    distributions, registry, packages = _metadata(root)
    lines: list[str] = []
    lines.extend(
        [
            "Partial-distribution registry",
            "-----------------------------",
            "",
            f"Current project-owned partial distributions: **{len(distributions.DISTRIBUTIONS)}**.",
            "",
            "The registry below is generated from the same ownership and packaging",
            "metadata used to build the distributions.  A new ``libs/`` distribution",
            "therefore cannot be silently omitted from this page.",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 22 24 12 12 44",
            "",
            "   * - Distribution",
            "     - Import surface",
            "     - Python",
            "     - Build",
            "     - Purpose",
        ]
    )
    for dist in distributions.DISTRIBUTIONS:
        package = packages[dist.name]
        surfaces = ", ".join(f"``{name}``" for name in _surfaces(dist))
        lines.extend(
            [
                f"   * - ``{dist.name}``",
                f"     - {surfaces}",
                f"     - {_python_requirement(package)}",
                f"     - {_build_kind(package)}",
                f"     - {dist.summary}",
            ]
        )

    lines.extend(
        [
            "",
            "Install from a checkout",
            "-----------------------",
            "",
            "Each entry under ``libs/`` is a buildable distribution.  From a repository",
            "checkout, install only the part you need with::",
            "",
            "   python -m pip install ./libs/<directory>",
            "",
            "The current directory mapping is:",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 34 24",
            "",
            "   * - Distribution",
            "     - Repository directory",
        ]
    )
    for dist in distributions.DISTRIBUTIONS:
        directory = registry.directory_of(dist.name)
        lines.extend(
            [
                f"   * - ``{dist.name}``",
                f"     - ``libs/{directory}/``",
            ]
        )

    lines.extend(
        [
            "",
            "Packaging infrastructure",
            "------------------------",
            "",
            "``libs/_tools`` is not an installable partial distribution.  It is the",
            "repository tooling that validates ownership, generates package metadata,",
            "builds the partial distributions, and verifies them in installed-wheel",
            "environments.  Maintainers can inspect the canonical registry with::",
            "",
            "   python -m libs._tools list",
            "",
            "and validate generated packaging files with::",
            "",
            "   python -m libs._tools check",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render(root: Path | None = None) -> str:
    """Render ``docs/source/affiliated/index.rst`` without writing it."""
    root = repo_root() if root is None else Path(root).resolve()
    template_path = root / "docs" / "source" / "affiliated" / "index.rst.in"
    template = template_path.read_text(encoding="utf-8")
    if template.count(_MARKER) != 1:
        raise ValueError(f"{template_path} must contain exactly one {_MARKER!r} marker")
    return template.replace(_MARKER, _render_registry(root).rstrip())


def _paths(root: Path) -> tuple[Path, Path]:
    directory = root / "docs" / "source" / "affiliated"
    return directory / "index.rst.in", directory / "index.rst"


def check(root: Path | None = None) -> tuple[bool, str]:
    """Return ``(is_current, expected_source)`` for the generated page."""
    root = repo_root() if root is None else Path(root).resolve()
    _template, output = _paths(root)
    expected = render(root)
    current = output.read_text(encoding="utf-8") if output.is_file() else ""
    return current == expected, expected


def _diff(current: str, expected: str, output: Path) -> str:
    return "".join(
        difflib.unified_diff(
            current.splitlines(True),
            expected.splitlines(True),
            fromfile=str(output),
            tofile=str(output) + " (generated)",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_affiliated_index.py",
        description=(
            "Generate the docs affiliated/partial-distributions index from the "
            "canonical libs registries."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Fail when the generated index is stale.")
    check_p.add_argument("--json", action="store_true", help="Emit machine-readable output.")

    render_p = sub.add_parser("render", help="Render the expected index to stdout.")
    render_p.add_argument("--json", action="store_true", help="Emit metadata as JSON.")

    sync_p = sub.add_parser(
        "sync", help="Preview synchronization; add --apply to write index.rst."
    )
    sync_p.add_argument("--apply", action="store_true", help="Write the generated index.")
    sync_p.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout=None,
    stderr=None,
) -> int:
    """Run the maintenance CLI and return a process exit code."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        root = repo_root()
        _template, output = _paths(root)
        current = output.read_text(encoding="utf-8") if output.is_file() else ""
        expected = render(root)
        stale = current != expected

        if args.command == "check":
            if args.json:
                json.dump({"path": str(output.relative_to(root)), "stale": stale}, stdout)
                stdout.write("\n")
            elif stale:
                stderr.write(f"stale: {output.relative_to(root)}\n")
            else:
                stdout.write("Affiliated index is up to date.\n")
            return 1 if stale else 0

        if args.command == "render":
            if args.json:
                distributions, _registry, _packages = _metadata(root)
                json.dump(
                    {
                        "path": str(output.relative_to(root)),
                        "distributions": [dist.name for dist in distributions.DISTRIBUTIONS],
                        "source": expected,
                    },
                    stdout,
                )
                stdout.write("\n")
            else:
                stdout.write(expected)
            return 0

        if args.command == "sync":
            if args.apply:
                if stale:
                    output.write_text(expected, encoding="utf-8")
                action = "written" if stale else "unchanged"
                if args.json:
                    json.dump(
                        {"path": str(output.relative_to(root)), "action": action}, stdout
                    )
                    stdout.write("\n")
                else:
                    stdout.write(f"{action}: {output.relative_to(root)}\n")
                return 0

            diff = _diff(current, expected, output) if stale else ""
            if args.json:
                json.dump(
                    {
                        "path": str(output.relative_to(root)),
                        "stale": stale,
                        "diff": diff,
                    },
                    stdout,
                )
                stdout.write("\n")
            elif stale:
                stdout.write(diff)
            else:
                stdout.write("Affiliated index is already synchronized.\n")
            return 0

        raise AssertionError(f"unhandled command {args.command!r}")
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
