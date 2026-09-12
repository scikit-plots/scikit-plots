"""YouTube-specific maintenance wrapper over the common family gate."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SUBSYSTEM = HERE.parent.parent
FAMILY = SUBSYSTEM.parent
CORE_TOOLS = FAMILY / "_maintenance_core" / "tools"
if str(CORE_TOOLS) not in sys.path:
    sys.path.insert(0, str(CORE_TOOLS))
from check_subsystem import check_subsystem
from paths import discover_runtime_sphinx_ext

_FORBIDDEN_CORE_IMPORTS = {"sphinx", "docutils", "sphinx_design", "myst_parser"}


def _core_dependency_errors(core: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(core.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as exc:
            errors.append(f"canonical core syntax error in {path.name}: {exc}")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level >= 2:
                    errors.append(
                        f"canonical core imports sibling runtime package: {path.name}:{node.lineno}"
                    )
                elif node.module:
                    names.append(node.module)
            for name in names:
                top = name.split(".", 1)[0]
                if top in _FORBIDDEN_CORE_IMPORTS:
                    errors.append(
                        f"canonical core imports optional Sphinx dependency {top}: {path.name}:{getattr(node, 'lineno', '?')}"
                    )
    return errors


def main() -> int:
    errors = check_subsystem(SUBSYSTEM / "MAINTENANCE.json")
    try:
        runtime = discover_runtime_sphinx_ext(__file__)
    except FileNotFoundError:
        runtime = None
    if runtime is not None:
        core = runtime / "_sphinx_youtube_core"
        gallery = runtime / "_sphinx_youtube_gallery"
        leaf = runtime / "_sphinxcontrib_youtube" / "utils.py"
        if not core.is_dir():
            errors.append("missing canonical runtime provider core")
        else:
            errors.extend(_core_dependency_errors(core))
        if leaf.is_file() and "_sphinx_youtube_gallery" in leaf.read_text(encoding="utf-8"):
            errors.append("leaf player still depends upward on _sphinx_youtube_gallery")
        for facade in (gallery / "reference.py", gallery / "_video_options.py"):
            if facade.is_file() and "_sphinx_youtube_core" not in facade.read_text(encoding="utf-8"):
                errors.append(
                    f"compatibility facade does not re-export canonical core: {facade.name}"
                )
    if errors:
        print("_sphinx_youtube_gallery maintenance drift: FAIL")
        for error in errors:
            print(f" - {error}")
        return 1
    print("_sphinx_youtube_gallery maintenance drift: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
