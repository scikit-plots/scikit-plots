"""Verify namespace portability, extension order, and dependency auto-loading.

This script never overwrites the historical ``verification.json`` beside it.
Pass ``--output PATH`` when a caller deliberately wants a new evidence file.
Exit code 2 means the Sphinx integration layer is unavailable in this environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate

ROOT = activate(__file__)
VIDEO_EXT = "_sphinxcontrib_youtube"
GRID_EXT = "_sphinx_gallery_grid"
CATALOG_EXT = "_sphinx_youtube_gallery"
COMPONENT_EXT = "_pydata_component_list"

VIDEO = ".. youtube:: https://www.youtube.com/watch?v=JXtISpdDPNY&t=90s\n"
GRID = ".. gallery-grid::\n\n   - title: Example\n     content: |\n       Content\n"
CAT = ".. youtube-gallery::\n\n   - id: JXtISpdDPNY\n     title: Example\n"


def _env(layout: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(layout) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _core_smoke(prefix: str, layout: Path) -> None:
    code = (
        f"from {prefix}._sphinx_youtube_core import reference as r; "
        f"from {prefix}._sphinx_youtube_core import video_options as o; "
        "assert r.parse_video_reference('JXtISpdDPNY').video_id == 'JXtISpdDPNY'; "
        "assert o.LEAF_VIDEO_SPEC['width']('640px') == '640px'"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], env=_env(layout), text=True, capture_output=True
    )
    if run.returncode:
        raise SystemExit(
            f"CORE_IMPORT_FAILED {prefix}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional path for new JSON evidence")
    args = parser.parse_args(argv)
    results: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        short = base / "short"
        long = base / "long"
        short.mkdir()
        (long / "scikitplot/_externals").mkdir(parents=True)
        shutil.copytree(ROOT, short / "_sphinx_ext")
        shutil.copytree(ROOT, long / "scikitplot/_externals/_sphinx_ext")
        (long / "scikitplot/__init__.py").touch()
        (long / "scikitplot/_externals/__init__.py").touch()

        layouts = [
            ("_sphinx_ext", short),
            ("scikitplot._externals._sphinx_ext", long),
        ]
        for prefix, layout in layouts:
            _core_smoke(prefix, layout)
            results.append({"namespace": prefix, "case": "youtube-core import", "result": "pass"})
        print("YOUTUBE_CORE_NAMESPACE_SMOKE_PASS (2 layouts)")

        missing = [name for name in ("sphinx", "docutils", "sphinx_design") if importlib.util.find_spec(name) is None]
        if missing:
            print("SPHINX_INTEGRATION_UNAVAILABLE: " + ", ".join(missing))
            return 2

        for prefix, layout in layouts:
            cases = [
                (VIDEO_EXT,),
                (GRID_EXT,),
                (CATALOG_EXT,),
                (COMPONENT_EXT,),
                *itertools.permutations((VIDEO_EXT, GRID_EXT, CATALOG_EXT)),
            ]
            for order in cases:
                number = len(results)
                src = base / f"src{number}"
                out = base / f"out{number}"
                src.mkdir()
                exts = [prefix + "." + name for name in order]
                (src / "conf.py").write_text(
                    "extensions = " + repr(exts) + "\nmaster_doc='index'\nproject='Verification'\n",
                    encoding="utf-8",
                )
                if CATALOG_EXT in order:
                    body = CAT + "\n" + VIDEO
                elif VIDEO_EXT in order:
                    body = VIDEO
                elif GRID_EXT in order:
                    body = GRID
                else:
                    body = "Plain document.\n"
                (src / "index.rst").write_text(
                    "Verification\n============\n\n" + body, encoding="utf-8"
                )
                run = subprocess.run(
                    [sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going", "-E", str(src), str(out)],
                    env=_env(layout), text=True, capture_output=True,
                )
                if run.returncode:
                    raise SystemExit(
                        f"FAILED {prefix} {order}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
                    )
                html = (out / "index.html").read_text(encoding="utf-8")
                if VIDEO_EXT in order or CATALOG_EXT in order:
                    assert "embed/JXtISpdDPNY?start=90" in html
                if CATALOG_EXT in order:
                    assert html.count("<iframe") == 2
                if GRID_EXT in order or CATALOG_EXT in order:
                    assert "sd-card" in html
                results.append({"namespace": prefix, "extensions": list(order), "result": "pass"})

        # Mixed names must be rejected before nodes/config can be registered.
        src = base / "mixed"
        src.mkdir()
        (src / "index.rst").write_text("Test\n====\n", encoding="utf-8")
        (src / "conf.py").write_text(
            "extensions=" + repr([
                "_sphinx_ext." + VIDEO_EXT,
                "scikitplot._externals._sphinx_ext." + VIDEO_EXT,
            ]),
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join((str(short), str(long), env.get("PYTHONPATH", "")))
        run = subprocess.run(
            [sys.executable, "-m", "sphinx", "-b", "html", str(src), str(base / "mixedout")],
            env=env, text=True, capture_output=True,
        )
        assert run.returncode and "Mixed scikit-plots extension namespaces" in run.stderr
        assert "already registered" not in run.stderr
        results.append({"case": "mixed namespaces rejected before registration", "result": "pass"})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
