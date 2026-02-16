"""Build this package example using scikitplot.cython."""

from __future__ import annotations

from pathlib import Path

from scikitplot.cython import build_package_from_paths


def main() -> int:
    here = Path(__file__).resolve().parent
    modules = {"meanvar": str(here / "meanvar.pyx"), "hist": str(here / "hist.pyx")}
    build_package_from_paths(
        modules, package_name="scikitplot_pkg_stats_basic", verbose=1
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
