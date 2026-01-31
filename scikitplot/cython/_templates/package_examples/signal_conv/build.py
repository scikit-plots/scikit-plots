"""Build this package example using scikitplot.cython."""

from __future__ import annotations

from pathlib import Path

from scikitplot.cython import build_package_from_paths


def main() -> int:
    here = Path(__file__).resolve().parent
    modules = {
        "window": str(here / "window.pyx"),
        "convolve1d": str(here / "convolve1d.pyx"),
    }
    build_package_from_paths(
        modules, package_name="scikitplot_pkg_signal_conv", verbose=1
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
