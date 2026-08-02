# scikitplot/annoy/_annoy/tests/test_no_orphan_pxi.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Guard against orphan Cython ``.pxi`` include files (CY-002).

A ``.pxi`` is only meaningful if some ``.pyx``/``.pxd`` actually pulls it in with a
Cython ``include`` directive. A ``.pxi`` that is copied into the build but never
included is dead weight whose constants drift from the real code (the removed
``annoylib.pxi`` declared ``DEFAULT_SCHEMA_VERSION = 1`` while the live code used
``0``). This test fails if any ``.pxi`` in the annoy Cython package is not consumed
by an ``include`` directive, and also fails if a stray ``annoylib.pxi`` reappears.
"""
import re
from pathlib import Path

_HERE = Path(__file__).resolve()
# .../scikitplot/annoy/_annoy/tests/this_file  ->  the _annoy package dir
_ANNOY_PKG = _HERE.parent.parent
_INCLUDE_RE = re.compile(r'^\s*include\s+["\']([^"\']+)["\']', re.MULTILINE)


def _included_pxi_names():
    names = set()
    for src in list(_ANNOY_PKG.glob("*.pyx*")) + list(_ANNOY_PKG.glob("*.pxd*")):
        text = src.read_text(encoding="utf-8", errors="ignore")
        for target in _INCLUDE_RE.findall(text):
            names.add(Path(target).name)
    return names


def test_the_stale_annoylib_pxi_is_gone():
    assert not (_ANNOY_PKG / "annoylib.pxi").exists(), (
        "annoylib.pxi was removed (CY-002, stale/unconsumed); do not reintroduce it"
    )


def test_no_orphan_pxi_in_annoy_package():
    included = _included_pxi_names()
    orphans = [p.name for p in _ANNOY_PKG.glob("*.pxi") if p.name not in included]
    assert not orphans, (
        f"orphan .pxi file(s) not consumed by any `include` directive: {orphans}. "
        "Either add an `include` in a .pyx/.pxd (and test it) or remove the file."
    )
