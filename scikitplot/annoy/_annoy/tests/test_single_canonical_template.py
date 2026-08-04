# scikitplot/annoy/_annoy/tests/test_single_canonical_template.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Guard: exactly one authoritative Cython template controls packaged code (CY-001).

Two near-identical templates (`annoylib.pyx.in` built + `annoylib_pyx.in` not
built) let a fix land in the dead copy and look complete during review — the dead
copy was in fact stale (raw `free(error)` where the canonical had migrated to
ScopedError, and missing supported_dtypes/float80). The alternate was removed.
This test fails if any sibling `.pyx.in`/`.pxd.in` template reappears next to the
canonical in the built package directory (`_annoy/`), keeping a single source of
truth. Quarantined copies under `backup_template/` are ignored — they are not in
the built directory and are unambiguously non-authoritative by name.
"""
from pathlib import Path

_ANNOY = Path(__file__).resolve().parent.parent  # the built _annoy package dir


def test_only_the_canonical_pyx_template_exists():
    pyx_templates = sorted(p.name for p in _ANNOY.glob("*.pyx.in"))
    assert pyx_templates == ["annoylib.pyx.in"], (
        f"expected exactly the canonical annoylib.pyx.in, found {pyx_templates}. "
        "A duplicate .pyx.in template can hide fixes in a non-built copy (CY-001)."
    )


def test_no_underscore_pyx_alternate():
    # the specific CY-001 trap: annoylib_pyx.in (note the underscore, not '.pyx.in')
    assert not (_ANNOY / "annoylib_pyx.in").exists(), (
        "annoylib_pyx.in was removed (CY-001 fix-trap); do not reintroduce it"
    )


def test_single_pxd_template():
    pxd_templates = sorted(p.name for p in _ANNOY.glob("*.pxd.in"))
    assert pxd_templates == ["annoylib.pxd.in"], (
        f"expected exactly annoylib.pxd.in, found {pxd_templates}"
    )
