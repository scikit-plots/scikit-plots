# scikitplot/annoy/_annoy/tests/test_pxd_abi_policy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Guard for the installed-.pxd ABI policy (CY-003).

The generated ``annoylib.pxd`` is installed (devel tag) and exposes the full
internal template-dispatch surface (100+ AnnoyIndex typedefs) that regenerates
whenever the dtype/index/metric matrix changes. The decision is that this cimport
surface is a PRIVATE implementation detail with no ABI-stability guarantee. This
test keeps the policy statement present in the template (hence the generated
``.pxd``), so the packaged file always documents its own instability.
"""
from pathlib import Path

_PXD_IN = Path(__file__).resolve().parent.parent / "annoylib.pxd.in"


def test_pxd_declares_internal_no_abi_stability():
    text = _PXD_IN.read_text(encoding="utf-8", errors="ignore")
    assert _PXD_IN.exists(), "annoylib.pxd.in missing"
    # the policy must be stated so the generated/installed .pxd carries it
    assert "NO ABI stability guarantee" in text
    assert "INTERNAL / PRIVATE cimport surface" in text
    # and it must point users at the supported Python API
    assert "scikitplot.annoy.Index" in text
