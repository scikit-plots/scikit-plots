"""
The plotting API imports without its optional dependencies.

Notes
-----
**User notes.** ``pip install scikit-plots`` (no extras) must run the README
quick start, ``scikitplot.metrics.plot_precision_recall``. Parts built on
seaborn (``scikitplot.seaborn``, ``scikitplot.decile``) need the ``core``
extra, and the error says so.

**Developer notes.** Each check runs in a fresh interpreter with ``seaborn``
blocked (``sys.modules["seaborn"] = None`` makes ``import seaborn`` raise
``ModuleNotFoundError``), so the result does not depend on what is installed
or already imported. The vendored ``externals._seaborn`` imports the real
seaborn, so first-party code that every plot reaches (``scikitplot._compat``)
must not import it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

BLOCK = "import sys; sys.modules['seaborn'] = None\n"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCK + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "module",
    ["scikitplot._compat", "scikitplot._compat._sklearn", "scikitplot.api", "scikitplot.api.metrics"],
)
def test_plotting_modules_import_without_seaborn(module):
    r = _run(f"import importlib; importlib.import_module({module!r}); print('ok')")
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]


def test_quick_start_runs_without_seaborn():
    r = _run(
        """
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import scikitplot as sp
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.eye(3)[y] * 0.8 + 0.2 / 3
        sp.metrics.plot_precision_recall(y, p)
        print("ok")
        """
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]


def test_seaborn_parts_name_the_missing_dependency():
    r = _run(
        """
        import scikitplot as sp
        try:
            sp.seaborn
        except AttributeError as e:
            print(e)
        """
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "No module named 'seaborn" in r.stdout  # 'seaborn' or 'seaborn._core'
    assert "has no attribute 'seaborn'" in r.stdout
    # The extra is named when the installed metadata lists it (not in an
    # in-place build without metadata).
    from scikitplot._distributions import dependency_hint

    hint = dependency_hint("seaborn")
    if hint is not None:
        assert hint in r.stdout
