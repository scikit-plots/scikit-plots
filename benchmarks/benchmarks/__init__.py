# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Benchmarks of scikit-plots itself (not of its tests).

Notes
-----
**User notes.** Two ways to run them; both read these files unchanged::

    # No asv needed: the installed scikit-plots of this environment.
    python .github/scripts/ci_bench.py run --out build/bench/head.json
    python .github/scripts/ci_bench.py compare build/bench/base.json build/bench/head.json

    # asv, as SciPy and scikit-learn do: builds each commit in its own environment.
    cd benchmarks && asv continuous --factor 1.1 main HEAD

**Developer notes.** Write benchmarks in the subset of asv that both
runners support: module-level functions or methods of classes whose names
start with ``time_``, ``peakmem_``, ``track_`` or ``timeraw_``; class
attributes ``params``, ``param_names`` and ``timeout``; ``setup`` and
``teardown`` (raise ``NotImplementedError`` in ``setup`` to skip a
parameter combination). Data is generated from fixed seeds in
:mod:`benchmarks.common`, never downloaded, so a run needs no network.
"""
