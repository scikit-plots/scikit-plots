"""
Pin/Alias: stable handles for cached builds
===========================================

.. currentmodule:: scikitplot.cython

Pins provide a human-friendly alias for a cached build key:

- ``pin(key, alias="fast_fft")``
- ``import_pinned("fast_fft")``

Pins are stored per cache directory (portable and deterministic).

What this example demonstrates
------------------------------
1) Compile a snippet and pin its cache key under an alias.
2) Import again using the alias (without remembering the key).
3) Show strict overwrite behavior.
4) List pins and remove a pin (unpin).
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
from __future__ import annotations

from scikitplot import cython

# %%
# Generate `python` Module from `python`
# --------------------------------------

report = cython.check_build_prereqs(numpy=False)

if not report.get('cython', {}).get('ok'):
    print("Skipping compilation because build prerequisites are missing.")
    problems = report.get("problems", [])
    if problems:
        print("Problems:", problems)
else:
    # 1) Compile a small module and capture its cache key.
    r = cython.compile_and_load_result("def f(int n):\n    return n*n\n", profile="fast-debug", verbose=0)
    key = r.key
    alias = "demo_square"

    print("Built key:", key)
    print("Pin alias:", alias)

    # 2) Pin the key (overwrite=True makes the example rerunnable).
    cython.pin(key, alias=alias, overwrite=True)

    # 3) Import via alias and verify correctness.
    if hasattr(cython, "import_pinned_result"):
        r2 = cython.import_pinned_result(alias)
        m = r2.module
        print("\nImported via alias (result API):")
        print("  module_name:", r2.module_name)
        print("  key        :", r2.key)
        print("  used_cache :", r2.used_cache)
    else:
        m = cython.import_pinned(alias)
        print("\nImported via alias (module API):", m.__name__)

    out = m.f(9)
    print("demo_square.f(9) =", out)
    print("Expected:", 81)
    print("Correct:", out == 81)

    # 4) Show pins dictionary (alias -> key).
    pins = cython.list_pins()
    print("\nPins:", pins)
    print("Alias resolves to this key:", pins.get(alias))

    # 5) Demonstrate strict overwrite behavior:
    #    Attempt to pin a different key to the same alias without overwrite.
    r_other = cython.compile_and_load_result("def f(int n):\n    return n*n + 1\n", profile="fast-debug", verbose=0)

    print("\nStrict collision demo:")
    try:
        cython.pin(r_other.key, alias=alias, overwrite=False)
        print("Unexpected: pin succeeded without overwrite.")
    except Exception as e:
        print("As expected, pin without overwrite failed:")
        print(" ", type(e).__name__ + ":", e)

    # Now overwrite intentionally.
    cython.pin(r_other.key, alias=alias, overwrite=True)
    m_over = cython.import_pinned(alias)
    print("After overwrite, demo_square.f(9) =", m_over.f(9))

    # 6) Unpin and verify alias is removed.
    if hasattr(cython, "unpin"):
        cython.unpin(alias)
        print("\nAfter unpin, pins:", cython.list_pins())
    else:
        print("\nNo unpin() API found in this build; skipping unpin demo.")

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
