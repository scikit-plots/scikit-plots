"""
Cython quickstart: compile_and_load
===================================

.. currentmodule:: scikitplot.cython

This example shows the minimal workflow:

1. Compile a small Cython snippet at runtime.
2. Call the compiled function immediately.
3. Inspect build/cache metadata (key, build dir, artifact path).
4. Demonstrate profiles, pin/alias, and restart-safe re-import by key.

Notes
-----
Runtime compilation requires:

- a working C/C++ compiler toolchain
- Python development headers (``Python.h``)
- the ``Cython`` package
- (optional) NumPy if ``numpy_support=True`` and your code uses NumPy C-API

If prerequisites are not available (common on doc builders),
this example prints a short diagnostic and exits gracefully.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
from __future__ import annotations

from scikitplot import cython

# %%

def _print_prereq_summary(report: dict) -> None:
    """Print a stable, user/dev friendly prereq summary."""
    print("Prereq ok:", report.get("ok", False))
    for k in ("cython", "setuptools", "numpy", "compiler", "python_headers"):
        if k in report:
            item = report.get(k, {})
            if isinstance(item, dict):
                ok = item.get("ok", False)
                ver = item.get("version", None)
                msg = item.get("message", None)
                s = f"  - {k}: ok={ok}"
                if ver:
                    s += f", version={ver}"
                if msg:
                    s += f", message={msg}"
                print(s)


# %%
# Basic environment diagnostic (safe to run on doc builders)
report = cython.check_build_prereqs()
_print_prereq_summary(report)

# %%
# Generate `python` Module from `python`
# --------------------------------------

if not report.get('cython', {}).get('ok'):
    print("\nCython runtime compilation prerequisites are missing.")
    # If your implementation provides a formatted string, prefer it.
    fmt = getattr(report, "format", None)
    if callable(fmt):
        print(fmt())
    else:
        problems = report.get("problems", [])
        if problems:
            print("Problems:", problems)
else:
    # ------------------------------------------------------------
    # 1) Compile a tiny Cython function and call it immediately.
    # ------------------------------------------------------------
    code_f = "def f(int n):\n    return n*n\n"

    try:
        m = cython.compile_and_load(code_f, profile="fast-debug")
    except Exception as e:
        # Docs-safe: show error without crashing the whole gallery build.
        print("\nCompilation failed (showing exception and exiting gracefully):")
        print(type(e).__name__ + ":", e)
    else:
        print("\nCompiled module name:", getattr(m, "__name__", "<unknown>"))
        print("f(10) =", m.f(10))

        # ------------------------------------------------------------
        # 2) If you need build metadata, use *_result API.
        # ------------------------------------------------------------
        code_g = "def g(int n):\n    return n+1\n"
        r = cython.compile_and_load_result(
            code_g,
            profile="fast-debug",
            numpy_support=True,
            numpy_required=False,
        )

        print("\nBuildResult (metadata):")
        print("  module_name :", r.module_name)
        print("  cache key   :", r.key)
        print("  build dir   :", r.build_dir)
        print("  artifact    :", r.artifact_path)
        print("  used_cache  :", r.used_cache)
        print("  created_utc :", r.created_utc)

        # Call g to prove the compiled module is live.
        print("g(10) =", r.module.g(10))

        # ------------------------------------------------------------
        # 3) Restart-safe workflow: import again by cache key.
        # ------------------------------------------------------------
        # This simulates "after kernel restart" behavior.
        try:
            m2 = cython.import_cached(r.key)
        except Exception as e:
            print("\nCould not re-import from cache by key:")
            print(type(e).__name__ + ":", e)
        else:
            print("\nRe-imported from cache key:")
            print("  module:", m2.__name__)
            print("  g(10) :", m2.g(10))

        # ------------------------------------------------------------
        # 4) Pin/Alias for friendly reuse.
        # ------------------------------------------------------------
        alias = "quickstart_g"
        try:
            cython.pin(r.key, alias=alias, overwrite=True)
            m3 = cython.import_pinned(alias)
        except Exception as e:
            print("\nPin/import by alias failed:")
            print(type(e).__name__ + ":", e)
        else:
            print("\nImported via alias:", alias)
            print("  module:", m3.__name__)
            print("  g(10) :", m3.g(10))

        # ------------------------------------------------------------
        # 5) Profile demonstration: release build gets a different key.
        # ------------------------------------------------------------
        r_rel = cython.compile_and_load_result(code_g, profile="release", numpy_support=True, numpy_required=False)
        print("\nProfile comparison (keys):")
        print("  fast-debug:", r.key)
        print("  release   :", r_rel.key)
        print("  keys differ:", r.key != r_rel.key)

        # Optional: print cache stats snapshot, if available.
        if hasattr(cython, "cache_stats"):
            print("\nCache stats snapshot:")
            print(cython.cache_stats())

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
