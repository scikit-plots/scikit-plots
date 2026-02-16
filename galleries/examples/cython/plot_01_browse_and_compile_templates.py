"""
Browse and compile templates
============================

.. currentmodule:: scikitplot.cython

This example shows how to:

- list available templates
- read a template source file
- compile a selected template
- inspect the compiled module
- handle missing templates or build prerequisites gracefully (docs-safe)

Notes
-----
Templates are shipped as package data under ``_templates/``. If your packaging
configuration excludes them, this example will print a clear message and exit.

Compilation requires build prerequisites; we check them first to keep gallery
builds stable.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

from typing import Any, Iterable

from scikitplot import cython

# %%


def _preview_text(s: str, n: int = 250) -> str:
    s2 = s.replace("\r\n", "\n")
    if len(s2) <= n:
        return s2
    return s2[:n] + "\n..."


def _safe_call_from_metadata(module: Any, demo_calls: Iterable[dict[str, Any]]) -> bool:
    """
    Execute strictly declared demo calls.

    The metadata schema is expected to include demo calls like:

    {"func": "square", "args": [12]}

    This function executes only declared calls, with no guessing.
    """
    any_called = False
    for call in demo_calls:
        func_name = call.get("func")
        args = call.get("args", [])
        kwargs = call.get("kwargs", {})
        if not isinstance(func_name, str):
            continue
        f = getattr(module, func_name, None)
        if not callable(f):
            continue
        try:
            out = f(*args, **kwargs)
        except Exception as e:
            print(f"Demo call failed: {module.__name__}.{func_name}{tuple(args)!r} -> {type(e).__name__}: {e}")
        else:
            print(f"Demo call: {module.__name__}.{func_name}(*{args!r}, **{kwargs!r}) -> {out!r}")
            any_called = True
    return any_called


def _public_names(module: Any, limit: int = 20) -> list[str]:
    names = sorted(
        n for n in getattr(module, "__dict__", {}).keys()
        if n and not n.startswith("_")
    )
    return names[:limit]


# %%
# List all available module tamplates
# -----------------------------------
# https://scikit-plots.github.io/dev/user_guide/cython/_templates/templates_index.html
templates = cython.list_templates()
print("Template count:", len(templates))
print("First 8:", templates[:8])


# %%
# Generate `python` Module from `cython`
# --------------------------------------

# Pick a template deterministically and safely.
if not templates:
    print(
        "No templates found.\n"
        "This usually means templates were not included as package data.\n"
        "Ensure your packaging includes scikitplot/cython/_templates/**."
    )
else:
    name = templates[0]
    path = cython.get_template_path(name)
    print("\nTemplate:", name)
    print("Path:", path)

    src = cython.read_template(name)
    print("\nSource preview:\n")
    print(_preview_text(src, 250))

    # Check build prerequisites (docs-safe).
    report = cython.check_build_prereqs(numpy=False)
    if not report.get('cython', {}).get('ok'):
        print("\nSkipping compilation because build prerequisites are missing.")
        problems = report.get("problems", [])
        if problems:
            print("Problems:", problems)
    else:
        # Compile the selected template.
        #
        # Prefer compile_template_result so we can display metadata.
        r = cython.compile_template_result(name, profile="fast-debug", verbose=0)
        print("\nBuildResult:")
        print("  module_name:", r.module_name)
        print("  key        :", r.key)
        print("  artifact   :", r.artifact_path)
        print("  used_cache :", r.used_cache)

        mod = r.module
        print("\nImported module:", mod.__name__)
        print("Public names:", _public_names(mod))

        # Strict demo execution:
        # 1) If template metadata declares demo calls, use them.
        # 2) Otherwise do NOT guess signatures; just print public names.
        info = getattr(cython, "read_template_info", None)
        if callable(info):
            meta = info(name)
            demo_calls = getattr(meta, "demo_calls", None)
            if demo_calls:
                print("\nRunning metadata-declared demo calls:")
                called = _safe_call_from_metadata(mod, demo_calls)
                if not called:
                    print("No declared calls could be executed.")
            else:
                print("\nNo demo_calls declared in template metadata; skipping execution.")
        else:
            print("\nTemplate metadata reader not available; skipping execution.")

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
