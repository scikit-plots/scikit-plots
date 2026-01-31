"""
Workflow templates (train / hpo / predict) + CLI entry template
===============================================================

.. currentmodule:: scikitplot.cython

Workflow templates are shipped as package data under
``_templates/workflow/``. They are designed to be copied into any project
directory and executed directly.

Each script follows canonical CLI structure:

- ``parse_args(argv=None)``
- ``main(argv=None) -> int``
- ``if __name__ == "__main__": raise SystemExit(main())``

This example lists workflows and copies one workflow to a temporary folder.
It also shows how to compile a Cython extension (a ``.so``/``.pyd``) and import
it from the copied workflow directory:

1) Single-module compile → copy artifact next to workflow → import by path.
2) Multi-module package build → copy package folder → import via sys.path.

Notes
-----
Compilation requires a working compiler toolchain and Python headers. During
Sphinx-Gallery builds, we skip compilation cleanly if prerequisites are missing.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from scikitplot import cython

# %%
# Generate Workflow Package `wf_ext_pkg`
# --------------------------------------
# Copy template workflow in local or use exist from local folder.

workflows = getattr(cython, "list_workflows", lambda: [])()
print("Workflows:", workflows)

if not workflows:
    print("No workflow templates bundled in this build.")
else:
    wf = workflows[0]
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        dest = Path(tmp) / "workflow_copy"
        cython.copy_workflow(wf, dest_dir=dest, overwrite=True)

        print("Copied workflow to:", dest)
        print("Files:", sorted(p.name for p in dest.rglob("*.py")))

        # Show how users would run the CLI entrypoint.
        cli = dest / "cli.py"
        print("\nRun:")
        print(f"  python {cli} train --help")
        print(f"  python {cli} hpo --help")
        print(f"  python {cli} predict --help")

        # ------------------------------------------------------------
        # Optional: compile and import a .so/.pyd into this copied folder
        # ------------------------------------------------------------
        # prereqs = getattr(cython, "check_build_prereqs", lambda **_: {"ok": False})()
        report = cython.check_build_prereqs(numpy=False)
        if not report.get('cython', {}).get('ok'):
            print("\nBuild prerequisites not available; skipping compilation demo.")
            # Print a compact diagnostic (safe for docs output)
            problems = report.get("problems", [])
            if problems:
                print("Prereq problems:", problems)
        else:
            # 1) Single-module extension compile -> copy artifact -> import by path
            code = (
                "def square(int n):\n"
                "    return n*n\n"
            )

            res = cython.compile_and_load_result(
                code,
                module_name="wf_ext_square",
                profile="fast-debug",
                numpy_support=True,
                numpy_required=False,
            )

            artifact_copy = dest / res.artifact_path.name
            artifact_copy.write_bytes(res.artifact_path.read_bytes())

            # Import the compiled extension from the copied workflow directory.
            m = cython.import_artifact_path(artifact_copy, module_name=res.module_name)
            print("\nSingle-module extension imported from copied folder:")
            print("  m.square(12) =", m.square(12))

            # 2) Multi-module package build -> copy package folder -> import via sys.path
            # This creates a package with two extension modules:
            #   wf_ext_pkg.alpha
            #   wf_ext_pkg.beta
            pkg = cython.build_package_from_code_result(
                package_name="wf_ext_pkg",
                modules={
                    "alpha": "def inc(int n):\n    return n + 1\n",
                    "beta":  "def dec(int n):\n    return n - 1\n",
                },
                profile="fast-debug",
                numpy_support=True,
                numpy_required=False,
            )

            # Copy the built package directory into the workflow folder.
            pkg_src_dir = pkg.build_dir / pkg.package_name
            pkg_dst_dir = dest / pkg.package_name

            # Strict copy (no heuristics): recreate files.
            pkg_dst_dir.mkdir(parents=True, exist_ok=True)
            for p in pkg_src_dir.rglob("*"):
                rel = p.relative_to(pkg_src_dir)
                out = pkg_dst_dir / rel
                if p.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.write_bytes(p.read_bytes())

            # Now import it like a normal package by adding `dest` to sys.path.
            sys.path.insert(0, str(dest))
            try:
                import wf_ext_pkg.alpha as alpha  # type: ignore
                import wf_ext_pkg.beta as beta    # type: ignore
                print("\nPackage extension imported from copied folder:")
                print("  alpha.inc(10) =", alpha.inc(10))
                print("  beta.dec(10)  =", beta.dec(10))
            finally:
                # Clean up sys.path modification
                if sys.path and sys.path[0] == str(dest):
                    sys.path.pop(0)

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
