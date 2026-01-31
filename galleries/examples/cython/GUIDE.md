Cython learning guide for scikitplot.cython templates
====================================================

This bundle is meant to be used with Sphinx-Gallery and as a hands-on learning path.

Beginner path (Python → Cython)
-------------------------------
1. Start with ``plot_00_quickstart_compile_and_load.py``.
2. Browse templates in ``plot_01_browse_and_compile_templates.py``.
3. Learn build profiles in ``plot_02_build_profiles.py``.
4. Learn cache reuse and pins in ``plot_03_cache_and_restart_reuse.py`` and ``plot_04_pin_alias.py``.

Key beginner concepts
---------------------
- ``def`` vs ``cdef`` vs ``cpdef``:
  - ``def``: Python-callable, Python semantics, slower loops unless typed.
  - ``cdef``: C-level only, fastest, not directly callable from Python.
  - ``cpdef``: hybrid; callable from Python and from Cython efficiently.
- Typed memoryviews: ``double[:]`` and ``int[:]`` let you write fast loops without NumPy.
- The GIL:
  - Pure Python objects require the GIL.
  - Numeric loops over typed buffers can often run ``nogil`` (advanced).

Intermediate path (performance + correctness)
---------------------------------------------
- Prefer typed memoryviews and local ``cdef`` variables.
- Use ``Py_ssize_t`` for indices.
- Avoid Python attribute access inside tight loops.
- Avoid implicit Python conversions in hot paths.

Advanced path (C/C++ integration)
---------------------------------
See:
- ``plot_06_multifile_support_files.py`` for `.pxi` + `.h` patterns.
- ``plot_07_cpp_mode_basics.py`` for `language="c++"` and `libcpp.vector`.
- Add safety directives explicitly when needed:
  - ``boundscheck=False``, ``wraparound=False`` (only when you guarantee correctness)
  - Keep these as explicit choices, not defaults.

Doc-build friendliness
----------------------
These examples call ``check_build_prereqs()`` first. If compilation prerequisites are not present,
they print a diagnostic and exit cleanly so Sphinx-Gallery builds do not fail.

Workflow templates
------------------
``plot_09_workflow_templates_cli.py`` demonstrates copying an executable-friendly workflow
template and shows how to run its CLI entry script.
