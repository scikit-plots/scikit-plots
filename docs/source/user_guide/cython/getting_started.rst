.. _cython-getting-started:

Getting started
===============

.. code-block:: python

   from scikitplot.cython import compile_and_load

   module = compile_and_load("def square(int n):\n    return n * n")
   assert module.square(9) == 81

The convenience function returns the imported module. Use the corresponding
``*_result`` API when you need cache keys, artifact paths, diagnostics, or
fingerprints for automation.

Before a production build, inspect :func:`runtime_capabilities`,
:func:`compiler_capabilities`, or :func:`check_build_prereqs` rather than
assuming a compiler/Cython toolchain exists.
